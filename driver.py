# driver.py
# Stephan DeLuna

import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingest.extract import loadDocument
from ingest.clean import cleanContent, joinTokens
from chunk.chunk import chunkText, organizeChunks
from model.summarize import summarizeWithLoadedModel, loadSummarizer
from model.repeat import repeatSummaries
from report.output import exportSurveyPaper

# - - - - - - - - - - #

INITIAL_CHAR_LIMIT = 8000
REPEAT_CHAR_LIMIT = 20000

# - - - - - - - - - - #

# load simple .env key=value pairs without making python-dotenv required
def loadEnvFile(envPath=".env"):
	if envPath == None or envPath.strip() == "":
		return

	if not os.path.exists(envPath):
		return

	with open(envPath, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()

			if line == "" or line.startswith("#"):
				continue

			if line.startswith("export "):
				line = line[7:].strip()

			if "=" not in line:
				continue

			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip()

			if key == "":
				continue

			if " #" in value:
				value = value.split(" #", 1)[0].strip()

			if len(value) >= 2:
				if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
					value = value[1:-1]

			if key not in os.environ:
				os.environ[key] = value

def getEnvValue(names, defaultValue=None):
	for name in names:
		value = os.getenv(name)
		if value != None and value.strip() != "":
			return value.strip()

	return defaultValue

# - - - - - - - - - - #

def strToBool(value):
	if isinstance(value, bool):
		return value

	value = value.lower().strip()

	if value in ["true", "t", "yes", "y", "1"]:
		return True
	if value in ["false", "f", "no", "n", "0"]:
		return False

	raise argparse.ArgumentTypeError("Boolean value expected: true or false")

def parseList(value):
	if value == None:
		return None

	value = value.strip()

	if value == "" or value.lower() == "none":
		return None

	items = []
	for item in value.split(","):
		item = item.strip()
		if item != "":
			items.append(item)

	if len(items) == 0:
		return None

	return items

def parseIntList(value):
	items = parseList(value)

	if items == None:
		return None

	intItems = []
	for item in items:
		intItems.append(int(item))

	return intItems

def makeSafeName(name):
	name = os.path.splitext(os.path.basename(name))[0]
	name = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
	name = name.strip("_")

	if name == "":
		name = "document"

	return name

def makeDisplayName(path):
	baseName = os.path.splitext(os.path.basename(path))[0]
	baseName = baseName.replace("_", " ").replace("-", " ")
	return baseName.title()

def makeUniqueNames(inputPaths):
	nameCounts = {}
	uniqueNames = []

	for inputPath in inputPaths:
		baseName = makeSafeName(inputPath)

		if baseName not in nameCounts:
			nameCounts[baseName] = 0
			uniqueNames.append(baseName)
		else:
			nameCounts[baseName] += 1
			uniqueNames.append(baseName + "_" + str(nameCounts[baseName]))

	return uniqueNames

# - - - - - - - - - - #

def parseArgs():
	loadEnvFile(".env")
	envModelName = getEnvValue(["MODEL_NAME", "MODEL", "HF_MODEL_NAME"], "Qwen/Qwen2.5-7B-Instruct")

	parser = argparse.ArgumentParser(description="LLM Summarization Pipeline")

	# input / output
	parser.add_argument(
		"--input",
		required=True,
		nargs="+",
		help="Path to one or more input files (txt/pdf/docx)"
	)

	parser.add_argument(
		"--output",
		default="storage",
		help="Output directory (default: storage)"
	)

	parser.add_argument(
		"--outputName",
		default=None,
		help="Base name for final output files"
	)

	parser.add_argument(
		"--title",
		default=None,
		help="Title for final survey output"
	)

	parser.add_argument(
		"--authorName",
		default="Stephan DeLuna",
		help="Author name for final survey output"
	)

	parser.add_argument(
		"--abstractText",
		default="",
		help="Optional abstract text for final survey output"
	)

	parser.add_argument(
		"--introductionText",
		default=None,
		help="Optional introduction text for final survey output"
	)

	parser.add_argument(
		"--conclusionText",
		default=None,
		help="Optional conclusion text for final survey output"
	)

	parser.add_argument(
		"--skipWaits",
		type=strToBool,
		default=False,
		help="Skip interactive pause steps (default: false)"
	)

	# cleanContent arguments
	parser.add_argument("--removeCitation", type=strToBool, default=True)
	parser.add_argument("--removeUnicode", type=strToBool, default=True)
	parser.add_argument("--removeSymbols", type=strToBool, default=True)
	parser.add_argument("--removeStutters", type=strToBool, default=True)
	parser.add_argument("--removeFillers", type=strToBool, default=True)
	parser.add_argument("--excludeRepeatWords", type=strToBool, default=True)
	parser.add_argument("--removeWordMorphemes", type=strToBool, default=False)
	parser.add_argument("--stutterList", default=None, help="Comma-separated custom stutter words")
	parser.add_argument("--fillerList", default=None, help="Comma-separated custom filler words")
	parser.add_argument("--morphemeList", default=None, help="Comma-separated custom morphemes")

	# chunkText arguments
	parser.add_argument("--charLimit", type=int, default=INITIAL_CHAR_LIMIT)
	parser.add_argument("--sectionType", default="chapter", choices=["chapter", "page", "paragraph"])
	parser.add_argument("--limitPercent", type=float, default=0.9)
	parser.add_argument("--overlapSize", type=int, default=200)

	# organizeChunks arguments
	parser.add_argument(
		"--orderType",
		default="firstToLast",
		choices=["firstToLast", "lastToFirst", "lastToFirstBySection", "firstToLastByTop"]
	)

	# loadSummarizer arguments
	parser.add_argument(
		"--model",
		default=envModelName,
		help="Hugging Face model name. Defaults to MODEL_NAME, MODEL, or HF_MODEL_NAME in .env; otherwise Qwen/Qwen2.5-7B-Instruct"
	)
	parser.add_argument("--deviceId", type=int, default=0)
	parser.add_argument("--parallelDevices", default=None, help="Comma-separated GPU IDs for file-level parallel summarization, example: 0,1")
	parser.add_argument("--parallelMode", default="file", choices=["none", "file"])
	parser.add_argument("--use4Bit", type=strToBool, default=True)
	parser.add_argument("--useYarn", type=strToBool, default=False)

	# summarizeWithLoadedModel / summarizeChunk arguments
	parser.add_argument("--promptText", default=None)
	parser.add_argument("--sentenceLimit", type=int, default=10)
	parser.add_argument("--maxNewTokens", type=int, default=400)
	parser.add_argument("--failOnNonEnglish", type=strToBool, default=False)
	parser.add_argument("--maxRetries", type=int, default=2)
	parser.add_argument("--forceEnglishRepair", type=strToBool, default=True)
	parser.add_argument("--doSample", type=strToBool, default=False)
	parser.add_argument("--temperature", type=float, default=0.3)
	parser.add_argument("--topP", type=float, default=0.9)

	# survey synthesis / generated report section arguments
	parser.add_argument("--surveyPromptText", default=None)
	parser.add_argument("--generateIntroduction", type=strToBool, default=True)
	parser.add_argument("--generateConclusion", type=strToBool, default=True)
	parser.add_argument("--introductionPromptText", default=None)
	parser.add_argument("--conclusionPromptText", default=None)
	parser.add_argument("--reportSectionSentenceLimit", type=int, default=6)
	parser.add_argument("--reportSectionMaxNewTokens", type=int, default=260)

	# repeatSummaries / reduction arguments
	parser.add_argument("--repeatCharLimit", type=int, default=REPEAT_CHAR_LIMIT)
	parser.add_argument("--repeatLimitPercent", type=float, default=0.9)
	parser.add_argument("--repeatOverlapSize", type=int, default=200)
	parser.add_argument("--repeatMaxRounds", type=int, default=5)
	parser.add_argument("--repeatSentenceLimit", type=int, default=6)
	parser.add_argument("--repeatMaxNewTokens", type=int, default=220)

	return parser.parse_args()

# - - - - - - - - - - #

def waitForNextStep(stepName, skipWaits=False):
	if skipWaits:
		return

	while True:
		userInput = input(f"\nPress space then Enter to continue to {stepName}: ")
		if userInput == " ":
			break

def saveText(path, text):
	with open(path, "w", encoding="utf-8") as f:
		f.write(text)

def saveChunks(path, chunks):
	with open(path, "w", encoding="utf-8") as f:
		for k, v in chunks.items():
			f.write(k + "\n")
			f.write("-" * 40 + "\n")
			f.write(v + "\n\n")

def saveSummaries(path, summaries):
	with open(path, "w", encoding="utf-8") as f:
		for k, v in summaries.items():
			f.write(k + "\n")
			f.write("-" * 40 + "\n")
			f.write(v + "\n\n")

# - - - - - - - - - - #

def getDeviceIds(args):
	deviceIds = parseIntList(args.parallelDevices)

	if deviceIds == None or len(deviceIds) == 0:
		deviceIds = [args.deviceId]

	return deviceIds

def getSurveyPromptText(args, isMultiFile):
	if args.surveyPromptText != None:
		return args.surveyPromptText

	if isMultiFile:
		return (
			"Write the survey synthesis in English only. "
			"Tie the provided document summaries together into one cohesive survey-style overview. "
			"Identify shared themes, major differences, and the overall relationship between the sources. "
			"Do not simply list each document one after another. "
			"Do not invent facts, citations, results, dates, or claims that are not supported by the provided summaries. "
			"Use a clear academic tone. "
			"Keep the synthesis under 8 sentences."
		)

	return args.promptText

def getDefaultIntroductionText(fileInfos):
	if len(fileInfos) == 1:
		return "This report summarizes the provided document using a multi-stage LLM pipeline."

	return "This report summarizes multiple provided documents using a multi-stage LLM pipeline and synthesizes their shared themes into a survey-style overview."

def getDefaultConclusionText(fileInfos):
	if len(fileInfos) == 1:
		return "This summary condenses the full document into a concise overview."

	return "This summary condenses the full document set into a concise overview while preserving individual source summaries for review."

def getIntroductionPromptText(args):
	if args.introductionPromptText != None:
		return args.introductionPromptText

	return (
		"Write an introduction paragraph for a survey-style report in English only. "
		"Base it only on the provided survey synthesis and individual document summaries. "
		"Explain what the report covers, what connects the documents, and why the document set is useful to review. "
		"Do not mention the prompt. "
		"Do not invent facts, citations, dates, or claims. "
		"Do not use bullet points or headings."
	)

def getConclusionPromptText(args):
	if args.conclusionPromptText != None:
		return args.conclusionPromptText

	return (
		"Write a conclusion paragraph for a survey-style report in English only. "
		"Base it only on the provided survey synthesis and individual document summaries. "
		"Tie the documents together, restate the main overall takeaway, and briefly note what the source set shows collectively. "
		"Do not mention the prompt. "
		"Do not invent facts, citations, dates, or claims. "
		"Do not use bullet points or headings."
	)

def buildReportContext(fileInfos, combinedSummaryText):
	contextText = "Survey Synthesis:\n"
	contextText += combinedSummaryText.strip() + "\n\n"
	contextText += "Individual Document Summaries:\n"

	for fileInfo in fileInfos:
		contextText += fileInfo["displayName"] + ":\n"
		contextText += fileInfo["finalSummaryText"].strip() + "\n\n"

	return contextText.strip()

# - - - - - - - - - - #

def processInputFile(inputPath, outputDir, outputBaseName, args):
	cleanedOutputPath = os.path.join(outputDir, f"{outputBaseName}_cleaned.txt")
	chunkOutputPath = os.path.join(outputDir, f"{outputBaseName}_chunks.txt")

	# --- INGEST ---
	rawText = loadDocument(inputPath)
	tokens, removedCitations, removedMorphemes = cleanContent(
		rawText,
		removeCitation=args.removeCitation,
		removeUnicode=args.removeUnicode,
		removeSymbols=args.removeSymbols,
		removeStutters=args.removeStutters,
		removeFillers=args.removeFillers,
		excludeRepeatWords=args.excludeRepeatWords,
		removeWordMorphemes=args.removeWordMorphemes,
		stutterList=parseList(args.stutterList),
		fillerList=parseList(args.fillerList),
		morphemeList=parseList(args.morphemeList)
	)
	cleanedText = joinTokens(tokens)

	saveText(cleanedOutputPath, cleanedText)

	print("\n--- AFTER INGEST ---")
	print("Input:", inputPath)
	print("Length:", len(cleanedText))
	print("Removed citations:", len(removedCitations))
	print("Removed morpheme words:", len(removedMorphemes))
	print("First 500 Char Cleaned")
	print(cleanedText[:500])

	# --- CHUNK ---
	chunks = chunkText(
		cleanedText,
		charLimit=args.charLimit,
		sectionType=args.sectionType,
		limitPercent=args.limitPercent,
		overlapSize=args.overlapSize
	)

	chunks = organizeChunks(chunks, orderType=args.orderType)

	saveChunks(chunkOutputPath, chunks)

	print("\n--- AFTER CHUNK ---")
	print("Input:", inputPath)
	print("Chunk count:", len(chunks))

	print("\n--- CHUNK DETAILS ---")
	for chunkName, chunkTextValue in chunks.items():
		print(f"\n{chunkName}")
		print(f"Length: {len(chunkTextValue)} chars")

	return {
		"inputPath": inputPath,
		"outputBaseName": outputBaseName,
		"displayName": makeDisplayName(inputPath),
		"cleanedOutputPath": cleanedOutputPath,
		"chunkOutputPath": chunkOutputPath,
		"chunks": chunks
	}

def summarizeOneFile(fileInfo, tokenizer, model, outputDir, args, deviceId=None):
	if deviceId == None:
		deviceId = args.deviceId

	summaryOutputPath = os.path.join(outputDir, f"{fileInfo['outputBaseName']}_summaries.txt")
	finalSummaryOutputPath = os.path.join(outputDir, f"{fileInfo['outputBaseName']}_final.txt")

	chunkInput = {}
	for k, v in fileInfo["chunks"].items():
		if k not in ["title", "front front"]:
			chunkInput[k] = v

	if len(chunkInput) == 0:
		chunkInput = fileInfo["chunks"]

	# --- FIRST SUMMARY ---
	summaries = summarizeWithLoadedModel(
		chunkInput,
		tokenizer,
		model,
		deviceId=deviceId,
		promptText=args.promptText,
		sentenceLimit=args.sentenceLimit,
		maxNewTokens=args.maxNewTokens,
		failOnNonEnglish=args.failOnNonEnglish,
		maxRetries=args.maxRetries,
		forceEnglishRepair=args.forceEnglishRepair,
		doSample=args.doSample,
		temperature=args.temperature,
		topP=args.topP
	)

	saveSummaries(summaryOutputPath, summaries)

	print("\n--- AFTER FIRST SUMMARY ---")
	print("Input:", fileInfo["inputPath"])
	print("GPU:", deviceId)
	print("Summary count:", len(summaries))

	# --- REPEAT / REDUCE ---
	_, finalSummary = repeatSummaries(
		summaries,
		tokenizer,
		model,
		promptText=args.promptText,
		sentenceLimit=args.repeatSentenceLimit,
		maxNewTokens=args.repeatMaxNewTokens,
		deviceId=deviceId,
		charLimit=args.repeatCharLimit,
		limitPercent=args.repeatLimitPercent,
		overlapSize=args.repeatOverlapSize,
		maxRounds=args.repeatMaxRounds,
		failOnNonEnglish=args.failOnNonEnglish,
		maxRetries=args.maxRetries,
		forceEnglishRepair=args.forceEnglishRepair,
		doSample=args.doSample,
		temperature=args.temperature,
		topP=args.topP
	)

	finalText = finalSummary.get("paper", "")

	print("\n--- FINAL SUMMARY ---")
	print("Input:", fileInfo["inputPath"])
	print("GPU:", deviceId)
	print(finalText)

	saveText(finalSummaryOutputPath, finalText)

	fileInfo["summaryOutputPath"] = summaryOutputPath
	fileInfo["finalSummaryOutputPath"] = finalSummaryOutputPath
	fileInfo["finalSummaryText"] = finalText
	fileInfo["summaryDeviceId"] = deviceId

	return fileInfo

def summarizeFileBatch(batchItems, workerInfo, outputDir, args):
	results = []

	for index, fileInfo in batchItems:
		print("\n--- GPU WORKER ---")
		print("GPU:", workerInfo["deviceId"])
		print("Input:", fileInfo["inputPath"])

		result = summarizeOneFile(
			fileInfo,
			workerInfo["tokenizer"],
			workerInfo["model"],
			outputDir,
			args,
			deviceId=workerInfo["deviceId"]
		)

		results.append((index, result))

	return results

def loadModelWorkers(modelName, deviceIds, args):
	workers = []

	for deviceId in deviceIds:
		print("\n--- LOAD MODEL ---")
		print("Model:", modelName)
		print("GPU:", deviceId)

		tokenizer, model = loadSummarizer(
			modelName=modelName,
			deviceId=deviceId,
			use4Bit=args.use4Bit,
			useYarn=args.useYarn
		)

		workers.append({
			"deviceId": deviceId,
			"tokenizer": tokenizer,
			"model": model
		})

	return workers

def summarizeFiles(fileInfos, workers, outputDir, args):
	if len(workers) == 1 or args.parallelMode == "none" or len(fileInfos) == 1:
		workerInfo = workers[0]
		summarizedFileInfos = []

		for fileInfo in fileInfos:
			summarizedFileInfos.append(
				summarizeOneFile(
					fileInfo,
					workerInfo["tokenizer"],
					workerInfo["model"],
					outputDir,
					args,
					deviceId=workerInfo["deviceId"]
				)
			)

		return summarizedFileInfos

	assignments = []
	for workerInfo in workers:
		assignments.append([])

	for i, fileInfo in enumerate(fileInfos):
		workerIndex = i % len(workers)
		assignments[workerIndex].append((i, fileInfo))

	orderedResults = [None] * len(fileInfos)

	print("\n--- PARALLEL SUMMARIZATION ---")
	print("Mode: file-level data parallelism")
	print("GPU devices:", [workerInfo["deviceId"] for workerInfo in workers])

	with ThreadPoolExecutor(max_workers=len(workers)) as executor:
		futures = []

		for workerIndex, workerInfo in enumerate(workers):
			if len(assignments[workerIndex]) > 0:
				future = executor.submit(
					summarizeFileBatch,
					assignments[workerIndex],
					workerInfo,
					outputDir,
					args
				)
				futures.append(future)

		for future in as_completed(futures):
			batchResults = future.result()

			for index, result in batchResults:
				orderedResults[index] = result

	return orderedResults

# - - - - - - - - - - #

def buildCombinedSummary(fileInfos, tokenizer, model, args, deviceId=None):
	if deviceId == None:
		deviceId = args.deviceId

	if len(fileInfos) == 1:
		return fileInfos[0]["finalSummaryText"]

	combinedInput = {}

	for i, fileInfo in enumerate(fileInfos):
		combinedInput["document" + str(i + 1)] = (
			"Document: " + fileInfo["displayName"] + "\n\n" + fileInfo["finalSummaryText"]
		)

	_, combinedFinal = repeatSummaries(
		combinedInput,
		tokenizer,
		model,
		promptText=getSurveyPromptText(args, True),
		sentenceLimit=args.repeatSentenceLimit,
		maxNewTokens=args.repeatMaxNewTokens,
		deviceId=deviceId,
		charLimit=args.repeatCharLimit,
		limitPercent=args.repeatLimitPercent,
		overlapSize=args.repeatOverlapSize,
		maxRounds=args.repeatMaxRounds,
		failOnNonEnglish=args.failOnNonEnglish,
		maxRetries=args.maxRetries,
		forceEnglishRepair=args.forceEnglishRepair,
		doSample=args.doSample,
		temperature=args.temperature,
		topP=args.topP
	)

	return combinedFinal.get("paper", "")

def generateReportParagraph(sourceText, promptText, tokenizer, model, args, deviceId):
	sectionInput = {
		"report": sourceText
	}

	_, finalSection = repeatSummaries(
		sectionInput,
		tokenizer,
		model,
		promptText=promptText,
		sentenceLimit=args.reportSectionSentenceLimit,
		maxNewTokens=args.reportSectionMaxNewTokens,
		deviceId=deviceId,
		charLimit=args.repeatCharLimit,
		limitPercent=args.repeatLimitPercent,
		overlapSize=args.repeatOverlapSize,
		maxRounds=args.repeatMaxRounds,
		failOnNonEnglish=args.failOnNonEnglish,
		maxRetries=args.maxRetries,
		forceEnglishRepair=args.forceEnglishRepair,
		doSample=args.doSample,
		temperature=args.temperature,
		topP=args.topP
	)

	return finalSection.get("paper", "")

def buildGeneratedReportSections(fileInfos, combinedSummaryText, tokenizer, model, args, deviceId):
	reportContext = buildReportContext(fileInfos, combinedSummaryText)

	if args.introductionText != None:
		introduction = args.introductionText
	elif args.generateIntroduction:
		print("\n--- GENERATING INTRODUCTION ---")
		introduction = generateReportParagraph(
			reportContext,
			getIntroductionPromptText(args),
			tokenizer,
			model,
			args,
			deviceId
		)
	else:
		introduction = getDefaultIntroductionText(fileInfos)

	if args.conclusionText != None:
		conclusion = args.conclusionText
	elif args.generateConclusion:
		print("\n--- GENERATING CONCLUSION ---")
		conclusion = generateReportParagraph(
			reportContext,
			getConclusionPromptText(args),
			tokenizer,
			model,
			args,
			deviceId
		)
	else:
		conclusion = getDefaultConclusionText(fileInfos)

	if introduction.strip() == "":
		introduction = getDefaultIntroductionText(fileInfos)

	if conclusion.strip() == "":
		conclusion = getDefaultConclusionText(fileInfos)

	return introduction, conclusion

def exportFinalOutput(fileInfos, combinedSummaryText, introduction, conclusion, outputDir, outputBaseName, args):
	finalSummaryOutputPath = os.path.join(outputDir, f"{outputBaseName}_final.txt")
	surveyTxtPath = os.path.join(outputDir, f"{outputBaseName}_survey.txt")
	surveyDocxPath = os.path.join(outputDir, f"{outputBaseName}_survey.docx")
	surveyPdfPath = os.path.join(outputDir, f"{outputBaseName}_survey.pdf")

	saveText(finalSummaryOutputPath, combinedSummaryText)

	if args.title != None:
		title = args.title
	elif len(fileInfos) == 1:
		title = fileInfos[0]["displayName"]
	else:
		title = "Multi File Survey Report"

	summaryDict = {
		"paper": combinedSummaryText
	}

	if len(fileInfos) > 1:
		for fileInfo in fileInfos:
			summaryDict[fileInfo["displayName"]] = fileInfo["finalSummaryText"]

	exportSurveyPaper(
		surveyTxtPath,
		"txt",
		title,
		args.authorName,
		introduction,
		summaryDict,
		conclusion,
		abstractText=args.abstractText
	)

	exportSurveyPaper(
		surveyDocxPath,
		"docx",
		title,
		args.authorName,
		introduction,
		summaryDict,
		conclusion,
		abstractText=args.abstractText
	)

	exportSurveyPaper(
		surveyPdfPath,
		"pdf",
		title,
		args.authorName,
		introduction,
		summaryDict,
		conclusion,
		abstractText=args.abstractText
	)

	return {
		"finalSummaryOutputPath": finalSummaryOutputPath,
		"surveyTxtPath": surveyTxtPath,
		"surveyDocxPath": surveyDocxPath,
		"surveyPdfPath": surveyPdfPath
	}

# - - - - - - - - - - #

def main():
	args = parseArgs()

	inputPaths = args.input
	outputDir = args.output
	modelName = args.model
	deviceIds = getDeviceIds(args)

	print("\n--- CONFIG ---")
	print("Model:", modelName)
	print("Output directory:", outputDir)
	print("Input files:", len(inputPaths))
	print("GPU devices:", deviceIds)

	os.makedirs(outputDir, exist_ok=True)

	uniqueNames = makeUniqueNames(inputPaths)

	if args.outputName != None:
		finalOutputBaseName = makeSafeName(args.outputName)
	elif len(inputPaths) == 1:
		finalOutputBaseName = uniqueNames[0]
	else:
		finalOutputBaseName = "multi_file_summary"

	fileInfos = []

	for i, inputPath in enumerate(inputPaths):
		fileInfo = processInputFile(
			inputPath,
			outputDir,
			uniqueNames[i],
			args
		)
		fileInfos.append(fileInfo)

	waitForNextStep("model loading", args.skipWaits)

	workers = loadModelWorkers(modelName, deviceIds, args)

	waitForNextStep("summarization", args.skipWaits)

	summarizedFileInfos = summarizeFiles(fileInfos, workers, outputDir, args)

	waitForNextStep("combined survey synthesis", args.skipWaits)

	combinedWorker = workers[0]

	combinedSummaryText = buildCombinedSummary(
		summarizedFileInfos,
		combinedWorker["tokenizer"],
		combinedWorker["model"],
		args,
		deviceId=combinedWorker["deviceId"]
	)

	waitForNextStep("introduction and conclusion", args.skipWaits)

	introduction, conclusion = buildGeneratedReportSections(
		summarizedFileInfos,
		combinedSummaryText,
		combinedWorker["tokenizer"],
		combinedWorker["model"],
		args,
		deviceId=combinedWorker["deviceId"]
	)

	waitForNextStep("output", args.skipWaits)

	outputPaths = exportFinalOutput(
		summarizedFileInfos,
		combinedSummaryText,
		introduction,
		conclusion,
		outputDir,
		finalOutputBaseName,
		args
	)

	print("\n--- OUTPUT COMPLETE ---")
	print("Saved to:", outputDir)
	print("Final summary:", outputPaths["finalSummaryOutputPath"])
	print("Survey txt:", outputPaths["surveyTxtPath"])
	print("Survey docx:", outputPaths["surveyDocxPath"])
	print("Survey pdf:", outputPaths["surveyPdfPath"])

# - - - - - - - - - - #

if __name__ == "__main__":
	main()

