# driver.py
# Stephan DeLuna

import os
import re
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingest.extract import loadDocument
from ingest.clean import cleanContent, joinTokens
from chunk.chunk import chunkText, organizeChunks
from chunk.section import countWords
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
		"--dataFileName",
		default="data.txt",
		help="Name of evaluator data file saved in the output directory (default: data.txt)"
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

def countTotalWords(content):
	wordCounts = countWords(content)
	totalWords = 0

	for word in wordCounts:
		totalWords += wordCounts[word]

	return totalWords

def makeThroughputMetric(runName, inputPath, deviceId, elapsedSeconds, inputChars, inputWords, chunkCount, outputChars, outputWords):
	if elapsedSeconds <= 0:
		elapsedSeconds = 0.000001

	return {
		"runName": runName,
		"inputPath": inputPath,
		"deviceId": deviceId,
		"elapsedSeconds": elapsedSeconds,
		"inputChars": inputChars,
		"inputWords": inputWords,
		"chunkCount": chunkCount,
		"outputChars": outputChars,
		"outputWords": outputWords,
		"charsPerSecond": inputChars / elapsedSeconds,
		"wordsPerSecond": inputWords / elapsedSeconds,
		"chunksPerSecond": chunkCount / elapsedSeconds
	}

def writeLoggedTextBlock(f, label, textValue):
	textValue = str(textValue)

	f.write(label + "\n")
	f.write("-" * 40 + "\n")
	f.write("Characters: " + str(len(textValue)) + "\n")
	f.write("Words: " + str(countTotalWords(textValue)) + "\n")
	f.write(textValue.strip() + "\n\n")

def writeLoggedSummaryDict(f, label, summaryDict):
	f.write(label + "\n")
	f.write("-" * 40 + "\n")

	if summaryDict == None or len(summaryDict) == 0:
		f.write("No entries recorded.\n\n")
		return

	for summaryName, summaryText in summaryDict.items():
		summaryText = str(summaryText)
		f.write("Entry: " + str(summaryName) + "\n")
		f.write("Characters: " + str(len(summaryText)) + "\n")
		f.write("Words: " + str(countTotalWords(summaryText)) + "\n")
		f.write(summaryText.strip() + "\n\n")

def writeOneReductionEntry(f, entry):
	f.write("Run: " + str(entry.get("runName", "")) + "\n")
	f.write("Stage: " + str(entry.get("stage", "")) + "\n")
	f.write("Round: " + str(entry.get("round", "")) + "\n")
	f.write("GPU: " + str(entry.get("deviceId", "")) + "\n")

	if entry.get("sectionName", None) != None:
		f.write("Section: " + str(entry.get("sectionName", "")) + "\n")

	if entry.get("inputKeys", None) != None:
		f.write("Input keys: " + ", ".join([str(x) for x in entry.get("inputKeys", [])]) + "\n")

	f.write("\n")

	if entry.get("inputSummaries", None) != None:
		writeLoggedSummaryDict(f, "Input Summaries Before Reduction", entry.get("inputSummaries", {}))

	if entry.get("combinedInput", None) != None:
		writeLoggedTextBlock(f, "Combined Input Text Before Reduction", entry.get("combinedInput", ""))

	if entry.get("inputText", None) != None:
		writeLoggedTextBlock(f, "Merged Input Text Before Reduction", entry.get("inputText", ""))

	if entry.get("rechunkedInput", None) != None:
		writeLoggedSummaryDict(f, "Rechunked Input Pieces Before Reduction", entry.get("rechunkedInput", {}))

	if entry.get("outputSummaries", None) != None:
		writeLoggedSummaryDict(f, "Output Summaries After Reduction", entry.get("outputSummaries", {}))

	if entry.get("mergedOutputText", None) != None:
		writeLoggedTextBlock(f, "Merged Output Text After Reduction", entry.get("mergedOutputText", ""))

	f.write("=" * 80 + "\n\n")

def writeReductionLogSection(f, sectionTitle, reductionLogs):
	f.write(sectionTitle + "\n")
	f.write("=" * 80 + "\n\n")

	if reductionLogs == None or len(reductionLogs) == 0:
		f.write("No intermediate repeat.py reductions were recorded for this section.\n\n")
		return

	for entry in reductionLogs:
		writeOneReductionEntry(f, entry)

def writeDataFile(path, fileInfos, gpuRunMetrics, combinedSummaryText, introduction, conclusion, args):
	with open(path, "w", encoding="utf-8") as f:
		f.write("Evaluator Data Report\n")
		f.write("=" * 80 + "\n\n")

		f.write("Run Configuration\n")
		f.write("-" * 80 + "\n")
		f.write("Model: " + str(args.model) + "\n")
		f.write("Output directory: " + str(args.output) + "\n")
		f.write("Input files: " + str(len(fileInfos)) + "\n")
		f.write("Parallel mode: " + str(args.parallelMode) + "\n")
		f.write("Parallel devices: " + str(args.parallelDevices) + "\n")
		f.write("Character limit: " + str(args.charLimit) + "\n")
		f.write("Section type: " + str(args.sectionType) + "\n")
		f.write("Order type: " + str(args.orderType) + "\n\n")

		f.write("GPU Throughput Runs\n")
		f.write("-" * 80 + "\n")
		if len(gpuRunMetrics) == 0:
			f.write("No GPU throughput metrics were recorded.\n\n")
		else:
			for metric in gpuRunMetrics:
				f.write("Run: " + str(metric.get("runName", "")) + "\n")
				f.write("Input: " + str(metric.get("inputPath", "")) + "\n")
				f.write("GPU: " + str(metric.get("deviceId", "")) + "\n")
				f.write("Elapsed seconds: " + format(metric.get("elapsedSeconds", 0), ".4f") + "\n")
				f.write("Input characters: " + str(metric.get("inputChars", 0)) + "\n")
				f.write("Input words: " + str(metric.get("inputWords", 0)) + "\n")
				f.write("Chunks processed: " + str(metric.get("chunkCount", 0)) + "\n")
				f.write("Output characters: " + str(metric.get("outputChars", 0)) + "\n")
				f.write("Output words: " + str(metric.get("outputWords", 0)) + "\n")
				f.write("Character throughput: " + format(metric.get("charsPerSecond", 0), ".2f") + " chars/sec\n")
				f.write("Word throughput: " + format(metric.get("wordsPerSecond", 0), ".2f") + " words/sec\n")
				f.write("Chunk throughput: " + format(metric.get("chunksPerSecond", 0), ".4f") + " chunks/sec\n")
				f.write("\n")

		f.write("Document Cleaning and Chunking Metrics\n")
		f.write("=" * 80 + "\n\n")
		for fileInfo in fileInfos:
			f.write("Document: " + fileInfo["displayName"] + "\n")
			f.write("Input path: " + fileInfo["inputPath"] + "\n")
			f.write("Raw characters before cleaning: " + str(fileInfo.get("rawCharCount", 0)) + "\n")
			f.write("Cleaned characters after cleaning: " + str(fileInfo.get("cleanedCharCount", 0)) + "\n")
			f.write("Word count before cleaning: " + str(fileInfo.get("rawWordCount", 0)) + "\n")
			f.write("Word count after cleaning: " + str(fileInfo.get("cleanedWordCount", 0)) + "\n")
			f.write("Removed citations: " + str(fileInfo.get("removedCitationCount", 0)) + "\n")
			f.write("Removed morpheme words: " + str(fileInfo.get("removedMorphemeCount", 0)) + "\n")
			f.write("Chunk count: " + str(len(fileInfo.get("chunks", {}))) + "\n")
			f.write("\nChunk Character Counts\n")
			for chunkName, chunkTextValue in fileInfo.get("chunks", {}).items():
				f.write("- " + chunkName + ": " + str(len(chunkTextValue)) + " characters\n")
			f.write("\n")

		f.write("Chunk Text Before and After Summarization\n")
		f.write("=" * 80 + "\n\n")
		for fileInfo in fileInfos:
			f.write("Document: " + fileInfo["displayName"] + "\n")
			f.write("Input path: " + fileInfo["inputPath"] + "\n")
			f.write("-" * 80 + "\n\n")

			for item in fileInfo.get("chunkSummaryPairs", []):
				f.write("Chunk: " + item.get("chunkName", "") + "\n")
				f.write("Chunk characters: " + str(item.get("chunkChars", 0)) + "\n")
				f.write("Summary characters: " + str(item.get("summaryChars", 0)) + "\n")
				f.write("\nBefore Summary - Chunk Text\n")
				f.write("-" * 40 + "\n")
				f.write(item.get("chunkText", "").strip() + "\n\n")
				f.write("After Summary - Summary Text\n")
				f.write("-" * 40 + "\n")
				f.write(item.get("summaryText", "").strip() + "\n")
				f.write("\n" + "=" * 80 + "\n\n")

		f.write("Repeat.py Intermediate Reduction Logs\n")
		f.write("=" * 80 + "\n\n")
		for fileInfo in fileInfos:
			writeReductionLogSection(
				f,
				"Document Repeat Reductions - " + fileInfo["displayName"],
				fileInfo.get("reductionLog", [])
			)

		writeReductionLogSection(
			f,
			"Global Repeat Reductions - Combined Survey / Report Sections",
			getattr(args, "globalReductionLogs", [])
		)

		f.write("Final Combined Survey Summary\n")
		f.write("=" * 80 + "\n")
		f.write(combinedSummaryText.strip() + "\n\n")
		f.write("Generated Introduction\n")
		f.write("=" * 80 + "\n")
		f.write(introduction.strip() + "\n\n")
		f.write("Generated Conclusion\n")
		f.write("=" * 80 + "\n")
		f.write(conclusion.strip() + "\n")

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
	rawWordCount = countTotalWords(rawText)
	rawCharCount = len(rawText)

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
	cleanedWordCount = countTotalWords(cleanedText)
	cleanedCharCount = len(cleanedText)

	saveText(cleanedOutputPath, cleanedText)

	print("\n--- AFTER INGEST ---")
	print("Input:", inputPath)
	print("Length:", len(cleanedText))
	print("Word count before cleaning:", rawWordCount)
	print("Word count after cleaning:", cleanedWordCount)
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
		"rawCharCount": rawCharCount,
		"cleanedCharCount": cleanedCharCount,
		"rawWordCount": rawWordCount,
		"cleanedWordCount": cleanedWordCount,
		"removedCitationCount": len(removedCitations),
		"removedMorphemeCount": len(removedMorphemes),
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

	runStartTime = time.time()
	inputChars = 0
	inputWords = 0

	for chunkName in chunkInput:
		inputChars += len(chunkInput[chunkName])
		inputWords += countTotalWords(chunkInput[chunkName])

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

	chunkSummaryPairs = []
	for chunkName, chunkTextValue in chunkInput.items():
		summaryText = summaries.get(chunkName, "")
		chunkSummaryPairs.append({
			"chunkName": chunkName,
			"chunkText": chunkTextValue,
			"summaryText": summaryText,
			"chunkChars": len(chunkTextValue),
			"summaryChars": len(summaryText)
		})

	saveSummaries(summaryOutputPath, summaries)

	print("\n--- AFTER FIRST SUMMARY ---")
	print("Input:", fileInfo["inputPath"])
	print("GPU:", deviceId)
	print("Summary count:", len(summaries))

	# --- REPEAT / REDUCE ---
	reductionLog = []
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
		topP=args.topP,
		reductionLog=reductionLog,
		reductionRunName="document-repeat-" + fileInfo["displayName"]
	)

	finalText = finalSummary.get("paper", "")
	runEndTime = time.time()
	elapsedSeconds = runEndTime - runStartTime
	outputChars = 0
	outputWords = 0

	for summaryName in summaries:
		outputChars += len(summaries[summaryName])
		outputWords += countTotalWords(summaries[summaryName])

	outputChars += len(finalText)
	outputWords += countTotalWords(finalText)

	gpuRunMetric = makeThroughputMetric(
		"document-summary",
		fileInfo["inputPath"],
		deviceId,
		elapsedSeconds,
		inputChars,
		inputWords,
		len(chunkInput),
		outputChars,
		outputWords
	)

	print("\n--- FINAL SUMMARY ---")
	print("Input:", fileInfo["inputPath"])
	print("GPU:", deviceId)
	print(finalText)

	saveText(finalSummaryOutputPath, finalText)

	fileInfo["summaryOutputPath"] = summaryOutputPath
	fileInfo["finalSummaryOutputPath"] = finalSummaryOutputPath
	fileInfo["finalSummaryText"] = finalText
	fileInfo["summaryDeviceId"] = deviceId
	fileInfo["chunkSummaryPairs"] = chunkSummaryPairs
	fileInfo["reductionLog"] = reductionLog
	fileInfo["gpuRunMetric"] = gpuRunMetric

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

	runStartTime = time.time()
	inputTextForMetric = "\n\n".join(combinedInput.values())
	combinedReductionLog = []

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
		topP=args.topP,
		reductionLog=combinedReductionLog,
		reductionRunName="combined-survey-synthesis"
	)

	if not hasattr(args, "globalReductionLogs"):
		args.globalReductionLogs = []
	args.globalReductionLogs.extend(combinedReductionLog)

	combinedSummaryText = combinedFinal.get("paper", "")
	elapsedSeconds = time.time() - runStartTime
	args.gpuRunMetrics.append(makeThroughputMetric(
		"combined-survey-synthesis",
		"multiple-documents",
		deviceId,
		elapsedSeconds,
		len(inputTextForMetric),
		countTotalWords(inputTextForMetric),
		len(combinedInput),
		len(combinedSummaryText),
		countTotalWords(combinedSummaryText)
	))

	return combinedSummaryText

def generateReportParagraph(sourceText, promptText, tokenizer, model, args, deviceId, runName="report-section"):
	sectionInput = {
		"report": sourceText
	}

	runStartTime = time.time()
	reportReductionLog = []

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
		topP=args.topP,
		reductionLog=reportReductionLog,
		reductionRunName=runName
	)

	if not hasattr(args, "globalReductionLogs"):
		args.globalReductionLogs = []
	args.globalReductionLogs.extend(reportReductionLog)

	sectionText = finalSection.get("paper", "")
	elapsedSeconds = time.time() - runStartTime
	args.gpuRunMetrics.append(makeThroughputMetric(
		runName,
		"report-context",
		deviceId,
		elapsedSeconds,
		len(sourceText),
		countTotalWords(sourceText),
		1,
		len(sectionText),
		countTotalWords(sectionText)
	))

	return sectionText

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
			deviceId,
			runName="generated-introduction"
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
			deviceId,
			runName="generated-conclusion"
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
	args.gpuRunMetrics = []
	args.globalReductionLogs = []

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

	for fileInfo in summarizedFileInfos:
		if "gpuRunMetric" in fileInfo:
			args.gpuRunMetrics.append(fileInfo["gpuRunMetric"])

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

	dataOutputPath = os.path.join(outputDir, args.dataFileName)
	writeDataFile(
		dataOutputPath,
		summarizedFileInfos,
		args.gpuRunMetrics,
		combinedSummaryText,
		introduction,
		conclusion,
		args
	)
	outputPaths["dataOutputPath"] = dataOutputPath

	print("\n--- OUTPUT COMPLETE ---")
	print("Saved to:", outputDir)
	print("Final summary:", outputPaths["finalSummaryOutputPath"])
	print("Survey txt:", outputPaths["surveyTxtPath"])
	print("Survey docx:", outputPaths["surveyDocxPath"])
	print("Survey pdf:", outputPaths["surveyPdfPath"])
	print("Evaluator data:", outputPaths["dataOutputPath"])

# - - - - - - - - - - #

if __name__ == "__main__":
	main()

