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
	# validate path/file
	if envPath == None or envPath.strip() == "":
		return
	if not os.path.exists(envPath):
		return

	# read file line-by-line
	with open(envPath, "r", encoding="utf-8") as f:
		for line in f:
			# clear whitespace
			line = line.strip()

			# skip empty lines and comments
			if line == "" or line.startswith("#"):
				continue

			# support "export KEY=value" syntax
			if line.startswith("export "):
				line = line[7:].strip()
			# ensure line contains "="
			if "=" not in line:
				continue
			# split into key and value
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip()

			if key == "":
				continue

			# remove inline comments
			if " #" in value:
				value = value.split(" #", 1)[0].strip()

			# remove surrounding quotes
			if len(value) >= 2:
				if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
					value = value[1:-1]

			# only set variable if it's not already in the env
			if key not in os.environ:
				os.environ[key] = value

# try multiple environment variable names, return first one that exists
def getEnvValue(names, defaultValue=None):
	for name in names:
		value = os.getenv(name)
		if value != None and value.strip() != "":
			return value.strip()
	# default if none are found
	return defaultValue

# - - - - - - - - - - #

# read string as a bool value
def strToBool(value):
	if isinstance(value, bool):
		return value

	# convert to lower case
	value = value.lower().strip()

	if value in ["true", "t", "yes", "y", "1"]:
		return True
	if value in ["false", "f", "no", "n", "0"]:
		return False

	# if string can't be read as either true or false
	raise argparse.ArgumentTypeError("Boolean value expected: true or false")

# turn a comma-separated string into a list
def parseList(value):
	# check if string is empty or "none"
	if value == None:
		return None
	value = value.strip()
	if value == "" or value.lower() == "none":
		return None

	# split string by comma
	items = []
	for item in value.split(","):
		item = item.strip()
		if item != "":
			items.append(item)

	# if resulting list is empty
	if len(items) == 0:
		return None

	return items

# converts comma-separated string into integers
def parseIntList(value):
	# convert into list, numbers remain as string
	items = parseList(value)
	# check if list is empty
	if items == None:
		return None

	# convert into integers
	intItems = []
	for item in items:
		intItems.append(int(item))

	return intItems

# create a filesystem-safe name
def makeSafeName(name):
	# extract base name without file extension
	name = os.path.splitext(os.path.basename(name))[0]
	# replace non-alphanumeric characters with "_"
	name = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
	# strip leading or trailing "_"
	name = name.strip("_")

	# if base name is empty, use default name
	if name == "":
		name = "document"

	return name

# create a human-friendly name
def makeDisplayName(path):
	# grab file's base name without file extension
	baseName = os.path.splitext(os.path.basename(path))[0]
	# insert spaces where "_" and "-" occur
	baseName = baseName.replace("_", " ").replace("-", " ")
	# capitalize first character of each word
	return baseName.title()

# prevents multiple inputs from having identical names
def makeUniqueNames(inputPaths):
	nameCounts = {}
	uniqueNames = []

	for inputPath in inputPaths:
		# create filesystem-safe name
		baseName = makeSafeName(inputPath)

		if baseName not in nameCounts:
			nameCounts[baseName] = 0
			uniqueNames.append(baseName)
		else:
			# duplicate names have an incrementing number appended to the end
			nameCounts[baseName] += 1
			uniqueNames.append(baseName + "_" + str(nameCounts[baseName]))

	return uniqueNames

# - - - - - - - - - - #

# read commandline inputs
def parseArgs():
	# load environment and define model
	loadEnvFile(".env")
	envModelName = getEnvValue(["MODEL_NAME", "MODEL", "HF_MODEL_NAME"], "Qwen/Qwen2.5-7B-Instruct")

	parser = argparse.ArgumentParser(description="LLM Summarization Pipeline")

	# input
	parser.add_argument(
		"--input",
		required=True,
		nargs="+",
		help="Path to one or more input files (txt/pdf/docx)"
	)
	# output
	parser.add_argument(
		"--output",
		default="storage",
		help="Output directory (default: storage)"
	)
	# file name of output
	parser.add_argument(
		"--outputName",
		default=None,
		help="Base name for final output files"
	)
	# title for output
	parser.add_argument(
		"--title",
		default=None,
		help="Title for final survey output"
	)
	# output author name
	parser.add_argument(
		"--authorName",
		default="Stephan DeLuna",
		help="Author name for final survey output"
	)
	# abstract text for output
	parser.add_argument(
		"--abstractText",
		default="",
		help="Optional abstract text for final survey output"
	)
	# introduction text for output
	parser.add_argument(
		"--introductionText",
		default=None,
		help="Optional introduction text for final survey output"
	)
	# conclusion text for output
	parser.add_argument(
		"--conclusionText",
		default=None,
		help="Optional conclusion text for final survey output"
	)
	# REMOVE. SKIPWAITS NOT BEING USED
	# skip pauses in pipeline
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

# REMOVE. SKIPWAITS NOT BEING USED
# if skipWaits is enabled, function is used to create pauses
def waitForNextStep(stepName, skipWaits=False):
	if skipWaits:
		return

	while True:
		userInput = input(f"\nPress space then Enter to continue to {stepName}: ")
		if userInput == " ":
			break

# function to save text
def saveText(path, text):
	with open(path, "w", encoding="utf-8") as f:
		f.write(text)

# function to save chunks
def saveChunks(path, chunks):
	with open(path, "w", encoding="utf-8") as f:
		for k, v in chunks.items():
			f.write(k + "\n")
			f.write("-" * 40 + "\n")
			f.write(v + "\n\n")

# function to save summaries
def saveSummaries(path, summaries):
	with open(path, "w", encoding="utf-8") as f:
		for k, v in summaries.items():
			f.write(k + "\n")
			f.write("-" * 40 + "\n")
			f.write(v + "\n\n")

# - - - - - - - - - - #

# grab integer list of available devices
def getDeviceIds(args):
	deviceIds = parseIntList(args.parallelDevices)

	# check if no devices
	if deviceIds == None or len(deviceIds) == 0:
		deviceIds = [args.deviceId]

	return deviceIds

# prompt passed to LLM to create final survey paper
def getSurveyPromptText(args, isMultiFile):
	# use prompt provided in commandline, if any
	if args.surveyPromptText != None:
		return args.surveyPromptText

	# if multiple files are used
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
	# if single file
	return args.promptText

# default introduction text
def getDefaultIntroductionText(fileInfos):
	# single file
	if len(fileInfos) == 1:
		return "This report summarizes the provided document using a multi-stage LLM pipeline."
	# multiple files
	return "This report summarizes multiple provided documents using a multi-stage LLM pipeline and synthesizes their shared themes into a survey-style overview."
# default conclusion text
def getDefaultConclusionText(fileInfos):
	# single file
	if len(fileInfos) == 1:
		return "This summary condenses the full document into a concise overview."
	# multiple files
	return "This summary condenses the full document set into a concise overview while preserving individual source summaries for review."

# generate LLM prompt for creating introduction text
def getIntroductionPromptText(args):
	# use user-provided prompt if any
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

# generate LLM prompt for creating conclusion text
def getConclusionPromptText(args):
	# use user-provided prompt if any
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

# build context for LLM in order to generate introduction and/or conclusion text
def buildReportContext(fileInfos, combinedSummaryText):
	# give complete summary for all files
	contextText = "Survey Synthesis:\n"
	contextText += combinedSummaryText.strip() + "\n\n"

	# separately display summaries for each document for clarity
	contextText += "Individual Document Summaries:\n"
	for fileInfo in fileInfos:
		contextText += fileInfo["displayName"] + ":\n"
		contextText += fileInfo["finalSummaryText"].strip() + "\n\n"

	return contextText.strip()

# - - - - - - - - - - #

# run pipeline on input file(s)
def processInputFile(inputPath, outputDir, outputBaseName, args):
	# create file paths for created files to be passed between modules
	cleanedOutputPath = os.path.join(outputDir, f"{outputBaseName}_cleaned.txt")
	chunkOutputPath = os.path.join(outputDir, f"{outputBaseName}_chunks.txt")

	# --- INGEST ---
	# extract text from file
	rawText = loadDocument(inputPath)
	# clean the text
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
	# save cleaned text
	saveText(cleanedOutputPath, cleanedText)

	# --- CHUNK ---
	# separate cleaned text into chunks
	chunks = chunkText(
		cleanedText,
		charLimit=args.charLimit,
		sectionType=args.sectionType,
		limitPercent=args.limitPercent,
		overlapSize=args.overlapSize
	)
	chunks = organizeChunks(chunks, orderType=args.orderType)

	# save chunked text
	saveChunks(chunkOutputPath, chunks)

	return {
		"inputPath": inputPath,
		"outputBaseName": outputBaseName,
		"displayName": makeDisplayName(inputPath),
		"cleanedOutputPath": cleanedOutputPath,
		"chunkOutputPath": chunkOutputPath,
		"chunks": chunks
	}

# function for summarizing one file
def summarizeOneFile(fileInfo, tokenizer, model, outputDir, args, deviceId=None):
	# determine what device to use
	if deviceId == None:
		deviceId = args.deviceId

	# create path for summary output and final summary output
	summaryOutputPath = os.path.join(outputDir, f"{fileInfo['outputBaseName']}_summaries.txt")
	finalSummaryOutputPath = os.path.join(outputDir, f"{fileInfo['outputBaseName']}_final.txt")

	# prepare chunk inputs, getting rid of special sections
	chunkInput = {}
	for k, v in fileInfo["chunks"].items():
		if k not in ["title", "front front"]:
			chunkInput[k] = v
	# if excluding special sections causes an empty chunk, just use entire chunk
	if len(chunkInput) == 0:
		chunkInput = fileInfo["chunks"]

	# --- FIRST SUMMARY ---
	# summarize chunks
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
	# save chunk summaries
	saveSummaries(summaryOutputPath, summaries)

	# --- REPEAT / REDUCE ---
	# repeat summarizing function and combine summaries
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
	# extract final summary text
	finalText = finalSummary.get("paper", "")

	# save final summary
	saveText(finalSummaryOutputPath, finalText)

	# update fileinfo with results
	fileInfo["summaryOutputPath"] = summaryOutputPath
	fileInfo["finalSummaryOutputPath"] = finalSummaryOutputPath
	fileInfo["finalSummaryText"] = finalText
	fileInfo["summaryDeviceId"] = deviceId

	return fileInfo

# parallel summarization
def summarizeFileBatch(batchItems, workerInfo, outputDir, args):
	results = []

	for index, fileInfo in batchItems:
		# workers summarize each file
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

# create gpu workers for parallelization
def loadModelWorkers(modelName, deviceIds, args):
	workers = []

	for deviceId in deviceIds:
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

# summarization controller
def summarizeFiles(fileInfos, workers, outputDir, args):
	# if only one gpu worker is available, run in sequential mode
	if len(workers) == 1 or args.parallelMode == "none" or len(fileInfos) == 1:
		workerInfo = workers[0]
		summarizedFileInfos = []
		# each file is summarized sequentially
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

	# multiple gpu workers are available, parallel mode is enabled
	assignments = []
	for workerInfo in workers:
		assignments.append([])

	for i, fileInfo in enumerate(fileInfos):
		workerIndex = i % len(workers)
		assignments[workerIndex].append((i, fileInfo))

	orderedResults = [None] * len(fileInfos)

	# create threads
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

# if multiple files, multiple final summaries from each file needs to be combined
def buildCombinedSummary(fileInfos, tokenizer, model, args, deviceId=None):
	# get device
	if deviceId == None:
		deviceId = args.deviceId

	if len(fileInfos) == 1:
		return fileInfos[0]["finalSummaryText"]

	combinedInput = {}

	for i, fileInfo in enumerate(fileInfos):
		combinedInput["document" + str(i + 1)] = (
			"Document: " + fileInfo["displayName"] + "\n\n" + fileInfo["finalSummaryText"]
		)

	# combine final summaries
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

# generate a paragraph for report
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

# builds introduction and conclusion of report
def buildGeneratedReportSections(fileInfos, combinedSummaryText, tokenizer, model, args, deviceId):
	# build LLM context prompt for report
	reportContext = buildReportContext(fileInfos, combinedSummaryText)

	# get introduction paragraph for report
	if args.introductionText != None:
		introduction = args.introductionText
	elif args.generateIntroduction:
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

	# get conclusion paragraph for report
	if args.conclusionText != None:
		conclusion = args.conclusionText
	elif args.generateConclusion:
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
	# generate final survey paper filepaths and standalone final summary path
	finalSummaryOutputPath = os.path.join(outputDir, f"{outputBaseName}_final.txt")
	surveyTxtPath = os.path.join(outputDir, f"{outputBaseName}_survey.txt")
	surveyDocxPath = os.path.join(outputDir, f"{outputBaseName}_survey.docx")
	surveyPdfPath = os.path.join(outputDir, f"{outputBaseName}_survey.pdf")

	saveText(finalSummaryOutputPath, combinedSummaryText)

	# determine survey title
	if args.title != None:
		title = args.title
	elif len(fileInfos) == 1:
		title = fileInfos[0]["displayName"]
	else:
		title = "Multi File Survey Report"

	# create dictionary for summaries
	summaryDict = {
		"paper": combinedSummaryText
	}
	# save individual summaries for each file and add to dictionary
	if len(fileInfos) > 1:
		for fileInfo in fileInfos:
			summaryDict[fileInfo["displayName"]] = fileInfo["finalSummaryText"]

	# generate final survey paper txt file
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
	# generate final survey paper docx file
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
	# generate final survey paper pdf file
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

# combine all functions to create modular pipeline
def main():
	# parse commandline arguments
	args = parseArgs()
	inputPaths = args.input
	outputDir = args.output
	modelName = args.model
	deviceIds = getDeviceIds(args)
	# create output directory
	os.makedirs(outputDir, exist_ok=True)
	# create unique file names
	uniqueNames = makeUniqueNames(inputPaths)

	# determine output name for summary
	if args.outputName != None:
		finalOutputBaseName = makeSafeName(args.outputName)
	elif len(inputPaths) == 1:
		finalOutputBaseName = uniqueNames[0]
	else:
		finalOutputBaseName = "multi_file_summary"

	# get file information
	fileInfos = []

	for i, inputPath in enumerate(inputPaths):
		fileInfo = processInputFile(
			inputPath,
			outputDir,
			uniqueNames[i],
			args
		)
		fileInfos.append(fileInfo)

	# REMOVE WAITSKIP
	waitForNextStep("model loading", args.skipWaits)
	# create gpu workers
	workers = loadModelWorkers(modelName, deviceIds, args)
	# REMOVE WAITSKIP
	waitForNextStep("summarization", args.skipWaits)
	# begin summarization
	summarizedFileInfos = summarizeFiles(fileInfos, workers, outputDir, args)
	# REMOVE WAITSKIP
	waitForNextStep("combined survey synthesis", args.skipWaits)

	combinedWorker = workers[0]
	# combine final summaries
	combinedSummaryText = buildCombinedSummary(
		summarizedFileInfos,
		combinedWorker["tokenizer"],
		combinedWorker["model"],
		args,
		deviceId=combinedWorker["deviceId"]
	)
	# REMOVE WAITSKIP
	waitForNextStep("introduction and conclusion", args.skipWaits)
	# generate introduction and conclusion paragraphs
	introduction, conclusion = buildGeneratedReportSections(
		summarizedFileInfos,
		combinedSummaryText,
		combinedWorker["tokenizer"],
		combinedWorker["model"],
		args,
		deviceId=combinedWorker["deviceId"]
	)
	# REMOVE WAITSKIP
	waitForNextStep("output", args.skipWaits)
	# export final survey paper files
	outputPaths = exportFinalOutput(
		summarizedFileInfos,
		combinedSummaryText,
		introduction,
		conclusion,
		outputDir,
		finalOutputBaseName,
		args
	)

# - - - - - - - - - - #

# execute pipeline in driver
if __name__ == "__main__":
	main()

