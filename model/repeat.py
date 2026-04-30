# repeat.py
# Stephan DeLuna
# reduces summaries in stages until one final paper summary remains

# - - - - - - - - - - #

from chunk.chunk import chunkText, organizeChunks
from model.summarize import summarizeWithLoadedModel

# - - - - - - - - - - #

# get main section name from a chunk key
def getSectionName(chunkKey):
	parts = chunkKey.split()

	if len(parts) < 2:
		return chunkKey

	labelType = parts[0]
	numberPart = parts[1]

	if "." in numberPart:
		mainNumber = numberPart.split(".")[0]
		return labelType + " " + mainNumber

	return chunkKey

# combine summaries by section
def combineSectionSummaries(summaries):
	groupedSummaries = {}

	for chunkKey, summaryText in summaries.items():
		sectionName = getSectionName(chunkKey)

		if sectionName not in groupedSummaries:
			groupedSummaries[sectionName] = []

		groupedSummaries[sectionName].append((chunkKey, summaryText))

	combinedSections = {}

	for sectionName, groupItems in groupedSummaries.items():
		groupItems.sort(key=lambda item: item[0])

		combinedText = ""
		for chunkKey, summaryText in groupItems:
			combinedText += summaryText.strip() + "\n\n"

		combinedSections[sectionName] = combinedText.strip()

	return combinedSections

# check whether any section still has multiple parts
def hasMultiPartSections(summaries):
	sectionCounts = {}

	for chunkKey in summaries:
		sectionName = getSectionName(chunkKey)

		if sectionName in sectionCounts:
			sectionCounts[sectionName] += 1
		else:
			sectionCounts[sectionName] = 1

	for sectionName in sectionCounts:
		if sectionCounts[sectionName] > 1:
			return True

	return False

# reduce section parts into one summary per section
def reduceSections(
	summaries,
	tokenizer,
	model,
	promptText=None,
	sentenceLimit=6,
	maxNewTokens=220,
	deviceId=0,
	failOnNonEnglish=False,
	maxRetries=2,
	forceEnglishRepair=True,
	doSample=False,
	temperature=0.3,
	topP=0.9,
	enableThinking=False,
	repetitionPenalty=1.05,
	noRepeatNgramSize=0,
	qualityPolish=False
):
	combinedSections = combineSectionSummaries(summaries)
	reducedSummaries = {}

	for sectionName, combinedText in combinedSections.items():
		sectionPieces = []

		for chunkKey in summaries:
			if getSectionName(chunkKey) == sectionName:
				sectionPieces.append(chunkKey)

		if len(sectionPieces) == 1:
			reducedSummaries[sectionName] = summaries[sectionPieces[0]]
			continue

		sectionInput = {
			sectionName: combinedText
		}

		sectionSummary = summarizeWithLoadedModel(
			sectionInput,
			tokenizer,
			model,
			deviceId=deviceId,
			promptText=promptText,
			sentenceLimit=sentenceLimit,
			maxNewTokens=maxNewTokens,
			failOnNonEnglish=failOnNonEnglish,
			maxRetries=maxRetries,
			forceEnglishRepair=forceEnglishRepair,
			doSample=doSample,
			temperature=temperature,
			topP=topP,
			enableThinking=enableThinking,
			repetitionPenalty=repetitionPenalty,
			noRepeatNgramSize=noRepeatNgramSize,
			qualityPolish=qualityPolish
		)

		reducedSummaries[sectionName] = sectionSummary[sectionName]

	return reducedSummaries

# merge all summaries into one text block
def mergeAllSummaries(summaries):
	fullText = ""

	for summaryName, summaryText in summaries.items():
		fullText += summaryText.strip() + "\n\n"

	return fullText.strip()

# reduce the whole paper until one final summary remains
def reducePaper(
	summaries,
	tokenizer,
	model,
	promptText=None,
	sentenceLimit=6,
	maxNewTokens=220,
	deviceId=0,
	charLimit=12000,
	limitPercent=0.9,
	overlapSize=200,
	maxRounds=5,
	failOnNonEnglish=False,
	maxRetries=2,
	forceEnglishRepair=True,
	doSample=False,
	temperature=0.3,
	topP=0.9,
	enableThinking=False,
	repetitionPenalty=1.05,
	noRepeatNgramSize=0,
	qualityPolish=False
):
	currentText = mergeAllSummaries(summaries)
	roundCount = 0

	while roundCount < maxRounds:
		if len(currentText) <= charLimit:
			finalInput = {
				"paper": currentText
			}

			finalSummary = summarizeWithLoadedModel(
				finalInput,
				tokenizer,
				model,
				deviceId=deviceId,
				promptText=promptText,
				sentenceLimit=sentenceLimit,
				maxNewTokens=maxNewTokens,
				failOnNonEnglish=failOnNonEnglish,
				maxRetries=maxRetries,
				forceEnglishRepair=forceEnglishRepair,
				doSample=doSample,
				temperature=temperature,
				topP=topP,
				enableThinking=enableThinking,
				repetitionPenalty=repetitionPenalty,
				noRepeatNgramSize=noRepeatNgramSize,
				qualityPolish=qualityPolish
			)

			return finalSummary

		rechunked = chunkText(
			currentText,
			charLimit=charLimit,
			sectionType="paragraph",
			limitPercent=limitPercent,
			overlapSize=overlapSize
		)

		rechunked = organizeChunks(rechunked, orderType="firstToLast")

		rechunkInput = {}
		partCount = 1

		for chunkName, chunkTextValue in rechunked.items():
			if chunkName == "title" or chunkName == "front front":
				continue

			rechunkInput["paper." + str(partCount)] = chunkTextValue
			partCount += 1

		reducedSummaries = summarizeWithLoadedModel(
			rechunkInput,
			tokenizer,
			model,
			deviceId=deviceId,
			promptText=promptText,
			sentenceLimit=sentenceLimit,
			maxNewTokens=maxNewTokens,
			failOnNonEnglish=failOnNonEnglish,
			maxRetries=maxRetries,
			forceEnglishRepair=forceEnglishRepair,
			doSample=doSample,
			temperature=temperature,
			topP=topP,
			enableThinking=enableThinking,
			repetitionPenalty=repetitionPenalty,
			noRepeatNgramSize=noRepeatNgramSize,
			qualityPolish=qualityPolish
		)

		currentText = mergeAllSummaries(reducedSummaries)
		roundCount += 1

	finalInput = {
		"paper": currentText
	}

	finalSummary = summarizeWithLoadedModel(
		finalInput,
		tokenizer,
		model,
		deviceId=deviceId,
		promptText=promptText,
		sentenceLimit=sentenceLimit,
		maxNewTokens=maxNewTokens,
		failOnNonEnglish=failOnNonEnglish,
		maxRetries=maxRetries,
		forceEnglishRepair=forceEnglishRepair,
		doSample=doSample,
		temperature=temperature,
		topP=topP,
		enableThinking=enableThinking,
		repetitionPenalty=repetitionPenalty,
		noRepeatNgramSize=noRepeatNgramSize,
		qualityPolish=qualityPolish
	)

	return finalSummary

# full reduce pipeline
def repeatSummaries(
	summaries,
	tokenizer,
	model,
	promptText=None,
	sentenceLimit=6,
	maxNewTokens=220,
	deviceId=0,
	charLimit=12000,
	limitPercent=0.9,
	overlapSize=200,
	maxRounds=5,
	failOnNonEnglish=False,
	maxRetries=2,
	forceEnglishRepair=True,
	doSample=False,
	temperature=0.3,
	topP=0.9,
	enableThinking=False,
	repetitionPenalty=1.05,
	noRepeatNgramSize=0,
	qualityPolish=False
):
	currentSummaries = summaries

	while hasMultiPartSections(currentSummaries):
		currentSummaries = reduceSections(
			currentSummaries,
			tokenizer,
			model,
			promptText=promptText,
			sentenceLimit=sentenceLimit,
			maxNewTokens=maxNewTokens,
			deviceId=deviceId,
			failOnNonEnglish=failOnNonEnglish,
			maxRetries=maxRetries,
			forceEnglishRepair=forceEnglishRepair,
			doSample=doSample,
			temperature=temperature,
			topP=topP,
			enableThinking=enableThinking,
			repetitionPenalty=repetitionPenalty,
			noRepeatNgramSize=noRepeatNgramSize,
			qualityPolish=qualityPolish
		)

	finalSummary = reducePaper(
		currentSummaries,
		tokenizer,
		model,
		promptText=promptText,
		sentenceLimit=sentenceLimit,
		maxNewTokens=maxNewTokens,
		deviceId=deviceId,
		charLimit=charLimit,
		limitPercent=limitPercent,
		overlapSize=overlapSize,
		maxRounds=maxRounds,
		failOnNonEnglish=failOnNonEnglish,
		maxRetries=maxRetries,
		forceEnglishRepair=forceEnglishRepair,
		doSample=doSample,
		temperature=temperature,
		topP=topP,
		enableThinking=enableThinking,
		repetitionPenalty=repetitionPenalty,
		noRepeatNgramSize=noRepeatNgramSize,
		qualityPolish=qualityPolish
	)

	return currentSummaries, finalSummary

# - - - - - - - - - - #

