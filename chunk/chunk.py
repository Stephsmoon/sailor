# chunk.py
# Stephan DeLuna
# chunks text into organized parts

# - - - - - - - - - - #

import re
from chunk.section import detectSections, detectLimit

# - - - - - - - - - - #

# get current section label from remaining and start
def detectFirstSectionName(content, sectionType="chapter"):
	lines = content.splitlines()
	checkedLines = 0
	for line in lines:
		strippedLine = line.strip()
		if strippedLine == "":
			continue
		checkedLines += 1
		# only check the first few real lines
		# - - - - - - - - - - #
		# difficult attempt and could be improved
		# this changes with every type of paper and could be
		# drastically improved or specialized for specific papers 
		# - - - - - - - - - - #
		if checkedLines > 3:
			break
		if sectionType == "chapter":
			if strippedLine.lower().startswith("chapter "):
				parts = strippedLine.split()
				if len(parts) >= 2:
					return "chapter", parts[1]
			if strippedLine.lower().startswith("part "):
				parts = strippedLine.split()
				if len(parts) >= 2:
					return "chapter", parts[1]
			# check for roman numerals
			match = re.match(r'^([IVXLC]+)\s+([A-Z][A-Z\'\?\!\,\-\s]+)$', strippedLine)
			if match:
				return "chapter", match.group(1)
		elif sectionType == "page":
			if strippedLine.lower().startswith("page "):
				parts = strippedLine.split()
				if len(parts) >= 2:
					return "page", parts[1]
			if strippedLine.isdigit():
				return "page", strippedLine
		elif sectionType == "paragraph":
			return "paragraph", "1"
	return None, None

# chunk text by section first, then by limit
def chunkText(content, charLimit, sectionType="chapter", limitPercent=0.9, overlapSize=0):
	chunks = {}
	sectionCounts = {}
	remainingText = content.strip()
	frontUsed = False
	currentType = None
	currentName = None
	# loop until no more text is left
	while remainingText != "":
		firstPart, secondPart, saveType, saveName, nextType, nextName = detectSections(
			remainingText,
			charLimit,
			sectionType=sectionType,
			overlapSize=0,
			currentType=currentType,
			currentName=currentName
		)
		# section split worked
		if firstPart != None and firstPart != "":
			baseKey = saveType + " " + str(saveName)
			if baseKey in sectionCounts:
				sectionCounts[baseKey] += 1
				chunkKey = baseKey + "." + str(sectionCounts[baseKey])
			else:
				sectionCounts[baseKey] = 0
				chunkKey = baseKey
			chunks[chunkKey] = firstPart
			# now advance current section to the next one
			if nextType != None and nextName != None:
				currentType = nextType
				currentName = nextName
			# front matter should not use overlap
			if saveType == "front":
				newRemainingText = remainingText[len(firstPart):].strip()
			else:
				newRemainingText = secondPart.strip()
			if len(newRemainingText) >= len(remainingText):
				break
			remainingText = newRemainingText
			continue
		# if remaining text already fits, save it and stop
		if len(remainingText) <= charLimit:
			firstType, firstName = detectFirstSectionName(remainingText, sectionType)
			if firstType != None and firstName != None:
				currentType = firstType
				currentName = firstName
			if currentType == None or currentName == None:
				baseKey = "chunk"
			else:
				baseKey = currentType + " " + str(currentName)
			if baseKey in sectionCounts:
				sectionCounts[baseKey] += 1
				chunkKey = baseKey + "." + str(sectionCounts[baseKey])
			else:
				sectionCounts[baseKey] = 0
				chunkKey = baseKey
			chunks[chunkKey] = remainingText
			break
		# section too large, split by limit
		firstType, firstName = detectFirstSectionName(remainingText, sectionType)
		if firstType != None and firstName != None:
			currentType = firstType
			currentName = firstName
		if currentType == None or currentName == None:
			baseKey = "chunk"
		else:
			baseKey = currentType + " " + str(currentName)
		firstPart, secondPart = detectLimit(
			remainingText,
			charLimit,
			limitPercent=limitPercent,
			overlapSize=overlapSize
		)
		if firstPart == "" or len(secondPart.strip()) >= len(remainingText):
			forceIndex = int(charLimit * limitPercent)
			if forceIndex <= 0 or forceIndex >= len(remainingText):
				forceIndex = charLimit
			firstPart = remainingText[:forceIndex].strip()
			secondStart = forceIndex - overlapSize
			if secondStart < 0:
				secondStart = 0
			secondPart = remainingText[secondStart:].strip()
		if baseKey in sectionCounts:
			sectionCounts[baseKey] += 1
			chunkKey = baseKey + "." + str(sectionCounts[baseKey])
		else:
			sectionCounts[baseKey] = 0
			chunkKey = baseKey
		chunks[chunkKey] = firstPart
		newRemainingText = secondPart.strip()
		if len(newRemainingText) >= len(remainingText):
			break
		remainingText = newRemainingText
	return chunks

# organize chunks in different reading orders
def organizeChunks(chunks, orderType="firstToLast"):
	parsedChunks = []
	for key, value in chunks.items():
		parts = key.split()
		if len(parts) < 2:
			labelType = "chunk"
			numberPart = "0"
		else:
			labelType = parts[0]
			numberPart = parts[1]
		hasSub = False
		mainNumber = 0
		subNumber = 0
		sortText = ""
		# numeric style like 2 or 2.1
		if re.match(r'^\d+(\.\d+)?$', numberPart):
			if "." in numberPart:
				mainPart, subPart = numberPart.split(".", 1)
				mainNumber = int(mainPart)
				subNumber = int(subPart)
				hasSub = True
			else:
				mainNumber = int(numberPart)
				subNumber = 0
		# roman numeral style like I or IV
		elif re.match(r'^[IVXLC]+(\.\d+)?$', numberPart):
			romanValues = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
			# function to easily convert roman numerals to integers
			def romanToInt(romanText):
				total = 0
				prevValue = 0

				for char in reversed(romanText):
					value = romanValues[char]
					if value < prevValue:
						total -= value
					else:
						total += value
					prevValue = value
				return total
			if "." in numberPart:
				mainPart, subPart = numberPart.split(".", 1)
				mainNumber = romanToInt(mainPart)
				subNumber = int(subPart)
				hasSub = True
			else:
				mainNumber = romanToInt(numberPart)
				subNumber = 0
		# text labels like front
		else:
			sortText = numberPart.lower()
		parsedChunks.append({
			"key": key,
			"value": value,
			"labelType": labelType,
			"mainNumber": mainNumber,
			"subNumber": subNumber,
			"hasSub": hasSub,
			"sortText": sortText
		})
	# define sorting rule
	def sortKey(item):
		if item["sortText"] == "front":
			return (-1, 0, 0)
		elif item["sortText"] == "chunk":
			return (9999, item["mainNumber"], item["subNumber"])
		else:
			return (0, item["mainNumber"], item["subNumber"])
	# - - - - - - - - - - #
	# additional feature which was briefly tested
	# proof of concept exists but could be further tested
	# - - - - - - - - - - #
	# sort normally (default)
	if orderType == "firstToLast":
		parsedChunks.sort(key=sortKey)
	# sort in reverse
	elif orderType == "lastToFirst":
		parsedChunks.sort(key=sortKey, reverse=True)
	# sort in reverse by section, keep subsections in the firstToLast order
	elif orderType == "lastToFirstBySection":
		groupedChunks = {}
		for item in parsedChunks:
			groupKey = (item["sortText"], item["mainNumber"])
			if groupKey not in groupedChunks:
				groupedChunks[groupKey] = []
			groupedChunks[groupKey].append(item)
		for groupKey in groupedChunks:
			groupedChunks[groupKey].sort(key=lambda item: item["subNumber"])
		reorderedChunks = []
		groupKeys = sorted(groupedChunks.keys(), reverse=True)
		for groupKey in groupKeys:
			for item in groupedChunks[groupKey]:
				reorderedChunks.append(item)
		parsedChunks = reorderedChunks
	# sort by top-level sections
	elif orderType == "firstToLastByTop":
		normalFlow = []
		delayedChunks = []
		for item in parsedChunks:
			if not item["hasSub"]:
				normalFlow.append(item)
			elif item["subNumber"] == 1:
				normalFlow.append(item)
			else:
				delayedChunks.append(item)
		normalFlow.sort(key=sortKey)
		delayedChunks.sort(key=sortKey)
		parsedChunks = normalFlow + delayedChunks
	else:
		parsedChunks.sort(key=sortKey)
	organizedChunks = {}
	for item in parsedChunks:
		organizedChunks[item["key"]] = item["value"]
	return organizedChunks

# - - - - - - - - - - #
