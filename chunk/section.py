# section.py
# Stephan DeLuna
# detects split points for chunking

# - - - - - - - - - - #

import re

# - - - - - - - - - - #

# split at next detected section inside limit
def detectSections(
	content,
	charLimit,
	sectionType="chapter",
	overlapSize=0,
	currentType=None,
	currentName=None
):
	lines = content.splitlines()
	currentIndex = 0
	foundSections = []

	for i in range(len(lines)):
		line = lines[i]
		strippedLine = line.strip()
		lineStart = currentIndex

		if strippedLine != "":
			foundType = None
			foundName = None

			# chapter clues
			if re.match(r'^(chapter)\s+([A-Za-z0-9IVXLC]+)', strippedLine, re.IGNORECASE):
				match = re.match(r'^(chapter)\s+([A-Za-z0-9IVXLC]+)', strippedLine, re.IGNORECASE)
				foundType = "chapter"
				foundName = match.group(2)

			elif re.match(r'^(part)\s+([A-Za-z0-9IVXLC]+)', strippedLine, re.IGNORECASE):
				match = re.match(r'^(part)\s+([A-Za-z0-9IVXLC]+)', strippedLine, re.IGNORECASE)
				foundType = "chapter"
				foundName = match.group(2)

			# roman numeral + uppercase title on same line
			elif re.match(r'^([IVXLC]+)\s+([A-Z][A-Z\'\?\!\,\-\s]+)$', strippedLine):
				match = re.match(r'^([IVXLC]+)\s+([A-Z][A-Z\'\?\!\,\-\s]+)$', strippedLine)
				foundType = "chapter"
				foundName = match.group(1)

			# page clues
			elif re.match(r'^(page)\s+(\d+)', strippedLine, re.IGNORECASE):
				match = re.match(r'^(page)\s+(\d+)', strippedLine, re.IGNORECASE)
				foundType = "page"
				foundName = match.group(2)

			elif re.match(r'^(\d+)$', strippedLine):
				match = re.match(r'^(\d+)$', strippedLine)
				foundType = "page"
				foundName = match.group(1)

			# paragraph clues
			elif line.startswith("    ") or line.startswith("\t"):
				foundType = "paragraph"
				foundName = str(len(foundSections) + 1)

			elif i > 0 and lines[i - 1].strip() == "":
				foundType = "paragraph"
				foundName = str(len(foundSections) + 1)

			if foundType != None:
				foundSections.append({
					"index": lineStart,
					"type": foundType,
					"name": foundName,
					"text": strippedLine
				})

		currentIndex += len(line) + 1

	filteredSections = []

	for section in foundSections:
		if section["type"] == sectionType:
			filteredSections.append(section)

	if len(filteredSections) == 0:
		return None, content, None, None, None, None

	firstSection = filteredSections[0]

	# text before the first detected section
	if firstSection["index"] > 0 and firstSection["index"] <= charLimit:
		firstPart = content[:firstSection["index"]].strip()

		secondStart = firstSection["index"] - overlapSize
		if secondStart < 0:
			secondStart = 0

		secondPart = content[secondStart:].strip()

		# save this under current section if we already have one
		if currentType != None and currentName != None:
			return firstPart, secondPart, currentType, currentName, firstSection["type"], firstSection["name"]

		# otherwise this is true front matter
		return firstPart, secondPart, "front", "front", firstSection["type"], firstSection["name"]

	# split at next section boundary inside limit
	for i in range(1, len(filteredSections)):
		sectionIndex = filteredSections[i]["index"]

		if sectionIndex <= charLimit:
			firstPart = content[:sectionIndex].strip()

			secondStart = sectionIndex - overlapSize
			if secondStart < 0:
				secondStart = 0

			secondPart = content[secondStart:].strip()

			return (
				firstPart,
				secondPart,
				filteredSections[i - 1]["type"],
				filteredSections[i - 1]["name"],
				filteredSections[i]["type"],
				filteredSections[i]["name"]
			)

	return None, content, None, None, None, None

# split by limit when section is too large
def detectLimit(content, charLimit, limitPercent=0.9, overlapSize=0):
	splitIndex = int(charLimit * limitPercent)

	if splitIndex >= len(content):
		return content.strip(), ""

	# prefer period
	bestIndex = -1
	checkIndex = splitIndex

	while checkIndex > 0:
		if content[checkIndex] == ".":
			bestIndex = checkIndex + 1
			break
		checkIndex -= 1

	# then prefer whitespace
	if bestIndex == -1:
		checkIndex = splitIndex

		while checkIndex > 0:
			if content[checkIndex].isspace():
				bestIndex = checkIndex
				break
			checkIndex -= 1

	# force split if needed
	if bestIndex == -1:
		bestIndex = splitIndex

	# make sure first part does not end in middle of word
	while bestIndex > 0 and bestIndex < len(content):
		if content[bestIndex - 1].isalnum() and content[bestIndex].isalnum():
			bestIndex -= 1
		else:
			break

	firstPart = content[:bestIndex].strip()

	secondStart = bestIndex - overlapSize
	if secondStart < 0:
		secondStart = 0

	# move overlap start back to the start of a whole word
	while secondStart > 0 and secondStart < len(content):
		if content[secondStart - 1].isalnum() and content[secondStart].isalnum():
			secondStart -= 1
		else:
			break

	secondPart = content[secondStart:].strip()

	return firstPart, secondPart

# - - - - - - - - - - #
