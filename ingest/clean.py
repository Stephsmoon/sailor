# clean.py
# Stephan DeLuna and Jocelyn Rogers
# clean text

# - - - - - - - - - - #

import re
import unicodedata

# - - - - - - - - - - #

naturals = {',', '.', '?', '!', ';', ':', '(', ')', '[', ']', '{', '}', '"'}
stutterWords = ["um", "uh", "oh", "er", "ah"]
fillerWords = ["very", "really", "highly", "like", "just", "totally", "literally", "seriously"]
repeatWords = ["really", "very", "highly", "just", "totally", "literally", "seriously", "like"]
morphemes = ["ing", "ed", "ly", "s"]

# - - - - - - - - - - #

# break content into words and punctuation
def breakContent(content):
	tokens = []
	currentWord = ""
	openStack = []
	bracketPairs = {
		'(': ')',
		'[': ']',
		'{': '}'
	}
	closingBrackets = {')', ']', '}'}
	for i in range(len(content)):
		char = content[i]
		prevChar = ""
		nextChar = ""
		if i > 0:
			prevChar = content[i - 1]
		if i + 1 < len(content):
			nextChar = content[i + 1]
		# keep building word
		if char.isalnum():
			currentWord += char
		# keep apostrophe inside a word
		elif char == "'" and prevChar.isalnum() and nextChar.isalnum():
			currentWord += char
		else:
			# save finished word
			if currentWord != "":
				tokens.append(currentWord)
				currentWord = ""
			# save punctuation as its own token
			if char in naturals:
				tokens.append(char)
				# track brackets
				if char in bracketPairs:
					openStack.append(char)
				elif char in closingBrackets:
					if len(openStack) > 0:
						lastOpen = openStack[-1]
						if bracketPairs[lastOpen] == char:
							openStack.pop()
			elif char == "\n":
				tokens.append("\n")
	# save final word
	if currentWord != "":
		tokens.append(currentWord)
	# add missing ending brackets
	while len(openStack) > 0:
		lastOpen = openStack.pop()
		tokens.append(bracketPairs[lastOpen])
	return tokens

# turn tokens back into text if needed later
def joinTokens(tokens):
	text = ""
	for i in range(len(tokens)):
		token = tokens[i]
		prevToken = None
		nextToken = None
		if i > 0:
			prevToken = tokens[i - 1]
		if i + 1 < len(tokens):
			nextToken = tokens[i + 1]
		if token == "\n":
			text += "\n"
		# punctuation that sticks to the word before it
		elif token in {",", ".", "?", "!", ";", ":", ")", "]", "}"}:
			text += token
			# add a space after punctuation only if next token is a normal word
			if nextToken != None and nextToken != "\n" and nextToken not in naturals and nextToken != '"':
				text += " "
		# opening brackets
		elif token in {"(", "[", "{"}:
			if text != "" and not text.endswith((" ", "\n")):
				text += " "
			text += token
		# quotes
		elif token == '"':
			# opening quote if start, after newline, or after space-like punctuation
			if prevToken == None or prevToken == "\n" or prevToken in {"(", "[", "{", ",", ";", ":"}:
				if text != "" and not text.endswith((" ", "\n", "(", "[", "{")):
					text += " "
				text += token
			# closing quote if after word-ending punctuation or word
			else:
				text += token
				# only add space after closing quote if next token is a word
				if nextToken != None and nextToken != "\n" and nextToken not in naturals:
					text += " "
		# regular words
		else:
			if text != "" and not text.endswith((" ", "\n", "(", "[", "{", '"')):
				text += " "
			text += token
	return text

# remove citation noise from text
def removeCitationNoise(content):
	removedCitations = []
	citationPatterns = [
		r'\[\d+\]',
		r'\(\d+\)',
		r'\[[^\]]*et al\.[^\]]*\]',
		r'\([A-Z][A-Za-z]+,\s*\d{4}\)',
		r'\([A-Z][A-Za-z]+\s+et al\.,\s*\d{4}\)'
	]
	for pattern in citationPatterns:
		matches = re.findall(pattern, content)
		for match in matches:
			removedCitations.append(match)
		content = re.sub(pattern, " ", content)
	return content, removedCitations

# remove bad unicode from text
def removeBadUnicode(content):
	content = unicodedata.normalize("NFKD", content)
	content = content.encode("ascii", "ignore").decode("ascii")
	return content

# remove weird symbols from text
def removeWeirdSymbols(content):
	content = re.sub(r'[^A-Za-z0-9\s,\.\?\!\;\:\(\)\[\]\{\}"\']', ' ', content)
	return content

# remove invalid word tokens
def removeBadTokens(tokens):
	newTokens = []
	for token in tokens:
		# keep punctuation and newlines
		if token in naturals or token == "\n":
			newTokens.append(token)
			continue
		# keep words with letters, numbers, and apostrophes
		if re.match(r"^[A-Za-z0-9']+$", token):
			hasLetter = False
			for char in token:
				if char.isalpha():
					hasLetter = True
					break
			if hasLetter:
				newTokens.append(token)
	return newTokens

# - - - - - - - - - - #

# remove filler and stutter words
def removeFiller(tokens, removeStutter=False, removeFillerWords=False, stutterList=None, fillerList=None):
	if stutterList == None:
		stutterList = stutterWords
	if fillerList == None:
		fillerList = fillerWords
	removeWords = []
	if removeStutter:
		for word in stutterList:
			removeWords.append(word.lower())
	if removeFillerWords:
		for word in fillerList:
			removeWords.append(word.lower())
	newTokens = []
	for token in tokens:
		if token in naturals or token == "\n":
			newTokens.append(token)
			continue
		if token.lower() not in removeWords:
			newTokens.append(token)
	return newTokens

# remove repeated words 
def removeRepetition(tokens):
	newTokens = []
	lastWord = None
	for token in tokens:
		if token in naturals or token == "\n":
			newTokens.append(token)
			lastWord = None
			continue
		lowerToken = token.lower()
		# skip if same as last word
		if lastWord == lowerToken:
			continue
		newTokens.append(token)
		lastWord = lowerToken
	return newTokens

# check and remove morphemes in one function
def removeMorphemes(tokens, morphemeList=None):
	if morphemeList == None:
		morphemeList = morphemes
	newTokens = []
	morphemeWords = []
	for token in tokens:
		if token in naturals or token == "\n":
			newTokens.append(token)
			continue
		newWord = token
		changedWord = False
		for morpheme in morphemeList:
			lowerWord = newWord.lower()
			lowerMorpheme = morpheme.lower()
			if lowerWord.endswith(lowerMorpheme) and len(newWord) > len(morpheme):
				morphemeWords.append(token)
				newWord = newWord[:-len(morpheme)]
				changedWord = True
				break
		if newWord != "":
			newTokens.append(newWord)
		elif not changedWord:
			newTokens.append(token)
	return newTokens, morphemeWords

# clean content and return ordered array
def cleanContent(
	content,
	removeCitation=True,
	removeUnicode=True,
	removeSymbols=True,
	removeStutters=False,
	removeFillers=False,
	excludeRepeatWords=False,
	removeWordMorphemes=False,
	stutterList=None,
	fillerList=None,
	morphemeList=None
):
	removedCitations = []
	removedMorphemeWords = []
	# clean raw string first
	if removeUnicode:
		content = removeBadUnicode(content)
	if removeCitation:
		content, removedCitations = removeCitationNoise(content)
	if removeSymbols:
		content = removeWeirdSymbols(content)
	# break into ordered array
	tokens = breakContent(content)
	# remove invalid tokens
	tokens = removeBadTokens(tokens)
	# optional cleanup steps
	if removeStutters or removeFillers:
		tokens = removeFiller(
			tokens,
			removeStutter=removeStutters,
			removeFillerWords=removeFillers,
			stutterList=stutterList,
			fillerList=fillerList
		)
	if excludeRepeatWords:
		tokens = removeRepetition(tokens)
	if removeWordMorphemes:
		tokens, removedMorphemeWords = removeMorphemes(tokens, morphemeList)
	return tokens, removedCitations, removedMorphemeWords

# - - - - - - - - - - #

