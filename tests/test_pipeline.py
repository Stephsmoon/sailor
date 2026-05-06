# Jocelyn Rogers

import os
import re
import pytest
from docx import Document
from ingest.extract import loadTxt, loadDocx, loadPdf
from ingest.clean import breakContent, joinTokens, removeCitationNoise, removeBadTokens, removeBadUnicode, removeFiller, removeRepetition, removeWeirdSymbols
from chunk.section import countWords, detectSections, detectLimit
from chunk.chunk import detectFirstSectionName, chunkText, organizeChunks
from report.output import buildBodyText, buildSurveyPaper, saveDocx,savePdf,saveTxt
from model.summarize import containsCJK, stripThinkingBlocks, cleanSummaryEdges, buildPrompt, prepareMessages, applyChatTemplate
from model.repeat import getSectionName,combineSectionSummaries,combineSectionSummaries,hasMultiPartSections

# download the attached AliceInWonderlandSample files and place into the storage folder, or maually change the variable to have the filepath
baseDir = os.path.dirname(os.path.dirname(__file__))
storage = os.path.join(baseDir, "storage")
filepathTXT = os.path.join(storage, "AliceInWonderlandSample.txt")
filepathPDF = os.path.join(storage, "AliceInWonderlandSample.pdf")
filepathDOCX = os.path.join(storage, "AliceInWonderlandSample.docx")

# ------
# HELPER
# ------

# check that the extracted text has key/recognizable words in the output
keyPhrases = ["chapter I", "alice was beginning","drink me","orange marmalade","white rabbit with pink eyes","cake","sister","drink me", "poison"]

# -----
# TESTS
# -----

# --------------
# INGEST - CLEAN
# --------------

# cleanContent is a wrapper function for other functions, so it does not need to be tested

input = "Hopefully, there should be (13) elements in this array."
breakContentInput = "Hopefully, there should be (13) elements in this array."
def test_breakContent():
    assert breakContent(breakContentInput) == ["Hopefully",",","there","should","be","(","13",")","elements","in","this","array","."]

input = ["Hopefully",",","there","should","be","(","13",")","elements","in","this","array","."]
def test_joinTokens():
    assert joinTokens(input) == "Hopefully, there should be (13) elements in this array."

inputText = "We are at a café 😊$$$ yeutr783tir. This is, like, seriously interesting. According to recent studies (Smith, 2020) and [12], This input includes weird unicode: café, naïve, coöperate, and emojis 😊🔥. Also strange symbols: @#$%^&*≈≠±§. There were bad tokens like th!s, 123abc!, and ???."
inputTokens = ["Um",",","this","is","a","really","interesting","interesting","café","😊","$$$","y!tr"]
def test_removeCitationNoise():
    output, removed = removeCitationNoise(inputText)
    for r in ["(Smith, 2020)", "[12]"]:
        assert r in removed

def test_removeBadTokens():
    assert removeBadTokens(inputTokens) == ["Um",",","this","is","a","really","interesting","interesting"]

def test_removeBadUnicode():
    assert removeBadUnicode(inputText) == "We are at a cafe $$$ yeutr783tir. This is, like, seriously interesting. According to recent studies (Smith, 2020) and [12], This input includes weird unicode: cafe, naive, cooperate, and emojis . Also strange symbols: @#$%^&*=. There were bad tokens like th!s, 123abc!, and ???."

def test_removeFiller():
    assert removeFiller(inputTokens) == [",","this","is","a","interesting","interesting","café","😊","$$$","y!tr"]

def test_removeRepetition():
    assert removeRepetition(inputTokens) == ["Um",",","this","is","a","really","interesting","café","😊","$$$","y!tr"]

def test_removeWeirdSymbols():
    for w in ["$$$", "@","#","$","%","=","&","*","^"]:
        assert w not in removeWeirdSymbols(inputText)

# ---------------
# CHUNK - SECTION
# ---------------

def test_countWords():
    assert countWords("There are four words.") == {"there":1,"are":1,"four":1,"words":1}

def test_detectSectionsChapter():
    _, output, _, _, _, _ = detectSections("Chapter I\nThis is the first chapter.",50)

    assert "chapter" in output
    assert "I" in output

def test_detectLimit():
    first, second = detectLimit("Hello world. This is a test.",20)

    assert first.endswith(".")
    assert second.startswith("This")

# -------------
# CHUNK - CHUNK
# -------------

def test_detectFirstSectionName():
    type, name = detectFirstSectionName("CHAPTER III\nSome text here.")

    assert type == "chapter"
    assert name == "III"

def test_chunkText():
    chunks = chunkText("CHAPTER I\nAlpha alpha alpha.\nCHAPTER II\nBeta beta beta.",50)

    assert "chapter I" in chunks
    assert "chapter II" in chunks
    assert "alpha" in chunks["chapter I"]
    assert "beta" in chunks["chapter II"]

def test_organizeChunks():
    organized = organizeChunks({"chapter 3":"third text.","chapter 1":"first text.","chapter 2":"second text."})
    keys = list(organized.keys())

    assert keys == ["chapter 1","chapter 2","chapter 3"]

# ---------------
# REPORT - OUTPUT
# ---------------

def test_buildBodyText():
    output = buildBodyText({"paper":"Overall summary text.","chapter 1":"First chapter summary text."})

    assert "Survey Summary" in output
    assert "Chapter 1" in output
    assert "Overall summary text." in output
    assert "First chapter summary text." in output

def test_buildSurveyPaper():
    output = buildSurveyPaper("My Title","Author Name","Intro text.", {"section a":"AAA","section b":"BBB"},"The end.", "Abstract text.")

    assert output.startswith("My Title")
    assert "Author Name" in output
    assert "Abstract" in output
    assert "Introduction" in output
    assert "Section A" in output
    assert "Conclusion" in output

def test_saveTxt():
    filepath = os.path.join(storage, "save.txt")
    saveTxt(filepath,"Hello world.")

    assert os.path.exists(filepath)

    with open(filepath,"r",encoding="utf-8") as f:
        assert f.read() == "Hello world."

def test_saveDocx():
    filepath = os.path.join(storage,"save.docx")
    saveDocx(filepath,"My Title","Author","Intro",{"section a":"AAA","section b":"BBB"},"Done")

    assert os.path.exists(filepath)

    doc = Document(filepath)
    text = "\n".join(p.text for p in doc.paragraphs)

    assert "My Title" in text
    assert "Author" in text
    assert "Intro" in text
    assert "AAA" in text
    assert "Done" in text

def test_savePdf():
    filepath = os.path.join(storage,"save.pdf")

    savePdf(filepath,"My Title","Author","Intro",{"section a":"AAA","section b":"BBB"},"Done")

    assert os.path.exists(filepath)
    assert os.path.getsize(filepath) > 0

# -----------------
# MODEL - SUMMARIZE
# -----------------

def test_containsCJK():
    assert containsCJK("Hello 世界") is True
    assert containsCJK("Just English text") is False

def test_stripThinkingBlocks():
    output = stripThinkingBlocks("Answer: <think>internal reasoning</think> final output.")
    assert "<think>" not in output
    assert "internal reasoning" not in output
    assert "final output" in output

def test_cleanSummaryEdges():
    output = cleanSummaryEdges("Summary: This is a test.\n\n More text! ")

    assert output == "This is a test. More text!"

def test_buildPrompt():
    output = buildPrompt("chunk text",sentenceLimit=4)

    assert "4 sentences" in output

def test_prepareMessages():
    prepared = prepareMessages([{"role":"system","content":"System message"},{"role":"user","content":"User message"}],False)

    assert "/no_think" in prepared[0]["content"]
    assert prepared[1]["content"].startswith("/no_think")

# --------------
# MODEL - REPEAT
# --------------

def test_getSectionName():
    assert getSectionName("chapter 3.2") == "chapter 3"
    assert getSectionName("chapter 5") == "chapter 5"
    assert getSectionName("front") == "front"

def test_combineSectionSummaries():
    output = combineSectionSummaries({"chapter 1.2":"Second part.","chapter 1.1":"First part.","chapter 2":"Another chapter."})

    assert "chapter 1" in output
    assert "chapter 2" in output
    assert output["chapter 1"].startswith("First part.")
    assert "Second part." in output["chapter 1"]

def test_hasMultiPartSections():
    assert hasMultiPartSections({"chapter 1.1":"A","chapter 1.2":"B"}) is True
    assert hasMultiPartSections({"chapter 1":"A","chapter 2":"B"}) is False
