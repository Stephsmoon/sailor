# LLM Multi-Document Survey Summarization Pipeline

This project summarizes one or more `.txt`, `.pdf`, or `.docx` files using a staged LLM pipeline:

```text
Document(s)
  -> ingest/extract text
  -> clean text
  -> chunk text
  -> summarize chunks
  -> recursively reduce summaries
  -> create final survey-style report
  -> export TXT, DOCX, and PDF outputs
```

The pipeline supports both single-document summarization and multi-document survey synthesis. For multi-document runs, each document is summarized individually first, then the individual summaries are combined into one higher-level survey synthesis. The final report can also include a generated introduction and conclusion.

---

## Supported Input Types

The document loader supports:

```text
.txt
.pdf
.docx
```

Unsupported file types will raise an error.

---

## Recommended Project Layout

A typical project layout should look like this:

```text
sailor-code/
├── driver.py
├── .env
├── README.md
├── ingest/
│   ├── extract.py
│   └── clean.py
├── chunk/
│   ├── chunk.py
│   └── section.py
├── model/
│   ├── summarize.py
│   └── repeat.py
├── report/
│   └── output.py
└── storage/
    ├── jfk.txt
    ├── alice.txt
    └── rag.pdf
```

Run commands from the project root, not from inside `storage/`.

---

## Python Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install torch transformers accelerate bitsandbytes pypdf python-docx reportlab
```

If your PyTorch/CUDA setup needs a specific wheel, install PyTorch using the command from the official PyTorch selector for your CUDA version first, then install the remaining packages.

---

## Hugging Face Setup

Some models require Hugging Face access approval or login.

Login if needed:

```bash
huggingface-cli login
```

Optional but recommended: set a Hugging Face cache directory in `.env`:

```env
HF_HOME=/mnt/caelus/huggingface
```

This keeps downloaded models out of the project folder.

---

## `.env` Model Configuration

The driver automatically reads a `.env` file from the project root.

Use one of these variable names:

```env
MODEL_NAME=Qwen/Qwen3-8B
```

or:

```env
MODEL=Qwen/Qwen3-8B
```

or:

```env
HF_MODEL_NAME=Qwen/Qwen3-8B
```

The priority is:

```text
1. --model command-line argument
2. MODEL_NAME from .env
3. MODEL from .env
4. HF_MODEL_NAME from .env
5. Qwen/Qwen2.5-7B-Instruct fallback
```

Example `.env`:

```env
MODEL_NAME=Qwen/Qwen3-8B
HF_HOME=/mnt/caelus/huggingface
```

The `.env` loader is built into `driver.py`, so `python-dotenv` is not required.

---

## Model Choice Notes

Recommended starting models:

```text
Qwen/Qwen3-8B
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-7B-Instruct
```

Possible but heavier:

```text
mistralai/Mixtral-8x7B-Instruct-v0.1
```

Important: the current dual-GPU mode uses file-level data parallelism. That means each GPU loads its own full copy of the model. This is excellent for smaller and mid-sized models, but heavy models like Mixtral may run out of VRAM if each 3090 has to hold a full model copy.

For a model that barely fits on one GPU, use:

```bash
--parallelMode none --deviceId 0
```

For models that fit comfortably on each GPU, use:

```bash
--parallelDevices 0,1 --parallelMode file
```

No supported model here is strictly English-only. English output is enforced by prompting, retry logic, CJK detection, and an English repair pass.

Qwen3 has a thinking mode. For this pipeline, thinking is disabled by default because chunking and recursive reduction already do the reasoning structure. Keep this setting unless you specifically want slower reasoning-style outputs:

```bash
--enableThinking false
```

If output quality looks rough, use the optional polish pass. This is slower because it adds an extra editing generation call, but it improves final readability:

```bash
--qualityPolish true
```

The default generation settings also avoid aggressive no-repeat filtering because that can make summaries sound broken or oddly spaced:

```bash
--repetitionPenalty 1.05 --noRepeatNgramSize 0
```

---

## Basic Single-File Run

Example using one PDF:

```bash
python driver.py \
  --input storage/rag.pdf \
  --output storage \
  --outputName rag_test \
  --skipWaits true
```

This produces files like:

```text
storage/rag_cleaned.txt
storage/rag_chunks.txt
storage/rag_summaries.txt
storage/rag_final.txt
storage/rag_test_final.txt
storage/rag_test_survey.txt
storage/rag_test_survey.docx
storage/rag_test_survey.pdf
```

---

## Basic Multi-File Run

Example using the three test files:

```bash
python driver.py \
  --input storage/jfk.txt storage/alice.txt storage/rag.pdf \
  --output storage \
  --outputName test_stack_summary \
  --skipWaits true
```

This produces individual summaries for each input file and one combined survey report:

```text
storage/jfk_cleaned.txt
storage/jfk_chunks.txt
storage/jfk_summaries.txt
storage/jfk_final.txt

storage/alice_cleaned.txt
storage/alice_chunks.txt
storage/alice_summaries.txt
storage/alice_final.txt

storage/rag_cleaned.txt
storage/rag_chunks.txt
storage/rag_summaries.txt
storage/rag_final.txt

storage/test_stack_summary_final.txt
storage/test_stack_summary_survey.txt
storage/test_stack_summary_survey.docx
storage/test_stack_summary_survey.pdf
```

---

## Dual-GPU Multi-File Run

Use this on the dual RTX 3090 setup when the selected model fits on each GPU:

```bash
python driver.py \
  --input storage/jfk.txt storage/alice.txt storage/rag.pdf \
  --output storage \
  --outputName test_stack_summary \
  --skipWaits true \
  --parallelDevices 0,1 \
  --parallelMode file \
  --use4Bit true
```

This assigns documents across GPUs in round-robin order. For example:

```text
GPU 0: jfk.txt, then rag.pdf
GPU 1: alice.txt
```

The combined survey synthesis, introduction, conclusion, and final exports are handled after the individual document summaries are complete.

---

## Full Example Run

This is the recommended capstone-style test command:

```bash
python driver.py \
  --input storage/jfk.txt storage/alice.txt storage/rag.pdf \
  --output storage \
  --outputName test_stack_summary \
  --skipWaits true \
  --parallelDevices 0,1 \
  --parallelMode file \
  --charLimit 8000 \
  --sectionType chapter \
  --limitPercent 0.9 \
  --overlapSize 200 \
  --orderType firstToLast \
  --use4Bit true \
  --maxNewTokens 400 \
  --repeatCharLimit 20000 \
  --repeatMaxNewTokens 500
```

If you want to override the `.env` model directly:

```bash
python driver.py \
  --input storage/jfk.txt storage/alice.txt storage/rag.pdf \
  --output storage \
  --outputName qwen3_test \
  --model Qwen/Qwen3-8B \
  --skipWaits true \
  --parallelDevices 0,1 \
  --parallelMode file \
  --use4Bit true
```

---

## Output Report Format

For one file, the final report is structured like:

```text
Title
Author

Introduction
Generated or default introduction.

Survey Summary
Final summary of the document.

Conclusion
Generated or default conclusion.
```

For multiple files, the final report is structured like:

```text
Multi File Survey Report
Author

Introduction
Generated introduction tying the document set together.

Survey Summary
Combined synthesis across all input documents.

Jfk
Individual final summary for jfk.txt.

Alice
Individual final summary for alice.txt.

Rag
Individual final summary for rag.pdf.

Conclusion
Generated conclusion tying the document set together.
```

The combined `Survey Summary` is meant to synthesize the whole document stack. The individual file sections are included for transparency and review.

---

## Important Command-Line Arguments

### Input and Output

```text
--input                  One or more input files
--output                 Output directory
--outputName             Base name for final combined output files
--title                  Optional report title
--authorName             Report author name
--abstractText           Optional abstract text
--skipWaits              Skip interactive pauses
```

### Cleaning Arguments

These map to `cleanContent`:

```text
--removeCitation
--removeUnicode
--removeSymbols
--removeStutters
--removeFillers
--excludeRepeatWords
--removeWordMorphemes
--stutterList
--fillerList
--morphemeList
```

Example:

```bash
--removeCitation true --removeUnicode true --removeSymbols true
```

Custom word lists use comma-separated values:

```bash
--fillerList very,really,literally
```

### Chunking Arguments

These map to `chunkText`:

```text
--charLimit
--sectionType
--limitPercent
--overlapSize
```

Valid `--sectionType` values:

```text
chapter
page
paragraph
```

Example:

```bash
--charLimit 8000 --sectionType chapter --limitPercent 0.9 --overlapSize 200
```

### Chunk Ordering Arguments

This maps to `organizeChunks`:

```text
--orderType
```

Valid values:

```text
firstToLast
lastToFirst
lastToFirstBySection
firstToLastByTop
```

### Model Loading Arguments

These map to `loadSummarizer`:

```text
--model
--deviceId
--parallelDevices
--parallelMode
--use4Bit
--useYarn
```

Examples:

```bash
--model Qwen/Qwen3-8B
```

```bash
--parallelDevices 0,1 --parallelMode file
```

```bash
--use4Bit true
```

### Summarization Arguments

These map to `summarizeWithLoadedModel` and `summarizeChunk`:

```text
--promptText
--sentenceLimit
--maxNewTokens
--failOnNonEnglish
--maxRetries
--forceEnglishRepair
--doSample
--temperature
--topP
--enableThinking
--repetitionPenalty
--noRepeatNgramSize
--qualityPolish
```

Recommended deterministic summarization:

```bash
--doSample false --temperature 0.3 --topP 0.9
```

The `temperature` and `topP` values only matter when `--doSample true`.

### Survey Synthesis and Report Section Arguments

```text
--surveyPromptText
--generateIntroduction
--generateConclusion
--introductionText
--conclusionText
--introductionPromptText
--conclusionPromptText
--reportSectionSentenceLimit
--reportSectionMaxNewTokens
```

To provide your own introduction and conclusion:

```bash
--introductionText "This report surveys the provided documents." \
--conclusionText "Together, these sources show the main trends in the document set."
```

To turn off generated introduction and conclusion:

```bash
--generateIntroduction false --generateConclusion false
```

### Recursive Reduction Arguments

These map to `repeatSummaries`:

```text
--repeatCharLimit
--repeatLimitPercent
--repeatOverlapSize
--repeatMaxRounds
--repeatSentenceLimit
--repeatMaxNewTokens
```

Example:

```bash
--repeatCharLimit 20000 --repeatMaxRounds 5 --repeatMaxNewTokens 500
```

---

## Interactive Mode

By default, the driver pauses between major stages.

To run without pauses:

```bash
--skipWaits true
```

To keep the original step-by-step behavior, omit `--skipWaits true`.

---

## How the Pipeline Works Internally

### 1. Ingest

The input file is loaded using `loadDocument`.

### 2. Clean

The raw text is cleaned with `cleanContent`, then rebuilt into a text string with `joinTokens`.

### 3. Chunk

The cleaned text is split by section boundaries first. If a section is too large, it is split by character limit.

### 4. Organize

Chunks are reordered using `organizeChunks`.

### 5. First Summary

Each chunk is summarized individually.

### 6. Recursive Reduction

Chunk summaries are merged by section and reduced until a final document-level summary remains.

### 7. Multi-Document Synthesis

For multiple files, the final document-level summaries are combined and reduced into one survey-style synthesis.

### 8. Report Generation

The report writer exports:

```text
.txt
.docx
.pdf
```

---

## GPU Architecture

The dual-GPU implementation uses file-level data parallelism.

It does this:

```text
GPU 0: full model copy -> assigned documents
GPU 1: full model copy -> assigned documents
```

It does not do this:

```text
GPU 0 + GPU 1: one model split across both GPUs
```

This architecture is simple and appropriate for the capstone because the workload naturally consists of many documents and many chunks.

Use file-level parallelism when:

```text
- you have multiple input files
- the model fits on each GPU
- you want better throughput
```

Avoid file-level parallelism when:

```text
- the model barely fits on one GPU
- the model is too large to duplicate
- you are running only one input file
```

---

## Troubleshooting

### CUDA out of memory

Try one or more of these:

```bash
--use4Bit true
--parallelMode none
--deviceId 0
--maxNewTokens 220
--repeatMaxNewTokens 220
--charLimit 6000
```

Also try a smaller model.

### Model downloads every time

Set `HF_HOME` in `.env`:

```env
HF_HOME=/mnt/caelus/huggingface
```

### Output contains Chinese or non-English text

The pipeline already tries to prevent this with English-only prompts, retry logic, CJK detection, and repair. For stricter behavior, use:

```bash
--failOnNonEnglish true
```

### Generated report is too short

Increase output token limits:

```bash
--maxNewTokens 500 --repeatMaxNewTokens 600 --reportSectionMaxNewTokens 400
```

### Generated report is too long

Decrease output token limits:

```bash
--maxNewTokens 220 --repeatMaxNewTokens 220 --reportSectionMaxNewTokens 180
```

### PDF formatting is plain

The PDF exporter is intentionally simple. The DOCX output is usually better for formatting and editing.

---

## Minimal Command Cheat Sheet

Single file:

```bash
python driver.py --input storage/rag.pdf --output storage --skipWaits true
```

Multiple files:

```bash
python driver.py --input storage/jfk.txt storage/alice.txt storage/rag.pdf --output storage --skipWaits true
```

Multiple files with two GPUs:

```bash
python driver.py --input storage/jfk.txt storage/alice.txt storage/rag.pdf --output storage --skipWaits true --parallelDevices 0,1 --parallelMode file
```

Override model:

```bash
python driver.py --input storage/rag.pdf --model Qwen/Qwen3-8B --skipWaits true
```

Use `.env` model:

```env
MODEL_NAME=Qwen/Qwen3-8B
```

Then run:

```bash
python driver.py --input storage/rag.pdf --skipWaits true
```

