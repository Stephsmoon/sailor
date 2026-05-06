# Sailor: Hierarchical LLM Document Stack Summarizer

**Repository:** https://github.com/Stephsmoon/sailor/tree/main

**Team Members:**

- Stephan DeLuna
- Jocelyn Rogers
- Cole May
- Jacob Ray
- Mason Fox

Sailor is a capstone project for summarizing long documents and stacks of related documents using a hierarchical Large Language Model pipeline. The system accepts `.txt`, `.pdf`, and `.docx` inputs, extracts and cleans text, splits documents into manageable chunks, summarizes each chunk, recursively reduces those summaries, and exports a final survey-style report. The main goal is to demonstrate a practical two-stage approach: first summarize each document individually, then aggregate the document-level summaries into a cohesive overview.

```text
Document(s)
  -> extract text
  -> clean text
  -> chunk text
  -> summarize chunks with an LLM
  -> recursively reduce summaries
  -> synthesize document stack
  -> export report files and evaluator logs
```

---

## Supported Input Types

The loader supports:

```text
.txt
.pdf
.docx
```

Unsupported file types will raise an error.

---

## Project Layout

A typical project layout is:

```text
sailor-code/
├── driver.py
├── README.md
├── .env.example
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
    ├── alice.txt
    ├── jfk.txt
    ├── rag.pdf
    └── testables/
```

Run commands from the project root:

```bash
cd /mnt/mars/Capstone/sailor-code
```

The `storage/` folder is used for local test inputs and generated outputs. It should usually be ignored by Git because it can contain large files, generated reports, logs, and model trial results.

---

## Python Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch transformers accelerate bitsandbytes pypdf python-docx reportlab tqdm fastapi uvicorn python-multipart
```

If installing on an NVIDIA GPU server, install the PyTorch CUDA build recommended for your CUDA version first, then install the remaining packages.

Recommended `requirements.txt`:

```txt
torch>=2.6.0
transformers>=4.51.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
pypdf>=4.0.0
python-docx>=1.1.0
reportlab>=4.0.0
tqdm>=4.66.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
sentencepiece>=0.2.0
safetensors>=0.4.0
```

---

## Hugging Face and `.env` Setup

Some Hugging Face models may require login or access approval.

```bash
huggingface-cli login
```

The driver reads model settings from a `.env` file in the project root. The supported model variable names are:

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

Optional cache path:

```env
HF_HOME=/mnt/caelus/huggingface
```

Model selection priority:

```text
1. --model command-line argument
2. MODEL_NAME from .env
3. MODEL from .env
4. HF_MODEL_NAME from .env
5. Qwen/Qwen3-8B fallback
```

The `.env` loader is built into `driver.py`, so `python-dotenv` is not required.

---

## Model Choice

The final prototype used:

```text
Qwen/Qwen3-8B
```

Qwen3-8B was selected because it provided a practical balance between summary quality, throughput, and VRAM usage on the available RTX 3090 hardware. The pipeline disables Qwen-style thinking behavior through the summarization template when supported, because this project already performs decomposition and recursive aggregation at the pipeline level. English-only prompting, retry behavior, and repair logic are used to reduce non-English or malformed output.

The code can still run other Hugging Face causal language models if they are compatible with the same Transformers interface and available GPU memory.

---

## Basic Single-File Run

Example using one PDF:

```bash
python driver.py \
  --input storage/rag.pdf \
  --output storage/results_rag \
  --outputName rag_test \
  --removeCitation true \
  --outputTypes txt,docx,pdf
```

This creates files such as:

```text
storage/results_rag/rag_cleaned.txt
storage/results_rag/rag_chunks.txt
storage/results_rag/rag_summaries.txt
storage/results_rag/rag_final.txt
storage/results_rag/rag_test_final.txt
storage/results_rag/rag_test_survey.txt
storage/results_rag/rag_test_survey.docx
storage/results_rag/rag_test_survey.pdf
storage/results_rag/data.txt
storage/results_rag/log.txt
```

---

## Basic Multi-File Run

Example using three files:

```bash
python driver.py \
  --input storage/jfk.txt storage/alice.txt storage/rag.pdf \
  --output storage/results_stack \
  --outputName test_stack_summary \
  --parallelDevices 0,1 \
  --parallelMode file \
  --removeCitation true \
  --outputTypes txt,docx,pdf
```

For multi-document runs, each file is summarized individually first. The final document-level summaries are then combined into a survey-style synthesis.

---

## Output Types

The final report can be exported as TXT, DOCX, PDF, or any comma-separated combination of those formats.

```bash
--outputTypes txt
```

```bash
--outputTypes txt,docx
```

```bash
--outputTypes txt,docx,pdf
```

The driver always saves the final combined summary as a `.txt` file. The selected `--outputTypes` control the survey report formats.

---

## Full Capstone-Style Run

```bash
python driver.py \
  --input storage/jfk.txt storage/alice.txt storage/rag.pdf \
  --output storage/results_full \
  --outputName test_stack_summary \
  --title "Multi-Document Survey Report" \
  --parallelDevices 0,1 \
  --parallelMode file \
  --removeCitation true \
  --removeUnicode true \
  --removeSymbols true \
  --removeStutters true \
  --removeFillers true \
  --excludeRepeatWords true \
  --sectionType chapter \
  --charLimit 8000 \
  --limitPercent 0.9 \
  --overlapSize 200 \
  --orderType firstToLast \
  --use4Bit true \
  --doSample false \
  --sentenceLimit 10 \
  --maxNewTokens 400 \
  --repeatCharLimit 20000 \
  --repeatMaxRounds 5 \
  --repeatMaxNewTokens 220 \
  --generateIntroduction true \
  --generateConclusion true \
  --outputTypes txt,docx,pdf
```

---

## Important Command-Line Arguments

### Input and Output

```text
--input            One or more input files
--output           Output directory
--outputName       Base name for final output files
--outputTypes      Comma-separated report formats: txt,docx,pdf
--dataFileName     Name of evaluator data file, default: data.txt
--title            Optional report title
--authorName       Optional report author name
--abstractText     Optional abstract text
```

### Cleaning Arguments

```text
--removeCitation
--removeUnicode
--removeSymbols
--removeStutters
--removeFillers
--excludeRepeatWords
--stutterList
--fillerList
```

Example:

```bash
--removeCitation true --removeUnicode true --removeSymbols true
```

Custom filler or stutter lists use comma-separated values:

```bash
--fillerList very,really,literally
```

### Chunking Arguments

```text
--charLimit
--sectionType
--limitPercent
--overlapSize
```

Valid section types:

```text
chapter
page
paragraph
```

Example:

```bash
--charLimit 8000 --sectionType chapter --limitPercent 0.9 --overlapSize 200
```

The current implementation uses section-aware and character-limit splitting with optional overlap. It is not a tokenizer-count chunker.

### Chunk Ordering Arguments

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

### Model and GPU Arguments

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
```

Recommended deterministic setting:

```bash
--doSample false
```

`temperature` and `topP` only affect generation when sampling is enabled.

### Survey and Report Arguments

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
--conclusionText "Together, these documents show the main themes in the source set."
```

To disable generated introduction and conclusion:

```bash
--generateIntroduction false --generateConclusion false
```

### Recursive Reduction Arguments

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
--repeatCharLimit 20000 --repeatMaxRounds 5 --repeatMaxNewTokens 220
```

---

## How the Pipeline Works

### 1. Ingest

`ingest/extract.py` loads `.txt`, `.pdf`, and `.docx` files into raw text. TXT files are read with fallback encodings to better support older text documents.

### 2. Clean

`ingest/clean.py` applies optional cleaning steps such as citation removal, Unicode normalization, symbol cleanup, filler-word removal, stutter removal, and repeated-word filtering.

### 3. Chunk

`chunk/chunk.py` and `chunk/section.py` split the cleaned text into section-aware chunks. If a detected section is too large, the system falls back to character-limit splitting with optional overlap.

### 4. Summarize Chunks

`model/summarize.py` loads the selected LLM and summarizes each chunk. Progress bars are shown when `tqdm` is installed.

### 5. Recursive Reduction

`model/repeat.py` combines chunk summaries into larger summaries. If the aggregated text is still too large, it is re-chunked and summarized again until a final document-level summary is produced.

### 6. Multi-Document Synthesis

For document stacks, the final document summaries are combined into one survey-style synthesis. The report can also include generated introduction and conclusion paragraphs.

### 7. Export

`report/output.py` exports the final report in the selected output formats.

---

## Generated Files

Each run can produce:

```text
<name>_cleaned.txt       Cleaned text after preprocessing
<name>_chunks.txt        Chunked text used for summarization
<name>_summaries.txt     First-pass chunk summaries
<name>_final.txt         Final document or combined summary
<name>_survey.txt        Survey report in TXT format
<name>_survey.docx       Survey report in DOCX format
<name>_survey.pdf        Survey report in PDF format
data.txt                 Evaluator data log
log.txt                  Runtime log
```

The exact survey formats depend on `--outputTypes`.

---

## Evaluator Data and Logging

The driver creates `data.txt` for evaluation and `log.txt` for runtime details.

`data.txt` records:

```text
run configuration
GPU throughput metrics
word count before and after cleaning
removed citation counts
chunk character lengths
chunk text before summarization
chunk summary after summarization
recursive reduction logs
final survey synthesis
introduction and conclusion text
```

`log.txt` stores detailed runtime information so the terminal can stay cleaner while preserving traceability.

---

## GPU Architecture

The dual-GPU implementation uses file-level data parallelism.

```text
GPU 0: full model copy -> assigned documents
GPU 1: full model copy -> assigned documents
```

It does not split one model across multiple GPUs. This design is simple and appropriate for the project because the workload naturally consists of multiple files and many chunks.

Use file-level parallelism when:

```text
- multiple input files are being summarized
- the selected model fits on each GPU
- faster multi-document processing is desired
```

Use single-GPU mode when:

```text
- summarizing one input file
- the model barely fits on one GPU
- avoiding duplicate model loads is preferred
```

Single-GPU example:

```bash
--parallelMode none --deviceId 0
```

Dual-GPU example:

```bash
--parallelDevices 0,1 --parallelMode file
```

---

## Optional API Server

The project can also be wrapped with a simple FastAPI server for file upload, job status checking, and result download. The API layer is intended to call the same `driver.py` pipeline rather than replacing it.

Install API dependencies:

```bash
pip install fastapi uvicorn python-multipart
```

Run the API server:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

Use one worker on the GPU server so multiple API workers do not accidentally load multiple model copies and exhaust VRAM.

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

Also consider a smaller model.

### Model downloads every run

Set `HF_HOME` in `.env`:

```env
HF_HOME=/mnt/caelus/huggingface
```

### Output contains non-English text

The pipeline uses English-only prompts, CJK detection, retry behavior, and repair logic. For stricter behavior:

```bash
--failOnNonEnglish true
```

### Report is too short

Increase token limits:

```bash
--maxNewTokens 500 --repeatMaxNewTokens 500 --reportSectionMaxNewTokens 400
```

### Report is too long

Decrease token limits:

```bash
--maxNewTokens 220 --repeatMaxNewTokens 220 --reportSectionMaxNewTokens 180
```

### TXT file encoding error

The text loader supports common fallback encodings, including UTF-8 and Windows-style encodings. If a source text still fails to load, convert it to UTF-8 before running the pipeline.

---

## Minimal Command Cheat Sheet

Single file:

```bash
python driver.py --input storage/rag.pdf --output storage/results_rag --outputName rag_test
```

Multiple files:

```bash
python driver.py --input storage/jfk.txt storage/alice.txt storage/rag.pdf --output storage/results_stack --outputName stack_test
```

Multiple files with two GPUs:

```bash
python driver.py --input storage/jfk.txt storage/alice.txt storage/rag.pdf --output storage/results_stack --outputName stack_test --parallelDevices 0,1 --parallelMode file
```

Only TXT survey output:

```bash
python driver.py --input storage/rag.pdf --output storage/results_rag --outputName rag_test --outputTypes txt
```

Use `.env` model:

```env
MODEL_NAME=Qwen/Qwen3-8B
```

Then run:

```bash
python driver.py --input storage/rag.pdf --output storage/results_rag --outputName rag_test
```

---

## Notes for Capstone Evaluation

This project should be understood as a working research prototype for hierarchical document-stack summarization. It demonstrates document ingestion, configurable cleaning, chunking, local LLM summarization, recursive aggregation, multi-document survey synthesis, GPU execution, report export, and evaluator logging. Generated reports are intended as survey-style drafts or review aids and should be reviewed by a human before being used as final academic writing.

