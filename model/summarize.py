# summarize.py
# Stephan DeLuna
# summarizes chunked text with an llm

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# - - - - - - - - - - #

defaultPrompt = (
    "Write the summary in English only. "
    "Do not use Chinese or any other language. "
    "If the source contains non-English text, translate it into English. "
    "Summarize the text into one clear paragraph. "
    "Preserve important details naturally, including who, what, when, where, and why when they are present. "
    "Be faithful to the text. "
    "Do not invent, assume, or add facts that are not stated. "
    "Do not quote large parts of the text. "
    "Do not repeat the input. "
    "Do not use labels, bullet points, headings, notes, or bracketed tags. "
    "Use normal English spelling, punctuation, and spacing. "
    "Keep the summary under 6 sentences."
)

# - - - - - - - - - - #

def updatePrompt(newPrompt):
    global defaultPrompt
    defaultPrompt = newPrompt.strip()

# - - - - - - - - - - #

def containsCJK(text):
    return re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text) is not None

def stripThinkingBlocks(text):
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<think>.*', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    return text

def cleanSummaryEdges(text):
    text = text.strip()
    text = stripThinkingBlocks(text)
    text = re.sub(r'^\s*(summary\s*:|english summary\s*:|final summary\s*:|answer\s*:|rewrite\s*:|polished summary\s*:|clean summary\s*:|introduction\s*:|conclusion\s*:)+\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,.;:!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\s+([\)\]\}])', r'\1', text)
    text = re.sub(r'([\(\[\{])\s+', r'\1', text)
    return text.strip()

# - - - - - - - - - - #

def loadSummarizer(
    modelName="Qwen/Qwen2.5-7B-Instruct",
    deviceId=0,
    use4Bit=True,
    useYarn=False
):
    tokenizer = AutoTokenizer.from_pretrained(
        modelName,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantConfig = None
    if use4Bit:
        quantConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quantConfig,
        device_map={"": deviceId}
    )

    model.eval()

    if useYarn:
        # placeholder for future YaRN config
        pass

    return tokenizer, model

# - - - - - - - - - - #

def buildPrompt(chunkText, promptText=None, sentenceLimit=6):
    if promptText is None:
        promptText = defaultPrompt

        # update sentence limit in the default behavior if caller changes it
        promptText = re.sub(
            r'Keep the summary under \d+ sentences\.',
            f'Keep the summary under {sentenceLimit} sentences.',
            promptText
        )

    return promptText

# - - - - - - - - - - #

def prepareMessages(messages, enableThinking=False):
    if enableThinking:
        return messages

    preparedMessages = []

    for message in messages:
        newMessage = dict(message)
        content = newMessage.get("content", "")

        # Qwen3 supports /no_think as a soft fallback. The hard switch is also passed below.
        if isinstance(content, str) and "/no_think" not in content:
            if newMessage.get("role") == "system":
                content = content.strip() + " /no_think"
            elif newMessage.get("role") == "user":
                content = "/no_think\n" + content

        newMessage["content"] = content
        preparedMessages.append(newMessage)

    return preparedMessages

def applyChatTemplate(tokenizer, messages, enableThinking=False):
    preparedMessages = prepareMessages(messages, enableThinking=enableThinking)

    try:
        return tokenizer.apply_chat_template(
            preparedMessages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enableThinking
        )
    except TypeError:
        # Older chat templates do not accept enable_thinking. The /no_think fallback remains in the prompt.
        return tokenizer.apply_chat_template(
            preparedMessages,
            tokenize=False,
            add_generation_prompt=True
        )

def generateFromMessages(
    messages,
    tokenizer,
    model,
    maxNewTokens=220,
    doSample=False,
    temperature=0.3,
    topP=0.9,
    enableThinking=False,
    repetitionPenalty=1.05,
    noRepeatNgramSize=0
):
    inputText = applyChatTemplate(
        tokenizer,
        messages,
        enableThinking=enableThinking
    )

    inputs = tokenizer(inputText, return_tensors="pt").to(model.device)

    generationArgs = {
        "max_new_tokens": maxNewTokens,
        "do_sample": doSample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True
    }

    if repetitionPenalty != None and repetitionPenalty != 1.0:
        generationArgs["repetition_penalty"] = repetitionPenalty

    if noRepeatNgramSize != None and noRepeatNgramSize > 0:
        generationArgs["no_repeat_ngram_size"] = noRepeatNgramSize

    if doSample:
        generationArgs["temperature"] = temperature
        generationArgs["top_p"] = topP

    outputIds = model.generate(**inputs, **generationArgs)

    inputLength = inputs["input_ids"].shape[1]
    newTokenIds = outputIds[0][inputLength:]
    summary = tokenizer.decode(newTokenIds, skip_special_tokens=True).strip()

    return cleanSummaryEdges(summary)

# - - - - - - - - - - #

def repairToEnglish(
    summaryText,
    tokenizer,
    model,
    maxNewTokens=220,
    enableThinking=False,
    repetitionPenalty=1.05,
    noRepeatNgramSize=0
):
    repairMessages = [
        {
            "role": "system",
            "content": (
                "You are an English-only copy editor. "
                "Output one polished English paragraph only. "
                "Use correct spelling, punctuation, and spacing."
            )
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following summary into clean English only. "
                "Preserve the meaning exactly. "
                "Do not add facts. "
                "Translate or remove any non-English characters. "
                "Fix broken word spacing and obvious spelling issues.\n\n"
                f"Summary:\n{summaryText}\n\n"
                "Clean English summary:"
            )
        }
    ]

    repaired = generateFromMessages(
        repairMessages,
        tokenizer,
        model,
        maxNewTokens=maxNewTokens,
        doSample=False,
        enableThinking=enableThinking,
        repetitionPenalty=repetitionPenalty,
        noRepeatNgramSize=noRepeatNgramSize
    )

    return repaired

def polishSummaryText(
    summaryText,
    tokenizer,
    model,
    maxNewTokens=220,
    enableThinking=False,
    repetitionPenalty=1.05,
    noRepeatNgramSize=0
):
    if summaryText.strip() == "":
        return summaryText

    polishMessages = [
        {
            "role": "system",
            "content": (
                "You are a careful copy editor. "
                "Output one polished English paragraph only. "
                "Do not add new facts."
            )
        },
        {
            "role": "user",
            "content": (
                "Polish the following summary for readability. "
                "Keep the same facts and meaning. "
                "Fix only English grammar, spelling, punctuation, and word spacing. "
                "Do not add details.\n\n"
                f"Summary:\n{summaryText}\n\n"
                "Polished summary:"
            )
        }
    ]

    polished = generateFromMessages(
        polishMessages,
        tokenizer,
        model,
        maxNewTokens=maxNewTokens,
        doSample=False,
        enableThinking=enableThinking,
        repetitionPenalty=repetitionPenalty,
        noRepeatNgramSize=noRepeatNgramSize
    )

    if polished.strip() == "":
        return summaryText

    return polished

# - - - - - - - - - - #

@torch.no_grad()
def summarizeChunk(
    chunkText,
    tokenizer,
    model,
    deviceId=0,
    promptText=None,
    sentenceLimit=6,
    maxNewTokens=220,
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
    instructionText = buildPrompt(chunkText, promptText, sentenceLimit)

    baseMessages = [
        {
            "role": "system",
            "content": (
                "You are a careful summarizer. "
                "Output only the summary in English. "
                "Never use Chinese or any other language. "
                "Use normal English spelling, punctuation, and spacing."
            )
        },
        {
            "role": "user",
            "content": instructionText + "\n\nText:\n" + chunkText + "\n\nSummary:"
        }
    ]

    # first attempt: deterministic unless the caller explicitly enables sampling
    summary = generateFromMessages(
        baseMessages,
        tokenizer,
        model,
        maxNewTokens=maxNewTokens,
        doSample=doSample,
        temperature=temperature,
        topP=topP,
        enableThinking=enableThinking,
        repetitionPenalty=repetitionPenalty,
        noRepeatNgramSize=noRepeatNgramSize
    )

    if not containsCJK(summary):
        if qualityPolish:
            summary = polishSummaryText(
                summary,
                tokenizer,
                model,
                maxNewTokens=maxNewTokens,
                enableThinking=enableThinking,
                repetitionPenalty=repetitionPenalty,
                noRepeatNgramSize=noRepeatNgramSize
            )
        return summary

    print("[WARNING] Non-English output detected on first pass.")
    print(summary[:300])

    # retries: deterministic, stronger wording
    retryCount = 0
    while retryCount < maxRetries:
        retryMessages = [
            {
                "role": "system",
                "content": (
                    "You are an English-only summarizer. "
                    "All output must be in English only. "
                    "Never output Chinese characters. "
                    "If source text contains another language, translate it into English. "
                    "Use normal English spelling, punctuation, and spacing."
                )
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following text in English only. "
                    "Use one paragraph. "
                    "Do not use Chinese or any other language. "
                    "Do not add facts. "
                    "Use correct English word spacing. "
                    f"Keep the summary under {sentenceLimit} sentences.\n\n"
                    f"Text:\n{chunkText}\n\n"
                    "English summary:"
                )
            }
        ]

        summary = generateFromMessages(
            retryMessages,
            tokenizer,
            model,
            maxNewTokens=maxNewTokens,
            doSample=False,
            enableThinking=enableThinking,
            repetitionPenalty=repetitionPenalty,
            noRepeatNgramSize=noRepeatNgramSize
        )

        if not containsCJK(summary):
            if qualityPolish:
                summary = polishSummaryText(
                    summary,
                    tokenizer,
                    model,
                    maxNewTokens=maxNewTokens,
                    enableThinking=enableThinking,
                    repetitionPenalty=repetitionPenalty,
                    noRepeatNgramSize=noRepeatNgramSize
                )
            return summary

        print(f"[WARNING] Retry {retryCount + 1} still produced non-English output.")
        print(summary[:300])
        retryCount += 1

    # repair pass on the generated summary itself
    if forceEnglishRepair:
        repaired = repairToEnglish(
            summary,
            tokenizer,
            model,
            maxNewTokens=maxNewTokens,
            enableThinking=enableThinking,
            repetitionPenalty=repetitionPenalty,
            noRepeatNgramSize=noRepeatNgramSize
        )

        if not containsCJK(repaired):
            print("[INFO] English repair pass succeeded.")
            if qualityPolish:
                repaired = polishSummaryText(
                    repaired,
                    tokenizer,
                    model,
                    maxNewTokens=maxNewTokens,
                    enableThinking=enableThinking,
                    repetitionPenalty=repetitionPenalty,
                    noRepeatNgramSize=noRepeatNgramSize
                )
            return repaired

        print("[WARNING] Repair pass still contains non-English output.")
        print(repaired[:300])

    # last resort: return best cleaned version
    return cleanSummaryEdges(summary)

# - - - - - - - - - - #

def summarizeWithLoadedModel(
    chunks,
    tokenizer,
    model,
    deviceId=0,
    promptText=None,
    sentenceLimit=6,
    maxNewTokens=220,
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
    finalSummaries = {}

    for chunkName, chunkText in chunks.items():
        print(f"Summarizing {chunkName}...")

        summary = summarizeChunk(
            chunkText,
            tokenizer,
            model,
            deviceId=deviceId,
            promptText=promptText,
            sentenceLimit=sentenceLimit,
            maxNewTokens=maxNewTokens,
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

        if containsCJK(summary):
            warningMessage = f"{chunkName} produced non-English output."

            if failOnNonEnglish:
                raise ValueError(warningMessage + f" Output: {summary[:300]}")

            print(f"[WARNING] {warningMessage}")

        finalSummaries[chunkName] = summary

    return finalSummaries

# - - - - - - - - - - #

def summarizeParallel(
    chunks,
    modelName="Qwen/Qwen2.5-7B-Instruct",
    promptText=None,
    sentenceLimit=6,
    maxNewTokens=220,
    use4Bit=True,
    useYarn=False,
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
    tokenizer, model = loadSummarizer(
        modelName=modelName,
        deviceId=deviceId,
        use4Bit=use4Bit,
        useYarn=useYarn
    )

    return summarizeWithLoadedModel(
        chunks,
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

# - - - - - - - - - - #

