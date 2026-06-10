---
title:  "Building an AI Agent for the GAIA Benchmark"
mathjax: true
layout: post
categories: media
---

## What is GAIA?

GAIA is a benchmark designed to test AI assistants on real-world tasks that require multi-step reasoning, tool use, and file processing. Unlike typical LLM benchmarks that test knowledge or reasoning in isolation, GAIA presents questions that demand an agent to search the web, read documents, transcribe audio, execute code, and synthesize information - often combining several of these in a single question. Answers are graded by exact string match, which makes formatting as important as correctness.

The benchmark is part of the Hugging Face Agents Course (Unit 4), where participants build an agent, deploy it on HF Spaces, and submit answers to a scoring API that evaluates 20 validation questions.

## Architecture Evolution

Here, I will describe the evolution of different attempts to build an AI agent for GAIA benchmark. The final AI agent can be found here: [AI agent github](https://github.com/tqv-notes/gaia_agent).

Important note: I implemented a custom tool-use agent directly on the Anthropic API, without using agent frameworks like smolagents, LangGraph, or LlamaIndex. The core is a ReAct-style loop - the LLM reasons, calls tools, observes results, and repeats - wrapped with deterministic preprocessing and postprocessing stages. This framework-free approach traded convenience for full control over file handling, error recovery, and answer formatting.

### Attempt 1: Naive LLM Loop (0%)

The starting point was a template with a `BasicAgent` that returned a default answer. Replacing it with a simple Claude API call and tool definitions got the structure right, but the first real run scored 0% - the model string was outdated (`claude-sonnet-4-20250514` had been deprecated), and every API call returned a 404.

Lesson: model identifiers change. Hardcoding them without a fallback is fragile.

### Attempt 2: Single-Loop Agent with Tools (25-35%)

After fixing the model name to `claude-sonnet-4-6`, the agent could reason and call tools: web search (DuckDuckGo), webpage fetching, and Python code execution. This got 25% on the first run and 35% with retry logic for rate limits.

The main failure modes were clear from the logs. File-dependent questions (chess images, audio recordings, Excel spreadsheets, Python scripts) all failed because the scoring API's file endpoint returned 404. YouTube videos were inaccessible due to SSL errors from the HF Space's network. And DuckDuckGo returned "No results" for roughly 80% of queries - likely IP-level throttling on the shared HF Space infrastructure.

### Attempt 3: Better Tools and Prompting (50%)

Three changes pushed the score to 50%. First, adding a Wikipedia tool that fetched full page HTML rather than relying on search snippets. Many GAIA questions are factual and Wikipedia contains the answer - but only if you get the complete page, not a 200-character search snippet. Second, adding a YouTube transcript tool using `youtube-transcript-api`. Third, improving the system prompt with concrete format examples, since GAIA's exact-match grading means `**broccoli**` (with markdown bold) is wrong even though the content is correct.

### Attempt 4: Deterministic Preprocessing (65%)

For this step, the key insight from detailed failure analysis: Stop asking the LLM to decide when and how to process files. Do it deterministically before the LLM ever sees the question.

The pipeline became:

```
question --> deobfuscate --> download file --> preprocess file -->
inject context into prompt --> agent reasons with tools -->
extract answer --> regex cleanup --> submit
```

Specific preprocessing steps, all executed in Python before the agent loop:

- **Python files**: executed via subprocess, both source code and stdout included in the prompt
- **Excel files**: read with pandas, columns/shape/first 30 rows/column sums all precomputed
- **Audio files**: transcribed with Whisper (tiny model to avoid OOM on free Spaces)
- **PDFs**: text extracted with PyPDF2
- **Images**: attached directly to Claude's vision API
- **Reversed text**: detected by counting English stopwords in the original vs. reversed string

The LLM formatter for answer cleanup was also replaced with deterministic regex. An earlier version used a cheap LLM call to clean up verbose answers, but this occasionally mutated correct answers - fatal for exact-match scoring.

### Attempt 5: Tavily Search and Dataset File Fix (75%)

Two final fixes addressed the remaining infrastructure issues. DuckDuckGo's unreliability from HF Spaces was mitigated by adding Tavily as the primary search engine (free tier, 1000 searches/month), with DDG as fallback. Tavily also returns a direct `answer` field that often contains exactly what GAIA needs.

The file download issue was traced to its root cause: the scoring API's `/files/{task_id}` endpoint returns `"No file path associated with task_id"` for all tasks. The files exist on the HuggingFace dataset repository, not the scoring API. I decided to change the download URL to: 
<div align="center">
https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{file_name}.
</div>
This should unlock the 5 file-dependent questions that have been failing since the beginning.

## Lessons Learned

**1. The LLM is not the bottleneck - the plumbing is.**

Most debugging time was spent on file downloads returning 404, search engines being throttled, YouTube blocking SSL connections, and model names being deprecated. The actual reasoning was usually correct when the agent had the right information.

**2. Deterministic preprocessing beats autonomous tool use.**

Asking an LLM "here's an audio file, please transcribe it" is unreliable. The model might ignore the instruction, hallucinate a transcript, forget the file path, or ask the user to re-upload. Running Whisper in a subprocess before the agent sees the question is 100% reliable. This principle applies to every file type.

**3. Exact-match grading demands deterministic answer cleanup.**

Markdown bold, trailing periods, verbose prefixes ("The answer is..."), and emoji all cause exact-match failures. Regex cleanup is safer than an LLM formatter because it never hallucinates or mutates the answer content.

**4. Search reliability varies wildly by environment.**

DuckDuckGo works locally but is heavily throttled from shared cloud IPs. The Wikipedia Python library occasionally returns empty JSON. YouTube blocks requests entirely from some data centers. Having multiple search backends (Tavily + DDG + direct Wikipedia page visits) is essential for robustness.

**5. Claude outperforms GPT-4.1 for autonomous agents in this architecture.**

A direct comparison showed Claude Sonnet at 50% vs. GPT-4.1 at 20% with the same tool set and prompts. Claude was notably better at deciding when to use tools, recovering from failed searches by trying alternative queries, and following file processing instructions. GPT-4.1's `tool_choice="auto"` was less reliable for multi-step reasoning.

## What Would Get to 80%+

Based on the failure analysis, reaching higher accuracy would require addressing the remaining hard questions: YouTube videos where transcripts aren't available (frame extraction + OCR), web pages that require JavaScript rendering (Playwright), and multi-hop research questions where the agent needs to chain 3-4 specific web lookups without getting lost. Adding a verification pass - re-checking the answer against the original question before submitting - would also catch formatting errors that slip through regex cleanup.

The gap between 65% and 90% is mostly engineering depth, not architectural novelty.
