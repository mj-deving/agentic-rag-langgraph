<div align="center">

# Agentic RAG on LangGraph

**A LangGraph agent that answers questions about the EU AI Act only from retrieved paragraphs, cites every claim, refuses when the corpus is silent, and is graded by the same deterministic gate as its Mastra twin. Same corpus, same cases, same prompt, same model. Fully local: no API key, no cloud.**

![Python](https://img.shields.io/badge/Python-3.13-3776ab?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-0.12-de5fe9)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1c3c3c)
![Ollama](https://img.shields.io/badge/LLM-local_Ollama-0f766e)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

[Quickstart](#quickstart) · [Architecture](#architecture) · [Results](#results) · [Mastra vs. LangGraph](#mastra-vs-langgraph-same-contract-what-each-framework-gives)

</div>

## Why this exists

[agentic-rag-mastra](https://github.com/mj-deving/agentic-rag-mastra) builds a citing, refusing RAG agent
on Mastra and grades it with a hard gate. This repo builds the **same contract on LangGraph** so the
two frameworks can be compared on equal terms. Only the framework changes:

| Held equal | Value |
|---|---|
| Corpus | 66 verbatim paragraphs of Regulation (EU) 2024/1689, copied byte for byte (`corpus/PROVENANCE.json`) |
| Cases | `golden/v1.jsonl` (40) and `golden/heldout.jsonl` (8), copied byte for byte |
| Prompt | v2, byte-identical to the TypeScript original (SHA-256 pinned in `tests/test_instructions.py`) |
| Model | `qwen3:8b` via Ollama, thinking off, temperature 0 |
| Retrieval | `bge-m3` embeddings, cosine, top-k 5 |
| Grading | `grading.ts` ported line by line, including the `runEvals` verdict arithmetic |

The contract: every statement carries a citation like `[chunk:art50-p4]`; a citation is only valid if
that chunk was retrieved **in the same run**; if the retrieved text does not support an answer, the
agent returns `[KEINE_GRUNDLAGE]`.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
ollama pull qwen3:8b
ollama pull bge-m3
uv sync
uv run ingest        # embeds 66 paragraphs into index/eu_ai_act_chunks.json
uv run ask "Welche Geldbuße droht bei Verstößen gegen die Transparenzpflichten nach Artikel 50?"
uv run eval          # 40 golden cases through the real agent, writes results/
GOLDEN=golden/heldout.jsonl uv run eval
```

`uv run pytest` needs neither Ollama nor network. `RETRIEVAL_ONLY=1 uv run eval` runs the
retrieval stage alone; `ONLY=<id>,<id>` limits a run to named cases; `PROMPT=v1` selects the baseline prompt.

## Architecture

```
START ──▶ agent ──(tool calls?)──▶ tools ──▶ agent ──▶ … ──▶ END
          ChatOllama + bound tool    ToolNode(search-regulation)
```

```
corpus/                       chunks, article snapshots, provenance; PROVENANCE.json pins every copy by hash
golden/                       the Mastra repo's golden and held-out sets, unchanged
src/agentic_rag_langgraph/
  rag.py                      vector store, search tool, StateGraph, ingest and ask entry points
  grading.py                  pure, deterministic grading logic plus the runEvals verdict rule
  instructions.py             prompt v1 and v2, verbatim
  eval.py                     retrieval-only stage + agent stage, gate, thresholds, results/
scripts/compare.py            code lines and run-by-run comparison with the Mastra repo
```

**The citation gate reads tool artifacts, not text.** The search tool returns its chunks twice
(`response_format="content_and_artifact"`): as JSON the model reads, and as a structured artifact on
the `ToolMessage`. `run()` collects retrieved ids only from those artifacts, on a fresh checkpointer
thread per question, so retrieval from another run can never satisfy the gate
(`tests/test_graph.py` checks this offline with a scripted model).

## Results

All numbers come from [`results/`](results/): `uv run eval` on a MacBook with `qwen3:8b` and `bge-m3`
through Ollama, top-k 5, temperature 0, prompt v2.

| Set | Citation gate | Refusal correct | Out-of-corpus refused | Citation recall | Fact coverage | Retrieval recall@5 | Verdict |
|---|---|---|---|---|---|---|---|
| golden (40) | 95.0 % | 85.0 % | 4 / 7 | 91.7 % | 90.0 % | 93.9 % | ❌ failed |
| held-out (8) | 100 % | 62.5 % | 2 / 4 | 100 % | 100 % | 100 % | 🟡 scored |

**The gate failed for the same reasons as on Mastra.** Two golden answers would mislead a reader:
`a113-application` claims the regulation applies from "17. Mai 2026" and cites `[chunk:art50-p1]`
without ever calling the search tool; `ooc-definition-ai-system` presents an Article 5 prohibition as
the definition of "KI-System". The eval exits non-zero, which is the point of a gate.

**Answers without search.** Because LangGraph keeps every tool call in the run state, the results
record the queries per case. In 8 of 40 golden cases and 2 of 8 held-out cases the model never
called the search tool. Three of the golden ones are wrong refusals of answerable questions
(`a5-social-scoring`, `a5-realtime-biometric`, `a26-biometric-48h`): the model refused before
looking. Forcing a tool call on the first step would be a framework-level fix; it was not built,
because it changes the agent and would need a fresh held-out set to judge.

## Mastra vs. LangGraph: same contract, what each framework gives

Mastra numbers are read from that repo's `results/` at commit `d19dfcd`, before its citation guard
was added; `uv run python scripts/compare.py ../agentic-rag-mastra d19dfcd` wrote
[`results/comparison.json`](results/comparison.json). That commit predates the public history of
[agentic-rag-mastra](https://github.com/mj-deving/agentic-rag-mastra). To reproduce against the
public repo, run `uv run python scripts/compare.py ../agentic-rag-mastra 4c3cbca`: the run-by-run
comparison comes out identical, while the Mastra code count reads 509 instead of 485, because that
commit already carries the `GUARD` switch (config, agent wiring, eval hooks).

| | Mastra | LangGraph |
|---|---|---|
| Golden: citation gate | 95.0 % | 95.0 % |
| Golden: refusal correct | 87.5 % | 85.0 % |
| Golden: out-of-corpus refused | 4 / 7 | 4 / 7 |
| Golden: citation recall | 91.9 % | 91.7 % |
| Golden: fact coverage | 87.1 % | 90.0 % |
| Golden: verdict | ❌ failed | ❌ failed |
| Held-out: gate / refusal / out-of-corpus refused | 100 % / 62.5 % / 2 / 4 | 100 % / 62.5 % / 2 / 4 |
| Retrieval recall@5 (golden / held-out) | 93.9 % / 100 % | 93.9 % / 100 % |
| Same answer-or-refuse decision per case | 39 of 40 golden, 8 of 8 held-out | |
| Gate failures | `a113-application`, `ooc-definition-ai-system` | the same two |
| Code lines, contract only (no blanks, comments, docstrings) | 485 | 372 |
| of which eval runner and scorers | 212 | 124 |
| of which grading (LangGraph includes the ported verdict rule) | 36 | 70 |

**Reading it.** With the model, prompt, data and grader held equal, the framework barely moves the
numbers: the two agents made the same answer-or-refuse decision on 39 of 40 golden questions and
failed the gate on the same two. The quality of this system is set by the 8B model and the prompt,
not by the orchestration layer. n = 40 is a regression check, not a benchmark; the Mastra repo
documents run-to-run variance at temperature 0, so a difference of one case is noise.

| Concern | Mastra | LangGraph |
|---|---|---|
| Agent loop with tool calling | free (`Agent`) | free (`StateGraph` + `ToolNode` + `tools_condition`), wired by hand |
| Typed tool input, validation errors returned to the model | free (Zod) | free (Pydantic `args_schema`, `ToolNode` error handling) |
| Structured tool output for the grader | extracted from the run output by a utility | free: `content_and_artifact` puts the chunks on the `ToolMessage` |
| Vector store | free (`LibSQLVector`) | free (`InMemoryVectorStore`, JSON dump/load; needs `numpy`) |
| Ingestion | a Mastra workflow | a function; no workflow engine needed for 66 chunks |
| Per-run state | free (memory/storage) | free (checkpointer, one thread per question) |
| Eval runner, gates, thresholds, verdict | free (`runEvals`, scorers) | **hand-built** (`eval.py`, `verdict()` in `grading.py`) |
| Local tracing UI | free (Mastra Studio, span tree) | **none without LangSmith**; the run state's messages are the trace, and the eval records queries and retrieved ids per case. OpenTelemetry was not added. |
| Definition of a valid answer | hand-built | hand-built, the same rule |

**What I had to add on LangGraph:** the eval harness. LangGraph has no counterpart to Mastra's
`runEvals`, so the loop over cases, the aggregation (not-scorable cases drop out of the mean, a gate
passes only at 100 %, failed beats scored beats passed) and the result files are 124 code lines in `eval.py` plus the `verdict()` port, all
my own code; `verdict()` is unit-tested against the arithmetic read from the `@mastra/core` bundle. What LangGraph
gives back is transparency: the graph is two nodes and three edges I can read, and the tool artifact
channel made the "retrieved in the same run" rule simpler to implement than on Mastra.

**Rough edges found while building:**

- `InMemoryVectorStore` needs `numpy` for similarity search but `langchain-core` does not declare it, so it had to be added by hand.
- `GenericFakeChatModel` has no `bind_tools`, so the offline graph test subclasses it with a no-op.
- `ChatOllama` calls Ollama's `think` switch `reasoning`; `reasoning=False` is what turns thinking off.
- `langgraph.prebuilt.create_react_agent` is deprecated since LangGraph 1.0 (`LangGraphDeprecatedSinceV10`); the graph here is built explicitly instead.
- The Mastra step limit was matched as 20 model steps (`recursion_limit=41`), read from an `stepCountIs(20)` in the `@mastra/core` bundle, not from documentation. No run came close: the most model steps any case used is 2.

## Corpus and scope

The corpus is a mirror's rendering of the consolidated text of Articles 4, 5, 6, 26, 50, 99 and 113,
not the Official Journal; each provenance file names the exact URL and retrieval date. Answers are
about this snapshot. This is not legal advice.

## Intent

[`ISA.md`](ISA.md) (German) is the spec this repo was built against: what "done" meant, the probe
that would falsify each criterion, the evidence recorded for it, and the limits left open on purpose.
