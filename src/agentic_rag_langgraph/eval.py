"""Eval runner, the counterpart of the Mastra repo's scripts/eval.ts.

Two stages over the same golden set:
 1. Retrieval only (no LLM): does the top-k contain the expected chunks?
 2. The real LangGraph agent, graded by grading.py with a hard gate and thresholds.
Writes results/<date>-<model>-<prompt>-<set>.{json,md} and exits non-zero unless the verdict is "passed".
LangGraph has no eval harness of its own, so the loop and the aggregation live here.
"""

import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from . import config
from .golden import GoldenCase, load_golden
from .grading import citation_recall, citation_validity, fact_coverage, refusal_correctness, verdict
from .rag import default_graph, embeddings, load_store, run, search

THRESHOLDS = {"refusal-correctness": 0.85, "citation-recall": 0.8, "fact-coverage": 0.7}
GATE = "citation-validity"


def retrieval_stage(store, cases: list[GoldenCase]) -> dict:
    rows = []
    for c in cases:
        if c.expected_outcome != "answered":
            continue
        top = [h["chunkId"] for h in search(store, c.question, config.TOP_K)]
        found = sum(1 for cid in c.expected_chunks if cid in top)
        rows.append({"id": c.id, "recall": found / len(c.expected_chunks), "top": top})
    mean = sum(r["recall"] for r in rows) / max(len(rows), 1)
    return {"recallAtK": mean, "k": config.TOP_K, "cases": rows}


def score_case(c: GoldenCase, answer: str, retrieved: set[str]) -> dict[str, float | None]:
    return {
        GATE: citation_validity(answer, retrieved),
        "refusal-correctness": refusal_correctness(c.expected_outcome, answer),
        "citation-recall": citation_recall(c.expected_chunks, answer),
        "fact-coverage": fact_coverage(c.expected_facts, answer),
    }


def agent_stage(graph, cases: list[GoldenCase]) -> list[dict]:
    rows = []
    for i, c in enumerate(cases, 1):
        started = time.monotonic()
        result = run(graph, c.question)
        scores = score_case(c, result.answer, result.retrieved_ids)
        rows.append({
            "id": c.id, "tags": c.tags, "expected": c.expected_outcome, "answer": result.answer,
            "scores": scores, "retrieved": sorted(result.retrieved_ids), "queries": result.queries,
            "modelSteps": result.steps, "ms": round((time.monotonic() - started) * 1000),
        })
        print(f"{i}/{len(cases)} {c.id} {json.dumps(scores)}", flush=True)
    return rows


def aggregate(rows: list[dict]) -> dict:
    return verdict(
        {GATE: [r["scores"][GATE] for r in rows]},
        {key: [r["scores"][key] for r in rows] for key in THRESHOLDS},
        THRESHOLDS,
    )


def pct(n: float | None) -> str:
    return "–" if n is None else f"{n * 100:.1f} %"


def table(headers: list[str], body: list[list]) -> str:
    lines = [f"| {' | '.join(headers)} |", f"|{'|'.join('---' for _ in headers)}|"]
    return "\n".join(lines + [f"| {' | '.join(str(x) for x in r)} |" for r in body])


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r["scores"][key] for r in rows if r["scores"][key] is not None]
    return sum(vals) / len(vals) if vals else None


def render_md(run_data: dict) -> str:
    rows, agg = run_data["items"], run_data["aggregate"]
    gate = next((g["score"] for g in agg["gateResults"] if g["id"] == GATE), None)
    tags = sorted({t for r in rows for t in r["tags"]})
    by_tag = [
        [t, len(tagged := [r for r in rows if t in r["tags"]]),
         *(pct(mean_of(tagged, k)) for k in (GATE, *THRESHOLDS))]
        for t in tags
    ]
    ooc = [r for r in rows if "out-of-corpus" in r["tags"]]
    ooc_refused = sum(1 for r in ooc if r["scores"]["refusal-correctness"] == 1)
    failures = [r for r in rows if any(s is not None and s < 1 for s in r["scores"].values())]
    failure_md = "\n".join(
        f"### {r['id']} ({r['expected']})\n\n{json.dumps(r['scores'])}\n\n"
        f"> {' '.join(r['answer'].split())[:600]}\n"
        for r in failures
    )
    return f"""# Eval run {run_data['stamp']}

Framework LangGraph, model `{run_data['model']}`, prompt `{run_data['prompt']}`, golden `{run_data['golden']}`, embeddings `{run_data['embeddingModel']}`, top-k {run_data['topK']}, {run_data['cases']} cases. Verdict: **{agg['verdict']}**.

## Aggregate

{table(["metric", "score", "rule"], [
    ["retrieval recall@k (no LLM)", pct(run_data["retrieval"]["recallAtK"]), "tracked"],
    [GATE, pct(gate), "gate: must be 100 %"],
    *([k, pct(agg["scores"].get(k)), f"≥ {pct(v)}"] for k, v in THRESHOLDS.items()),
])}

Out-of-corpus refused: {ooc_refused} / {len(ooc)}.

## By tag

{table(["tag", "n", "validity", "refusal", "recall", "facts"], by_tag)}

## Cases below 1.0 ({len(failures)})

{failure_md}"""


def main() -> None:
    golden_path = os.environ.get("GOLDEN", "golden/v1.jsonl")
    cases = load_golden(config.ROOT / golden_path)
    if only := os.environ.get("ONLY"):
        cases = [c for c in cases if c.id in only.split(",")]

    retrieval = retrieval_stage(load_store(embeddings()), cases)
    print(f"retrieval recall@{retrieval['k']}: {pct(retrieval['recallAtK'])}")
    if os.environ.get("RETRIEVAL_ONLY") == "1":
        for r in retrieval["cases"]:
            if r["recall"] < 1:
                print(f"  miss {r['id']}: top {', '.join(r['top'])}")
        sys.exit(0)

    rows = agent_stage(default_graph(), cases)
    agg = aggregate(rows)
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    run_data = {
        "stamp": stamp, "framework": "langgraph", "model": config.CHAT_MODEL, "prompt": config.PROMPT_VERSION,
        "golden": golden_path, "embeddingModel": config.EMBEDDING_MODEL, "topK": config.TOP_K,
        "cases": len(cases), "thresholds": THRESHOLDS, "verdict": agg["verdict"], "aggregate": agg,
        "retrieval": retrieval, "items": rows,
    }
    out = config.ROOT / "results"
    out.mkdir(exist_ok=True)
    model_slug = re.sub(r"[^a-z0-9]+", "-", config.CHAT_MODEL, flags=re.I)
    slug = f"{stamp[:10]}-{model_slug}-{config.PROMPT_VERSION}-{Path(golden_path).stem}"
    (out / f"{slug}.json").write_text(json.dumps(run_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{slug}.md").write_text(render_md(run_data), encoding="utf-8")
    print(f"\nverdict: {agg['verdict']}\nwrote results/{slug}.{{json,md}}")
    sys.exit(0 if agg["verdict"] == "passed" else 1)
