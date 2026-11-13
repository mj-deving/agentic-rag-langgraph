"""Pure grading logic, a line-by-line port of the Mastra repo's src/lib/grading.ts.

Nothing here calls a model, so the eval gate is deterministic: the same answer text and retrieval
always score the same. `verdict` ports the aggregation rule of Mastra's `runEvals`, so both repos
reach "passed", "scored" or "failed" by the same arithmetic.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

REFUSAL_MARKER = "[KEINE_GRUNDLAGE]"

_CITATION = re.compile(r"\[chunk:([a-z0-9-]+)\]")

ExpectedOutcome = Literal["answered", "refused"]


def parse_citations(answer: str) -> list[str]:
    """Unique chunk ids cited as `[chunk:<id>]`, in order of first appearance."""
    return list(dict.fromkeys(_CITATION.findall(answer)))


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in answer


def citation_validity(answer: str, retrieved_ids: Iterable[str]) -> int:
    """1 when every cited chunk was actually retrieved in this run, else 0.

    An answer that is neither a refusal nor cites anything scores 0: an uncited claim is the
    failure this gate exists for. A refusal without citations scores 1.
    """
    cited = parse_citations(answer)
    if not cited:
        return 1 if is_refusal(answer) else 0
    retrieved = set(retrieved_ids)
    return 1 if all(cid in retrieved for cid in cited) else 0


def refusal_correctness(expected: ExpectedOutcome, answer: str) -> int:
    """1 when the agent answered/refused as expected."""
    return 1 if (expected == "refused") == is_refusal(answer) else 0


def citation_recall(expected_chunks: Sequence[str], answer: str) -> float | None:
    """Share of expected chunks the answer cites.

    None (not scorable) for cases that expect a refusal, and for answered cases the agent wrongly
    refused (refusal correctness counts those).
    """
    if not expected_chunks or is_refusal(answer):
        return None
    cited = set(parse_citations(answer))
    return sum(1 for cid in expected_chunks if cid in cited) / len(expected_chunks)


# JS `\d` and `\b` are ASCII-only even with the `u` flag; Python's are Unicode. Spelled out so the
# port matches the TS regex `(?<=\d)[.\s ](?=\d{3}\b)` on every input, not only on German text.
_THOUSANDS = re.compile(r"(?<=[0-9])[.\s](?=[0-9]{3}(?![A-Za-z0-9_]))")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and drop thousands separators between digits."""
    return _WHITESPACE.sub(" ", _THOUSANDS.sub("", text.lower()))


def fact_coverage(facts: Sequence[str] | None, answer: str) -> float | None:
    """Share of expected facts present in the answer.

    A fact may list alternatives separated by `|` ("35000000|35 millionen"); any one counts.
    None when the case lists no facts or the agent refused.
    """
    if not facts or is_refusal(answer):
        return None
    text = normalize(answer)
    hits = [f for f in facts if any(normalize(alt.strip()) in text for alt in f.split("|"))]
    return len(hits) / len(facts)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def verdict(
    gate_scores: Mapping[str, Sequence[float | None]],
    threshold_scores: Mapping[str, Sequence[float | None]],
    thresholds: Mapping[str, float],
) -> dict:
    """Mastra `runEvals` aggregation: None scores are not scorable and drop out of the mean.

    A metric with no scorable item is skipped. A gate passes only at a mean of 1. Any failed gate
    makes the verdict "failed"; otherwise a missed threshold makes it "scored"; else "passed".
    """
    gate_results, threshold_results, not_scorable = [], [], {}
    for key, raw in [*gate_scores.items(), *threshold_scores.items()]:
        skipped = sum(1 for s in raw if s is None)
        if skipped:
            not_scorable[key] = skipped
    for key, raw in gate_scores.items():
        scores = [s for s in raw if s is not None]
        if scores:
            avg = _mean(scores)
            gate_results.append({"id": key, "passed": avg >= 1, "score": avg})
    for key, raw in threshold_scores.items():
        scores = [s for s in raw if s is not None]
        if scores:
            avg = _mean(scores)
            threshold_results.append(
                {"id": key, "passed": avg >= thresholds[key], "averageScore": avg, "threshold": thresholds[key]}
            )
    if not all(g["passed"] for g in gate_results):
        result = "failed"
    elif not all(t["passed"] for t in threshold_results):
        result = "scored"
    else:
        result = "passed"
    return {
        "verdict": result,
        "scores": {t["id"]: t["averageScore"] for t in threshold_results},
        "gateResults": gate_results,
        "thresholdResults": threshold_results,
        "notScorable": not_scorable,
    }
