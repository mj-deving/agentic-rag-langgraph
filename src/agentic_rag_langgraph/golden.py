"""Golden-set loader with the same schema checks as the Mastra repo's zod schema."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expected_outcome: str
    expected_chunks: list[str]
    tags: list[str]
    expected_facts: list[str] | None = None


def _check(raw: dict, where: str) -> GoldenCase:
    def strs(value, key, min_len=0):
        if not isinstance(value, list) or len(value) < min_len or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{where}: {key} must be a list of strings (min {min_len})")
        return value

    for key in ("id", "question"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"{where}: {key} must be a non-empty string")
    if raw.get("expected_outcome") not in ("answered", "refused"):
        raise ValueError(f"{where}: expected_outcome must be answered or refused")
    facts = raw.get("expected_facts")
    return GoldenCase(
        id=raw["id"],
        question=raw["question"],
        expected_outcome=raw["expected_outcome"],
        expected_chunks=strs(raw.get("expected_chunks"), "expected_chunks"),
        tags=strs(raw.get("tags"), "tags", 1),
        expected_facts=None if facts is None else strs(facts, "expected_facts"),
    )


def load_golden(path: str | Path) -> list[GoldenCase]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [_check(json.loads(line), f"{path}:{i + 1}") for i, line in enumerate(lines) if line.strip()]
