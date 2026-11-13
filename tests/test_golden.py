import json

import pytest

from agentic_rag_langgraph.config import ROOT
from agentic_rag_langgraph.golden import load_golden

CHUNKS = {c["id"]: c["text"].lower() for c in json.loads((ROOT / "corpus/chunks.json").read_text(encoding="utf-8"))}
SETS = {name: load_golden(ROOT / f"golden/{name}.jsonl") for name in ("v1", "heldout")}


def test_golden_v1_size_and_refusals():
    golden = SETS["v1"]
    assert len(golden) >= 30
    assert sum(c.expected_outcome == "refused" for c in golden) >= 6


def test_golden_v1_covers_every_required_tag():
    tags = {t for c in SETS["v1"] for t in c.tags}
    assert {"single", "multi-hop", "numerical", "ambiguous", "out-of-corpus"} <= tags


@pytest.mark.parametrize("name", SETS)
def test_ids_unique_and_chunks_exist(name):
    cases = SETS[name]
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        assert all(cid in CHUNKS for cid in c.expected_chunks), c.id
        if c.expected_outcome == "answered":
            assert c.expected_chunks, c.id
        else:
            assert c.expected_chunks == [], c.id


@pytest.mark.parametrize("name", SETS)
def test_every_expected_fact_appears_in_an_expected_chunk(name):
    for c in SETS[name]:
        for fact in c.expected_facts or []:
            alts = fact.split("|")
            assert any(a.lower() in CHUNKS[cid] for a in alts for cid in c.expected_chunks), f"{c.id}: {fact}"


def test_loader_rejects_bad_outcome(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"x","question":"q","expected_outcome":"maybe","expected_chunks":[],"tags":["t"]}\n')
    with pytest.raises(ValueError, match="expected_outcome"):
        load_golden(bad)
