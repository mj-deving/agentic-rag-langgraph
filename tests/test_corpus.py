import hashlib
import json

import pytest

from agentic_rag_langgraph.config import ROOT

PROVENANCE = json.loads((ROOT / "corpus" / "PROVENANCE.json").read_text(encoding="utf-8"))
CHUNKS = json.loads((ROOT / "corpus" / "chunks.json").read_text(encoding="utf-8"))
ARTICLES = sorted({c["article"] for c in CHUNKS})


def sha256(path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("path", sorted(PROVENANCE["sha256"]))
def test_copy_is_byte_identical_to_recorded_hash(path):
    assert sha256(path) == PROVENANCE["sha256"][path]


def test_every_copied_file_is_recorded():
    copied = {p.relative_to(ROOT).as_posix() for p in (ROOT / "corpus" / "sources").iterdir()}
    copied |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "golden").glob("*.jsonl")}
    copied.add("corpus/chunks.json")
    assert copied == set(PROVENANCE["sha256"])


def test_covers_at_least_six_articles():
    assert len(ARTICLES) >= 6


@pytest.mark.parametrize("article", ARTICLES)
def test_snapshot_hash_matches_its_provenance(article):
    prov = json.loads((ROOT / f"corpus/sources/art-{article}.provenance.json").read_text(encoding="utf-8"))
    assert sha256(f"corpus/sources/art-{article}.source.txt") == prov["sha256"]
    assert prov["retrievalUrl"].startswith("https://")


def test_every_chunk_is_a_verbatim_substring_of_its_snapshot():
    for c in CHUNKS:
        assert c["text"] in (ROOT / f"corpus/sources/art-{c['article']}.source.txt").read_text(encoding="utf-8")


def test_chunk_ids_are_unique_and_non_empty():
    assert len({c["id"] for c in CHUNKS}) == len(CHUNKS)
    assert all(len(c["text"]) > 20 for c in CHUNKS)
