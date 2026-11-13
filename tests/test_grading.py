"""Mirrors the Mastra repo's test/grading.test.ts case for case, plus the runEvals verdict rule."""

import pytest

from agentic_rag_langgraph.grading import (
    REFUSAL_MARKER,
    citation_recall,
    citation_validity,
    fact_coverage,
    is_refusal,
    normalize,
    parse_citations,
    refusal_correctness,
    verdict,
)

RETRIEVED = {"art50-p1", "art50-p4", "art99-p3"}


class TestParseCitations:
    def test_dedupes_and_keeps_first_seen_order(self):
        assert parse_citations("A [chunk:art50-p4] B [chunk:art50-p1] C [chunk:art50-p4]") == ["art50-p4", "art50-p1"]

    def test_ignores_look_alikes(self):
        assert parse_citations("Artikel 50 Absatz 1, chunk:art50-p1, [Chunk:art50-p1], [chunk: art50-p1]") == []


class TestCitationValidity:
    def test_every_cited_chunk_retrieved(self):
        assert citation_validity("X [chunk:art50-p1] Y [chunk:art99-p3]", RETRIEVED) == 1

    def test_one_cited_chunk_not_retrieved_even_if_it_exists_in_corpus(self):
        assert citation_validity("X [chunk:art50-p1] Y [chunk:art26-p6]", RETRIEVED) == 0

    def test_invented_id(self):
        assert citation_validity("X [chunk:art999-p1]", RETRIEVED) == 0

    def test_uncited_answer(self):
        assert citation_validity("Ja, das ist Pflicht.", RETRIEVED) == 0

    def test_refusal_without_citations(self):
        assert citation_validity(f"{REFUSAL_MARKER} Dazu steht nichts im Korpus.", RETRIEVED) == 1

    def test_empty_retrieval_and_uncited_answer(self):
        assert citation_validity("Antwort", set()) == 0

    def test_refusal_that_cites_an_unretrieved_chunk(self):
        # Not in the TS suite; pins the port to the TS rule that citations are checked first.
        assert citation_validity(f"{REFUSAL_MARKER} [chunk:art26-p6]", RETRIEVED) == 0


class TestRefusalCorrectness:
    @pytest.mark.parametrize(
        ("expected", "answer", "score"),
        [
            ("refused", REFUSAL_MARKER, 1),
            ("refused", "Antwort [chunk:art50-p1]", 0),
            ("answered", "Antwort [chunk:art50-p1]", 1),
            ("answered", REFUSAL_MARKER, 0),
        ],
    )
    def test_matrix(self, expected, answer, score):
        assert refusal_correctness(expected, answer) == score

    def test_marker_must_be_exact(self):
        assert is_refusal("[keine_grundlage]") is False


class TestCitationRecall:
    def test_partial_multi_hop_recall(self):
        assert citation_recall(["art26-p11", "art50-p1"], "A [chunk:art50-p1]") == 0.5

    def test_extra_citations_do_not_lower_recall(self):
        assert citation_recall(["art50-p1"], "A [chunk:art50-p1] [chunk:art50-p4]") == 1

    def test_not_scorable_for_refusal_cases_and_wrong_refusals(self):
        assert citation_recall([], "A") is None
        assert citation_recall(["art50-p1"], REFUSAL_MARKER) is None


class TestFactCoverage:
    def test_thousands_separators_and_alternatives(self):
        facts = ["35000000|35 Millionen", "7 %|7 Prozent"]
        assert fact_coverage(facts, "bis zu 35.000.000 EUR oder 7 % des Umsatzes") == 1
        assert fact_coverage(facts, "bis zu 35 Millionen Euro") == 0.5

    def test_case_insensitive(self):
        assert fact_coverage(["Arbeitnehmervertreter"], "die ARBEITNEHMERVERTRETER") == 1

    def test_not_scorable_without_facts_or_on_refusal(self):
        assert fact_coverage(None, "x") is None
        assert fact_coverage(["x"], REFUSAL_MARKER) is None

    def test_normalize_keeps_ordinary_numbers_apart(self):
        assert normalize("2. August 2026") == "2. august 2026"

    # Expected values produced by running the TypeScript normalize() on the same inputs.
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("15 000 000", "15000000"),
            ("1.0000", "1.0000"),
            ("1.000x", "1.000x"),
            ("1.000ä", "1000ä"),  # JS \b is ASCII-only, so "ä" ends the digit group
            ("1.000_", "1.000_"),
            ("1.000,50", "1000,50"),
            ("35.000.000 EUR", "35000000 eur"),
            ("٣.000", "٣.000"),  # JS \d is ASCII-only
            ("1 000", "1000"),
        ],
    )
    def test_normalize_matches_js_regex_semantics(self, raw, expected):
        assert normalize(raw) == expected


class TestVerdict:
    T = {"refusal-correctness": 0.85}

    def test_failed_gate_wins_over_passing_thresholds(self):
        r = verdict({"citation-validity": [1, 0]}, {"refusal-correctness": [1, 1]}, self.T)
        assert r["verdict"] == "failed"
        assert r["gateResults"] == [{"id": "citation-validity", "passed": False, "score": 0.5}]

    def test_missed_threshold_is_scored_not_failed(self):
        r = verdict({"citation-validity": [1, 1]}, {"refusal-correctness": [1, 0]}, self.T)
        assert r["verdict"] == "scored"

    def test_threshold_is_inclusive(self):
        r = verdict({}, {"refusal-correctness": [0.85]}, self.T)
        assert r["verdict"] == "passed"

    def test_none_drops_out_of_mean_and_is_counted(self):
        r = verdict({}, {"refusal-correctness": [1, None, 1]}, self.T)
        assert r["scores"]["refusal-correctness"] == 1
        assert r["notScorable"] == {"refusal-correctness": 1}

    def test_metric_with_no_scorable_item_is_skipped(self):
        r = verdict({}, {"refusal-correctness": [None, None]}, self.T)
        assert r["thresholdResults"] == []
        assert r["verdict"] == "passed"
