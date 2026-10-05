"""Tests for ViDoRe V3 qrels/corpus parsing helpers."""

from mira.evaluation.vidore import (
    ENGLISH_SUBSETS,
    SUBSETS,
    build_relevance,
    corpus_keys,
    relevant_set,
)

CORPUS = [
    {"corpus_id": 0, "doc_id": "book_a", "page_number_in_doc": 0},
    {"corpus_id": 1, "doc_id": "book_a", "page_number_in_doc": 1},
    {"corpus_id": 2, "doc_id": "book_b", "page_number_in_doc": 0},
]

QRELS = [
    {"query_id": 10, "corpus_id": 1, "score": 2},
    {"query_id": 10, "corpus_id": 2, "score": 1},
    {"query_id": 11, "corpus_id": 0, "score": 1},
]


def test_corpus_keys():
    assert corpus_keys(CORPUS) == {0: ("book_a", 0), 1: ("book_a", 1), 2: ("book_b", 0)}


def test_build_relevance_spans_documents():
    rel = build_relevance(QRELS, corpus_keys(CORPUS))
    # ViDoRe relevant pages can cross documents; the custom queries.jsonl format can't
    assert rel[10] == {("book_a", 1): 2, ("book_b", 0): 1}
    assert rel[11] == {("book_a", 0): 1}


def test_relevant_set_counts_any_grade():
    grades = {("a", 0): 1, ("a", 1): 2}
    assert relevant_set(grades) == {("a", 0), ("a", 1)}
    assert relevant_set({}) == set()


def test_english_subsets_exclude_french_corpora():
    assert "computer_science" in ENGLISH_SUBSETS
    for french in ("physics", "energy", "finance_fr"):
        assert french not in ENGLISH_SUBSETS
    assert set(ENGLISH_SUBSETS) < set(SUBSETS)
