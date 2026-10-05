"""
ViDoRe V3 helpers: pure parsing/mapping, unit-testable without HF downloads.

The dataset-specific loading (datasets.load_dataset) lives in scripts/. The 8
public subsets; energy, physics and finance_fr have French pages, where Mira's
English-stemmer BM25 channel is not meaningful — prefer the English subsets.
"""

from typing import Dict, Iterable, Tuple

Page = Tuple[str, int]  # (doc_id, page_number_in_doc), 0-indexed page

SUBSETS = (
    "hr",
    "computer_science",
    "physics",       # French corpus
    "energy",        # French corpus
    "industrial",
    "pharmaceuticals",
    "finance_en",
    "finance_fr",    # French corpus
)

ENGLISH_SUBSETS = tuple(s for s in SUBSETS if s not in ("physics", "energy", "finance_fr"))

# Relevance grades in the qrels (score column)
FULLY_RELEVANT = 2
CRITICALLY_RELEVANT = 1


def corpus_keys(corpus_rows: Iterable[dict]) -> Dict[int, Page]:
    """Map the qrels' flat corpus_id -> (doc_id, page_number_in_doc)."""
    return {row["corpus_id"]: (row["doc_id"], row["page_number_in_doc"]) for row in corpus_rows}


def build_relevance(qrels: Iterable[dict], keys: Dict[int, Page]) -> Dict[int, Dict[Page, int]]:
    """query_id -> {(doc_id, page_number): grade}, grade 2=Fully, 1=Critically Relevant."""
    relevance: Dict[int, Dict[Page, int]] = {}
    for row in qrels:
        relevance.setdefault(row["query_id"], {})[keys[row["corpus_id"]]] = row["score"]
    return relevance


def relevant_set(grades: Dict[Page, int]) -> set:
    """Pages counting as relevant for binary metrics (recall, MRR): any grade >= 1."""
    return {page for page, grade in grades.items() if grade >= CRITICALLY_RELEVANT}
