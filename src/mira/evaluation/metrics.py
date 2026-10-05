"""
Retrieval metrics over ranked page keys.

A ranking is a list of (document_id, page_num), best first. Relevance is either
binary (a set of pages) or graded (a page -> grade dict, as in ViDoRe V3 where
2 = Fully Relevant and 1 = Critically Relevant).
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

Page = Tuple[str, int]


def first_relevant_rank(ranking: List[Page], relevant: Set[Page]) -> Optional[int]:
    """1-based rank of the first relevant page, or None if absent."""
    return next((i for i, page in enumerate(ranking, start=1) if page in relevant), None)


def recall_at_k(ranking: List[Page], relevant: Set[Page], k: int) -> float:
    """Fraction of relevant pages found in the top k."""
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: List[Page], relevant: Set[Page]) -> float:
    """1 / rank of first relevant page (0 if none); averaged over queries this is MRR."""
    rank = first_relevant_rank(ranking, relevant)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranking: List[Page], relevant: Set[Page], k: int) -> float:
    """Binary-relevance nDCG@k: rewards relevant pages ranked higher, normalized to [0, 1]."""
    dcg = sum(1 / math.log2(i + 1) for i, page in enumerate(ranking[:k], start=1) if page in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def ndcg_at_k_graded(ranking: List[Page], grades: Dict[Page, float], k: int) -> float:
    """Graded-relevance nDCG@k (linear gains), ViDoRe's headline metric."""
    dcg = sum(grades[page] / math.log2(i + 1) for i, page in enumerate(ranking[:k], start=1) if page in grades)
    ideal = sum(g / math.log2(i + 1) for i, g in enumerate(sorted(grades.values(), reverse=True)[:k], start=1))
    return dcg / ideal if ideal else 0.0


def paired_bootstrap(a: List[float], b: List[float], n: int = 10000, seed: int = 0) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap on the per-query difference a - b.

    Returns (mean difference, ci_low, ci_high, p) where p is the two-sided
    fraction of resamples on the wrong side of zero. Use to test whether the
    adaptive/fixed gap on per-query nDCG is real or noise.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("paired_bootstrap needs two non-empty arrays of equal length")
    resamples = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    diffs = a[resamples].mean(axis=1) - b[resamples].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float((a - b).mean()), float(lo), float(hi), float(min(p, 1.0))
