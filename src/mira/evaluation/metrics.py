"""
Retrieval metrics over ranked page keys with binary relevance.

A ranking is a list of (document_id, page_num), best first; `relevant` is the
set of pages that answer the query (usually one, sometimes a span).
"""

import math
from typing import List, Optional, Set, Tuple

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
