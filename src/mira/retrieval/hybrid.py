"""
Hybrid retrieval (Phase 4): visual (ColQwen MaxSim) + lexical (BM25) rankings
fused with query-adaptive weighted Reciprocal Rank Fusion.

    RRF(d) = w_visual / (k + rank_visual(d)) + w_lexical / (k + rank_lexical(d))

Rank-based, so BM25 and MaxSim scores never need a common scale.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .colqwen import ColQwenEmbedder
from .lexical import BM25Index
from .qdrant_store import QdrantMultivectorStore, SearchResult

# Words suggesting the answer lives in a figure/table/layout rather than prose
VISUAL_CUES = frozenset("""
    figure fig chart graph plot diagram table image picture photo drawing schematic
    layout curve trend bar axis shown show shows illustrated pinout block
""".split())

# ponytail: hand-set shift, tune on data/eval/queries.jsonl (Phase 4 ablation)
WEIGHT_SHIFT = 0.5

_IDENTIFIER = re.compile(r"""
    (?i: 0x[0-9a-f]+         # hex: 0x2D
       | \w*[a-z]\w*\d\w*      # letters then digits: STM32F401RE, I2C, BME280
       | \w*\d\w*[a-z]\w*      # digits then letters: 3V3, 74HC595
       | \w+_\w+ )             # snake/pin names: VDD_IO, USB_VBUS
  | \b[A-Z]{5,}\b            # long all-caps register/signal names, case-sensitive: TXPOWER, PWRKEY (not EU, OECD)
""", re.VERBOSE)

# Letter+digit tokens that are quantities or dates, not part numbers: 2ns, 100mA, 1950s, 3rd, Q4, FY2024
_NOT_IDENTIFIER = re.compile(r"""
    \d+(\.\d+)?(ns|us|ms|s|hz|khz|mhz|ghz|mv|v|ma|ua|a|mw|w|kb|mb|gb|tb|bits?|mm|cm|km|kg|db|dbm|k|x)
  | \d+(st|nd|rd|th|s)
  | [qh][1-4] | fy\d+
""", re.VERBOSE | re.IGNORECASE)


@dataclass
class QueryWeights:
    """Fusion weights for one query, plus the features that produced them (for logging/ablation)."""
    visual: float
    lexical: float
    identifiers: List[str] = field(default_factory=list)
    quoted: bool = False
    visual_cues: List[str] = field(default_factory=list)


def query_weights(query: str) -> QueryWeights:
    """
    Query-adaptive fusion weights from cheap surface features (no model call).

    Identifiers or quoted text -> lean lexical; figure/chart/table words -> lean visual.
    Both or neither -> equal weights.
    """
    identifiers = [
        m.group() for m in _IDENTIFIER.finditer(query) if not _NOT_IDENTIFIER.fullmatch(m.group())
    ]
    quoted = bool(re.search(r'"[^"]+"', query))
    cues = [w for w in re.findall(r"[a-z]+", query.lower()) if w in VISUAL_CUES]

    shift = WEIGHT_SHIFT * ((bool(identifiers) or quoted) - bool(cues))
    return QueryWeights(
        visual=1.0 - shift,
        lexical=1.0 + shift,
        identifiers=identifiers,
        quoted=quoted,
        visual_cues=cues,
    )


def rrf(
    rankings: List[List[Tuple[str, int]]],
    weights: List[float],
    k: int = 60,
) -> Dict[Tuple[str, int], float]:
    """
    Weighted Reciprocal Rank Fusion.

    Args:
        rankings: One list of page keys (document_id, page_num) per channel, best first
        weights: One weight per channel
        k: Damping constant; 60 is the standard value from the RRF paper

    Returns:
        Fused score per page key (pages missing from a ranking get 0 from it)
    """
    scores: Dict[Tuple[str, int], float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return scores


@dataclass
class HybridResult:
    """A fused result, with each channel's rank kept for analysis."""
    document_id: str
    page_num: int
    score: float
    visual_rank: Optional[int]  # 1-based; None if not in visual candidates
    lexical_rank: Optional[int]
    payload: dict  # Qdrant payload when the page came from visual search, else BM25's


MODES = ("adaptive", "fixed", "visual", "lexical")


class HybridRetriever:
    """
    Query both channels, fuse with weighted RRF.

    ponytail: visual channel is Qdrant's exact MaxSim over every page. The design doc's coarse
    pooled-vector stage only pays off once that is measurably slow at full-corpus size.
    """

    def __init__(
        self,
        embedder: Optional[ColQwenEmbedder] = None,
        store: Optional[QdrantMultivectorStore] = None,
        text_index: Optional[BM25Index] = None,
        candidates: int = 50,
        rrf_k: int = 60,
    ):
        """
        Args:
            embedder: ColQwen embedder (loaded lazily, so mode="lexical" runs without a GPU)
            store: Qdrant store (created if None)
            text_index: BM25 index (default location if None)
            candidates: Results taken from each channel before fusion
            rrf_k: RRF damping constant
        """
        self._embedder = embedder
        self.store = store or QdrantMultivectorStore()
        self.text_index = text_index or BM25Index()
        self.candidates = candidates
        self.rrf_k = rrf_k

    @property
    def embedder(self) -> ColQwenEmbedder:
        if self._embedder is None:
            self._embedder = ColQwenEmbedder()
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_filter: Optional[str] = None,
        mode: str = "adaptive",
    ) -> List[HybridResult]:
        """
        Args:
            query: Text query
            top_k: Number of fused results
            document_filter: Optional document ID to restrict both channels to
            mode: "adaptive" (query-dependent weights), "fixed" (1:1),
                  "visual" or "lexical" (single channel) - the last three are ablation baselines

        Returns:
            HybridResults, best first
        """
        return self.search_modes(query, top_k, document_filter, modes=(mode,))[mode]

    def search_modes(
        self,
        query: str,
        top_k: int = 10,
        document_filter: Optional[str] = None,
        modes: Tuple[str, ...] = MODES,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, List[HybridResult]]:
        """
        Like search(), for several modes at once: each channel is queried once, then fused per mode.
        Pass query_embedding when the caller already has it (evidence cropping reuses it).
        """
        for mode in modes:
            if mode not in MODES:
                raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        visual, lexical = self.channels(
            query, document_filter, query_embedding,
            visual=any(m != "lexical" for m in modes), lexical=any(m != "visual" for m in modes),
        )
        return self.fuse_modes(query, visual, lexical, modes, top_k)

    def fuse_modes(
        self, query: str, visual: List[SearchResult], lexical: List[SearchResult],
        modes: Tuple[str, ...] = MODES, top_k: int = 10,
    ) -> Dict[str, List[HybridResult]]:
        """Fuse precomputed channel results (from channels()) for each mode."""
        return {
            m: self._fuse(query, visual if m != "lexical" else [], lexical if m != "visual" else [], m, top_k)
            for m in modes
        }

    def channels(
        self,
        query: str,
        document_filter: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        visual: bool = True,
        lexical: bool = True,
    ) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Each channel's top `candidates` results before fusion (an empty list for a channel not asked for)."""
        visual_results: List[SearchResult] = []
        lexical_results: List[SearchResult] = []
        if visual:
            if query_embedding is None:
                query_embedding = self.embedder.embed_query(query)
            visual_results = self.store.search(query_embedding, top_k=self.candidates, document_filter=document_filter)
        if lexical:
            lexical_results = self.text_index.search(query, top_k=self.candidates, document_filter=document_filter)
        return visual_results, lexical_results

    def _fuse(
        self, query: str, visual: List[SearchResult], lexical: List[SearchResult], mode: str, top_k: int
    ) -> List[HybridResult]:
        if mode == "adaptive":
            w = query_weights(query)
            weights = [w.visual, w.lexical]
        else:
            weights = [1.0, 1.0]  # single-channel modes have an empty other ranking

        visual_keys = [(r.document_id, r.page_num) for r in visual]
        lexical_keys = [(r.document_id, r.page_num) for r in lexical]
        scores = rrf([visual_keys, lexical_keys], weights, k=self.rrf_k)

        visual_rank = {key: i for i, key in enumerate(visual_keys, start=1)}
        lexical_rank = {key: i for i, key in enumerate(lexical_keys, start=1)}
        payloads = {(r.document_id, r.page_num): r.payload for r in lexical + visual}  # visual wins

        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            HybridResult(
                document_id=key[0],
                page_num=key[1],
                score=scores[key],
                visual_rank=visual_rank.get(key),
                lexical_rank=lexical_rank.get(key),
                payload=payloads[key],
            )
            for key in best
        ]
