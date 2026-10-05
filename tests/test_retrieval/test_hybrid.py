"""Tests for hybrid retrieval: adaptive weights, RRF, fusion modes, metrics."""

import numpy as np
import pytest

from mira.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from mira.retrieval import BM25Index, PageEmbedding, QdrantMultivectorStore
from mira.retrieval.hybrid import HybridRetriever, query_weights, rrf


class TestQueryWeights:

    def test_identifier_leans_lexical(self):
        w = query_weights("ADXL345 register 0x2D")
        assert w.lexical > w.visual
        assert w.identifiers == ["ADXL345", "0x2D"]

    def test_pin_name_is_identifier(self):
        assert query_weights("what drives USB_VBUS").identifiers == ["USB_VBUS"]

    def test_quoted_leans_lexical(self):
        w = query_weights('pages mentioning "deep sleep"')
        assert w.quoted and w.lexical > w.visual

    def test_visual_cue_leans_visual(self):
        w = query_weights("which chart shows the highest revenue")
        assert w.visual > w.lexical
        assert w.visual_cues == ["chart", "shows"]

    def test_both_or_neither_is_balanced(self):
        assert query_weights("block diagram of the RP2040").visual == 1.0
        assert query_weights("how does attention work").visual == 1.0


class TestRRF:

    def test_scores_match_formula(self):
        scores = rrf([[("a", 0), ("b", 0)], [("b", 0)]], [1.0, 2.0], k=60)
        assert scores[("a", 0)] == pytest.approx(1 / 61)
        assert scores[("b", 0)] == pytest.approx(1 / 62 + 2 / 61)

    def test_weights_must_match_rankings(self):
        with pytest.raises(ValueError):
            rrf([[("a", 0)]], [1.0, 1.0])


class _FakeEmbedder:
    """Query embedding fixed, so we control which page wins MaxSim."""

    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, query):
        return self.vector[None, :]


@pytest.fixture
def retriever(tmp_path):
    """
    Visual channel ranks 'diagram' page first; BM25 ranks 'regs' page first,
    so each fusion mode has a different winner.
    """
    rng = np.random.default_rng(0)
    q = rng.standard_normal(128).astype(np.float32)

    store = QdrantMultivectorStore(url=":memory:", collection_name="hybrid_test")
    store.create_collection()
    pages = {0: q + 0.1 * rng.standard_normal(128),   # diagram: close to query
             1: -q}                                     # regs: opposite
    store.upsert_pages(
        [PageEmbedding("mcu", n, emb[None, :].astype(np.float32), (1, 1), 0, (10, 10), "native")
         for n, emb in pages.items()],
        ["block diagram", "register map"],
    )

    text_index = BM25Index(str(tmp_path))
    text_index.add_pages("mcu", [(0, "block diagram of the clock tree"),
                                 (1, "register 0x2D POWER_CTL register map")])
    return HybridRetriever(embedder=_FakeEmbedder(q), store=store, text_index=text_index)


class TestHybridRetriever:

    def test_single_channel_modes(self, retriever):
        assert retriever.search("register 0x2D", mode="visual")[0].page_num == 0
        assert retriever.search("register 0x2D", mode="lexical")[0].page_num == 1

    def test_adaptive_identifier_query_follows_lexical(self, retriever):
        # "clock" puts page 0 second in BM25, so both channels rank both pages, in opposite
        # orders: fixed weights tie exactly, and the identifier must break it toward lexical.
        query = "register 0x2D clock"
        fixed = retriever.search(query, mode="fixed")
        assert fixed[0].score == pytest.approx(fixed[1].score)
        adaptive = retriever.search(query, mode="adaptive")
        assert adaptive[0].page_num == 1
        assert adaptive[0].score > adaptive[1].score
        assert (adaptive[0].visual_rank, adaptive[0].lexical_rank) == (2, 1)

    def test_adaptive_visual_query_follows_visual(self, retriever):
        # Channels disagree ("register" puts page 1 first in BM25), so fixed weights tie exactly;
        # the cue word "diagram" must break the tie toward the visual winner.
        assert retriever.search("register diagram", mode="lexical")[0].page_num == 1
        fixed = retriever.search("register diagram", mode="fixed")
        assert fixed[0].score == pytest.approx(fixed[1].score)
        adaptive = retriever.search("register diagram", mode="adaptive")
        assert adaptive[0].page_num == 0
        assert adaptive[0].score > adaptive[1].score

    def test_lexical_only_page_keeps_bm25_payload(self, retriever):
        result = retriever.search("register", mode="lexical")[0]
        assert result.visual_rank is None
        assert "POWER_CTL" in result.payload["native_text"]

    def test_rejects_unknown_mode(self, retriever):
        with pytest.raises(ValueError):
            retriever.search("x", mode="bm25")


class TestMetrics:
    ranking = [("d", 3), ("d", 7), ("d", 1)]

    def test_recall(self):
        assert recall_at_k(self.ranking, {("d", 7)}, 1) == 0
        assert recall_at_k(self.ranking, {("d", 7)}, 2) == 1
        assert recall_at_k(self.ranking, {("d", 7), ("d", 9)}, 3) == 0.5

    def test_reciprocal_rank(self):
        assert reciprocal_rank(self.ranking, {("d", 7)}) == 0.5
        assert reciprocal_rank(self.ranking, {("x", 0)}) == 0

    def test_ndcg(self):
        assert ndcg_at_k(self.ranking, {("d", 3)}, 10) == 1
        assert ndcg_at_k(self.ranking, {("d", 7)}, 10) == pytest.approx(1 / np.log2(3))
