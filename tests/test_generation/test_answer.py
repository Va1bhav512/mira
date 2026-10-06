"""Tests for the answering pipeline with stub retriever/generator (no models)."""

import numpy as np
import pytest
from PIL import Image

from mira.generation import answer_query, parse_reply
from mira.retrieval import BM25Index, HybridRetriever, PageEmbedding, QdrantMultivectorStore


def test_parse_reply_json_and_unknown_ids():
    reply = 'Sure.\n```json\n{"answer": "3.3 V", "evidence_ids": ["E2", "E9"]}\n```'
    assert parse_reply(reply, ["E1", "E2"]) == ("3.3 V", ["E2"])


def test_parse_reply_falls_back_to_raw_text():
    assert parse_reply("The supply is 3.3 V.", ["E1"]) == ("The supply is 3.3 V.", [])
    assert parse_reply('{"answer": 3}', ["E1"]) == ('{"answer": 3}', [])
    assert parse_reply('{"answer": "x", "evidence_ids": "E1"}', ["E1"]) == ("x", [])


class StubEmbedder:
    def __init__(self, query):
        self.query = query
        self.calls = 0

    def embed_query(self, q):
        self.calls += 1
        return self.query


class StubGenerator:
    def generate(self, query, evidence):
        self.evidence = evidence
        return "42", [evidence[0][0]], '{"answer": "42", "evidence_ids": ["E1"]}'


@pytest.fixture
def retriever(tmp_path):
    """One indexed 10x10-patch page whose bottom-right 4x4 block matches the query token."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal(8)
    q /= np.linalg.norm(q)
    patches = rng.standard_normal((10, 10, 8))
    patches[6:, 6:] = q
    patches /= np.linalg.norm(patches, axis=-1, keepdims=True)
    emb = np.concatenate([rng.standard_normal((2, 8)), patches.reshape(100, 8)])  # 2 prompt tokens first

    store = QdrantMultivectorStore(url=":memory:", collection_name="t")
    store.create_collection()
    store.upsert_pages([PageEmbedding("doc", 3, emb.astype(np.float32), (10, 10), 2, (400, 400), "native")], ["text"])
    text = BM25Index(str(tmp_path / "bm25"))
    text.add_pages("doc", [(3, "the answer is forty two")])
    return HybridRetriever(embedder=StubEmbedder(q[None]), store=store, text_index=text)


def test_store_get_page_round_trip(retriever):
    page = retriever.store.get_page("doc", 3)
    assert page.patch_grid == (10, 10) and page.image_token_start == 2 and page.image_dims == (400, 400)
    assert page.patch_embeddings.shape == (10, 10, 8)
    with pytest.raises(KeyError):
        retriever.store.get_page("doc", 4)


@pytest.mark.parametrize("strategy", ["page", "max_patch", "heatmap"])
def test_answer_cites_retriever_boxes(retriever, strategy):
    seen = []
    gen = StubGenerator()

    def page_image(doc, page, box):
        seen.append((doc, page, box))
        return Image.new("RGB", (10, 10))

    ans = answer_query("answer", retriever, gen, page_image, top_pages=1, strategy=strategy)

    assert ans.answer == "42"
    assert retriever.embedder.calls == 1  # embedding reused for retrieval and heatmap
    assert [e[0] for e in gen.evidence] == [e.id for e in ans.evidence]
    (doc, page, box), = seen[:1]
    assert ans.sources[0] == {"document": "doc", "page": 3, "bbox": [round(v, 4) for v in box]}
    if strategy == "page":
        assert box == (0.0, 0.0, 1.0, 1.0)
    else:
        x0, y0, x1, y1 = box
        assert x0 >= 0.5 and y0 >= 0.5  # the matching block starts at 0.6 (minus padding)


def test_unknown_strategy_rejected(retriever):
    with pytest.raises(ValueError):
        answer_query("q", retriever, StubGenerator(), lambda *a: None, strategy="nope")
