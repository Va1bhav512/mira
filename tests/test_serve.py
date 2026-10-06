"""API and UI wiring with stub models (no GPU)."""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("gradio")
from fastapi.testclient import TestClient

from mira.retrieval import BM25Index, HybridRetriever, PageEmbedding, QdrantMultivectorStore
from mira.serve import Mira, build_app, build_ui, overlay

DIM = 128


class StubEmbedder:
    def __init__(self, q):
        self.q = q

    def embed_query(self, query):
        return self.q


class StubGenerator:
    def generate(self, query, evidence):
        return "42", [evidence[0][0]], '{"evidence_ids": ["E1"], "answer": "42"}'


@pytest.fixture
def mira(tmp_path):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(DIM)
    patches = rng.standard_normal((4, 4, DIM))
    patches[2:, 2:] = q
    store = QdrantMultivectorStore(url=":memory:", collection_name="serve")
    store.create_collection()
    store.upsert_pages([PageEmbedding("doc", 0, patches.reshape(16, DIM).astype(np.float32), (4, 4), 0, (100, 100), "native")], ["x"])
    text = BM25Index(str(tmp_path))
    text.add_pages("doc", [(0, "register 0x2D power control")])
    retriever = HybridRetriever(embedder=StubEmbedder(q[None].astype(np.float32)), store=store, text_index=text)
    return Mira(retriever, StubGenerator(), str(tmp_path), page_image=lambda doc, page, box: Image.new("RGB", (20, 20)))


def test_query_endpoint(mira):
    client = TestClient(build_app(mira, examples=[]))
    body = client.post("/query", json={"question": "what does 0x2D control", "top_pages": 1, "include_images": True}).json()
    assert body["answer"] == "42"
    assert body["sources"][0]["document"] == "doc"
    assert body["retrieved"][0] == {"document": "doc", "page": 0, "score": body["retrieved"][0]["score"],
                                    "visual_rank": 1, "lexical_rank": 1}
    assert body["weights"]["identifiers"] == ["0x2D"]
    assert body["evidence"][0]["image_png_b64"]
    assert "_answer" not in body


def test_ui_builds(mira):
    assert build_ui(mira, examples=["what does 0x2D control"]) is not None


def test_overlay_marks_heat_and_boxes():
    page = Image.new("RGB", (100, 100), "white")
    heat = np.zeros((4, 4)); heat[3, 3] = 1.0
    out = np.asarray(overlay(page, heat, [(0.0, 0.0, 0.5, 0.5)]))
    assert out[95, 95, 1] < 200          # hot corner tinted red
    assert out[2, 25].tolist() != [255, 255, 255]  # box edge drawn
    assert out[30, 75].tolist() == [255, 255, 255]  # elsewhere untouched
