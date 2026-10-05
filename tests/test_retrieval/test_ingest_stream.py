"""Tests for DocumentIndexer.index_page_stream (non-PDF sources, e.g. ViDoRe)."""

import numpy as np
from PIL import Image

from mira.pdf import ProcessedPage
from mira.retrieval import BM25Index, DocumentIndexer, QdrantMultivectorStore


class _FakeEmbedder:
    """Stands in for ColQwen so ingest runs on CPU: 2x2 patch grid after 1 prompt token."""

    def embed_images(self, images, batch_size=4):
        rng = np.random.default_rng(0)
        return [(rng.standard_normal((5, 128)).astype(np.float32), (2, 2), 1) for _ in images]


def _page(page_num, text):
    return ProcessedPage(
        page_num=page_num,
        image=Image.new("RGB", (32, 32), color="white"),
        native_text=text,
        text=text,
        text_source="native",
        is_usable=True,
        char_count=len(text),
    )


def test_index_page_stream_populates_both_stores(tmp_path):
    store = QdrantMultivectorStore(url=":memory:", collection_name="stream_test")
    store.create_collection()
    indexer = DocumentIndexer(
        embedder=_FakeEmbedder(), store=store, text_index=BM25Index(str(tmp_path))
    )

    # A generator (lazy image decoding, as in scripts/index_vidore.py) over 3 pages
    pages = (
        _page(n, text)
        for n, text in enumerate([
            "the transformer architecture",
            "multi-head attention scores",
            "residual connections and layer norm",
        ])
    )
    result = indexer.index_page_stream(pages, document_id="paper", chunk_size=2)

    assert result.pages_indexed == 3
    assert store.get_collection_info()["points_count"] == 3
    reloaded = BM25Index(str(tmp_path))
    assert len(reloaded.pages) == 3
    hit = reloaded.search("layer norm")[0]
    assert (hit.document_id, hit.page_num) == ("paper", 2)


def test_index_document_still_works_via_stream(tmp_path):
    """The PDF path must keep working through the refactored stream sink."""
    from pathlib import Path

    import pytest

    pdf = Path(__file__).parents[2] / "data" / "samples" / "test.pdf"
    if not pdf.exists():
        pytest.skip("No sample PDF found")

    store = QdrantMultivectorStore(url=":memory:", collection_name="stream_pdf_test")
    store.create_collection()
    indexer = DocumentIndexer(
        embedder=_FakeEmbedder(), store=store, text_index=BM25Index(str(tmp_path))
    )
    result = indexer.index_document(str(pdf), document_id="attention", dpi=36, chunk_size=4)

    assert result.pages_indexed == 15
    assert store.get_collection_info()["points_count"] == 15
    assert len(BM25Index(str(tmp_path)).pages) == 15
