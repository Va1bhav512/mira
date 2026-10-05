"""Tests for BM25 lexical retrieval."""

from pathlib import Path

import numpy as np
import pytest

from mira.retrieval import DocumentIndexer, QdrantMultivectorStore
from mira.retrieval.lexical import BM25Index, tokenize


class TestTokenize:
    """Identifiers must survive tokenization intact; plain words are stemmed."""

    def test_part_number_kept_whole(self):
        assert tokenize("STM32F401RE") == ["stm32f401re"]

    def test_hex_address_kept_whole(self):
        assert tokenize("at 0x7FA93C") == ["0x7fa93c"]

    def test_compound_kept_whole_plus_parts(self):
        assert tokenize("VDD_IO") == ["vdd_io", "vdd", "io"]

    def test_superscript_normalized(self):
        assert tokenize("I²C") == tokenize("I2C") == ["i2c"]

    def test_ligature_normalized(self):
        assert tokenize("conﬁguration") == tokenize("configuration")

    def test_line_break_hyphenation_rejoined(self):
        assert tokenize("trans-\nformer") == tokenize("transformer")

    def test_plain_words_stemmed_stopwords_dropped(self):
        assert tokenize("The registers") == ["regist"]

    def test_trailing_punctuation_not_part_of_token(self):
        assert tokenize("pins.") == tokenize("pins")


def _index(tmp_path):
    index = BM25Index(str(tmp_path))
    index.add_pages("stm32", [
        (0, "STM32F401RE datasheet. Flash memory up to 512 Kbytes."),
        (1, "GPIO pins PA0-PA15 and the VDD_IO supply rail."),
    ])
    index.add_pages("esp32", [
        (0, "ESP32 datasheet. Wi-Fi and Bluetooth radio, GPIO pins."),
    ])
    return index


class TestBM25Index:

    def test_identifier_query_finds_page(self, tmp_path):
        results = _index(tmp_path).search("stm32f401re flash size")
        assert (results[0].document_id, results[0].page_num) == ("stm32", 0)

    def test_compound_part_matches(self, tmp_path):
        results = _index(tmp_path).search("VDD IO")
        assert (results[0].document_id, results[0].page_num) == ("stm32", 1)

    def test_document_filter(self, tmp_path):
        results = _index(tmp_path).search("GPIO pins", document_filter="esp32")
        assert [r.document_id for r in results] == ["esp32"]

    def test_no_match_returns_empty(self, tmp_path):
        assert _index(tmp_path).search("zigbee") == []

    def test_save_load_round_trip(self, tmp_path):
        index = _index(tmp_path)
        index.save()
        reloaded = BM25Index(str(tmp_path))
        assert reloaded.pages == index.pages
        assert reloaded.search("bluetooth")[0].document_id == "esp32"

    def test_add_after_search_rebuilds(self, tmp_path):
        index = _index(tmp_path)
        assert index.search("zigbee") == []
        index.add_pages("cc2530", [(0, "Zigbee SoC")])
        assert index.search("zigbee")[0].document_id == "cc2530"

    def test_delete_document(self, tmp_path):
        index = _index(tmp_path)
        index.delete_document("stm32")
        assert all(r.document_id == "esp32" for r in index.search("GPIO datasheet"))


class _FakeEmbedder:
    """Stands in for ColQwen so ingest runs on CPU: 2x2 patch grid after 1 prompt token."""
    
    def embed_images(self, images, batch_size=4):
        rng = np.random.default_rng(0)
        return [(rng.standard_normal((5, 128)).astype(np.float32), (2, 2), 1) for _ in images]


def test_ingest_populates_bm25_and_qdrant(tmp_path):
    pdf = Path(__file__).parents[2] / "data" / "samples" / "test.pdf"
    if not pdf.exists():
        pytest.skip("No sample PDF found")
    
    store = QdrantMultivectorStore(url=":memory:", collection_name="ingest_test")
    store.create_collection()
    indexer = DocumentIndexer(
        embedder=_FakeEmbedder(), store=store, text_index=BM25Index(str(tmp_path))
    )
    
    result = indexer.index_document(str(pdf), document_id="attention", dpi=36, chunk_size=4)
    
    assert result.pages_indexed == 15
    assert store.get_collection_info()["points_count"] == 15
    # Saved to disk, and a fresh index finds the BLEU results table text
    reloaded = BM25Index(str(tmp_path))
    assert len(reloaded.pages) == 15
    assert reloaded.search("BLEU EN-DE newstest2014")[0].document_id == "attention"
