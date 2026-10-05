"""Tests for visual retrieval module."""

import pytest
from pathlib import Path
import numpy as np
import torch

from mira.retrieval import (
    ColQwenEmbedder,
    PageEmbedding,
    QdrantMultivectorStore,
    VisualRetriever,
)


class TestColQwenEmbedder:
    """Tests for ColQwen embedding model."""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        # fp16 model needs ~7.5 GB VRAM; on CPU it loads ~12 GB in fp32
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 10e9:
            pytest.skip("ColQwen tests need a GPU with >=10 GB VRAM")
        # No try/except: a load failure on a capable GPU is a real failure, not a skip
        return ColQwenEmbedder()
    
    def test_embedder_loads_model(self, embedder):
        """Test that model loads successfully."""
        assert embedder.model is not None
        assert embedder.processor is not None
    
    def test_embed_query_returns_correct_shape(self, embedder):
        """Test query embedding shape."""
        query = "attention mechanism"
        embedding = embedder.embed_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 128  # ColQwen dimension
        assert embedding.shape[0] > 0     # At least one token
    
    @pytest.mark.slow
    def test_embed_images_batch(self, embedder, sample_pdf):
        """Test batch image embedding."""
        from mira.pdf import process_pdf
        
        pages = process_pdf(sample_pdf, dpi=72)
        images = [p.image for p in pages[:2]]
        
        embeddings = embedder.embed_images(images, batch_size=2)
        
        assert len(embeddings) == 2
        for emb, (rows, cols), start in embeddings:
            assert emb.shape[1] == 128
            assert rows * cols > 100  # Many patches
            assert start + rows * cols <= emb.shape[0]
            assert np.abs(emb).sum(axis=1).min() > 0  # no zero padding rows
    
    @pytest.mark.slow
    def test_fp16_overflow_page_is_recovered(self, embedder):
        """Page 251 of the scanned NASA report overflows to NaN in fp16 on a T4."""
        pdf = Path(__file__).parents[2] / "data" / "samples" / "19700022229.pdf"
        if not pdf.exists():
            pytest.skip("19700022229.pdf not available")
        from mira.pdf import render_page
        
        (emb, _, _), = embedder.embed_images([render_page(str(pdf), 251, dpi=150)])
        
        assert not np.isnan(emb).any()
        assert embedder.model.dtype == torch.float16  # fp16 restored after the bf16 retry


class TestQdrantStore:
    """Tests for Qdrant multivector storage."""
    
    @pytest.fixture
    def store(self):
        """Create in-memory Qdrant store."""
        store = QdrantMultivectorStore(
            url=":memory:",
            collection_name="test_collection"
        )
        store.create_collection()
        return store
    
    def test_collection_created(self, store):
        """Test that collection is created."""
        info = store.get_collection_info()
        assert info['status'] == 'green'
        assert info['points_count'] == 0
    
    def test_upsert_and_search(self, store):
        """Test inserting and searching."""
        # Create mock embedding
        embedding = np.random.randn(100, 128).astype(np.float32)
        page_emb = PageEmbedding(
            document_id="test_doc",
            page_num=0,
            embeddings=embedding,
            patch_grid=(10, 10),
            image_token_start=0,
            image_dims=(1000, 1000),
            text_source="native"
        )
        
        # Upsert
        store.upsert_pages([page_emb], ["Test content"])
        
        # Check count
        info = store.get_collection_info()
        assert info['points_count'] == 1
        
        # Search
        query = np.random.randn(5, 128).astype(np.float32)
        results = store.search(query, top_k=1)
        
        assert len(results) == 1
        assert results[0].document_id == "test_doc"


def _stub_embedder(nan_in_fp16=(), nan_in_bf16=()):
    """ColQwenEmbedder with the model stubbed out: chosen inputs embed to NaN per dtype."""
    e = ColQwenEmbedder.__new__(ColQwenEmbedder)
    e.dtype = e.current = torch.float16
    e.loads, e.calls = [], []
    
    def load(dtype):
        e.loads.append(dtype)
        e.current = dtype
    
    def batch(images):
        e.calls.append(list(images))
        nan = nan_in_fp16 if e.current == torch.float16 else nan_in_bf16
        return [(np.full((3, 128), np.nan if img in nan else 1.0, np.float32), (1, 1), 0) for img in images]
    
    e._load, e._embed_image_batch = load, batch
    return e


class TestNaNFallback:
    
    def test_only_nan_images_redone_in_bf16(self):
        e = _stub_embedder(nan_in_fp16={"b"})
        results = e.embed_images(["a", "b", "c"], batch_size=3)
        assert not any(np.isnan(emb).any() for emb, _, _ in results)
        assert e.calls == [["a", "b", "c"], ["b"]]
        assert e.loads == [torch.bfloat16, torch.float16]
    
    def test_no_nan_no_reload(self):
        e = _stub_embedder()
        e.embed_images(["a", "b"])
        assert e.loads == []
    
    def test_raises_if_still_nan_and_restores_fp16(self):
        e = _stub_embedder(nan_in_fp16={"b"}, nan_in_bf16={"b"})
        with pytest.raises(ValueError):
            e.embed_images(["a", "b"])
        assert e.current == torch.float16


class TestEmbeddingGeneration:
    """Tests for embedding generation."""
    
    def test_patch_embeddings_reshape(self):
        """Image tokens after the prompt prefix reshape into the patch grid."""
        embeddings = np.arange(30 * 128, dtype=np.float32).reshape(30, 128)
        page_emb = PageEmbedding(
            document_id="test_doc",
            page_num=0,
            embeddings=embeddings,
            patch_grid=(4, 5),
            image_token_start=3,
            image_dims=(500, 400),
            text_source="native"
        )
        
        patches = page_emb.patch_embeddings
        assert patches.shape == (4, 5, 128)
        assert np.array_equal(patches[0, 0], embeddings[3])
        assert np.array_equal(patches[3, 4], embeddings[22])
    
    def test_cache_round_trip_is_lossless_for_fp16_outputs(self, tmp_path):
        """Cache stores fp16; embeddings from the fp16 model must come back exactly, as float32."""
        from mira.retrieval.embeddings import _load_from_cache, _save_to_cache
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((5, 128)).astype(np.float16).astype(np.float32)
        _save_to_cache(tmp_path, PageEmbedding("d", 0, emb, (2, 2), 1, (10, 10), "native"))

        loaded = _load_from_cache(tmp_path, 0)
        assert loaded.embeddings.dtype == np.float32
        assert np.array_equal(loaded.embeddings, emb)

    def test_cache_rejects_nan_embeddings(self, tmp_path):
        """A page cached before the NaN fallback existed must be recomputed, not reused."""
        from mira.retrieval.embeddings import _load_from_cache, _save_to_cache
        emb = np.full((5, 128), np.nan, np.float32)
        _save_to_cache(tmp_path, PageEmbedding("d", 0, emb, (2, 2), 1, (10, 10), "native"))
        assert _load_from_cache(tmp_path, 0) is None


@pytest.fixture
def sample_pdf():
    """Path to sample PDF for testing."""
    test_pdf = Path(__file__).parent.parent.parent / "data" / "samples" / "test.pdf"
    if test_pdf.exists():
        return str(test_pdf)
    pytest.skip("No sample PDF found")
