"""Tests for visual retrieval module."""

import pytest
from pathlib import Path
import numpy as np

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
        # Skip if CUDA not available and too slow for CPU
        try:
            return ColQwenEmbedder(use_4bit=True)
        except Exception as e:
            pytest.skip(f"Could not load ColQwen model: {e}")
    
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
        for emb in embeddings:
            assert emb.shape[1] == 128
            assert emb.shape[0] > 100  # Many patches


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
            image_dims=(1000, 1000),
            text_source="native"
        )
        
        # Upsert
        store.upsert_page(page_emb, native_text="Test content")
        
        # Check count
        info = store.get_collection_info()
        assert info['points_count'] == 1
        
        # Search
        query = np.random.randn(5, 128).astype(np.float32)
        results = store.search(query, top_k=1)
        
        assert len(results) == 1
        assert results[0].document_id == "test_doc"


class TestEmbeddingGeneration:
    """Tests for embedding generation."""
    
    def test_estimate_patch_grid(self):
        """Test patch grid estimation."""
        from mira.retrieval.embeddings import _estimate_patch_grid
        
        # Test with known dimensions
        grid = _estimate_patch_grid(400, (1000, 1000))
        
        assert isinstance(grid, tuple)
        assert len(grid) == 2
        assert grid[0] * grid[1] <= 400


# Skip slow tests unless explicitly requested
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def sample_pdf():
    """Path to sample PDF for testing."""
    test_pdf = Path(__file__).parent.parent.parent / "data" / "samples" / "test.pdf"
    if test_pdf.exists():
        return str(test_pdf)
    pytest.skip("No sample PDF found")
