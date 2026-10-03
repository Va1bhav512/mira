#!/usr/bin/env python
"""
Quick verification of Phase 2 implementation.
Tests imports and basic functionality without GPU.
"""

import sys


def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    try:
        from mira.retrieval import (
            ColQwenEmbedder,
            PageEmbedding,
            generate_embeddings,
            QdrantMultivectorStore,
            SearchResult,
            DocumentIndexer,
            IndexingResult,
            VisualRetriever,
            RetrievalResult,
        )
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_qdrant_connection():
    """Test Qdrant connection (cloud or local)."""
    print("\nTesting Qdrant connection...")
    
    try:
        from mira.retrieval import QdrantMultivectorStore
        
        store = QdrantMultivectorStore()
        collections = store.client.get_collections()
        
        print(f"  ✓ Connected to Qdrant")
        print(f"    Collections: {[c.name for c in collections.collections]}")
        return True
    except Exception as e:
        print(f"  ✗ Qdrant connection failed: {e}")
        print("    Make sure .env has QDRANT_CLUSTER_ENDPOINT and QDRANT_CLUSTER_API_KEY")
        return False


def test_create_collection():
    """Test creating collection."""
    print("\nTesting collection creation...")
    
    try:
        from mira.retrieval import QdrantMultivectorStore
        
        store = QdrantMultivectorStore(collection_name="mira_test")
        store.create_collection(exist_ok=True)
        
        info = store.get_collection_info()
        print(f"  ✓ Collection ready: {info['status']}")
        return True
    except Exception as e:
        print(f"  ✗ Collection creation failed: {e}")
        return False


def test_page_embedding_dataclass():
    """Test PageEmbedding dataclass."""
    print("\nTesting PageEmbedding...")
    
    try:
        from mira.retrieval import PageEmbedding
        import numpy as np
        
        emb = PageEmbedding(
            document_id="test",
            page_num=0,
            embeddings=np.random.randn(100, 128),
            patch_grid=(10, 10),
            image_token_start=0,
            image_dims=(1000, 1000),
            text_source="native"
        )
        
        print(f"  ✓ PageEmbedding created")
        print(f"    Embeddings shape: {emb.embeddings.shape}")
        return True
    except Exception as e:
        print(f"  ✗ PageEmbedding failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Mira Phase 2 Verification")
    print("="*60)
    
    results = [
        test_imports(),
        test_qdrant_connection(),
        test_create_collection(),
        test_page_embedding_dataclass(),
    ]
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All tests passed!")
        print("="*60)
        print("\nPhase 2 is ready to use!")
        print("\nNext steps:")
        print("  1. Run: uv run python examples/index_and_search.py data/samples/test.pdf")
        print("  2. This will index the PDF and run test searches")
        return 0
    else:
        print("✗ Some tests failed")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
