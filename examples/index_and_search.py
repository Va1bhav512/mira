#!/usr/bin/env python
"""
Example: Index a PDF and search for relevant pages.

Usage:
    uv run python examples/index_and_search.py data/samples/test.pdf
"""

import sys
from pathlib import Path

from mira.retrieval import DocumentIndexer, VisualRetriever
from mira.retrieval.qdrant_store import QdrantMultivectorStore


def main(pdf_path: str):
    """Index a PDF and demonstrate search."""
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"Error: PDF not found at {pdf_file}")
        sys.exit(1)
    
    # Initialize
    print("Initializing Mira Visual Retrieval...")
    indexer = DocumentIndexer()
    
    # Setup Qdrant collection
    indexer.setup()
    
    # Index document (use lower DPI for faster initial testing)
    result = indexer.index_document(
        pdf_path=str(pdf_file),
        dpi=150,
        batch_size=2,
        cache_dir=".cache/embeddings"
    )
    
    # Get collection info
    store = QdrantMultivectorStore()
    info = store.get_collection_info()
    print(f"\nCollection stats: {info}")
    
    # Search examples
    print("\n" + "="*60)
    print("SEARCH EXAMPLES")
    print("="*60)
    
    retriever = VisualRetriever()
    
    queries = [
        "attention mechanism",
        "transformer architecture",
        "encoder decoder",
        "multi-head attention",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Page {result.page_num} (score: {result.score:.4f})")
            print(f"     Text preview: {result.native_text[:80]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/index_and_search.py <pdf_path>")
        print("\nFirst-time setup:")
        print("  1. Make sure Qdrant is running (local: docker run -p 6333:6333 qdrant/qdrant)")
        print("  2. Or set QDRANT_CLUSTER_ENDPOINT and QDRANT_CLUSTER_API_KEY in .env")
        sys.exit(1)
    
    main(sys.argv[1])
