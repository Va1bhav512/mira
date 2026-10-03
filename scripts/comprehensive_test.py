#!/usr/bin/env python
"""
Comprehensive test for Phase 2 with multiple documents.
"""

import sys
from pathlib import Path
import time
from mira.pdf import process_pdf
from mira.retrieval import DocumentIndexer, VisualRetriever, QdrantMultivectorStore
from mira.retrieval.colqwen import ColQwenEmbedder


def test_phase1_processing(pdf_path: str):
    """Test Phase 1: PDF processing."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Processing {Path(pdf_path).name}")
    print(f"{'='*60}")
    
    start = time.time()
    pages = process_pdf(pdf_path, dpi=150)
    duration = time.time() - start
    
    # Stats
    native_count = sum(1 for p in pages if p.text_source == 'native')
    ocr_count = sum(1 for p in pages if p.text_source == 'ocr')
    total_chars = sum(p.char_count for p in pages)
    
    print(f"✓ Processed {len(pages)} pages in {duration:.1f}s")
    print(f"  Text sources: {native_count} native, {ocr_count} OCR")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Time per page: {duration/len(pages):.2f}s")
    
    return pages, duration


def test_phase2_embedding(pages, pdf_path: str, embedder):
    """Test Phase 2: Embedding generation (with cache)."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Embedding {Path(pdf_path).name}")
    print(f"{'='*60}")
    
    from mira.retrieval.embeddings import generate_embeddings
    
    doc_id = Path(pdf_path).stem
    start = time.time()
    
    embeddings = generate_embeddings(
        pages=pages,
        embedder=embedder,
        batch_size=4,
        document_id=doc_id,
        cache_dir=".cache/embeddings"
    )
    
    duration = time.time() - start
    
    # Stats
    total_patches = sum(e.embeddings.shape[0] for e in embeddings)
    avg_patches = total_patches / len(embeddings)
    
    print(f"✓ Generated embeddings in {duration:.1f}s")
    print(f"  Total patches: {total_patches:,}")
    print(f"  Avg patches/page: {avg_patches:.0f}")
    print(f"  Embedding shape: [{embeddings[0].embeddings.shape[0]}, {embeddings[0].embeddings.shape[1]}]")
    
    return embeddings, duration


def test_search_queries(retriever, queries: list):
    """Test search functionality."""
    print(f"\n{'='*60}")
    print(f"Testing Search Queries")
    print(f"{'='*60}")
    
    total_time = 0
    
    for query in queries:
        start = time.time()
        results = retriever.search(query, top_k=3)
        duration = time.time() - start
        total_time += duration
        
        print(f"\nQuery: '{query}'")
        print(f"  Latency: {duration*1000:.0f}ms")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.document_id}/p{result.page_num} (score: {result.score:.4f})")
            print(f"     Preview: {result.native_text[:60]}...")
    
    print(f"\nAvg search latency: {total_time/len(queries)*1000:.0f}ms")


def main():
    """Run comprehensive tests."""
    print("="*60)
    print("Mira Phase 1 & 2 Comprehensive Test")
    print("="*60)
    
    # Get all PDFs
    pdf_dir = Path("data/samples")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"\nFound {len(pdfs)} PDFs to test")
    
    # Initialize components
    print("\nInitializing components...")
    
    try:
        embedder = ColQwenEmbedder(use_4bit=True)
        print("✓ ColQwen embedder loaded (4-bit mode)")
    except Exception as e:
        print(f"✗ Failed to load embedder: {e}")
        print("\nNote: GPU required for embedding. Testing Phase 1 only.")
        embedder = None
    
    indexer = DocumentIndexer(embedder=embedder)
    indexer.setup()
    
    store = QdrantMultivectorStore()
    retriever = VisualRetriever(embedder=embedder, store=store)
    
    # Test stats
    all_stats = {
        'phase1_time': 0,
        'phase2_time': 0,
        'total_pages': 0,
        'total_patches': 0,
    }
    
    # Process first 3 documents fully (with embeddings)
    # Process remaining documents just Phase 1 (to save time)
    full_process_count = 3
    
    for i, pdf_path in enumerate(pdfs[:full_process_count]):
        print(f"\n\n{'='*60}")
        print(f"Document {i+1}/{min(len(pdfs), full_process_count)}: {pdf_path.name}")
        print(f"{'='*60}")
        
        # Phase 1
        pages, p1_time = test_phase1_processing(str(pdf_path))
        all_stats['phase1_time'] += p1_time
        all_stats['total_pages'] += len(pages)
        
        # Phase 2 (if embedder available)
        if embedder:
            embeddings, p2_time = test_phase2_embedding(pages, str(pdf_path), embedder)
            all_stats['phase2_time'] += p2_time
            all_stats['total_patches'] += sum(e.embeddings.shape[0] for e in embeddings)
            
            # Index to Qdrant
            for emb, page in zip(embeddings, pages):
                store.upsert_page(emb, native_text=page.text)
            
            print(f"✓ Indexed {len(embeddings)} pages to Qdrant")
    
    # Remaining docs - Phase 1 only
    print(f"\n\n{'='*60}")
    print(f"Processing remaining {len(pdfs) - full_process_count} PDFs (Phase 1 only)")
    print(f"{'='*60}")
    
    for pdf_path in pdfs[full_process_count:full_process_count+5]:  # Limit to 5 more
        pages, p1_time = test_phase1_processing(str(pdf_path))
        all_stats['phase1_time'] += p1_time
        all_stats['total_pages'] += len(pages)
    
    # Test search
    if embedder:
        test_queries = [
            "transformer architecture",
            "datasheet specifications",
            "temperature sensor",
            "power consumption",
            "GPIO pins"
        ]
        test_search_queries(retriever, test_queries)
    
    # Final stats
    print(f"\n\n{'='*60}")
    print(f"FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Documents tested: {min(len(pdfs), full_process_count + 5)}")
    print(f"Total pages processed: {all_stats['total_pages']}")
    if embedder:
        print(f"Total patches embedded: {all_stats['total_patches']:,}")
        print(f"Avg patches/page: {all_stats['total_patches']/all_stats['total_pages']:.0f}")
    print(f"\nTiming:")
    print(f"  Phase 1 total: {all_stats['phase1_time']:.1f}s")
    if embedder:
        print(f"  Phase 2 total: {all_stats['phase2_time']:.1f}s")
        print(f"  Total time: {all_stats['phase1_time'] + all_stats['phase2_time']:.1f}s")
        print(f"  Time per page: {(all_stats['phase1_time'] + all_stats['phase2_time'])/all_stats['total_pages']:.2f}s")
    
    # Qdrant stats
    if embedder:
        try:
            info = store.get_collection_info()
            print(f"\nQdrant:")
            print(f"  Points stored: {info['points_count']}")
            print(f"  Status: {info['status']}")
        except:
            pass
    
    print(f"\n{'='*60}")
    print("✓ Testing Complete!")
    print(f"{'='*60}")
    
    # Collection info
    print(f"\nIndexed Documents:")
    try:
        docs = store.list_documents()
        for doc in docs[:10]:
            print(f"  - {doc}")
        if len(docs) > 10:
            print(f"  ... and {len(docs) - 10} more")
    except:
        pass


if __name__ == "__main__":
    main()
