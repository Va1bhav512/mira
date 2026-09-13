from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from time import time

from mira.pdf import process_pdf, ProcessedPage
from .colqwen import ColQwenEmbedder
from .embeddings import generate_embeddings, PageEmbedding
from .qdrant_store import QdrantMultivectorStore


@dataclass
class IndexingResult:
    """Result of document indexing."""
    document_id: str
    pages_indexed: int
    total_patches: int
    duration_seconds: float
    text_source_distribution: dict


class DocumentIndexer:
    """Index PDF documents into Qdrant."""
    
    def __init__(
        self,
        embedder: Optional[ColQwenEmbedder] = None,
        store: Optional[QdrantMultivectorStore] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize document indexer.
        
        Args:
            embedder: ColQwen embedder (created if None)
            store: Qdrant store (created if None)
            qdrant_url: Qdrant URL (from env if None)
            qdrant_api_key: Qdrant API key (from env if None)
        """
        self.embedder = embedder or ColQwenEmbedder()
        self.store = store or QdrantMultivectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
    
    def setup(self):
        """Ensure Qdrant collection exists."""
        self.store.create_collection(exist_ok=True)
    
    def index_document(
        self,
        pdf_path: str,
        document_id: Optional[str] = None,
        dpi: int = 150,
        batch_size: int = 4,
        cache_dir: Optional[str] = None
    ) -> IndexingResult:
        """
        Full pipeline: PDF → Process → Embed → Store.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Unique ID (default: filename)
            dpi: Rendering resolution (lower for faster processing)
            batch_size: Batch size for embedding
            cache_dir: Optional cache directory
        
        Returns:
            IndexingResult with stats
        """
        start_time = time()
        
        # Get document ID from filename
        if document_id is None:
            document_id = Path(pdf_path).stem
        
        print(f"\n{'='*60}")
        print(f"Indexing: {pdf_path}")
        print(f"Document ID: {document_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Process PDF (Phase 1)
        print("Step 1: Processing PDF...")
        pages = process_pdf(pdf_path, dpi=dpi)
        print(f"  → Processed {len(pages)} pages")
        
        # Step 2: Generate embeddings
        print("\nStep 2: Generating embeddings...")
        embeddings = generate_embeddings(
            pages=pages,
            embedder=self.embedder,
            batch_size=batch_size,
            document_id=document_id,
            cache_dir=cache_dir
        )
        print(f"  → Generated {len(embeddings)} page embeddings")
        
        # Step 3: Store in Qdrant
        print("\nStep 3: Storing in Qdrant...")
        for page_emb, page in zip(embeddings, pages):
            self.store.upsert_page(
                page_embedding=page_emb,
                native_text=page.text
            )
        print(f"  → Stored {len(embeddings)} pages in Qdrant")
        
        # Calculate stats
        duration = time() - start_time
        total_patches = sum(e.embeddings.shape[0] for e in embeddings)
        
        # Text source distribution
        text_sources = {}
        for page in pages:
            text_sources[page.text_source] = text_sources.get(page.text_source, 0) + 1
        
        result = IndexingResult(
            document_id=document_id,
            pages_indexed=len(pages),
            total_patches=total_patches,
            duration_seconds=duration,
            text_source_distribution=text_sources
        )
        
        print(f"\n{'='*60}")
        print(f"Indexing Complete!")
        print(f"  Pages: {result.pages_indexed}")
        print(f"  Patches: {result.total_patches}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Text sources: {result.text_source_distribution}")
        print(f"{'='*60}\n")
        
        return result
