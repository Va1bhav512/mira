from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from time import time

from mira.pdf import iter_processed_pages
from .colqwen import ColQwenEmbedder
from .embeddings import generate_embeddings
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
        cache_dir: Optional[str] = None,
        chunk_size: int = 32
    ) -> IndexingResult:
        """
        Full pipeline: PDF → Process → Embed → Store.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Unique ID (default: filename)
            dpi: Rendering resolution (lower for faster processing)
            batch_size: Batch size for embedding
            cache_dir: Optional cache directory
            chunk_size: Pages rendered and held in memory at once
        
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
        
        # Render/embed/store in chunks so a 700-page PDF isn't held in RAM at once
        page_count = 0
        total_patches = 0
        text_sources = {}
        for pages in iter_processed_pages(pdf_path, dpi=dpi, chunk_size=chunk_size):
            print(f"Pages {pages[0].page_num}-{pages[-1].page_num}")
            page_count += len(pages)
            
            embeddings = generate_embeddings(
                pages=pages,
                embedder=self.embedder,
                batch_size=batch_size,
                document_id=document_id,
                cache_dir=cache_dir
            )
            self.store.upsert_pages(embeddings, [page.text for page in pages])
            for page_emb, page in zip(embeddings, pages):
                total_patches += page_emb.embeddings.shape[0]
                text_sources[page.text_source] = text_sources.get(page.text_source, 0) + 1
        
        duration = time() - start_time
        
        result = IndexingResult(
            document_id=document_id,
            pages_indexed=page_count,
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
