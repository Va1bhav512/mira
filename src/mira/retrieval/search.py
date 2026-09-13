from typing import List, Optional
from dataclasses import dataclass

from .colqwen import ColQwenEmbedder
from .qdrant_store import QdrantMultivectorStore, SearchResult


@dataclass
class RetrievalResult:
    """Result from visual retrieval."""
    document_id: str
    page_num: int
    score: float
    native_text: str
    text_source: str
    patch_grid: tuple
    image_dims: tuple


class VisualRetriever:
    """Query interface for visual retrieval."""
    
    def __init__(
        self,
        embedder: Optional[ColQwenEmbedder] = None,
        store: Optional[QdrantMultivectorStore] = None
    ):
        """
        Initialize visual retriever.
        
        Args:
            embedder: ColQwen embedder
            store: Qdrant store
        """
        self.embedder = embedder or ColQwenEmbedder()
        self.store = store or QdrantMultivectorStore()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        document_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Search for pages matching query.
        
        Args:
            query: Text query
            top_k: Number of results
            document_filter: Optional document ID to filter results
        
        Returns:
            List of RetrievalResult objects
        """
        # Encode query
        print(f"Searching for: '{query}'")
        query_embedding = self.embedder.embed_query(query)
        print(f"  → Query embedding shape: {query_embedding.shape}")
        
        # Search Qdrant
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_filter=document_filter
        )
        
        # Convert to RetrievalResult
        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                document_id=result.document_id,
                page_num=result.page_num,
                score=result.score,
                native_text=result.payload.get('native_text', ''),
                text_source=result.payload.get('text_source', 'unknown'),
                patch_grid=(
                    result.payload['patch_grid']['rows'],
                    result.payload['patch_grid']['cols']
                ),
                image_dims=(
                    result.payload['image_dims']['width'],
                    result.payload['image_dims']['height']
                )
            ))
        
        return retrieval_results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        document_filter: Optional[str] = None
    ) -> List[List[RetrievalResult]]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of text queries
            top_k: Number of results per query
            document_filter: Optional document ID filter
        
        Returns:
            List of result lists
        """
        all_results = []
        
        for query in queries:
            results = self.search(
                query=query,
                top_k=top_k,
                document_filter=document_filter
            )
            all_results.append(results)
        
        return all_results
