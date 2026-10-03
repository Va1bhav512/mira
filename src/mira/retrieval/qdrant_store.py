from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    MultiVectorConfig,
    MultiVectorComparator,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from typing import List, Optional, Dict, Any
import os
import uuid
from dotenv import load_dotenv
import numpy as np
from dataclasses import dataclass

from .embeddings import PageEmbedding

load_dotenv()


@dataclass
class SearchResult:
    """Result from visual retrieval."""
    document_id: str
    page_num: int
    score: float
    payload: Dict[str, Any]


class QdrantMultivectorStore:
    """Manage multivector storage in Qdrant."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "mira_pages"
    ):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant cluster URL (default: from env)
            api_key: Qdrant API key (default: from env)
            collection_name: Name of collection to use
        """
        # Get from environment if not provided
        url = url or os.getenv('QDRANT_CLUSTER_ENDPOINT')
        api_key = api_key or os.getenv('QDRANT_CLUSTER_API_KEY')
        
        if not url:
            # Fallback to local Qdrant
            url = "http://localhost:6333"
            api_key = None
            print("Using local Qdrant at localhost:6333")
        
        # Initialize client
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
            print(f"Connected to Qdrant cloud: {url}")
        else:
            self.client = QdrantClient(url=url)
        
        self.collection_name = collection_name
    
    def create_collection(self, exist_ok: bool = True):
        """
        Create collection with multivector schema.
        
        Args:
            exist_ok: If True, don't error if collection exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]
            
            if self.collection_name in existing_names:
                if exist_ok:
                    print(f"Collection '{self.collection_name}' already exists")
                    return
                else:
                    self.client.delete_collection(self.collection_name)
            
            # Create collection with multivector support
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=128,  # ColQwen patch embedding dimension
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    )
                )
            )
            print(f"Created collection '{self.collection_name}'")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def upsert_page(self, page_embedding: PageEmbedding, native_text: str = ""):
        """
        Insert or update a page's embeddings.
        
        Args:
            page_embedding: PageEmbedding object
            native_text: Extracted text from the page
        """
        # Qdrant only accepts unsigned ints or UUIDs; uuid5 keeps re-indexing idempotent
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{page_embedding.document_id}/page/{page_embedding.page_num}"
        ))
        
        # Prepare payload
        payload = {
            "document_id": page_embedding.document_id,
            "page_num": page_embedding.page_num,
            "text_source": page_embedding.text_source,
            "patch_grid": {
                "rows": page_embedding.patch_grid[0],
                "cols": page_embedding.patch_grid[1]
            },
            "image_token_start": page_embedding.image_token_start,
            "image_dims": {
                "width": page_embedding.image_dims[0],
                "height": page_embedding.image_dims[1]
            },
            "native_text": native_text[:10000],  # Limit text size
        }
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=page_embedding.embeddings.tolist(),
            payload=payload
        )
        
        # Upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        MaxSim search against indexed pages.
        
        Args:
            query_embedding: Query embedding, shape [num_tokens, 128]
            top_k: Number of results to return
            document_filter: Optional document ID to filter results
        
        Returns:
            List of SearchResult objects
        """
        # Build filter if needed
        query_filter = None
        if document_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_filter)
                    )
                ]
            )
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        # Convert to SearchResult
        search_results = []
        for result in results.points:
            search_results.append(SearchResult(
                document_id=result.payload['document_id'],
                page_num=result.payload['page_num'],
                score=result.score,
                payload=result.payload
            ))
        
        return search_results
    
    def delete_document(self, document_id: str):
        """Delete all pages for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        print(f"Deleted document: {document_id}")
    
    def get_collection_info(self):
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        result = {
            "points_count": info.points_count,
            "status": info.status.value
        }
        # vectors_count may not exist in all Qdrant versions
        if hasattr(info, 'vectors_count'):
            result["vectors_count"] = info.vectors_count
        return result
    
    def list_documents(self) -> List[str]:
        """List all indexed documents."""
        # Scroll through collection to get unique document IDs
        from qdrant_client.models import ScrollResult
        
        document_ids = set()
        offset = None
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            for result in results:
                document_ids.add(result.payload['document_id'])
            
            if offset is None:
                break
        
        return list(document_ids)
