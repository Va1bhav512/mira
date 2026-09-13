"""
Visual Retrieval Module for Mira.

Provides ColQwen visual embeddings and Qdrant multivector storage.
"""

from .colqwen import ColQwenEmbedder
from .embeddings import PageEmbedding, generate_embeddings
from .qdrant_store import QdrantMultivectorStore, SearchResult
from .ingest import DocumentIndexer, IndexingResult
from .search import VisualRetriever, RetrievalResult

__all__ = [
    'ColQwenEmbedder',
    'PageEmbedding',
    'generate_embeddings',
    'QdrantMultivectorStore',
    'SearchResult',
    'DocumentIndexer',
    'IndexingResult',
    'VisualRetriever',
    'RetrievalResult',
]
