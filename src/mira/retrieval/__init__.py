"""
Visual Retrieval Module for Mira.

Provides ColQwen visual embeddings with Qdrant multivector storage,
and BM25 lexical retrieval over page text.
"""

from .colqwen import ColQwenEmbedder
from .embeddings import PageEmbedding, generate_embeddings
from .qdrant_store import QdrantMultivectorStore, SearchResult
from .ingest import DocumentIndexer, IndexingResult
from .search import VisualRetriever, RetrievalResult
from .lexical import BM25Index, tokenize

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
    'BM25Index',
    'tokenize',
]
