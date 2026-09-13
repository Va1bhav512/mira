from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from mira.pdf import ProcessedPage
from .colqwen import ColQwenEmbedder


@dataclass
class PageEmbedding:
    """Embedding data for a single page."""
    document_id: str
    page_num: int
    embeddings: np.ndarray  # Shape: [num_patches, 128]
    patch_grid: Tuple[int, int]  # (rows, cols)
    image_dims: Tuple[int, int]  # (width, height)
    text_source: str


def generate_embeddings(
    pages: List[ProcessedPage],
    embedder: ColQwenEmbedder,
    batch_size: int = 4,
    document_id: str = "unknown",
    cache_dir: Optional[str] = None
) -> List[PageEmbedding]:
    """
    Generate embeddings for processed pages.
    
    Args:
        pages: List of ProcessedPage from Phase 1
        embedder: ColQwen model wrapper
        batch_size: Pages to process together (GPU memory limit)
        document_id: Document identifier
        cache_dir: Optional cache directory to avoid recomputation
    
    Returns:
        List of PageEmbedding objects
    """
    if not pages:
        return []
    
    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / document_id
        cache_path.mkdir(parents=True, exist_ok=True)
        
        cached = _load_from_cache(cache_path, len(pages))
        if cached:
            print(f"Loaded {len(cached)} embeddings from cache")
            return cached
    
    # Generate embeddings
    images = [page.image for page in pages]
    
    print(f"Generating embeddings for {len(pages)} pages...")
    
    # Process in batches
    all_embeddings = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Embedding pages"):
        batch_images = images[i:i + batch_size]
        batch_pages = pages[i:i + batch_size]
        
        # Embed batch
        batch_emb = embedder.embed_images(batch_images, batch_size=len(batch_images))
        
        # Create PageEmbedding objects
        for j, (emb, page) in enumerate(zip(batch_emb, batch_pages)):
            # Estimate patch grid from embedding shape
            num_patches = emb.shape[0]
            patch_grid = _estimate_patch_grid(num_patches, page.image.size)
            
            all_embeddings.append(PageEmbedding(
                document_id=document_id,
                page_num=page.page_num,
                embeddings=emb,
                patch_grid=patch_grid,
                image_dims=page.image.size,
                text_source=page.text_source
            ))
    
    # Save to cache
    if cache_dir:
        _save_to_cache(cache_path, all_embeddings)
    
    return all_embeddings


def _estimate_patch_grid(num_patches: int, image_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Estimate patch grid dimensions from embedding count and image size.
    
    Args:
        num_patches: Number of patch embeddings
        image_size: (width, height) of image
    
    Returns:
        (rows, cols) grid dimensions
    """
    width, height = image_size
    
    #_aspect ratio
    aspect = width / height
    
    # Estimate grid (ColQwen uses 14x14 pixel patches typically)
    # Total patches = rows * cols
    # rows = cols / aspect
    
    import math
    cols = int(math.sqrt(num_patches * aspect))
    rows = int(num_patches / cols)
    
    return (rows, cols)


def _save_to_cache(cache_path: Path, embeddings: List[PageEmbedding]):
    """Save embeddings to cache."""
    for emb in embeddings:
        # Save embeddings as numpy
        emb_path = cache_path / f"page_{emb.page_num:04d}.npy"
        np.save(emb_path, emb.embeddings)
        
        # Save metadata as JSON
        meta_path = cache_path / f"page_{emb.page_num:04d}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'document_id': emb.document_id,
                'page_num': emb.page_num,
                'patch_grid': emb.patch_grid,
                'image_dims': emb.image_dims,
                'text_source': emb.text_source,
                'embedding_shape': emb.embeddings.shape
            }, f)
    
    print(f"Cached {len(embeddings)} embeddings to {cache_path}")


def _load_from_cache(cache_path: Path, expected_pages: int) -> Optional[List[PageEmbedding]]:
    """Load embeddings from cache if available."""
    embeddings = []
    
    for page_num in range(expected_pages):
        emb_path = cache_path / f"page_{page_num:04d}.npy"
        meta_path = cache_path / f"page_{page_num:04d}_meta.json"
        
        if not emb_path.exists() or not meta_path.exists():
            return None
        
        # Load embedding
        emb_data = np.load(emb_path)
        
        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        embeddings.append(PageEmbedding(
            document_id=meta['document_id'],
            page_num=meta['page_num'],
            embeddings=emb_data,
            patch_grid=tuple(meta['patch_grid']),
            image_dims=tuple(meta['image_dims']),
            text_source=meta['text_source']
        ))
    
    return embeddings
