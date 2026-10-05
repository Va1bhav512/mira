from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

from mira.pdf import ProcessedPage
from .colqwen import ColQwenEmbedder


@dataclass
class PageEmbedding:
    """Embedding data for a single page."""
    document_id: str
    page_num: int
    embeddings: np.ndarray  # Shape: [num_tokens, 128]
    patch_grid: Tuple[int, int]  # (rows, cols) of image tokens
    image_token_start: int  # embeddings[start:start+rows*cols] is the patch grid
    image_dims: Tuple[int, int]  # (width, height)
    text_source: str

    @property
    def patch_embeddings(self) -> np.ndarray:
        """Image-token embeddings shaped [rows, cols, 128], for heatmaps."""
        rows, cols = self.patch_grid
        start = self.image_token_start
        return self.embeddings[start:start + rows * cols].reshape(rows, cols, -1)


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
        cache_dir: Optional cache directory to avoid recomputation (per page)

    Returns:
        List of PageEmbedding objects, in the same order as pages
    """
    cache_path = Path(cache_dir) / document_id if cache_dir else None

    results = {}
    to_embed = []
    for page in pages:
        cached = _load_from_cache(cache_path, page.page_num) if cache_path else None
        if cached:
            results[page.page_num] = cached
        else:
            to_embed.append(page)

    if len(results):
        print(f"Loaded {len(results)} embeddings from cache")

    # One call for all uncached pages: embed_images batches internally, and its bf16
    # NaN fallback then reloads the model at most once per call instead of once per batch
    if to_embed:
        print(f"Embedding {len(to_embed)} pages")
        embedded = embedder.embed_images([p.image for p in to_embed], batch_size=batch_size)

        for (emb, patch_grid, image_token_start), page in zip(embedded, to_embed, strict=True):
            page_emb = PageEmbedding(
                document_id=document_id,
                page_num=page.page_num,
                embeddings=emb,
                patch_grid=patch_grid,
                image_token_start=image_token_start,
                image_dims=page.image.size,
                text_source=page.text_source
            )
            results[page.page_num] = page_emb
            if cache_path:
                _save_to_cache(cache_path, page_emb)

    return [results[p.page_num] for p in pages]


def _save_to_cache(cache_path: Path, emb: PageEmbedding):
    """Save one page's embedding to cache."""
    cache_path.mkdir(parents=True, exist_ok=True)
    np.save(cache_path / f"page_{emb.page_num:04d}.npy", emb.embeddings)
    with open(cache_path / f"page_{emb.page_num:04d}_meta.json", 'w') as f:
        json.dump({
            'document_id': emb.document_id,
            'page_num': emb.page_num,
            'patch_grid': emb.patch_grid,
            'image_token_start': emb.image_token_start,
            'image_dims': emb.image_dims,
            'text_source': emb.text_source,
        }, f)


def _load_from_cache(cache_path: Path, page_num: int) -> Optional[PageEmbedding]:
    """Load one page's embedding from cache if available."""
    emb_path = cache_path / f"page_{page_num:04d}.npy"
    meta_path = cache_path / f"page_{page_num:04d}_meta.json"
    if not emb_path.exists() or not meta_path.exists():
        return None

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Caches written before image_token_start existed have a guessed grid
    if 'image_token_start' not in meta:
        return None

    # Caches written before the bf16 NaN fallback can hold fp16-overflowed pages
    embeddings = np.load(emb_path)
    if np.isnan(embeddings).any():
        return None

    return PageEmbedding(
        document_id=meta['document_id'],
        page_num=meta['page_num'],
        embeddings=embeddings,
        patch_grid=tuple(meta['patch_grid']),
        image_token_start=meta['image_token_start'],
        image_dims=tuple(meta['image_dims']),
        text_source=meta['text_source']
    )
