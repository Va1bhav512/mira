"""
Query-Adaptive Evidence Cropping (Phase 5): where on a retrieved page is the evidence?

    S_i(r, c) = q_i . p(r, c)                      similarity map of query token i
    H(r, c)   = sum_i alpha_i * topk_i(S_i)(r, c)  keep each token's top-k patches only
    alpha_i   = max S_i - mean S_i                 peakedness: "the" or ColQwen's prompt and
                                                   padding tokens match everywhere about
                                                   equally and get ~0; an identifier spikes

then threshold H, group adjacent hot patches (connected components after a 1-patch
dilation), and return each group's padded bounding box. Boxes are normalized (x0, y0,
x1, y1) in [0, 1] page coordinates: ColQwen's processor resizes without padding, so the
patch grid spans the whole page and patch (r, c) covers [c/cols, (c+1)/cols] x [r/rows, ...].
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

Box = Tuple[float, float, float, float]  # normalized (x0, y0, x1, y1)

# THRESHOLD and MIN_SIZE tuned on ViDoRe V3 hr (zone F1 0.231 -> 0.495, with the ink mask), checked on
# held-out computer_science (0.206 -> 0.418; whole page 0.318). Report hr as the tuning subset.
TOP_K = 16           # patches kept per query token
THRESHOLD = 0.1      # fraction of the heatmap max that counts as "hot"
MAX_REGIONS = 2      # 3 adds +0.003 F1 for 50% more images to the VLM
MIN_MASS = 0.25      # drop regions with less than this fraction of the best region's heat
PAD = 0.02           # padding added on each side, as a fraction of the page
MIN_SIZE = 0.25      # smallest crop side, as a fraction of the page: the VLM needs context
INK_STD = 8.0        # pixel std (0-255) above which a patch counts as having ink


@dataclass
class Region:
    box: Box
    score: float  # summed heat in the region


def ink_mask(image: Image.Image, grid: Tuple[int, int], min_std: float = INK_STD) -> np.ndarray:
    """
    [rows, cols] True where the page has ink. Blank patches (margins, whitespace) can't hold
    evidence, yet ColQwen's embeddings for them match almost any query token ("sink" patches),
    so the heatmap must ignore them.
    """
    rows, cols = grid
    cell = 8
    pixels = np.asarray(image.convert("L").resize((cols * cell, rows * cell), Image.BILINEAR), dtype=np.float32)
    return pixels.reshape(rows, cell, cols, cell).std(axis=(1, 3)) > min_std


def heatmap(query: np.ndarray, patches: np.ndarray, top_k: int = TOP_K, content: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Args:
        query: Query token embeddings [n_tokens, dim]
        patches: Page patch embeddings [rows, cols, dim] (PageEmbedding.patch_embeddings)
        top_k: Patches kept per query token
        content: Optional [rows, cols] mask of patches that may hold evidence (ink_mask);
            the others get no heat and don't compete for a token's top-k

    Returns:
        H [rows, cols], scaled so its max is 1 (all zeros if nothing matched)
    """
    rows, cols, dim = patches.shape
    flat = patches.reshape(-1, dim)
    keep = content.reshape(-1) if content is not None and content.any() else np.ones(len(flat), bool)
    sims = query @ flat[keep].T  # [n_tokens, kept patches]
    alpha = sims.max(axis=1) - sims.mean(axis=1)

    k = min(top_k, sims.shape[1])
    kth = np.partition(sims, -k, axis=1)[:, -k][:, None]
    heat = np.zeros(len(flat))
    heat[keep] = (alpha[:, None] * np.where(sims >= kth, sims.clip(min=0), 0.0)).sum(axis=0)
    peak = heat.max()
    return (heat / peak if peak > 0 else heat).reshape(rows, cols)


def _patch_box(rows_idx: np.ndarray, cols_idx: np.ndarray, grid: Tuple[int, int], pad: float, min_size: float) -> Box:
    """Normalized box around the given patches, padded and grown to min_size, clipped to the page."""
    rows, cols = grid
    x0, x1 = cols_idx.min() / cols - pad, (cols_idx.max() + 1) / cols + pad
    y0, y1 = rows_idx.min() / rows - pad, (rows_idx.max() + 1) / rows + pad

    def grow(lo, hi):
        if hi - lo < min_size:
            mid = (lo + hi) / 2
            lo, hi = mid - min_size / 2, mid + min_size / 2
        # Shift back inside the page rather than shrink
        shift = max(0.0, -lo) - max(0.0, hi - 1.0)
        return max(0.0, lo + shift), min(1.0, hi + shift)

    x0, x1 = grow(x0, x1)
    y0, y1 = grow(y0, y1)
    return (float(x0), float(y0), float(x1), float(y1))


def evidence_regions(
    heat: np.ndarray,
    threshold: float = THRESHOLD,
    max_regions: int = MAX_REGIONS,
    min_mass: float = MIN_MASS,
    pad: float = PAD,
    min_size: float = MIN_SIZE,
) -> List[Region]:
    """Hot connected regions of a heatmap, best first (at least one if the heatmap is non-zero)."""
    hot = heat >= threshold * heat.max() if heat.max() > 0 else np.zeros_like(heat, bool)
    # Dilate so patches one apart (a table row split by whitespace) join one group
    labels, n = ndimage.label(ndimage.binary_dilation(hot), structure=np.ones((3, 3)))
    labels[~hot] = 0  # boxes cover hot patches only, not the dilation halo

    regions = []
    for label in range(1, n + 1):
        r, c = np.nonzero(labels == label)
        regions.append(Region(_patch_box(r, c, heat.shape, pad, min_size), float(heat[r, c].sum())))
    regions.sort(key=lambda reg: reg.score, reverse=True)
    return [reg for reg in regions[:max_regions] if reg.score >= min_mass * regions[0].score]


def max_patch_region(heat: np.ndarray, pad: float = PAD, min_size: float = MIN_SIZE) -> Region:
    """Baseline: a box around the single hottest patch (what the heatmap algorithm improves on)."""
    r, c = np.unravel_index(heat.argmax(), heat.shape)
    return Region(_patch_box(np.array([r]), np.array([c]), heat.shape, pad, min_size), float(heat[r, c]))


def to_pixels(box: Box, width: int, height: int) -> Tuple[int, int, int, int]:
    """Normalized box -> integer pixel (x0, y0, x1, y1) on a width x height image."""
    x0, y0, x1, y1 = box
    return round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)
