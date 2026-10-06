"""Tests for Query-Adaptive Evidence Cropping on synthetic embeddings."""

import numpy as np
import pymupdf
import pytest

from mira.evaluation import zone_f1
from mira.evidence import crop_image, evidence_regions, heatmap, max_patch_region, render_region, to_pixels

DIM = 16


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def page_with_blocks(rows=20, cols=10, seed=0):
    """Random background patches; a 'table' block at rows 12-15, cols 2-6 matching token A."""
    rng = np.random.default_rng(seed)
    patches = unit(rng.standard_normal((rows, cols, DIM)))
    a = unit(rng.standard_normal(DIM))
    patches[12:16, 2:7] = unit(a + 0.1 * rng.standard_normal((4, 5, DIM)))
    return patches, a


def test_heatmap_peaks_on_matching_block():
    patches, a = page_with_blocks()
    heat = heatmap(a[None], patches)
    assert heat.max() == pytest.approx(1.0)
    r, c = np.unravel_index(heat.argmax(), heat.shape)
    assert 12 <= r < 16 and 2 <= c < 7


def test_uniform_token_gets_no_weight():
    """A token that matches every patch equally ('the', padding) must not move the heatmap."""
    patches, a = page_with_blocks()
    # Extra dimension every patch shares equally; a token along it scores the same everywhere
    patches = unit(np.concatenate([patches, np.ones(patches.shape[:2] + (1,))], axis=-1))
    query = np.stack([np.append(a, 0.0), np.eye(DIM + 1)[-1]])
    assert np.allclose(heatmap(query, patches), heatmap(query[:1], patches))


def test_region_covers_block_tightly():
    patches, a = page_with_blocks()
    heat = heatmap(a[None], patches, top_k=20)
    regions = evidence_regions(heat, pad=0.0, min_size=0.0)
    x0, y0, x1, y1 = regions[0].box
    # block spans cols 2-6 of 10, rows 12-15 of 20
    assert x0 == pytest.approx(0.2) and x1 == pytest.approx(0.7)
    assert y0 == pytest.approx(0.6) and y1 == pytest.approx(0.8)


def test_heatmap_region_beats_single_max_patch_on_zone_f1():
    patches, a = page_with_blocks()
    heat = heatmap(a[None], patches, top_k=20)
    gold = [[to_pixels((0.2, 0.6, 0.7, 0.8), 1000, 2000)]]
    region = [to_pixels(r.box, 1000, 2000) for r in evidence_regions(heat)]
    single = [to_pixels(max_patch_region(heat).box, 1000, 2000)]
    assert zone_f1(region, gold, 1000, 2000) > zone_f1(single, gold, 1000, 2000)


def test_regions_split_and_rank_by_mass():
    heat = np.zeros((10, 10))
    heat[0:3, 0:3] = 1.0   # big block
    heat[8, 8] = 0.9       # small far block
    regions = evidence_regions(heat, pad=0.0, min_size=0.0, min_mass=0.0)
    assert len(regions) == 2
    assert regions[0].box == (0.0, 0.0, 0.3, 0.3)
    assert regions[1].box == pytest.approx((0.8, 0.8, 0.9, 0.9))
    # the small one is below min_mass of the big one by default
    assert len(evidence_regions(heat, pad=0.0, min_size=0.0)) == 1


def test_gap_of_one_patch_joins_one_region():
    heat = np.zeros((10, 10))
    heat[2, 1:4] = 1.0
    heat[4, 1:4] = 1.0  # one empty row between: same table
    assert len(evidence_regions(heat, min_mass=0.0)) == 1


def test_min_size_grows_box_inside_page():
    heat = np.zeros((10, 10))
    heat[0, 9] = 1.0  # top-right corner patch
    (region,) = evidence_regions(heat, pad=0.0, min_size=0.3)
    x0, y0, x1, y1 = region.box
    assert x1 == pytest.approx(1.0) and x1 - x0 == pytest.approx(0.3)
    assert y0 == pytest.approx(0.0) and y1 - y0 == pytest.approx(0.3)


def test_zero_heatmap_has_no_regions():
    assert evidence_regions(np.zeros((4, 4))) == []


def test_crop_image_and_render_region(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (200, 400))
    assert crop_image(img, (0.25, 0.5, 0.75, 1.0)).size == (100, 200)

    pdf = tmp_path / "p.pdf"
    doc = pymupdf.open()
    doc.new_page(width=200, height=400)  # points: 72 DPI
    doc.save(pdf)
    crop = render_region(str(pdf), 0, (0.0, 0.0, 0.5, 0.5), dpi=144)
    assert crop.size == (200, 400)  # half the page at 2x


def test_zone_f1():
    assert zone_f1([(0, 0, 10, 10)], [[(0, 0, 10, 10)]], 20, 20) == 1.0
    assert zone_f1([(0, 0, 10, 10)], [[(10, 10, 20, 20)]], 20, 20) == 0.0
    # half overlap: |P&G|=50, |P|=|G|=100 -> 0.5; best of two annotators wins
    assert zone_f1([(0, 0, 10, 10)], [[(10, 10, 20, 20)], [(5, 0, 15, 10)]], 20, 20) == 0.5
    # an annotator's boxes merge into one zone (overlap counted once)
    assert zone_f1([(0, 0, 10, 10)], [[(0, 0, 10, 10), (0, 0, 5, 5)]], 20, 20) == 1.0
