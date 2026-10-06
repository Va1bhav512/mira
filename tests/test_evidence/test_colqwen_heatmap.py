"""Real ColQwen heatmap: a query lands on the page region that holds its words (GPU >= 10 GB)."""

import pytest
from PIL import Image, ImageDraw, ImageFont

from mira.evidence import evidence_regions, heatmap
from mira.retrieval import PageEmbedding


@pytest.mark.slow
def test_heatmap_finds_the_paragraph():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 10e9:
        pytest.skip("ColQwen tests need a GPU with >=10 GB VRAM")
    from mira.retrieval import ColQwenEmbedder

    page = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=28)
    draw.text((60, 80), "Annual Report: Company Overview", fill="black", font=font)
    draw.text((60, 160), "Our offices are open Monday to Friday.", fill="black", font=font)
    # Evidence in the bottom-right quarter
    draw.text((470, 850), "Battery capacity:", fill="black", font=font)
    draw.text((470, 900), "5000 mAh lithium", fill="black", font=font)

    embedder = ColQwenEmbedder()
    (emb, grid, start), = embedder.embed_images([page])
    patches = PageEmbedding("d", 0, emb, grid, start, page.size, "native").patch_embeddings
    regions = evidence_regions(heatmap(embedder.embed_query("What is the battery capacity?"), patches))

    x0, y0, x1, y1 = regions[0].box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert cx > 0.5 and cy > 0.6, regions
