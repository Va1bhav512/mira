#!/usr/bin/env python
"""
Report figures.

  crop  : before/after heatmaps on V3 pages (needs dump_heatmaps.py output, plus page images: a
          pickle {(doc_id, page): PIL image}, as fetched for analyze_cropping.py --images)
  fusion: nDCG vs the visual share of fixed RRF weights, per dataset (needs rescore_fusion.py
          output saved as JSON: {"dataset": [[share, score], ...]}); uses matplotlib

Usage:
    python scripts/make_figures.py crop .cache/kaggle/mira-diag/heatmaps_hr.pkl images_hr.pkl docs/figures 2 4
    uv run --with matplotlib python scripts/make_figures.py fusion curves.json docs/figures/fusion_weights.png
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mira.evidence import evidence_regions
from mira.evidence.localize import heatmap, ink_mask


def panel(page, heat, gold, regions, title):
    w, h = page.size
    alpha = Image.fromarray((heat * 170).astype(np.uint8)).resize((w, h), Image.NEAREST)
    red = Image.new("RGBA", (w, h), (230, 20, 20, 0))
    red.putalpha(alpha)
    out = Image.alpha_composite(page.convert("RGBA"), red)
    draw = ImageDraw.Draw(out)
    for annotator in gold:
        for box in annotator:
            draw.rectangle(box, outline=(0, 160, 60, 255), width=8)
    for reg in regions:
        x0, y0, x1, y1 = reg.box
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(20, 60, 230, 255), width=8)
    out = out.convert("RGB").resize((w // 3, h // 3))
    canvas = Image.new("RGB", (out.width, out.height + 40), "white")
    canvas.paste(out, (0, 40))
    ImageDraw.Draw(canvas).text((10, 8), title, fill="black", font=ImageFont.load_default(size=22))
    return canvas


def crop_figures(dump, images_path, out_dir, *indices):
    data = pickle.load(open(dump, "rb"))
    images = pickle.load(open(images_path, "rb"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in map(int, indices):
        q = data["queries"][i]
        key = next(iter(q["boxes"]))
        page_data, gold = data["pages"][key], q["boxes"][key]
        rows, cols = page_data["grid"]
        patches = page_data["patches"].astype(np.float32).reshape(rows, cols, -1)
        emb = q["embedding"].astype(np.float32)
        gray = images[key]
        page = gray.convert("RGB").resize(page_data["size"])
        before = heatmap(emb, patches)
        after = heatmap(emb, patches, content=ink_mask(gray, (rows, cols)))
        a = panel(page, before, gold, evidence_regions(before, threshold=0.3, min_size=0.15), "Before: all patches")
        b = panel(page, after, gold, evidence_regions(after), "After: ink mask (blank patches ignored)")
        fig = Image.new("RGB", (a.width + b.width + 20, a.height + 50), "white")
        fig.paste(a, (0, 50))
        fig.paste(b, (a.width + 20, 50))
        ImageDraw.Draw(fig).text((10, 10), f"Q: {q['query'][:110]}", fill="black", font=ImageFont.load_default(size=20))
        path = out_dir / f"crop_before_after_{i}.png"
        fig.save(path)
        print(f"Saved {path}  (red: heatmap, green: annotators, blue: Mira's evidence boxes)")


def fusion_figure(curves_json, out_png):
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = json.load(open(curves_json))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, points in curves.items():
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("visual share of RRF weight (0 = BM25 only, 1 = ColQwen only)")
    ax.set_ylabel("nDCG")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    {"crop": crop_figures, "fusion": fusion_figure}[sys.argv[1]](*sys.argv[2:])
