#!/usr/bin/env python
"""
CPU-side cropping diagnosis on scripts/dump_heatmaps.py output: for each heatmap variant,
how often the hottest patch lands inside the annotators' zone (pointing accuracy, vs chance =
the zone's share of the page), and zone F1 of the evidence boxes.

Usage:
    python scripts/analyze_cropping.py .cache/kaggle/mira-diag/heatmaps_hr.pkl [--images images_hr.pkl] [--sweep]

--images: {(doc_id, page): grayscale PIL page} to add ink-masked variants (blank patches ignored).
--sweep: grid over the region parameters for the best variant (use on the tuning subset only).
"""

import argparse
import pickle
import re
from statistics import mean

import numpy as np

from mira.evaluation import zone_f1
from mira.evidence import evidence_regions, to_pixels
from mira.evidence.localize import heatmap, ink_mask

# Query tokens that carry no content: ColQwen's prompt ("Query: ") and its padding/augmentation tokens
_NON_CONTENT = re.compile(r"^(<\|.*\|>|Ġ?Query|:|Ġ?[^\w]*)$")


def content_mask(tokens):
    return np.array([not _NON_CONTENT.match(t) for t in tokens])


def zone_mask(annotators, size, grid):
    """Per annotator: boolean [rows, cols] of patches whose centre lies inside that annotator's boxes."""
    (w, h), (rows, cols) = size, grid
    cy = (np.arange(rows) + 0.5) / rows * h
    cx = (np.arange(cols) + 0.5) / cols * w
    masks = []
    for boxes in annotators:
        m = np.zeros((rows, cols), bool)
        for x0, y0, x1, y1 in boxes:
            m |= (cy[:, None] >= y0) & (cy[:, None] < y1) & (cx[None, :] >= x0) & (cx[None, :] < x1)
        masks.append(m)
    return masks


def variants(q_emb, tokens, patches_flat, grid, image=None):
    rows, cols = grid
    keep = content_mask(tokens)
    flat = patches_flat.astype(np.float32)
    grid_patches = flat.reshape(rows, cols, -1)
    centred = flat - flat.mean(axis=0)  # remove what every patch shares (blank-page direction)
    out = {
        "current": heatmap(q_emb, grid_patches),
        "content_tokens": heatmap(q_emb[keep], grid_patches),
        "centred_patches": heatmap(q_emb[keep], centred.reshape(rows, cols, -1)),
        "transposed_grid": heatmap(q_emb, flat.reshape(cols, rows, -1).transpose(1, 0, 2)),
    }
    if image is not None:
        ink = ink_mask(image, grid)
        out["ink_mask"] = heatmap(q_emb, grid_patches, content=ink)
        out["ink_mask+content_tokens"] = heatmap(q_emb[keep], grid_patches, content=ink)
    return out


SWEEP = {"threshold": (0.2, 0.3, 0.4, 0.5), "max_regions": (1, 2, 3), "min_size": (0.15, 0.25, 0.35), "pad": (0.02, 0.05)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--images")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    data = pickle.load(open(args.dump, "rb"))
    pages = data["pages"]
    images = pickle.load(open(args.images, "rb")) if args.images else {}
    kept = []  # (heat of the last variant, annotators, size) for the sweep

    point, chance, f1 = {}, [], {}
    for q in data["queries"]:
        q_emb = q["embedding"].astype(np.float32)
        for key, annotators in q["boxes"].items():
            page = pages[key]
            masks = zone_mask(annotators, page["size"], page["grid"])
            chance.append(np.logical_or.reduce(masks).mean())  # a hit counts against any annotator
            for name, heat in variants(q_emb, q["tokens"], page["patches"], page["grid"], images.get(key)).items():
                r, c = np.unravel_index(heat.argmax(), heat.shape)
                point.setdefault(name, []).append(any(m[r, c] for m in masks))
                w, h = page["size"]
                boxes = [to_pixels(reg.box, w, h) for reg in evidence_regions(heat)]
                f1.setdefault(name, []).append(zone_f1(boxes, annotators, w, h))
            kept.append((heat, annotators, page["size"]))

    print(f"{args.dump}: {len(chance)} query-page pairs; chance pointing = {mean(chance):.3f}")
    print(f"{'variant':24s}{'pointing':>10s}{'zone F1':>9s}")
    for name in point:
        print(f"{name:24s}{mean(point[name]):10.3f}{mean(f1[name]):9.3f}")

    if args.sweep:
        import itertools
        print(f"\nRegion-parameter sweep on the last variant ({name}), top 8 by zone F1:")
        results = []
        for values in itertools.product(*SWEEP.values()):
            params = dict(zip(SWEEP, values))
            scores = []
            for heat, annotators, (w, h) in kept:
                boxes = [to_pixels(reg.box, w, h) for reg in evidence_regions(heat, **params)]
                scores.append(zone_f1(boxes, annotators, w, h))
            results.append((mean(scores), params))
        for score, params in sorted(results, key=lambda r: -r[0])[:8]:
            print(f"  {score:.3f}  {params}")


if __name__ == "__main__":
    main()
