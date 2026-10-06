#!/usr/bin/env python
"""
Dump raw ColQwen embeddings for ViDoRe V3 gold pages that annotators boxed, so cropping
(Phase 5) can be diagnosed and tuned on CPU without re-running the GPU.

Output (pickle): {"pages": {(doc_id, page): {"patches": fp16 [rows*cols, 128], "grid": (rows, cols),
"size": (width, height)}}, "queries": [{"query_id", "query", "content_type", "embedding": fp16
[n_tokens, 128], "tokens": [str], "boxes": {(doc_id, page): [[(x0, y0, x1, y1), ...] per annotator]}}]}

Usage (GPU):
    python scripts/dump_heatmaps.py --vidore hr --limit 150 --out $OUT/heatmaps_hr.pkl
"""

import argparse
import pickle

import numpy as np
from datasets import load_dataset

from mira.evaluation.vidore import annotator_boxes, corpus_keys
from mira.retrieval import ColQwenEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vidore", required=True, help="ViDoRe V3 subset")
    parser.add_argument("--limit", type=int, default=150, help="First N English queries that have boxes")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ds = f"vidore/vidore_v3_{args.vidore}"
    corpus = load_dataset(ds, data_dir="corpus", split="test")
    light = corpus.remove_columns([c for c in corpus.column_names if c not in ("corpus_id", "doc_id", "page_number_in_doc")])
    keys = corpus_keys(light)
    row_of = {key: i for i, key in enumerate((r["doc_id"], r["page_number_in_doc"]) for r in light)}
    boxes = annotator_boxes(load_dataset(ds, data_dir="qrels", split="test"), keys)
    queries = [q for q in load_dataset(ds, data_dir="queries", split="test")
               if q["language"] == "english" and q["query_id"] in boxes][: args.limit]

    embedder = ColQwenEmbedder()
    needed = sorted({page for q in queries for page in boxes[q["query_id"]]})
    print(f"{len(queries)} queries, {len(needed)} pages to embed", flush=True)

    pages = {}
    for i in range(0, len(needed), 16):
        chunk = needed[i:i + 16]
        images = [corpus[row_of[key]]["image"].convert("RGB") for key in chunk]
        for key, image, (emb, grid, start) in zip(chunk, images, embedder.embed_images(images)):
            rows, cols = grid
            pages[key] = {"patches": emb[start:start + rows * cols].astype(np.float16), "grid": grid, "size": image.size}
        print(f"{min(i + 16, len(needed))}/{len(needed)} pages", flush=True)

    tokenizer = embedder.processor.tokenizer
    out = []
    for q in queries:
        inputs = embedder.processor.process_queries([q["query"]])
        ids = inputs["input_ids"][0][inputs["attention_mask"][0].bool()].tolist()
        emb = embedder.embed_query(q["query"])
        assert len(ids) == len(emb), "token ids and embeddings out of step"
        out.append({
            "query_id": q["query_id"], "query": q["query"], "content_type": q.get("content_type") or [],
            "embedding": emb.astype(np.float16), "tokens": tokenizer.convert_ids_to_tokens(ids),
            "boxes": boxes[q["query_id"]],
        })

    with open(args.out, "wb") as f:
        pickle.dump({"pages": pages, "queries": out}, f)
    print(f"Saved {args.out}: {len(out)} queries, {len(pages)} pages")


if __name__ == "__main__":
    main()
