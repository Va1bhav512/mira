#!/usr/bin/env python
"""
Phase 5: score evidence localization against ViDoRe V3 human bounding boxes.

For each English query and each relevant page that annotators boxed (oracle
pages, so this isolates cropping from retrieval), compare the evidence boxes
each strategy produces with the annotators' zones: pixel-level F1, best
annotator (the V3 paper's protocol; inter-annotator ceiling 0.602).

Strategies: page (whole page, the no-crop baseline), max_patch (box around
the single hottest patch), heatmap (Query-Adaptive Evidence Cropping).

Usage (subset indexed with scripts/index_vidore.py; needs ColQwen for queries):
    uv run python scripts/eval_cropping.py --vidore hr --qdrant-url http://localhost:6333
"""

import argparse
from collections import defaultdict
from statistics import mean

from datasets import load_dataset

from mira.evaluation import zone_f1
from mira.evaluation.vidore import annotator_boxes, corpus_keys
from mira.evidence import to_pixels
from mira.generation import STRATEGIES, page_boxes
from mira.retrieval import ColQwenEmbedder, QdrantMultivectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vidore", required=True, help="ViDoRe V3 subset")
    parser.add_argument("--qdrant-url", help="Qdrant server URL (default: cloud cluster from .env)")
    parser.add_argument("--collection", help="Qdrant collection (default: vidore_v3_<subset>)")
    parser.add_argument("--limit", type=int, help="Only the first N queries (smoke runs)")
    args = parser.parse_args()

    subset = args.vidore
    ds = f"vidore/vidore_v3_{subset}"
    queries = [q for q in load_dataset(ds, data_dir="queries", split="test") if q["language"] == "english"]
    qrels = load_dataset(ds, data_dir="qrels", split="test")
    corpus = load_dataset(ds, data_dir="corpus", split="test")
    light = corpus.remove_columns([c for c in corpus.column_names if c not in ("corpus_id", "doc_id", "page_number_in_doc")])
    boxes_by_query = annotator_boxes(qrels, corpus_keys(light))
    queries = [q for q in queries if q["query_id"] in boxes_by_query][: args.limit]

    store = QdrantMultivectorStore(url=args.qdrant_url, collection_name=args.collection or f"vidore_v3_{subset}")
    embedder = ColQwenEmbedder()

    # scores[bucket][strategy] -> per (query, page) F1; content type comes from the query
    scores = defaultdict(lambda: defaultdict(list))
    for i, q in enumerate(queries, start=1):
        query_embedding = embedder.embed_query(q["query"])
        content_types = q.get("content_type") or []
        buckets = ["ALL"] + [f"content:{c}" for c in content_types]
        for (doc_id, page_num), annotators in boxes_by_query[q["query_id"]].items():
            page = store.get_page(doc_id, page_num)
            width, height = page.image_dims
            for strategy in STRATEGIES:
                pred = [to_pixels(b, width, height) for b in page_boxes(query_embedding, page, strategy)]
                f1 = zone_f1(pred, annotators, width, height)
                for b in buckets:
                    scores[b][strategy].append(f1)
        if i % 25 == 0:
            print(f"{i}/{len(queries)} queries", flush=True)

    print(f"\nZone F1 vs V3 annotators ({subset}, oracle pages; human ceiling 0.602)")
    print(f"{'bucket':24s}{'pages':>7s}" + "".join(f"{s:>11s}" for s in STRATEGIES))
    for bucket in ["ALL"] + sorted(b for b in scores if b != "ALL"):
        n = len(scores[bucket][STRATEGIES[0]])
        print(f"{bucket:24s}{n:7d}" + "".join(f"{mean(scores[bucket][s]):11.3f}" for s in STRATEGIES))


if __name__ == "__main__":
    main()
