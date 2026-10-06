#!/usr/bin/env python
"""
Re-score fusion on CPU from per-query channel rankings (eval_retrieval.py --dump-rankings):
fixed visual:lexical weight ratios from all-BM25 to all-visual, and adaptive fusion at several
shift sizes. No GPU, no index.

Usage:
    python scripts/rescore_fusion.py .cache/kaggle/mira-full/rankings_custom.jsonl --queries data/eval/queries.jsonl
    python scripts/rescore_fusion.py .cache/kaggle/mira-full/rankings_hr.jsonl --vidore hr
"""

import argparse
import json
from statistics import mean

from mira.evaluation import ndcg_at_k, ndcg_at_k_graded
from mira.retrieval import hybrid


def fuse(row, w_visual, w_lexical, top_k=10):
    scores = hybrid.rrf([[tuple(p) for p in row["visual"]], [tuple(p) for p in row["lexical"]]], [w_visual, w_lexical])
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rankings")
    parser.add_argument("--queries", help="Custom queries.jsonl (binary relevance)")
    parser.add_argument("--vidore", help="ViDoRe V3 subset (graded relevance from its qrels)")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.rankings)]
    if args.vidore:
        from datasets import load_dataset

        from mira.evaluation.vidore import build_relevance, corpus_keys
        ds = f"vidore/vidore_v3_{args.vidore}"
        corpus = load_dataset(ds, data_dir="corpus", split="test")
        light = corpus.remove_columns([c for c in corpus.column_names if c not in ("corpus_id", "doc_id", "page_number_in_doc")])
        grades = build_relevance(load_dataset(ds, data_dir="qrels", split="test"), corpus_keys(light))
        by_text = {q["query"]: grades.get(q["query_id"], {}) for q in load_dataset(ds, data_dir="queries", split="test")
                   if q["language"] == "english"}
        rel = [by_text[r["query"]] for r in rows]
        metric, name = (lambda ranking, g: ndcg_at_k_graded(ranking, g, 5)), "nDCG@5"
    else:
        labels = {q["query"]: {(q["document_id"], p) for p in q["pages"]} for q in map(json.loads, open(args.queries))}
        rel = [labels[r["query"]] for r in rows]
        metric, name = (lambda ranking, g: ndcg_at_k(ranking, g, 10)), "nDCG@10"

    def score(weights_of):
        return mean(metric(fuse(r, *weights_of(r["query"])), g) for r, g in zip(rows, rel) if g)

    print(f"{args.rankings}: {len(rows)} queries, {name}")
    print("fixed visual share (0 = BM25 only, 1 = visual only):")
    for share in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        print(f"  {share:4.2f}  {score(lambda q: (share * 2, (1 - share) * 2)):.3f}")
    print("adaptive, by shift size:")
    original = hybrid.WEIGHT_SHIFT
    for shift in (0.25, 0.5, 0.75):
        hybrid.WEIGHT_SHIFT = shift
        print(f"  {shift:4.2f}  {score(lambda q: (hybrid.query_weights(q).visual, hybrid.query_weights(q).lexical)):.3f}")
    hybrid.WEIGHT_SHIFT = original


if __name__ == "__main__":
    main()
