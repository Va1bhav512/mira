#!/usr/bin/env python
"""
Compare retrieval modes (adaptive / fixed / visual / lexical) on labelled queries.

Usage:
    uv run python scripts/eval_retrieval.py                    # all modes (needs GPU + indexed corpus)
    uv run python scripts/eval_retrieval.py --modes lexical    # CPU only
    uv run python scripts/eval_retrieval.py --qdrant-path /content/cache/qdrant --bm25-path /content/cache/bm25

Query file: one JSON object per line,
    {"query": "...", "document_id": "DS_stm32f401re", "pages": [12], "type": "identifier"}
pages are 0-indexed; type is identifier | visual | semantic.
"""

import argparse
import json
from collections import defaultdict
from statistics import mean

from mira.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from mira.retrieval import BM25Index, QdrantMultivectorStore
from mira.retrieval.hybrid import MODES, HybridRetriever

METRICS = {
    "R@1": lambda r, rel: recall_at_k(r, rel, 1),
    "R@5": lambda r, rel: recall_at_k(r, rel, 5),
    "R@10": lambda r, rel: recall_at_k(r, rel, 10),
    "MRR": reciprocal_rank,
    "nDCG@10": lambda r, rel: ndcg_at_k(r, rel, 10),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--qdrant-path", help="On-disk embedded Qdrant (default: cloud from .env)")
    parser.add_argument("--bm25-path", default=".cache/bm25")
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]

    retriever = HybridRetriever(
        store=QdrantMultivectorStore(path=args.qdrant_path),
        text_index=BM25Index(args.bm25_path),
    )

    # scores[mode][query type][metric] -> per-query values
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for q in queries:
        relevant = {(q["document_id"], p) for p in q["pages"]}
        for mode in args.modes:
            ranking = [(r.document_id, r.page_num) for r in retriever.search(q["query"], top_k=10, mode=mode)]
            for name, metric in METRICS.items():
                value = metric(ranking, relevant)
                scores[mode][q["type"]][name].append(value)
                scores[mode]["ALL"][name].append(value)

    types = sorted({q["type"] for q in queries}) + ["ALL"]
    for qtype in types:
        n = len(scores[args.modes[0]][qtype]["MRR"])
        print(f"\n{qtype} ({n} queries)")
        print(f"{'mode':10s}" + "".join(f"{m:>9s}" for m in METRICS))
        for mode in args.modes:
            print(f"{mode:10s}" + "".join(f"{mean(scores[mode][qtype][m]):9.3f}" for m in METRICS))


if __name__ == "__main__":
    main()
