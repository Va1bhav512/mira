#!/usr/bin/env python
"""
Compare retrieval modes (adaptive / fixed / visual / lexical) on labelled queries.

Custom corpus (data/samples, indexed with scripts/index_corpus.py):
    uv run python scripts/eval_retrieval.py                    # all modes (needs GPU)
    uv run python scripts/eval_retrieval.py --modes lexical    # CPU only

ViDoRe V3 subset (indexed with scripts/index_vidore.py):
    uv run python scripts/eval_retrieval.py --vidore computer_science
    uv run python scripts/eval_retrieval.py --vidore hr --qdrant-url http://localhost:6333

Custom query file: one JSON object per line,
    {"query": "...", "document_id": "DS_stm32f401re", "pages": [12], "type": "identifier"}
pages are 0-indexed; type is identifier | visual | semantic.

ViDoRe mode uses English queries, graded nDCG (qrels scores 2=Fully, 1=Critically
Relevant; recall/MRR count any grade >= 1) and prints a paired bootstrap of
adaptive vs fixed nDCG@5 - the significance test the custom set is too small for.
"""

import argparse
import json
from collections import defaultdict
from statistics import mean

from mira.evaluation import (
    ndcg_at_k,
    ndcg_at_k_graded,
    paired_bootstrap,
    recall_at_k,
    reciprocal_rank,
)
from mira.retrieval import BM25Index, QdrantMultivectorStore
from mira.retrieval.hybrid import MODES, HybridRetriever

METRICS = {
    "R@1": lambda r, rel: recall_at_k(r, rel, 1),
    "R@5": lambda r, rel: recall_at_k(r, rel, 5),
    "R@10": lambda r, rel: recall_at_k(r, rel, 10),
    "MRR": reciprocal_rank,
    "nDCG@10": lambda r, rel: ndcg_at_k(r, rel, 10),
}

# ViDoRe's headline is graded nDCG@5; with ~9 relevant pages per query, R@1 is not meaningful
VIDORE_METRICS = {
    "R@5": lambda r, grades: recall_at_k(r, {p for p, g in grades.items() if g >= 1}, 5),
    "R@10": lambda r, grades: recall_at_k(r, {p for p, g in grades.items() if g >= 1}, 10),
    "MRR": lambda r, grades: reciprocal_rank(r, {p for p, g in grades.items() if g >= 1}),
    "nDCG@5": lambda r, grades: ndcg_at_k_graded(r, grades, 5),
    "nDCG@10": lambda r, grades: ndcg_at_k_graded(r, grades, 10),
}


def print_table(modes, bucket_name, n, scores, metric_names):
    if n == 0:
        return
    print(f"\n{bucket_name} ({n} queries)")
    print(f"{'mode':10s}" + "".join(f"{m:>9s}" for m in metric_names))
    for mode in modes:
        values = scores[mode]
        print(f"{mode:10s}" + "".join(f"{mean(values[m]):9.3f}" if values[m] else f"{'-':>9s}" for m in metric_names))


def report_bootstrap(per_query_ndcg, queries_n):
    diff, lo, hi, p = paired_bootstrap(per_query_ndcg["adaptive"], per_query_ndcg["fixed"])
    print(f"\nadaptive - fixed nDCG: mean {diff:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], "
          f"paired bootstrap p={p:.4f} (n={queries_n})")


def run_custom(args):
    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]

    retriever = HybridRetriever(
        store=QdrantMultivectorStore(url=args.qdrant_url),
        text_index=BM25Index(args.bm25_path),
    )

    # scores[mode][query type][metric] -> per-query values
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_query_ndcg = defaultdict(list)
    for q in queries:
        relevant = {(q["document_id"], p) for p in q["pages"]}
        for mode in args.modes:
            ranking = [(r.document_id, r.page_num) for r in retriever.search(q["query"], top_k=10, mode=mode)]
            for name, metric in METRICS.items():
                value = metric(ranking, relevant)
                scores[mode][q["type"]][name].append(value)
                scores[mode]["ALL"][name].append(value)
            per_query_ndcg[mode].append(ndcg_at_k(ranking, relevant, 10))

    types = sorted({q["type"] for q in queries}) + ["ALL"]
    for qtype in types:
        n = len(scores[args.modes[0]][qtype]["MRR"])
        print_table(args.modes, qtype, n, {m: scores[m][qtype] for m in args.modes}, METRICS)
    if {"adaptive", "fixed"} <= set(args.modes):
        report_bootstrap(per_query_ndcg, len(queries))


def run_vidore(args):
    from datasets import load_dataset

    from mira.evaluation.vidore import build_relevance, corpus_keys

    subset = args.vidore
    collection = args.collection or f"vidore_v3_{subset}"
    bm25_path = args.bm25_path_vidore or f".cache/bm25_vidore/{subset}"

    print(f"Loading vidore/vidore_v3_{subset} queries and qrels...")
    queries_ds = load_dataset(f"vidore/vidore_v3_{subset}", data_dir="queries", split="test")
    qrels_ds = load_dataset(f"vidore/vidore_v3_{subset}", data_dir="qrels", split="test")
    corpus_ds = load_dataset(f"vidore/vidore_v3_{subset}", data_dir="corpus", split="test")

    # Drop the image column so no page is decoded just to read the id mapping
    light = corpus_ds.remove_columns([c for c in corpus_ds.column_names if c not in ("corpus_id", "doc_id", "page_number_in_doc")])
    grades_by_query = build_relevance(qrels_ds, corpus_keys(light))

    queries = [q for q in queries_ds if q["language"] == "english"]
    print(f"{len(queries)} English queries over {collection} ({len(light)} pages)")

    retriever = HybridRetriever(
        store=QdrantMultivectorStore(url=args.qdrant_url, collection_name=collection),
        text_index=BM25Index(bm25_path),
    )

    # scores[bucket name][mode][metric] -> per-query values
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    bucket_counts = defaultdict(int)
    per_query_ndcg = defaultdict(list)
    for q in queries:
        grades = grades_by_query.get(q["query_id"], {})
        if not grades:
            continue
        # content_type is a list (["Chart", "Table"]): the query counts under each
        content_types = q.get("content_type") or []
        if isinstance(content_types, str):
            content_types = [content_types]
        buckets = {"ALL", f"format:{q['query_format']}"} | {f"content:{c}" for c in content_types}
        for b in buckets:
            bucket_counts[b] += 1
        for mode in args.modes:
            ranking = [(r.document_id, r.page_num) for r in retriever.search(q["query"], top_k=10, mode=mode)]
            for name, metric in VIDORE_METRICS.items():
                value = metric(ranking, grades)
                for b in buckets:
                    scores[b][mode][name].append(value)
            per_query_ndcg[mode].append(ndcg_at_k_graded(ranking, grades, 5))

    def bucket_table(bucket):
        print_table(args.modes, bucket, bucket_counts[bucket], {m: scores[bucket][m] for m in args.modes}, VIDORE_METRICS)

    bucket_table("ALL")
    for b in sorted(bucket_counts):
        if b != "ALL":
            bucket_table(b)
    if {"adaptive", "fixed"} <= set(args.modes):
        report_bootstrap(per_query_ndcg, len(per_query_ndcg["adaptive"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--qdrant-url", help="Qdrant server URL (default: cloud cluster from .env)")
    parser.add_argument("--bm25-path", default=".cache/bm25", help="BM25 dir for the custom corpus")
    parser.add_argument("--vidore", help="ViDoRe V3 subset to evaluate instead of the custom corpus")
    parser.add_argument("--collection", help="Qdrant collection for --vidore (default: vidore_v3_<subset>)")
    parser.add_argument("--bm25-path-vidore", help="BM25 dir for --vidore (default: .cache/bm25_vidore/<subset>)")
    args = parser.parse_args()

    if args.vidore:
        run_vidore(args)
    else:
        run_custom(args)


if __name__ == "__main__":
    main()
