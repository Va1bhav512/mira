#!/usr/bin/env python
"""
Phase 6: answer ViDoRe V3 queries end to end (hybrid retrieval -> evidence -> Qwen2.5-VL)
under each context strategy, and save answers beside V3's reference answers.

Usage (subset indexed with scripts/index_vidore.py; GPU with ColQwen + 4-bit VLM, e.g. a T4):
    uv run python scripts/eval_generation.py --vidore hr --limit 50 --qdrant-url http://localhost:6333

Writes one JSON line per (query, strategy) to --out, and prints mean latency per
strategy plus how often the cited evidence includes a gold-relevant page.

ponytail: answer correctness isn't scored here. V3 scores answers with an LLM judge
against the reference answer; run one over the JSONL once a judge is chosen.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from datasets import load_dataset

from mira.evaluation.vidore import build_relevance, corpus_keys
from mira.evidence import crop_image
from mira.generation import STRATEGIES, VLMGenerator, answer_query
from mira.retrieval import BM25Index, HybridRetriever, QdrantMultivectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vidore", required=True, help="ViDoRe V3 subset")
    parser.add_argument("--qdrant-url", help="Qdrant server URL (default: cloud cluster from .env)")
    parser.add_argument("--collection", help="Qdrant collection (default: vidore_v3_<subset>)")
    parser.add_argument("--bm25-path", help="BM25 dir (default: .cache/bm25_vidore/<subset>)")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--top-pages", type=int, default=3)
    parser.add_argument("--limit", type=int, default=50, help="First N English queries")
    parser.add_argument("--out", help="JSONL output (default: data/eval/generation_<subset>.jsonl)")
    args = parser.parse_args()

    subset = args.vidore
    ds = f"vidore/vidore_v3_{subset}"
    queries = [q for q in load_dataset(ds, data_dir="queries", split="test") if q["language"] == "english"][: args.limit]
    qrels = load_dataset(ds, data_dir="qrels", split="test")
    corpus = load_dataset(ds, data_dir="corpus", split="test")
    light = corpus.remove_columns([c for c in corpus.column_names if c not in ("corpus_id", "doc_id", "page_number_in_doc")])
    grades = build_relevance(qrels, corpus_keys(light))
    row_of = {(r["doc_id"], r["page_number_in_doc"]): i for i, r in enumerate(light)}

    def page_image(doc, page, box):
        return crop_image(corpus[row_of[(doc, page)]]["image"].convert("RGB"), box)

    retriever = HybridRetriever(
        store=QdrantMultivectorStore(url=args.qdrant_url, collection_name=args.collection or f"vidore_v3_{subset}"),
        text_index=BM25Index(args.bm25_path or f".cache/bm25_vidore/{subset}"),
    )
    generator = VLMGenerator()

    out = Path(args.out or f"data/eval/generation_{subset}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    latency, grounded = defaultdict(list), defaultdict(list)
    with out.open("w") as f:
        for i, q in enumerate(queries, start=1):
            relevant = set(grades.get(q["query_id"], {}))
            for strategy in args.strategies:
                ans = answer_query(q["query"], retriever, generator, page_image, top_pages=args.top_pages, strategy=strategy)
                latency[strategy].append(sum(ans.timings.values()))
                grounded[strategy].append(any((s["document"], s["page"]) in relevant for s in ans.sources))
                f.write(json.dumps({
                    "query_id": q["query_id"], "query": q["query"], "strategy": strategy,
                    "answer": ans.answer, "reference": q["answer"], "sources": ans.sources,
                    "evidence": [{"id": e.id, "document": e.document_id, "page": e.page_num, "bbox": e.box} for e in ans.evidence],
                    "raw": ans.raw, "timings": ans.timings,
                }) + "\n")
                f.flush()
            print(f"{i}/{len(queries)} {q['query'][:60]!r}", flush=True)

    print(f"\n{'strategy':12s}{'latency s':>11s}{'cites gold page':>17s}  (n={len(queries)})")
    for s in args.strategies:
        print(f"{s:12s}{mean(latency[s]):11.2f}{mean(grounded[s]):17.3f}")
    print(f"Answers: {out}")


if __name__ == "__main__":
    main()
