#!/usr/bin/env python
"""
Index a ViDoRe V3 subset: page images -> Qdrant (ColQwen), markdown text -> BM25.

Pages are keyed (doc_id, page_number_in_doc) in both stores, matching the keys
that scripts/eval_retrieval.py --vidore builds from the qrels.

Usage:
    uv run python scripts/index_vidore.py --subset computer_science
    uv run python scripts/index_vidore.py --subset hr --qdrant-url http://localhost:6333

Resumable: documents already in the BM25 index are skipped (BM25 is saved only
after a document finishes). Per-page embedding cache lives under --cache-dir.

Note: physics, energy and finance_fr have French pages; Mira's BM25 stemmer is
English, so those subsets are not meaningful for hybrid eval yet.
"""

import argparse
from collections import defaultdict
from typing import Iterator, List, Tuple

from datasets import load_dataset

from mira.evaluation.vidore import SUBSETS
from mira.pdf import ProcessedPage
from mira.retrieval import BM25Index, ColQwenEmbedder, DocumentIndexer, QdrantMultivectorStore

MIN_CHARS = 50


def doc_pages(corpus, rows: List[Tuple[int, int]]) -> Iterator[ProcessedPage]:
    """Yield ProcessedPage for (page_number, corpus row index), decoding images lazily."""
    for page_num, row_idx in rows:
        row = corpus[row_idx]
        text = row["markdown"] or ""
        yield ProcessedPage(
            page_num=page_num,
            image=row["image"].convert("RGB"),
            native_text=text,
            text=text,
            text_source="native",  # Docling parser output, not OCR
            is_usable=len(text.strip()) >= MIN_CHARS,
            char_count=len(text),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--qdrant-url", help="Qdrant server URL (default: cloud cluster from .env)")
    parser.add_argument("--collection", help="Qdrant collection (default: vidore_v3_<subset>)")
    parser.add_argument("--bm25-path", help="BM25 dir (default: .cache/bm25_vidore/<subset>)")
    parser.add_argument("--cache-dir", help="Embedding cache dir (default: .cache/embeddings_vidore/<subset>)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()

    collection = args.collection or f"vidore_v3_{args.subset}"
    bm25_path = args.bm25_path or f".cache/bm25_vidore/{args.subset}"
    cache_dir = args.cache_dir or f".cache/embeddings_vidore/{args.subset}"

    print(f"Loading vidore/vidore_v3_{args.subset} corpus...")
    corpus = load_dataset(f"vidore/vidore_v3_{args.subset}", data_dir="corpus", split="test")

    # Read metadata columns from the arrow table: no image decoding here
    table = corpus.data
    doc_ids = table.column("doc_id").to_pylist()
    page_nums = table.column("page_number_in_doc").to_pylist()
    docs = defaultdict(list)
    for row_idx, (doc_id, page_num) in enumerate(zip(doc_ids, page_nums)):
        docs[doc_id].append((page_num, row_idx))

    text_index = BM25Index(bm25_path)
    store = QdrantMultivectorStore(url=args.qdrant_url, collection_name=collection)
    indexer = DocumentIndexer(embedder=ColQwenEmbedder(), store=store, text_index=text_index)
    indexer.setup()

    done = {doc_id for doc_id, _ in text_index.pages}
    for i, doc_id in enumerate(sorted(docs), start=1):
        rows = sorted(docs[doc_id])
        if doc_id in done:
            print(f"[{i}/{len(docs)}] skip {doc_id} (already indexed)")
            continue
        print(f"[{i}/{len(docs)}] {doc_id} ({len(rows)} pages)")
        indexer.index_page_stream(
            doc_pages(corpus, rows),
            document_id=doc_id,
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            chunk_size=args.chunk_size,
        )

    print(store.get_collection_info(), f"BM25 pages: {len(text_index.pages)}")


if __name__ == "__main__":
    main()
