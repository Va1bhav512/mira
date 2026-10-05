#!/usr/bin/env python
"""
Index every PDF in a directory: ColQwen -> Qdrant, page text -> BM25.

Usage:
    uv run python scripts/index_corpus.py                                  # cloud Qdrant from .env
    uv run python scripts/index_corpus.py --qdrant-path /content/cache/qdrant \\
        --bm25-path /content/cache/bm25 --cache-dir /content/cache/embeddings

Resumable: documents already in the BM25 index are skipped. BM25 is saved only
after a document finishes, so its presence means Qdrant has every page too.
"""

import argparse
from pathlib import Path

from mira.retrieval import BM25Index, ColQwenEmbedder, DocumentIndexer, QdrantMultivectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data/samples")
    parser.add_argument("--qdrant-path", help="On-disk embedded Qdrant (default: cloud from .env)")
    parser.add_argument("--bm25-path", default=".cache/bm25")
    parser.add_argument("--cache-dir", default=".cache/embeddings")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    text_index = BM25Index(args.bm25_path)
    store = QdrantMultivectorStore(path=args.qdrant_path)
    indexer = DocumentIndexer(embedder=ColQwenEmbedder(), store=store, text_index=text_index)
    indexer.setup()

    done = {doc for doc, _ in text_index.pages}
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    for i, pdf in enumerate(pdfs, start=1):
        if pdf.stem in done:
            print(f"[{i}/{len(pdfs)}] skip {pdf.stem} (already indexed)")
            continue
        print(f"[{i}/{len(pdfs)}] {pdf.stem}")
        indexer.index_document(str(pdf), dpi=args.dpi, batch_size=args.batch_size, cache_dir=args.cache_dir)

    print(store.get_collection_info(), f"BM25 pages: {len(text_index.pages)}")


if __name__ == "__main__":
    main()
