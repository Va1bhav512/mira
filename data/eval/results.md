# Retrieval results: Phase 4 fusion modes

Final run 2026-10-07 on a Kaggle T4 (`scripts/kaggle_full_eval.sh`; log `data/eval/kaggle_run.log`).
The first run (Colab, 2026-10-05) differed only on identifier queries; between them the identifier
rule stopped treating quantities, dates and quarters (`2ns`, `q4`, `1950s`) as part numbers and
started matching long all-caps names. That rule change was driven by ViDoRe V3 false positives,
not by these queries.

**Setup**
- Corpus: all 22 PDFs in `data/samples` (2,674 pages; 3 scanned, OCR'd for BM25), rendered at 150 DPI.
- Visual: ColQwen2.5 (`vidore/colqwen2.5-v0.2`, fp16 with bf16 retry on overflow), exact MaxSim in a Qdrant server.
- Lexical: page-level BM25 (`bm25s`, identifier-preserving tokenizer).
- Fusion: weighted RRF, k=60, 50 candidates per channel. Adaptive shift = 0.5, hand-set before this run and **not tuned on these queries**.
- Queries: `queries.jsonl`, 47 queries (15 identifier, 16 visual, 16 semantic), pages 0-indexed.

| Query type | Mode | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| **All (47)** | **adaptive** | **0.638** | **0.904** | **0.947** | **0.847** | **0.844** |
| | fixed (1:1) | 0.585 | 0.894 | 0.936 | 0.814 | 0.824 |
| | visual only | 0.543 | 0.883 | 0.926 | 0.791 | 0.801 |
| | lexical only | 0.532 | 0.798 | 0.894 | 0.724 | 0.751 |
| Identifier (15) | adaptive | 0.567 | 0.933 | 0.933 | 0.824 | 0.830 |
| | fixed | 0.467 | 0.900 | 0.933 | 0.753 | 0.801 |
| | visual | 0.567 | 0.867 | 0.867 | 0.850 | 0.800 |
| | lexical | 0.500 | 0.900 | 0.967 | 0.774 | 0.805 |
| Visual (16) | adaptive | 0.719 | 0.906 | 1.000 | 0.872 | 0.889 |
| | fixed | 0.656 | 0.906 | 0.969 | 0.842 | 0.857 |
| | visual | 0.719 | 0.938 | 0.938 | 0.875 | 0.891 |
| | lexical | 0.500 | 0.656 | 0.844 | 0.655 | 0.691 |
| Semantic (16) | adaptive | 0.625 | 0.875 | 0.906 | 0.842 | 0.812 |
| | fixed | 0.625 | 0.875 | 0.906 | 0.842 | 0.812 |
| | visual | 0.344 | 0.844 | 0.969 | 0.653 | 0.711 |
| | lexical | 0.594 | 0.844 | 0.875 | 0.745 | 0.759 |

**Reading it**
- Overall ordering on every metric: adaptive > fixed > visual > lexical. Fusing the two channels beats either alone, and adaptive weighting adds a little on top.
- Adaptive = fixed on semantic queries, as expected: they rarely contain identifiers or figure words, so the weights stay 1:1.
- On visual queries, adaptive matches visual-only (MRR 0.872 vs 0.875) while fixed fusion loses a little: leaning visual recovers what equal weighting gives away to BM25.
- ColQwen alone has the best identifier MRR (0.850): it reads part numbers off the rendered page well. BM25 still has the best identifier R@10 (0.967).

**Significance and weight sweep**
- Paired bootstrap, adaptive − fixed nDCG@10: **+0.020, 95% CI [−0.003, +0.048], p = 0.099** (n = 47). The direction is consistent with the per-type tables, but at this size it is not significant at 0.05.
- Re-scoring fixed weights on CPU from the dumped rankings (`scripts/rescore_fusion.py`): nDCG@10 is 0.751 (BM25 only), 0.797, 0.815, **0.824 at 1:1**, 0.820, 0.805, 0.801 (visual only) as the visual share goes 0 → 1. So 1:1 is already the best fixed weighting here, and adaptive (0.844) beats every fixed weighting. Shift sizes 0.25 / 0.5 / 0.75 give 0.839 / 0.844 / 0.849; 0.5 was kept, since picking the best on these queries would be tuning on the test set.

**Caveats (say these in the report)**
- 47 queries is small. Adaptive vs fixed overall R@1 (0.638 vs 0.585) is a difference of 2–3 queries.
- Queries and labels were written by Claude from the PDFs (visual ones from rendered figures), then spot-checked; they are not an independent benchmark.
- In born-digital PDFs, figures carry real text labels, so BM25 already finds most "visual" targets; scanned or photo-heavy corpora should favour the visual channel more.
