# Mira Progress Report

**Last updated:** October 7, 2026

---

## Executive Summary

✅ **All phases built and evaluated.** PDF processing, visual and BM25 indexing, query-adaptive
fusion, evidence cropping, Qwen2.5-VL generation, a local answer judge, and the API + Gradio
demo all run end to end on a Kaggle T4. The full evaluation (custom set + three ViDoRe V3
subsets) ran in one ~4.5 h session on 2026-10-07.

Report material: [docs/REPORT.md](docs/REPORT.md). Result tables:
[data/eval/results.md](data/eval/results.md), [data/eval/vidore_results.md](data/eval/vidore_results.md).

---

## Phase status

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Project setup | ✅ uv, pytest (93 CPU tests), GitHub Actions CI |
| 1 | PDF processing | ✅ 22 PDFs, 2,674 pages (3 scanned, EasyOCR) |
| 2 | Visual indexing | ✅ ColQwen2.5 + Qdrant multivector, exact MaxSim; ink mask stored per page |
| 3 | Text indexing | ✅ `bm25s`, identifier-preserving tokenizer |
| 4 | Hybrid retrieval | ✅ Weighted RRF with query-adaptive weights; evaluated on both benchmarks |
| 5 | Evidence cropping | ✅ Heatmap over ink patches; zone F1 0.483 (hr), 0.408 (cs, held out) |
| 6 | Generation + API/demo | ✅ Qwen2.5-VL-3B 4-bit with region citations; FastAPI `/query` + Gradio `/ui`; Kaggle demo verified |
| 7 | Evaluation | ✅ Retrieval, cropping, generation, LLM-judge correctness, bootstrap tests |

---

## Headline results

| Question | Answer |
|---|---|
| Does fusion beat either channel? | **Custom set: yes.** adaptive nDCG@10 0.844 vs visual 0.801, BM25 0.751. **ViDoRe: no**, visual-only is best (e.g. hr nDCG@5 0.576 vs 0.558). |
| Does adaptive beat fixed fusion? | Custom: +0.020 (p = 0.10). ViDoRe: −0.002 to −0.005 (significant only on hr). |
| Does cropping find the evidence? | Yes after the ink-mask fix: zone F1 0.483 / 0.408 vs whole page 0.358 / 0.303 (hr / cs). Not on slide decks (pharmaceuticals: page 0.672 vs 0.486). |
| Do crops hurt answers? | Heatmap crops ≈ whole pages (pooled −0.033, p = 0.29); single-patch crops clearly hurt (−0.194, p < 0.001). |

---

## What changed in the last stretch

- **Cropping fix:** ColQwen's blank-margin "sink" patches won every token's top-k; the heatmap now
  scores only patches with ink. Pointing accuracy went from chance (0.25) to 0.60 on hr.
- **Fusion rule:** quantities, dates and quarters are no longer treated as identifiers; long
  all-caps names are.
- **Answer judge:** local Qwen2.5-7B-Instruct, 4-bit, correct / partial / incorrect.
- **API + demo:** `python -m mira.serve`; `scripts/kaggle_demo.sh` for a Kaggle notebook.
- **Kaggle runner:** batch kernels with code embedded, Qdrant on the VM; non-fatal eval steps;
  shared VLM image budget after a CUDA OOM; rankings dumped for CPU re-scoring.
- Fixes found by the final run: 60 s Qdrant client timeout, truncated-JSON answer salvage,
  kernel size limit, per-kernel log names, demo index rebuilt from cached embeddings.

---

## Open items

- **Hand-check the custom labels** (47 queries; written from the PDFs and treated as true so far).
- **Human-grade a sample of answers** to calibrate the judge (my check: 25 / 30 agreement).
- Fusion: data-dependent default weight or a learned router (tune on the custom set, test on V3).
- Cropping: whole-page fallback when the heat is spread out (slides).

## Known issues / debt

- Embedding cache is keyed by (document_id, page_num) only — changing DPI/model reuses stale entries.
- `HybridResult.payload` shape differs between visual-only and lexical-only hits (visual wins when both).
- BM25 has no prefix matching: `STM32F401RE` ≠ `STM32F401RET6` (FAQ documents the candidate fix).
- Exact MaxSim scans every page; fine at a few thousand pages, needs HNSW/two-stage beyond.
- Generation results predate the truncated-reply parser fix (3% of answers judged as raw JSON).
