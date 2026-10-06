# Mira Progress Report

**Last updated:** October 6, 2026

---

## Executive Summary

✅ **Phases 0–4 complete:** PDF processing, visual indexing (ColQwen + Qdrant), BM25 lexical indexing, and query-adaptive hybrid fusion are built, tested on a Colab T4, and evaluated on a 47-query labelled set.
🔨 **Phase 7 in progress:** custom-corpus eval done; ViDoRe V3 harness built (this change), full run pending.
🔨 **Phases 5–6 implemented, GPU-unverified:** evidence cropping and Qwen2.5-VL generation are built and unit-tested on CPU; the real-model tests and V3 evals have not run yet.

---

## Phase status

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Project setup | ✅ Complete |
| 1 | PDF processing | ✅ Complete — 22 PDFs, 2,674 pages (incl. 3 scanned, OCR'd) |
| 2 | Visual indexing | ✅ Complete — verified on Colab T4 |
| 3 | Text indexing (BM25) | ✅ Complete — `bm25s`, identifier-preserving tokenizer |
| 4 | Retrieval core (RRF + adaptive fusion) | ✅ Complete — evaluated, see below |
| 5 | Evidence cropping | 🔨 Implemented (`mira/evidence/`), CPU-tested; V3 zone-F1 eval (`scripts/eval_cropping.py`) not yet run |
| 6 | Generation (VLM) | 🔨 Implemented (`mira/generation/`), CPU-tested with stubs; real-model test + `scripts/eval_generation.py` not yet run |
| 7 | Evaluation | 🔨 Custom eval ✅; ViDoRe V3 harness ✅, first run pending |

---

## Phase 4: retrieval ablation (real numbers)

Full table, setup and caveats: [data/eval/results.md](data/eval/results.md).
2,674 pages, 47 labelled queries, Colab T4:

| Mode | R@1 | R@5 | MRR | nDCG@10 |
|------|----:|----:|----:|--------:|
| **adaptive** | **0.628** | **0.904** | **0.836** | **0.843** |
| fixed (1:1) | 0.585 | 0.894 | 0.814 | 0.824 |
| visual only | 0.543 | 0.883 | 0.791 | 0.801 |
| lexical only | 0.532 | 0.798 | 0.724 | 0.751 |

**Open caveat:** adaptive vs fixed is a 2-query difference at R@1 on n=47. The paired
bootstrap significance test is now built into `scripts/eval_retrieval.py`; the ViDoRe
V3 run (~300 queries/subset) is what makes it meaningful.

---

## ViDoRe V3 harness (this change)

**Built, not yet run on GPU.** V3 is the main benchmark: 8 public subsets, human-verified
qrels with graded relevance (2=Fully, 1=Critically), bounding boxes (→ Phase 5 scoring,
human ceiling F1 0.602), reference answers (→ Phase 6 scoring), and per-query
type/format/content labels for adaptive-fusion breakdowns.

- `scripts/index_vidore.py --subset computer_science` — indexes page images → Qdrant,
  markdown → BM25, keyed `(doc_id, page_number_in_doc)`; resumable; per-subset stores.
- `scripts/eval_retrieval.py --vidore hr` — English queries; graded nDCG@5 (headline),
  R@5/10, MRR; breakdowns by query format and content type; **paired bootstrap
  adaptive-vs-fixed nDCG@5**.
- `DocumentIndexer.index_page_stream()` — new sink; `index_document` is now a thin PDF
  wrapper. Any (image + text) page source can be indexed.
- `scripts/test_colab.sh --eval-vidore computer_science,hr` — full GPU run with
  checkpointing, same pattern as `--eval`.
- Validated locally: dataset loading, qrels/corpus mapping (hr: 318 English queries,
  1,110 pages), bucket logic, empty-text BM25 pages. 54 tests pass.

**Subset plan:** first run computer_science (1,360p) + hr (1,110p) ≈ current corpus size.
Final report adds pharmaceuticals (English, charts/tables). physics/energy/finance_fr are
French corpora — BM25's English stemmer doesn't apply; skip or handle separately.

**Honest expectations:** adaptive may ≈ fixed on V3 (natural-language queries rarely
contain identifiers); the custom datasheet set stays the identifier-heavy out-of-domain
evidence. ColQwen has a distribution advantage on ViDoRe-style data (but V3 postdates
ColQwen2.5, so no memorization).

---

## Phases 5–6 (this change)

Details: FAQ, "Phase 5" and "Phase 6".

- `mira/evidence/`: per-token similarity maps from the stored patch vectors (`store.get_page`),
  top-k per token, peakedness weights, threshold → connected regions → padded normalized boxes.
  PDF regions re-rendered at 300 DPI; image sources cropped.
- `mira/generation/`: `VLMGenerator` (Qwen2.5-VL-3B, 4-bit NF4, `generation` extra) returns
  `{answer, evidence_ids}`; `answer_query` builds citations from the retriever's boxes.
  Context strategies: `page` / `max_patch` / `heatmap`.
- Evals: `eval_cropping.py` (V3 paper's zone F1, best annotator, oracle pages) and
  `eval_generation.py` (end-to-end answers saved beside V3 references, latency, gold-page citation rate).
  Both are wired into `test_colab.sh --eval-vidore`.
- **Verified:** 81 CPU tests pass (synthetic heatmaps, region grouping, zone F1, V3 box parsing,
  pipeline with stub models, Qdrant round trip), and the Qwen processor/chat-template plumbing
  was checked locally.
- **Not verified:** the real ColQwen heatmap test, the real Qwen2.5-VL test, and both evals.
  The local GPU is 4 GB and the HF download is too slow here; the GPU run is on hold.
- Open points: thresholds are hand-set; whether fp16 compute on a T4 makes Qwen2.5-VL overflow;
  answer correctness needs an LLM judge (V3's protocol); none has been chosen yet.

---

## Known issues / debt

- Embedding cache is keyed by (document_id, page_num) only — changing DPI/model reuses stale entries (fine at current fixed settings).
- `HybridResult.payload` shape differs between visual-only and lexical-only hits (matters for Phase 5 payload plumbing).
- BM25 has no prefix matching: `STM32F401RE` ≠ `STM32F401RET6` (FAQ documents the candidate fix).
- No CI (Phase 0 claimed it); `scripts/test_colab.sh` is the de-facto gate.

---

## Next steps

1. **Run V3 eval** (CS + HR) via `scripts/test_colab.sh --eval-vidore computer_science,hr`; commit `data/eval/vidore_eval.log` and a results table.
2. **Verify Phases 5–6 on a GPU** — the slow tests (`test_colqwen_heatmap`, `test_vlm`), then `eval_cropping.py` (page vs max_patch vs heatmap zone F1, vs the 0.602 human ceiling) and `eval_generation.py`.
3. **Answer scoring** — pick an LLM judge for `data/eval/generation_<subset>.jsonl`.
4. Tune `WEIGHT_SHIFT` on the custom set only (never on V3).
