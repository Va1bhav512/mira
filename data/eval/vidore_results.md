# ViDoRe V3 results: retrieval, cropping, generation

Run 2026-10-06 on a Kaggle T4 via `scripts/test_kaggle.sh --eval-vidore hr,computer_science`
(87/87 tests passed first). Full log: `data/eval/kaggle_vidore.log` (gitignored, regenerate with
the same command). Generated answers: `generation_<subset>.jsonl`.

**Setup**
- Subsets: `hr` (1,110 pages, 318 English queries) and `computer_science` (1,360 pages, 215).
- Retrieval is the same as the custom set: ColQwen2.5 fp16, exact MaxSim, BM25 on V3's `markdown` page text, RRF k=60, 50 candidates per channel. Nothing was tuned on V3.
- Metrics: graded nDCG (2 = fully relevant, 1 = critically relevant), as in ViDoRe. Recall and MRR count any grade as relevant.

## Retrieval (Phase 4)

| Subset | Mode | R@5 | R@10 | MRR | **nDCG@5** | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| hr (318) | adaptive | 0.496 | 0.637 | 0.707 | 0.558 | 0.591 |
| | fixed | 0.499 | 0.637 | 0.712 | 0.562 | 0.594 |
| | **visual** | **0.532** | **0.651** | 0.705 | **0.576** | **0.601** |
| | lexical | 0.429 | 0.544 | 0.613 | 0.475 | 0.496 |
| computer_science (215) | adaptive | 0.644 | 0.781 | 0.875 | 0.722 | 0.753 |
| | fixed | 0.647 | 0.781 | 0.878 | 0.725 | 0.754 |
| | **visual** | **0.654** | **0.809** | 0.873 | **0.734** | **0.768** |
| | lexical | 0.562 | 0.683 | 0.805 | 0.627 | 0.656 |

Adaptive vs fixed (paired bootstrap on per-query nDCG@5):
- hr: −0.0045, 95% CI [−0.0084, −0.0013], p = 0.003. Small but consistently negative.
- computer_science: −0.0033, 95% CI [−0.0099, +0.0011], p = 0.22. Not significant.

**Reading it**
- **On ViDoRe, visual-only beats both fusions.** That's the opposite of the custom datasheet set (`results.md`), where fusion won. With 1:1 RRF, BM25 is a weaker second channel here: its nDCG@5 is ~0.1 lower, and equal weighting pulls good visual rankings down.
- **BM25 does help where its strength lies.** On computer_science *keyword* queries (35), fixed fusion scores 0.824 nDCG@5 against 0.776 for visual-only.
- **Why adaptive loses slightly:** it changes weights on only ~7% of V3 queries (hr: 15 toward lexical, 7 toward visual; cs: 9 and 5). Most lexical shifts come from tokens the identifier rule mistakes for part numbers: years and units such as `q4`, `1950s`, `2ns`, `S80`. Those push weight toward the weaker channel. A tighter rule, plus a visual-leaning default when no identifier is present, are the obvious next steps. They should be **tuned on the custom set or a held-out V3 subset**, not on these two.
- ColQwen has a training-distribution advantage on ViDoRe-style pages (see the FAQ). The datasheet set is where lexical matching matters.

## Evidence cropping (Phase 5)

Zone F1 against V3 annotator boxes, on the gold pages (so retrieval is not a factor). This is the V3 paper's protocol: merged zones, pixel F1, best annotator. Human agreement is 0.602.

| Subset | Page-zones | page (no crop) | max_patch | heatmap |
|---|---:|---:|---:|---:|
| hr | 1,728 | **0.358** | 0.054 | 0.232 |
| computer_science | 1,049 | **0.303** | 0.059 | 0.208 |

- The heatmap is about 4× better than the single-patch baseline, but **still below simply returning the whole page**. Gold zones are large (a whole-page box's F1 of 0.36 implies they cover ~22% of the page on average), so pixel F1 rewards big boxes.
- The crop thresholds are untuned guesses.

**Precision/recall split** (computed locally from the evidence boxes in `generation_*.jsonl`, on retrieved pages that are gold and annotated; hr n=59, cs n=88):

| Subset | Strategy | F1 | Precision | Recall | Gold area | Crop area |
|---|---|---:|---:|---:|---:|---:|
| hr | heatmap | 0.233 | 0.305 | 0.269 | 0.256 | 0.180 |
| | max_patch | 0.040 | 0.235 | 0.024 | 0.264 | 0.022 |
| computer_science | heatmap | 0.225 | 0.283 | 0.255 | 0.221 | 0.144 |
| | max_patch | 0.061 | 0.269 | 0.039 | 0.213 | 0.022 |

A randomly placed box has precision equal to the gold area fraction, and recall equal to its own area fraction. Against that baseline:
- Heatmap precision is only ~1.2× chance, and recall ~1.5–1.8× chance.
- The single hottest patch is **at or below chance**.

So on real pages the heatmap barely localizes, and the boxes are also smaller than the gold zones (14–18% of the page vs 22–26%). Untuned thresholds alone don't explain a hottest patch that lands no better than a random one. Possible causes:
1. "Attention-sink" patches (blank areas) that match every query token.
2. Non-content query tokens (the prompt plus ColQwen's padding/augmentation tokens).
3. A coordinate bug. The synthetic GPU test can't rule out a transposed grid, because its evidence sits in a corner that a transpose maps onto itself.

Next: render heatmap overlays against the gold boxes for a few pages, and test a transposed-grid control and per-patch normalization.

## Generation (Phase 6), 50 queries per subset

| Subset | Context | Latency (s) | Cites a gold page |
|---|---|---:|---:|
| hr | page | 14.3 | 0.48 |
| | max_patch | 6.7 | 0.64 |
| | heatmap | 11.7 | 0.60 |
| computer_science | page | 14.1 | 0.84 |
| | max_patch | 8.2 | 0.76 |
| | heatmap | 11.7 | 0.80 |

- "Cites a gold page" measures grounding, **not answer correctness**. Correctness needs an LLM judge against V3's reference answers (the answers in the JSONL are ready for one).
- No citation at all: 0–6 of 50 per setting. Nearly all of these are replies that weren't valid JSON, so the parser fell back to raw text.
- With n = 50 per cell, differences under about ±0.1 are noise.

**Caveats:** two subsets; crop and fusion parameters set before seeing V3; generation is not yet scored for correctness.

V3 queries, qrels and reference answers: CC BY 4.0, from `vidore/vidore_v3_hr` and `vidore/vidore_v3_computer_science`.
