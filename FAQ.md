# Mira FAQ

Technical Q&A about the multimodal RAG architecture and implementation decisions.

---

## Project Overview

### What is Mira?

Mira is a multimodal retrieval-augmented generation (RAG) system for visually rich technical documents (PDFs). Unlike traditional RAG that converts PDFs to text, Mira maintains both visual and lexical representations, enabling retrieval over tables, diagrams, charts, and complex layouts.

### Why not just use traditional text-based RAG?

Traditional RAG pipelines:
```
PDF → OCR → Text chunks → Dense embeddings → Vector search → LLM
```

This destroys information in:
- Tables (structure lost)
- Charts and graphs (visual data lost)
- Diagrams (spatial relationships lost)
- Multi-column layouts (reading order scrambled)
- Equations (formatting lost)

Mira keeps the visual representation intact and uses late-interaction retrieval (ColQwen) to match queries against visual patches.

### What's unique about this approach?

Two novel contributions:
1. **Query-Adaptive Hybrid Fusion**: Dynamically adjusts weights between visual (ColQwen) and lexical (BM25) retrieval based on query characteristics (identifiers → BM25 heavy, "show the diagram" → visual heavy).

2. **Query-Adaptive Evidence Cropping**: Uses similarity heatmaps to extract relevant regions at high DPI before passing to VLM, improving detail recognition on small tables/figures.

---

## Architecture

### How does the system work at a high level?

```
Ingestion:
PDF → Render pages → ColQwen embeddings (visual)
                   → Native text → BM25 index (lexical)

Query:
            ┌─ Query encoder → MaxSim search ─┐
Query text ─┤                                  ├─ RRF fusion → Top pages → VLM
            └─ Tokenize → BM25 search ─────────┘
```

### Why use two retrieval systems (ColQwen AND BM25)?

They're complementary:
- **BM25**: Exact keyword matching. Excellent for identifiers ("STM32F401RE"), model numbers, specific terms. Fails on visual queries ("show the revenue graph").
- **ColQwen**: Visual semantic matching. Understands diagrams, charts, layouts. May miss exact string matches.

Example:
```
Query: "Find STM32F401RE specifications"
- BM25: Recalls page with exact "STM32F401RE" text
- ColQwen: May match visual similarity to "specification table"

Query: "Show the architecture diagram"
- BM25: Fails (no diagram keywords)
- ColQwen: Matches visual layout of diagram
```

### Why ColQwen2.5 instead of original ColPali?

ColQwen2.5 outperforms ColPali on ViDoRe benchmarks:
- Better performance on document retrieval tasks
- Dynamic resolution handling (no fixed-size preprocessing)
- Newer architecture with ongoing improvements
- Stronger multilingual support

---

## Phase 1: PDF Processing

### Why does ProcessedPage have only one image per page?

Each page is rendered as a single high-DPI image (300 DPI). ColQwen internally splits this into ~400-800 patches (16×16 pixel regions), each becoming a 128-dimensional embedding.

```
Page 3 (single image 2200×3400 pixels)
    │
    ▼ ColQwen encoder
    │
Patch grid (42 rows × 28 cols = 1176 patches)
    │
    ▼ Each patch → 128-D vector
    │
Result: [1176, 128] embeddings for this page
```

Separate images per figure/table aren't needed because:
1. ColQwen handles patch decomposition internally
2. Phase 5 (evidence cropping) extracts regions post-retrieval
3. Fixed grid enables spatial attribution for citations

### Why not split images into figures/tables during ingestion?

Reasons:
1. **Requires layout analysis model** (e.g., LayoutLMv3), adding complexity
2. **ColQwen already does implicit segmentation** via patch attention
3. **Late interaction scores patches**, so it finds relevant regions automatically
4. **Would need to maintain correspondence** between figures and pages for citations

Phase 5 solves region extraction better:
```
Query → ColQwen retrieval → Page with all patches
                             ↓
                  Similarity heatmap (patch-level scores)
                             ↓
                  Cluster high-score regions → Crop
```

### Why track text_source (native vs OCR)?

Critical for query-adaptive fusion in Phase 4.

Native text is reliable for exact matching:
```
"STM32F401RE" in PDF → "STM32F401RE" extracted (perfect)
```

OCR text may have errors:
```
"STM32F401RE" in scanned PDF → "STM32F401 RE" (OCR error)
"0x7FA93C" → "0x7 FA93C" (space inserted)
```

**Fusion strategy:**
```python
if page.text_source == "native":
    bm25_weight = 1.5  # Trust exact matching
    visual_weight = 0.7
elif page.text_source == "ocr":
    bm25_weight = 0.8  # OCR unreliable
    visual_weight = 1.2
```

This improves recall on:
- Technical identifiers
- Part numbers
- Hex addresses
- Code snippets

---

## Phase 2: Visual Indexing

### How does Qdrant store ColQwen embeddings?

Qdrant supports multivector storage (late interaction).

**Traditional vector DB:**
```
page_3 → [768-D vector] → One vector per page
```

**Qdrant multivector:**
```
page_3 → [[128-D], [128-D], ..., [128-D]] → 600 vectors per page
```

Schema:
```python
{
  "id": uuid5("attention_paper/page/3"),  # Qdrant IDs must be int or UUID
  "vector": [  # Shape: [num_patches, 128]
    [0.12, -0.34, ...],  # patch_0
    [0.45, 0.23, ...],   # patch_1
    ...,
    [0.67, -0.89, ...]   # patch_599
  ],
  "payload": {
    "document_id": "attention_paper",
    "page_num": 3,
    "text_source": "native",
    "image_width": 2200,
    "image_height": 3400,
    "native_text": "The dominant...",
    "patch_grid": {"rows": 42, "cols": 28},
    "image_token_start": 4  # vector[4:4+rows*cols] are the image patches
  }
}
```

### What is MaxSim scoring?

MaxSim is the scoring function for late interaction:

```
Query: "attention mechanism"
     ↓
Query tokens: ["attention", "mechanism"]
     ↓
Query embeddings: [q1, q2]  (each 128-D)

For each document page with patches [d0, d1, ..., dN]:

Score = max(q1·d0, q1·d1, ..., q1·dN)  # Best match for q1
      + max(q2·d0, q2·d1, ..., q2·dN)  # Best match for q2
      = sum over query tokens of max similarity to any patch
```

Intuition: For each query token, find the most similar patch in the document, then sum.

Advantages over single-vector:
- Fine-grained matching (token-level)
- Preserves spatial information
- Better handles multi-modal queries

### Why not use CLIP or other single-vector vision models?

| Approach | Q: "STM32F401RE clock speed" | Q: "Show architecture diagram" |
|----------|------------------------------|--------------------------------|
| CLIP (single vector) | May miss exact text | Good visual match |
| ColQwen (late interaction) | Matches "STM32F401RE" patch to query token | Matches diagram patch to "architecture" |

Late interaction allows different query tokens to match different parts of the page:
- "STM32F401RE" matches the identifier patch
- "clock speed" matches the specification table patch

Single vector would compress entire page, losing this fine-grained alignment.

### Why store `patch_grid` and `image_token_start` with every page?

ColQwen's output for a page is not just image patches. The sequence is:

```
[prompt tokens ...][image tokens: rows × cols][more prompt tokens ...]
 ^ 0               ^ image_token_start
```

All of these vectors are stored and used for MaxSim, because that's how the model was trained to score pages. But Phase 5 (evidence cropping) needs to know *which* vectors are image patches and *where* on the page each one sits. So at embedding time we record:

- `patch_grid = (rows, cols)`: read from the processor's `image_grid_thw` (in 14-px patches) divided by the 2×2 spatial merge. It is **not** guessed from the vector count.
- `image_token_start`: position of the first image token.

Then `embeddings[start : start + rows*cols].reshape(rows, cols, 128)` is the page's patch grid (`PageEmbedding.patch_embeddings`). Padding vectors (all zeros, added when a batch mixes page sizes) are stripped before storage.

### Why fp16 instead of 4-bit quantization for ColQwen?

ColQwen2.5 is a ~3B model: ~7.5 GB in fp16, which fits a 16 GB T4 with room for batches. 4-bit would need `bitsandbytes` and can degrade embedding quality, for no practical gain on a T4. (T4s lack bf16 support, hence fp16.) A 4 GB laptop GPU can't hold it either way, so GPU work runs on Colab via `scripts/test_colab.sh`.

---

## Phase 3: Text Indexing

### Why include BM25 when we have visual retrieval?

BM25 solves problems ColQwen struggles with:

**Exact string matching:**
```
Query: "Find RFC-9110 reference"
- ColQwen: May find "RFC" but not exact "RFC-9110"
- BM25: Exact match on "RFC-9110" token
```

**Rare identifiers:**
```
Query: "GHSA-xxxx-xxxx vulnerability"
- ColQwen: No semantic similarity to vulnerability concept
- BM25: Exact match on identifier
```

**Hybrid query handling:**
```
Query: "STM32F401RE pin diagram"
- BM25: Finds "STM32F401RE" mentions
- ColQwen: Finds "diagram" visuals
- Fusion: Intersection gives best page
```

### Why not use dense text embeddings (e.g., BGE, E5)?

Dense text embeddings would create a third retrieval signal:
- Visual: ColQwen
- Lexical: BM25
- Dense text: BGE/E5

This adds complexity without clear benefit because:
1. ColQwen already provides semantic matching (but visual)
2. BM25 provides exact matching (which dense models can't guarantee)
3. Dense text would overlap with both, diluting the fusion logic

Two complementary signals (visual + lexical) is cleaner and more interpretable for ablation studies.

### How does the BM25 tokenizer handle identifiers?

Off-the-shelf tokenizers split and stem in ways that wreck part numbers. `mira.retrieval.lexical.tokenize` instead:

| Input | Tokens | Why |
|-------|--------|-----|
| `STM32F401RE` | `stm32f401re` | Anything with a digit is kept whole, never stemmed |
| `0x7FA93C` | `0x7fa93c` | Hex addresses kept whole |
| `VDD_IO` | `vdd_io`, `vdd`, `io` | Compound kept whole **plus** its parts, so "VDD IO" also matches |
| `I²C` | `i2c` | NFKC normalization (also fixes ligatures like `ﬁ`) |
| `trans-\nformer` | `transform` | Words hyphenated across a line break are rejoined |
| `The registers` | `regist` | Plain words: stopwords dropped, then stemmed |

**Known limitation:** BM25 matches whole tokens, so `STM32F401RE` does not match the orderable variant `STM32F401RET6` or the family wildcard `STM32F401xE` (both appear in the STM32 datasheet). Query-time prefix expansion over the vocabulary is a candidate fix, to be measured on the labelled queries rather than added blind.

### Why page-level BM25 with `bm25s`, not chunks or Qdrant sparse vectors?

- **Page-level**: ColQwen retrieves pages, so both rankings use the same unit `(document_id, page_num)` and fuse directly. Finding the region *within* a page is Phase 5's job.
- **Local `bm25s` over Qdrant sparse vectors**: Qdrant could store BM25-style sparse vectors and fuse server-side, but the adaptive weighting is our contribution. Doing fusion in Python keeps it easy to implement, inspect and ablate, and BM25 then runs offline on CPU.
- **Storage**: one `pages.jsonl` of page texts; the BM25 matrix is rebuilt in memory on first search (~1.7 s for 2,243 pages, searches then take <1 ms). No index format to version.

---

## Phase 4: Retrieval & Fusion

### What is RRF (Reciprocal Rank Fusion)?

RRF combines ranked lists from multiple retrievers:

```
ColQwen ranking: [page_3, page_7, page_1, ...]
BM25 ranking:     [page_7, page_3, page_5, ...]

RRF score for page_3:
  = 1/(k + rank_visual) + 1/(k + rank_lexical)
  = 1/(60 + 1) + 1/(60 + 2)
  = 0.0164 + 0.0161
  = 0.0325
```

Parameter `k` (usually 60) dampens rank differences.

Advantages:
- No score normalization needed (BM25 and MaxSim have different scales)
- Robust to outliers
- Simple to implement

### How does query-adaptive fusion work?

`mira.retrieval.hybrid.query_weights(query)` looks at cheap surface features of the query (no model call) and shifts weight between the two channels:

| Query feature | Example | Weights (visual, lexical) |
|---------------|---------|---------------------------|
| Identifier or quoted text | `ADXL345 register 0x2D`, `"deep sleep"` | 0.5, 1.5 |
| Visual cue word (figure, chart, diagram, table, shown, ...) | `which chart shows revenue` | 1.5, 0.5 |
| Both, or neither | `block diagram of the RP2040`, `how does attention work` | 1.0, 1.0 |

Identifiers are detected by regex: hex (`0x2D`), letter/digit mixes (`STM32F401RE`, `I2C`, `3V3`), and snake-case pin names (`USB_VBUS`). The weights then go into weighted RRF:

```
score(page) = w_visual / (60 + rank_visual) + w_lexical / (60 + rank_lexical)
```

`HybridRetriever.search(query, mode=...)` supports four modes for the ablation: `adaptive`, `fixed` (1:1), `visual` only, `lexical` only. Each result keeps both channel ranks, so you can see *why* a page won.

The shift of 0.5 is hand-set. It should be tuned (and adaptive shown to beat fixed) on the labelled query set with `scripts/eval_retrieval.py`; that comparison is this contribution's evidence.

This is the first novel contribution of Mira.

### Why two-stage retrieval (coarse → exact MaxSim)? Is it implemented?

**Not yet, deliberately.** Today the visual channel is Qdrant's exact MaxSim over every page. The coarse stage only pays off once that is measurably slow; with ~2,000 pages it probably isn't, and the latency we see now is mostly network (the Qdrant cluster is in São Paulo). Measure on the full corpus first. The reasoning for when it becomes worthwhile:

MaxSim is expensive:
```
Query with 10 tokens × 1M patches = 10M dot products
```

Two-stage approach:
```
Stage 1: Coarse retrieval (fast)
  - BM25: top-100
  - Pooled ColQwen vector: top-100
  - Union → ~150 candidates

Stage 2: Exact MaxSim rerank (slow but accurate)
  - Full MaxSim on 150 candidates
  - Return top-10
```

This reduces computation by ~100x for large corpora.

---

## Phase 5: Evidence Cropping

### Why crop regions instead of sending full pages to VLM?

Full page at 300 DPI:
- Size: 2200×3400 pixels (expensive for VLM)
- Small table might be 10% of pixels (hard to read)
- VLM attention diluted across entire page

Evidence crop:
```
Full page 2200×3400 → Crop to 600×800 relevant region → Re-render at 600 DPI
```

Result:
- Smaller input to VLM (faster)
- Higher resolution in relevant area (better OCR of small text)
- Better attention on evidence (more accurate answers)

### How does similarity heatmap work?

ColQwen provides similarity maps showing patch-level scores:

```python
# For query token "architecture"
similarities = colqwen.get_similarity_map(query="architecture", page=page_3)

# Result: 2D array [rows, cols] with scores
# [[0.1, 0.2, 0.05, ...],
#  [0.3, 0.85, 0.82, ...],  ← High scores = relevant region
#  [0.15, 0.78, 0.80, ...],
#  ...]
```

Aggregate across all meaningful query tokens (excluding stopwords):
```
H(x,y) = Σ α_i × S_i(x,y)

where:
- S_i(x,y) = similarity at (x,y) for token i
- α_i = weight (higher for rare/important tokens)
```

Then threshold, cluster, pad, and crop.

---

## Phase 6: Generation

### Why not force VLM to output bounding boxes?

Original plan:
```json
{
  "answer": "...",
  "bbox": [742, 119, 1102, 691]  ← VLM generates this
}
```

Problems:
1. VLM may hallucinate coordinates
2. Doesn't prove evidence supports answer
3. VLMs are bad at precise coordinate generation
4. Retriever already knows the region

Better approach (Mira's design):
```python
# Retriever provides:
evidence = {
  "page_id": 3,
  "bbox": [742, 119, 1102, 691],
  "crop": cropped_image
}

# VLM generates:
answer = vlm.generate(cropped_image, query)

# System constructs:
response = {
  "answer": answer,
  "evidence": [
    {"document": "attention_paper", "page": 3, "bbox": [742, 119, 1102, 691]}
  ]
}
```

Deterministic, verifiable, no hallucination.

### Why Qwen2.5-VL-3B instead of 7B?

For a student project:
- 3B fits on T4 GPU (Kaggle/Colab free tier)
- Faster inference (important for demo)
- Retrieval quality is more important academically than model size

Can compare 3B vs 7B in evaluation if hardware permits, but retrieval experiments (fusion, cropping) are more interesting.

---

## Evaluation

### What datasets for evaluation?

Primary:
- **ViDoRe benchmark**: Standard for visual document retrieval (7 datasets, Recall@k, nDCG)
- **DocVQA**: Visual question answering on documents
- **Custom**: Scanned datasheets, financial reports

For Mira's novel contributions, create custom splits:
- Identifier-heavy queries (tests BM25)
- Visual queries (tests ColQwen)
- Hybrid queries (tests fusion)

### What metrics?

Retrieval:
- Recall@5, Recall@10
- nDCG@10
- MRR (Mean Reciprocal Rank)
- "Exact ID recall" (custom: if query contains identifier, did we retrieve the exact page?)

Generation:
- RAGAS faithfulness (answer supported by evidence)
- RAGAS answer correctness
- Custom: citation accuracy (bbox actually contains answer)

### How do we check retrieval results? How should the labelled queries be written?

Spot checks (reading a few top pages and confirming they contain the answer) catch bugs but aren't evidence. The evidence is `scripts/eval_retrieval.py` over a hand-labelled file, `data/eval/queries.jsonl`:

```json
{"query": "ADXL345 register 0x2D", "document_id": "adxl345", "pages": [12], "type": "identifier"}
```

`pages` are 0-indexed; `type` is `identifier`, `visual` or `semantic`. The script reports Recall@1/5/10, MRR and nDCG@10 per type, for each mode.

Writing the queries:
- **Don't copy wording from the page text.** Queries lifted from the text favour BM25 and inflate its numbers. Write them the way a user would ask.
- **Write `visual` queries by looking at the figure or table**, not the caption.
- **Don't put the document name in every query.** "ADXL345 register 0x2D" is easy because "ADXL345" alone narrows it to one document. Mix in queries that don't name the part.
- Label every page that answers the query, not just the first one you find.

### What is ablation testing and why important?

Ablation removes components to test contribution:

*Illustrative numbers only, not results. Replace with real numbers from `scripts/eval_retrieval.py`.*

| System | Recall@5 | Notes |
|--------|----------|-------|
| ColQwen only | 0.72 | Baseline |
| BM25 only | 0.45 | Baseline |
| Fixed RRF (w=1,1) | 0.78 | Fusion helps |
| **Adaptive RRF** | **0.82** | This is your contribution |
| Full page to VLM | 0.71 accuracy | Baseline |
| **Evidence crop + VLM** | **0.85 accuracy** | Your contribution |

Tables prove:
- You didn't just "assemble existing components"
- Your novel algorithms (adaptive fusion, evidence cropping) add measurable value

---

## Implementation

### What's the estimated code size?

Realistic estimate for complete system:
- Source code (`src/`): 3,500-4,500 lines
- Tests: 1,000-1,500 lines
- Scripts/examples: 250-350 lines
- Total: ~5,000-6,000 lines

Breakdown:
- PDF processing: ~350 lines (Phase 1) ✓
- Visual indexing: ~400 lines
- Text indexing: ~200 lines
- Fusion/retrieval: ~350 lines
- Evidence cropping: ~500 lines
- VLM generation: ~500 lines
- Evaluation: ~600 lines

### Can this run on free GPUs?

Yes:
- ColQwen inference: T4 (Colab/Kaggle free tier)
- Qdrant: Local Docker or Qdrant Cloud free tier
- Qwen2.5-VL-3B: T4 with 4-bit quantization
- BM25: CPU (no GPU needed)

For full evaluation with larger models:
- A100 (Colab Pro or Kaggle upgrade) for 7B VLM
- Or use 4-bit quantization throughout

### What's the division between two project members?

From initial_instructions.txt:

**Partner 1: Retrieval and Representation**
- ColQwen visual embedding pipeline
- Page-level coarse candidate retrieval
- Multivector storage in Qdrant
- Exact MaxSim implementation/reranking
- Quantization experiments
- BM25 + visual fusion
- Adaptive fusion algorithm
- Retrieval benchmarking

**Partner 2: Evidence Localization and Generation**
- PDF native-text extraction (Phase 1) ✓
- OCR fallback (Phase 1) ✓
- Similarity-map extraction
- Spatial aggregation algorithm
- High-resolution PDF-region renderer
- Dynamic crop strategy
- VLM serving
- Evidence/citation pipeline
- Generation benchmarking

Both have substantial algorithmic work, systems engineering, and experiments.

---

## Future Extensions

### What if I want to scale to millions of pages?

Current design works for thousands. For millions:

1. **Quantized vectors**: INT8 or binary compression (Qdrant supports this)
2. **Caching**: Cache common query embeddings
3. **Distributed Qdrant**: Shard across multiple nodes
4. **Batch processing**: Parallel ingestion with Ray/Dask
5. **Approximate MaxSim**: Investigate compressed multivector search

### Could this work for other document types?

Yes:
- **Presentations (PPTX)**: Render slides, same pipeline
- **Web pages**: Screenshot rendering, same pipeline
- **Scientific papers with LaTeX**: Could extract equations separately
- **Medical images**: Would need domain-specific training

### What about real-time queries?

Current design:
- Ingestion: Slow (minutes per PDF)
- Query: ~500ms (ColQwen embedding + MaxSim + VLM)

For real-time:
- Pre-compute all embeddings
- Cache query embeddings for repeated queries
- Use lighter VLM or API-based generation
- Skip evidence crop for speed (trade accuracy)

---

## Troubleshooting

### OCR is too slow. What to do?

Options:
1. Only OCR if native text < 50 chars (current default)
2. Increase threshold to 100 or 200 chars
3. Use Tesseract instead of EasyOCR (faster, less accurate)
4. Parallelize OCR across CPU cores
5. Skip OCR entirely for digitally-generated PDFs

### Qdrant multivector search is slow. What to do?

Check:
1. Are you using multivector config correctly?
2. Is HNSW index built?
3. Try limiting vectors per point (truncate to top-500 patches by norm)

### VLM gives wrong answers. What to check?

Potential issues:
1. Evidence crop doesn't contain relevant info (check heatmap)
2. Cropped resolution too low (increase DPI)
3. VLM prompt needs refinement
4. Retrieval failed (wrong page retrieved)

Debug:
```python
# Check retrieval
for page in retrieved_pages:
    print(f"Page {page.page_num}: {page.text[:100]}...")

# Check crop
crop_image.save("debug_crop.png")

# Test VLM directly
answer = vlm.generate(crop_image, query)
print(answer)
```
