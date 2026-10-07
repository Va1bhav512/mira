# Mira - Multimodal RAG for Visually Rich PDFs

## What is Mira?

Mira is a multimodal retrieval-augmented generation (RAG) system designed for visually rich technical documents. Unlike traditional RAG that converts PDFs to plain text (destroying tables, charts, and layout information), Mira maintains both visual and lexical representations of documents.

**Key insight:** Some information in PDFs cannot be represented as text:
- Tables with complex structures
- Charts and graphs with visual data
- Diagrams showing spatial relationships
- Multi-column layouts
- Equations and mathematical notation

Mira uses late-interaction visual retrieval (ColQwen2.5) combined with exact lexical matching (BM25) to find relevant content, then dynamically crops evidence regions for detailed VLM analysis.

---

## Why Mira?

**Traditional RAG limitations:**

```
Traditional Pipeline:
PDF → OCR → Text chunks → Dense embeddings → Vector search → LLM

Problems:
- Tables lose structure (column alignment gone)
- Charts become "Figure 3: Revenue" (actual data lost)
- Diagrams become alt-text (spatial info lost)
- OCR errors propagate to retrieval
```

**Mira's approach:**

```
Mira Pipeline:
PDF → [Visual] Render pages → ColQwen patch embeddings
    → [Lexical] Native text → BM25 index
    
Query → Hybrid retrieval (visual + lexical) → Evidence crop → VLM → Answer

Advantages:
- Visual: Understands layout, diagrams, charts
- Lexical: Exact matching for identifiers, model numbers
- Evidence crop: Hi-res region for VLM (not compressed full page)
```

---

## Novel Contributions

1. **Query-Adaptive Hybrid Fusion**: Dynamically adjusts weights between visual and lexical retrieval based on query characteristics. Identifier-heavy queries use more BM25; visual queries use more ColQwen.

2. **Query-Adaptive Evidence Cropping**: Uses similarity heatmaps to extract relevant regions at high DPI before VLM generation, improving detail recognition on small tables/figures.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PHASE                          │
└─────────────────────────────────────────────────────────────┘
                         │
            PDF Document │
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
Phase 1:            Phase 2:            Phase 3:
Render Pages        Visual Index        Text Index
(Page images)       (ColQwen)           (BM25)
    │                    │                    │
    └────────────────────┴────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Qdrant + BM25      │
              │  Storage            │
              └─────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      QUERY PHASE                             │
└─────────────────────────────────────────────────────────────┘
                         │
                 User Query
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Query Encoder    Text Tokenizer    Feature Extract
   (ColQwen)         (BM25)            (for weights)
        │                │                │
        ▼                ▼                ▼
   MaxSim Search    BM25 Search    Weight Calculator
        │                │                │
        └────────────────┴────────────────┘
                         │
                         ▼
                 Phase 4: RRF Fusion
                 (Query-adaptive weights)
                         │
                         ▼
                    Top-K Pages
                         │
                         ▼
                 Phase 5: Evidence Crop
                 (Similarity heatmap → Region)
                         │
                         ▼
                 Phase 6: VLM Generation
                 (Qwen2.5-VL-3B)
                         │
                         ▼
                    Final Answer
                    + Citations
```

---

## Project Phases

| Phase | Focus | Deliverable | Status |
|-------|-------|-------------|--------|
| 0 | Project setup | Code structure, dependencies, CI | ✅ Complete |
| 1 | PDF processing | Renderer, text extraction, OCR fallback | ✅ Complete |
| 2 | Visual indexing | ColQwen embeddings, Qdrant setup | ✅ Complete (verified on Colab T4) |
| 3 | Text indexing | BM25 implementation, native text pipeline | ✅ Complete |
| 4 | Retrieval core | MaxSim, RRF, query-adaptive fusion | ✅ Built and evaluated ([results](data/eval/results.md)) |
| 5 | Evidence cropping | Heatmaps, spatial aggregation, hi-res crop | ✅ Built and evaluated; tuned on V3 hr ([V3 results](data/eval/vidore_results.md)) |
| 6 | Generation | VLM serving, citations, API | ✅ Answers with region citations; FastAPI + Gradio demo |
| 7 | Evaluation | Custom ablation; ViDoRe V3 hr, computer_science, pharmaceuticals; LLM judge | ✅ Complete ([report](docs/REPORT.md)) |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Va1bhav512/mira.git
cd mira

# Install dependencies
uv sync

# Install dev dependencies (for testing)
uv sync --all-extras
```

### Process a PDF (Phase 1)

```python
from mira.pdf import process_pdf

# Process PDF with hi-res rendering
pages = process_pdf("document.pdf", dpi=300)

# Access results
for page in pages:
    print(f"Page {page.page_num}:")
    print(f"  Text source: {page.text_source}")  # 'native' or 'ocr'
    print(f"  Characters: {page.char_count}")
    print(f"  Usable: {page.is_usable}")
    
    # Save page image
    page.image.save(f"output/page_{page.page_num}.png")
```

### Index and search (Phases 2-3)

```python
from mira.retrieval import BM25Index, DocumentIndexer, VisualRetriever

# Visual embeddings -> Qdrant, page text -> BM25 (.cache/bm25/pages.jsonl). Needs a GPU.
indexer = DocumentIndexer()
indexer.setup()
indexer.index_document("data/samples/DS_stm32f401re.pdf")

# Exact-term search (CPU only; identifiers like 0x2D or VDD_IO are kept whole)
for r in BM25Index().search("STM32F401RE flash size", top_k=5):
    print(r.document_id, r.page_num, r.score)
```

### Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=mira

# GPU tests + index/search end-to-end on a Colab T4, using your local
# working tree (needs the colab CLI and Qdrant creds in .env)
scripts/test_colab.sh            # session "mira"; reused if already running
scripts/test_colab.sh --eval     # full-corpus custom-set eval
scripts/test_colab.sh --eval-vidore computer_science,hr   # ViDoRe V3 eval
colab stop -s mira               # release the VM when done

# Phase 1 stats over every PDF in data/samples (CPU, ~2 min)
uv run python scripts/phase1_report.py
```

ColQwen tests skip locally unless the GPU has >=10 GB VRAM.

### GPU runs on Kaggle (recommended: 30 GPU-h/week, 12 h sessions, no mid-run reclaim)

```bash
scripts/test_kaggle.sh                                    # pytest + example on a T4
scripts/test_kaggle.sh --eval-vidore hr,computer_science  # ViDoRe retrieval/cropping/generation
scripts/test_kaggle.sh --dataset <you>/mira-custom-embeddings \
    --run 'bash scripts/kaggle_full_eval.sh' mira-full    # everything + a demo check
kaggle quota                                              # GPU hours left this week
```

Code travels inside the kernel; only `data/samples` goes up, once, as a private dataset. Qdrant runs
on the VM, so no credentials leave your machine. Logs go to `data/eval/kaggle_<kernel>.log`, and
`--run` outputs to `.cache/kaggle/<kernel>/`.

### API and demo

```bash
# On a GPU box with the custom corpus indexed (scripts/index_corpus.py):
uv sync --extra generation --extra demo
uv run python -m mira.serve --qdrant-url http://localhost:6333 --bm25-path .cache/bm25
#   POST /query  {"question": "...", "mode": "adaptive", "strategy": "heatmap"}
#   /ui          Gradio demo: answer + citations, retrieved pages with heatmap and evidence
#                boxes, the crops the VLM read, visual/BM25 ranks and fusion weights
```

On Kaggle, run the demo in a notebook (GPU T4, Internet on, inputs `<you>/mira-samples` and
`<you>/mira-custom-embeddings`; the index is rebuilt from the cached embeddings in ~6 min):

```
!git clone -q --depth 1 https://github.com/Va1bhav512/mira /kaggle/working/mira
!bash /kaggle/working/mira/scripts/kaggle_demo.sh     # prints a public gradio.live link
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| PDF Rendering | PyMuPDF (fitz) | Hi-DPI page rendering, native text extraction |
| OCR Fallback | EasyOCR | Text extraction from scanned documents |
| Visual Encoder | ColQwen2.5 | Late-interaction visual embeddings |
| Vector Database | Qdrant | Multivector storage, MaxSim search |
| Lexical Search | BM25 (BM25s or Whoosh) | Exact text matching |
| Generation | Qwen2.5-VL-3B | Visual question answering |
| Framework | Python 3.12+, uv package manager | Modern dependency management |

---

## Project Structure

```
mira/
├── src/mira/
│   ├── pdf/              # Phase 1: PDF processing
│   │   ├── renderer.py   # PDF → image
│   │   ├── extractor.py   # Native text extraction
│   │   ├── ocr.py        # OCR fallback
│   │   └── pipeline.py   # Unified processing
│   ├── retrieval/        # Phase 2-4: Visual + text indexing
│   ├── evidence/         # Phase 5: Evidence cropping
│   ├── generation/       # Phase 6: VLM answers + citations
│   ├── evaluation/       # Phase 7: Metrics, ViDoRe loaders, answer judge
│   └── serve.py          # API + Gradio demo
├── tests/
│   └── test_pdf/         # PDF module tests
├── examples/
│   └── process_pdf.py    # Example usage
├── data/
│   └── samples/          # Sample PDFs
├── FAQ.md                # Detailed Q&A
└── README.md             # This file
```

---

## Documentation

- **FAQ.md**: Detailed Q&A about architecture decisions, implementation choices
- **src/mira/pdf/README.md**: Phase 1 module documentation
- **initial_instructions.txt**: Original architecture design (in .gitignore)

---

## Team

Two-person B.Tech minor project:

**Partner 1: Retrieval & Representation**
- ColQwen embedding pipeline
- Qdrant multivector storage
- MaxSim reranking
- Adaptive fusion
- Retrieval benchmarking

**Partner 2: Localization & Generation**
- PDF text extraction ✓
- OCR fallback ✓
- Similarity heatmaps
- Evidence cropping
- VLM generation
- Citation pipeline

---

## Running on Free GPUs

Mira is designed to run on free-tier GPUs:

- **Kaggle**: 2× T4 (30 hours/week)
- **Google Colab**: T4 (free), A100 (pay-as-you-go)
- **Qdrant**: Local Docker or free cloud tier

Requirements:
- ColQwen inference: T4 (✓)
- Qwen2.5-VL-3B: T4 with 4-bit quantization (✓)
- BM25: CPU only (✓)
- Full pipeline: T4 sufficient (✓)

---

## License

[Add your license]

---

## References

- [ColPali Paper](https://arxiv.org/abs/2407.01449)
- [ColVision Repository](https://github.com/illuin-tech/colpali)
- [Qdrant Multivector Documentation](https://qdrant.tech/documentation/manage-data/vectors/)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [ViDoRe Benchmark](https://github.com/illuin-tech/vidore-benchmark)
