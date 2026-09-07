# Phase 1: PDF Processing Pipeline

## Overview

This phase implements the foundation for processing PDF documents:
- **Rendering**: Convert PDF pages to high-resolution images
- **Text Extraction**: Extract native PDF text using PyMuPDF
- **OCR Fallback**: Use EasyOCR when native text is insufficient

## Components

### `renderer.py`
Renders PDF pages to PIL Images at configurable DPI.

```python
from mira.pdf import render_page, render_pages

# Render single page
img = render_page("document.pdf", page_num=0, dpi=300)

# Render multiple pages
images = render_pages("document.pdf", pages=[0, 1, 2], dpi=300)
```

### `extractor.py`
Extracts native text from PDFs.

```python
from mira.pdf import extract_text

pages = extract_text("document.pdf")
for page in pages:
    print(f"Page {page.page_num}: {page.char_count} chars, usable={page.is_usable}")
```

### `ocr.py`
Performs OCR as fallback for pages without usable native text.

```python
from mira.pdf import ocr_page, should_use_ocr
from PIL import Image

# Check if OCR needed
if should_use_ocr(native_text, min_chars=50):
    text = ocr_page(image)
```

### `pipeline.py`
Unified processing pipeline combining all components.

```python
from mira.pdf import process_pdf

# Process entire PDF
pages = process_pdf("document.pdf", dpi=300, min_chars=50)

for page in pages:
    print(f"Page {page.page_num}:")
    print(f"  Source: {page.text_source}")
    print(f"  Chars: {page.char_count}")
    print(f"  Usable: {page.is_usable}")
    
    # Save image
    page.image.save(f"page_{page.page_num}.png")
```

## Testing

```bash
# Run tests (requires sample PDF)
uv run pytest tests/test_pdf/ -v

# Run with coverage
uv run pytest tests/test_pdf/ --cov=mira.pdf
```

## Dependencies

- `pymupdf`: PDF rendering and text extraction
- `pillow`: Image handling
- `easyocr`: OCR fallback (GPU-accelerated)

## Output Structure

`ProcessedPage` contains:
- `page_num`: Page number (0-indexed)
- `image`: PIL Image at specified DPI
- `text`: Extracted text (native or OCR)
- `text_source`: 'native', 'ocr', or 'none'
- `is_usable`: Whether text meets minimum quality
- `char_count`: Number of characters

## Next Steps

Phase 2 will use these processed pages for:
- ColQwen visual embedding generation
- Multi-representation indexing
- Qdrant vector storage setup
