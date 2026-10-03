"""
Unified PDF Processing Pipeline.

Combines rendering, text extraction, and OCR fallback into single workflow.
"""

from typing import Iterator, List, Optional
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import pymupdf

from .renderer import render_pages
from .extractor import extract_text, is_text_usable, PageText
from .ocr import ocr_fallback, should_use_ocr


@dataclass
class ProcessedPage:
    """Fully processed page with image and text."""
    page_num: int
    image: Image.Image
    native_text: str
    text: str
    text_source: str  # 'native', 'ocr', or 'none'
    is_usable: bool
    char_count: int


def process_pdf(
    pdf_path: str,
    dpi: int = 300,
    min_chars: int = 50,
    pages: Optional[List[int]] = None,
    force_ocr: bool = False
) -> List[ProcessedPage]:
    """
    Process a PDF file: render pages, extract text, apply OCR fallback if needed.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for image rendering (default: 300)
        min_chars: Minimum characters for usable text (default: 50)
        pages: Specific page numbers to process (None = all pages)
        force_ocr: Force OCR even if native text exists (for testing)
    
    Returns:
        List of ProcessedPage objects with image and extracted text
    
    Example:
        >>> pages = process_pdf("document.pdf", dpi=300)
        >>> for page in pages:
        ...     print(f"Page {page.page_num}: {page.text_source}, {page.char_count} chars")
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Step 1: Render pages to images
    images = render_pages(str(pdf_path), pages=pages, dpi=dpi)
    
    # Step 2: Extract native text
    native_texts = extract_text(str(pdf_path), min_chars=min_chars)
    
    # Filter to requested pages if specified
    if pages is not None:
        native_texts = [t for t in native_texts if t.page_num in pages]
    
    # Step 3: Determine which pages need OCR
    pages_need_ocr = []
    processed_pages = []
    
    for i, (image, native_data) in enumerate(zip(images, native_texts)):
        page_num = native_data.page_num if native_texts else (pages[i] if pages else i)
        native_text = native_data.text if native_texts else ""
        
        if force_ocr or should_use_ocr(native_text, min_chars):
            pages_need_ocr.append((page_num, image))
        else:
            processed_pages.append(ProcessedPage(
                page_num=page_num,
                image=image,
                native_text=native_text,
                text=native_text,
                text_source='native',
                is_usable=True,
                char_count=len(native_text)
            ))
    
    # Step 4: Run OCR on pages that need it
    if pages_need_ocr:
        ocr_results = ocr_fallback(pages_need_ocr, min_chars=min_chars)
        
        for (page_num, image), ocr_text in zip(pages_need_ocr, ocr_results):
            processed_pages.append(ProcessedPage(
                page_num=page_num,
                image=image,
                native_text="",
                text=ocr_text.text,
                text_source='ocr',
                is_usable=ocr_text.is_usable,
                char_count=ocr_text.char_count
            ))
    
    # Sort by page number
    processed_pages.sort(key=lambda p: p.page_num)
    
    return processed_pages


def iter_processed_pages(
    pdf_path: str,
    dpi: int = 300,
    chunk_size: int = 32,
    min_chars: int = 50,
) -> Iterator[List[ProcessedPage]]:
    """
    Yield process_pdf results chunk_size pages at a time.
    
    Use this instead of process_pdf for whole documents: a 700-page PDF
    rendered at once is several GB of page images.
    ponytail: process_pdf re-extracts text for the whole doc per chunk; cheap vs rendering
    """
    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count
    
    for start in range(0, page_count, chunk_size):
        pages = list(range(start, min(start + chunk_size, page_count)))
        yield process_pdf(pdf_path, dpi=dpi, min_chars=min_chars, pages=pages)


def save_processed_page(page: ProcessedPage, output_dir: str, format: str = 'PNG') -> str:
    """
    Save processed page image to disk.
    
    Args:
        page: ProcessedPage object
        output_dir: Directory to save image
        format: Image format (PNG, JPEG)
    
    Returns:
        Path to saved image
    """
    from pathlib import Path
    output_path = Path(output_dir) / f"page_{page.page_num:04d}.{format.lower()}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.image.save(output_path, format=format)
    return str(output_path)
