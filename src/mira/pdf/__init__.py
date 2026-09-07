"""
PDF Processing Module for Mira.

Provides tools for PDF rendering, text extraction, and OCR fallback.
"""

from .renderer import render_page, render_pages
from .extractor import extract_text, is_text_usable, PageText
from .ocr import ocr_page, ocr_fallback, should_use_ocr
from .pipeline import process_pdf, save_processed_page, ProcessedPage

__all__ = [
    'render_page',
    'render_pages',
    'extract_text',
    'is_text_usable',
    'PageText',
    'ocr_page',
    'ocr_fallback',
    'should_use_ocr',
    'process_pdf',
    'save_processed_page',
    'ProcessedPage',
]
