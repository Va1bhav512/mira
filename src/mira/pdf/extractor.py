import pymupdf
from typing import List
from dataclasses import dataclass


@dataclass
class PageText:
    """Structured text data for a single PDF page."""
    page_num: int
    text: str
    char_count: int
    is_usable: bool


def extract_text(pdf_path: str, min_chars: int = 50) -> List[PageText]:
    """
    Extract native text from PDF pages.
    
    Args:
        pdf_path: Path to PDF file
        min_chars: Minimum characters for text to be considered usable
    
    Returns:
        List of PageText objects with extracted text and metadata
    """
    doc = pymupdf.open(pdf_path)
    pages_text = []
    
    for page_num, page in enumerate(doc):
        text_str = page.get_text()
        usable = is_text_usable(text_str, min_chars)
        
        pages_text.append(PageText(
            page_num=page_num,
            text=text_str,
            char_count=len(text_str),
            is_usable=usable
        ))
    
    doc.close()
    return pages_text


def is_text_usable(text: str, min_chars: int = 50) -> bool:
    """
    Check if extracted text is usable.
    
    Args:
        text: Extracted text
        min_chars: Minimum character threshold
    
    Returns:
        True if text is usable, False otherwise
    """
    if len(text.strip()) < min_chars:
        return False
    return True
