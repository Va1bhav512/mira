from typing import List, Tuple
from dataclasses import dataclass
from PIL import Image
import easyocr
import numpy as np

from .extractor import PageText


@dataclass
class OCRResult:
    """OCR result for a single page."""
    page_num: int
    text: str
    char_count: int
    confidence: float
    source: str = "ocr"


def ocr_page(image: Image.Image, reader: easyocr.Reader = None, languages: List[str] = None) -> str:
    """
    Perform OCR on a single page image.
    
    Args:
        image: PIL Image of the page
        reader: Optional pre-initialized EasyOCR reader (for efficiency)
        languages: List of language codes (default: ['en'])
    
    Returns:
        Extracted text string
    """
    if languages is None:
        languages = ['en']
    
    if reader is None:
        reader = easyocr.Reader(languages, gpu=True)
    
    img_array = np.array(image)
    results = reader.readtext(img_array, detail=0, paragraph=True)
    
    return ' '.join(results) if results else ""


def ocr_fallback(
    pages: List[Tuple[int, Image.Image]], 
    min_chars: int = 50,
    languages: List[str] = None
) -> List[PageText]:
    """
    Perform OCR fallback for pages with insufficient native text.
    
    Args:
        pages: List of (page_num, image) tuples
        min_chars: Minimum characters for text to be usable
        languages: List of language codes for OCR
    
    Returns:
        List of PageText objects with OCR-extracted text
    """
    if languages is None:
        languages = ['en']
    
    reader = easyocr.Reader(languages, gpu=True)
    pages_text = []
    
    for page_num, image in pages:
        text = ocr_page(image, reader)
        usable = len(text.strip()) >= min_chars
        
        pages_text.append(PageText(
            page_num=page_num,
            text=text,
            char_count=len(text),
            is_usable=usable
        ))
    
    return pages_text


def should_use_ocr(native_text: str, min_chars: int = 50) -> bool:
    """
    Determine if OCR should be used based on native text quality.
    
    Args:
        native_text: Extracted native text
        min_chars: Minimum character threshold
    
    Returns:
        True if OCR should be used, False otherwise
    """
    return len(native_text.strip()) < min_chars
