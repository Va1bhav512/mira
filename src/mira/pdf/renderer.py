from typing import Optional, List
import pymupdf

from PIL import Image

def render_page(pdf_path: str, page_num: int, dpi: int = 300) -> Image.Image:
    """
    Render a single PDF page to a PIL Image.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        dpi: Resolution for rendering (default: 300)
    
    Returns:
        PIL Image of the page
    """
    doc = pymupdf.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = pix.pil_image()
    doc.close()
    return img

def render_pages(pdf_path: str, pages: Optional[List[int]] = None, dpi: int = 300) -> List[Image.Image]:
    """
    Render multiple PDF pages to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        pages: List of page numbers (0-indexed). If None, renders all pages.
        dpi: Resolution for rendering (default: 300)
    
    Returns:
        List of PIL Images
    """
    doc = pymupdf.open(pdf_path)
    images = []
    
    if pages is not None:
        for i in pages:
            page = doc[i]
            pix = page.get_pixmap(dpi=dpi)
            img = pix.pil_image()
            images.append(img)
    else:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = pix.pil_image()
            images.append(img)
    
    doc.close()
    return images
