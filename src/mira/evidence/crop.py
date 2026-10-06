"""Turn a normalized evidence box into pixels the VLM reads."""

import pymupdf
from PIL import Image

from .localize import Box, to_pixels


def render_region(pdf_path: str, page_num: int, box: Box, dpi: int = 300) -> Image.Image:
    """Re-render just the box from the PDF at high DPI: sharper small text than upscaling the indexed image."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_num]
        r = page.rect
        x0, y0, x1, y1 = box
        clip = pymupdf.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
        return page.get_pixmap(dpi=dpi, clip=clip).pil_image()


def crop_image(image: Image.Image, box: Box) -> Image.Image:
    """Crop the box from a page image (sources with no PDF, e.g. ViDoRe page images)."""
    return image.crop(to_pixels(box, *image.size))
