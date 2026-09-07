"""Tests for PDF processing module."""

import pytest
from pathlib import Path
from PIL import Image

from mira.pdf import (
    render_page,
    render_pages,
    extract_text,
    is_text_usable,
    PageText,
    process_pdf,
)


# Test fixtures - create sample PDF path
@pytest.fixture
def sample_pdf():
    """Path to a sample PDF for testing."""
    # You'll need to provide an actual PDF file for integration tests
    test_pdf = Path(__file__).parent.parent.parent / "data" / "samples" / "test.pdf"
    if test_pdf.exists():
        return str(test_pdf)
    pytest.skip("No sample PDF found at data/samples/test.pdf")


class TestRenderer:
    """Tests for PDF rendering."""
    
    def test_render_page_returns_image(self, sample_pdf):
        """Test that render_page returns a PIL Image."""
        img = render_page(sample_pdf, 0, dpi=72)
        assert isinstance(img, Image.Image)
        assert img.mode == 'RGB'
    
    def test_render_pages_returns_list(self, sample_pdf):
        """Test that render_pages returns a list of images."""
        images = render_pages(sample_pdf, pages=[0], dpi=72)
        assert isinstance(images, list)
        assert len(images) == 1
        assert all(isinstance(img, Image.Image) for img in images)
    
    def test_render_all_pages(self, sample_pdf):
        """Test rendering all pages when pages=None."""
        images = render_pages(sample_pdf, pages=None, dpi=72)
        assert len(images) > 0


class TestExtractor:
    """Tests for text extraction."""
    
    def test_extract_text_returns_structure(self, sample_pdf):
        """Test that extract_text returns proper structure."""
        texts = extract_text(sample_pdf, min_chars=0)
        assert isinstance(texts, list)
        assert all(isinstance(t, PageText) for t in texts)
        assert all(hasattr(t, 'page_num') for t in texts)
        assert all(hasattr(t, 'text') for t in texts)
    
    def test_is_text_usable(self):
        """Test text usability check."""
        assert is_text_usable("This is a valid text with enough characters.", min_chars=10) == True
        assert is_text_usable("Short", min_chars=10) == False
        assert is_text_usable("    ", min_chars=1) == False  # Whitespace only


class TestPipeline:
    """Tests for unified processing pipeline."""
    
    def test_process_pdf_returns_processed_pages(self, sample_pdf):
        """Test that process_pdf returns ProcessedPage objects."""
        pages = process_pdf(sample_pdf, dpi=72, min_chars=10)
        assert isinstance(pages, list)
        assert len(pages) > 0
        assert all(hasattr(p, 'page_num') for p in pages)
        assert all(hasattr(p, 'image') for p in pages)
        assert all(hasattr(p, 'text') for p in pages)
        assert all(hasattr(p, 'text_source') for p in pages)
    
    def test_text_source_is_native_or_ocr(self, sample_pdf):
        """Test that text_source is either 'native' or 'ocr'."""
        pages = process_pdf(sample_pdf, dpi=72)
        for page in pages:
            assert page.text_source in ['native', 'ocr', 'none']


class TestTextUsability:
    """Edge cases for text usability."""
    
    def test_empty_text(self):
        assert is_text_usable("", min_chars=50) == False
    
    def test_whitespace_only(self):
        assert is_text_usable("   \n\t  ", min_chars=1) == False
    
    def test_exactly_min_chars(self):
        text = "x" * 50
        assert is_text_usable(text, min_chars=50) == True
    
    def test_below_threshold(self):
        text = "x" * 49
        assert is_text_usable(text, min_chars=50) == False
