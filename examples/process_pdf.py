#!/usr/bin/env python
"""
Example: Process a PDF document.

Usage:
    uv run python examples/process_pdf.py data/samples/document.pdf
"""

import sys
from pathlib import Path
from mira.pdf import process_pdf, save_processed_page


def main(pdf_path: str):
    """Process a PDF and display results."""
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"Error: PDF not found at {pdf_file}")
        sys.exit(1)
    
    print(f"Processing: {pdf_file.name}")
    print("=" * 60)
    
    # Process PDF
    pages = process_pdf(str(pdf_file), dpi=300, min_chars=50)
    
    # Display results
    print(f"\nTotal pages: {len(pages)}")
    print(f"\nPage Details:")
    print("-" * 60)
    
    for page in pages:
        print(f"Page {page.page_num:3d} | "
              f"Source: {page.text_source:6s} | "
              f"Chars: {page.char_count:5d} | "
              f"Usable: {page.is_usable}")
        
        # Save first few pages as example
        if page.page_num < 3:
            output_dir = Path("output/pages")
            output_path = save_processed_page(page, str(output_dir))
            print(f"  → Saved: {output_path}")
    
    # Summary
    native_count = sum(1 for p in pages if p.text_source == 'native')
    ocr_count = sum(1 for p in pages if p.text_source == 'ocr')
    
    print("\n" + "=" * 60)
    print(f"Summary: {native_count} pages with native text, {ocr_count} needed OCR")
    print(f"Average chars per page: {sum(p.char_count for p in pages) / len(pages):.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/process_pdf.py <pdf_path>")
        sys.exit(1)
    
    main(sys.argv[1])
