#!/usr/bin/env python
"""
Phase 1 corpus report: run process_pdf over data/samples and print per-PDF
page/char/timing stats (skips OCR-heavy PDFs, which are slow on CPU).

Usage: uv run python scripts/phase1_report.py
"""

import sys
from pathlib import Path
import time
from mira.pdf import iter_processed_pages


def main():
    print("="*60)
    print("Phase 1 Corpus Report")
    print("="*60)
    
    pdf_dir = Path("data/samples")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"\nScanning {len(pdfs)} PDFs...")
    
    # Phase 1: Process all PDFs (skip heavy OCR ones)
    print("\n" + "="*60)
    print("Phase 1: PDF Processing")
    print("="*60)
    
    results = []
    skip_ocr_heavy = ["19700022229.pdf", "19730018163.pdf", "ina219.pdf"]
    
    for pdf in pdfs:
        if pdf.name in skip_ocr_heavy:
            print(f"Skipping {pdf.name} (OCR-heavy, slow on CPU)")
            continue
        
        try:
            start = time.time()
            page_count = native = ocr = chars = 0
            for pages in iter_processed_pages(str(pdf), dpi=150):
                page_count += len(pages)
                native += sum(1 for p in pages if p.text_source == 'native')
                ocr += sum(1 for p in pages if p.text_source == 'ocr')
                chars += sum(p.char_count for p in pages)
            duration = time.time() - start
            
            results.append({
                'name': pdf.name,
                'pages': page_count,
                'native': native,
                'ocr': ocr,
                'chars': chars,
                'time': duration
            })
            
            print(f"{pdf.name[:50]:50s} | {page_count:3d}p | {native:3d} native | {chars:7,d} chars | {duration:.1f}s")
            
        except Exception as e:
            print(f"{pdf.name[:50]:50s} | ERROR: {str(e)[:30]}")
    
    # Stats
    total_pages = sum(r['pages'] for r in results)
    total_chars = sum(r['chars'] for r in results)
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print("\n" + "="*60)
    print("Phase 1 Summary")
    print("="*60)
    print(f"PDFs processed: {len(results)}/{len(pdfs)}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per PDF: {avg_time:.1f}s")
    print(f"Average time per page: {total_time/total_pages:.2f}s" if total_pages > 0 else "")


if __name__ == "__main__":
    main()
