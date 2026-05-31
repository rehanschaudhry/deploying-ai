"""
Step 4: PDF parser.

Reads a single .pdf file using pymupdf (fitz), extracts text page by page,
strips whitespace, skips empty pages, and saves clean JSON of pages with
metadata.

Usage:
    python scripts/parse_pdf.py <relative_path_from_repo_root>

Example:
    python scripts/parse_pdf.py 01_materials/slides/05_rag.pdf
"""

import json
import sys
from pathlib import Path
import fitz  # this is the pymupdf package, imported under the name 'fitz'


# --- Configuration -------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "parsed_pdfs"


# --- Parser --------------------------------------------------------------------
def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse a PDF into a list of page dictionaries.

    Each page dict contains:
      - page_number (1-indexed, human convention)
      - content (extracted text, stripped)
      - char_count
      - line_count
    """
    parsed_pages = []

    # fitz.open() returns a Document object. Using `with` ensures it gets
    # closed cleanly, even if something goes wrong mid-parse.
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            # page.get_text() extracts text using pymupdf's default layout
            # algorithm, which handles most multi-column layouts reasonably.
            content = page.get_text().strip()

            if not content:
                # Page is empty (or only had images). Skip it.
                continue

            parsed_pages.append({
                "page_number": page_index + 1,  # +1 because PDFs are 1-indexed for humans
                "content": content,
                "char_count": len(content),
                "line_count": content.count("\n") + 1,
            })

    return parsed_pages


# --- Main ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/parse_pdf.py <relative_path_from_repo_root>")
        sys.exit(1)

    rel_path = sys.argv[1]
    pdf_path = REPO_ROOT / rel_path

    if not pdf_path.exists():
        print(f"ERROR: file not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: not a PDF: {pdf_path}")
        sys.exit(1)

    print(f"Parsing: {pdf_path}")
    pages = parse_pdf(pdf_path)

    # --- Summary --------------------------------------------------------------
    total_chars = sum(p["char_count"] for p in pages)
    print(f"\nPages with content (after filtering empties): {len(pages)}")
    print(f"  total chars: {total_chars:,}")
    if pages:
        avg_chars = total_chars // len(pages)
        print(f"  avg chars per page: {avg_chars:,}")

    # --- Show the first page as a sanity check --------------------------------
    if pages:
        first = pages[0]
        preview = first["content"][:300]
        print(f"\nFirst page (page_number={first['page_number']}):")
        print("  " + preview.replace("\n", "\n  "))

        # Also show the last page, since failure modes often appear at boundaries.
        last = pages[-1]
        if last["page_number"] != first["page_number"]:
            preview = last["content"][:300]
            print(f"\nLast page (page_number={last['page_number']}):")
            print("  " + preview.replace("\n", "\n  "))

    # --- Save parsed output ---------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_filename = pdf_path.stem + "_parsed.json"
    output_path = OUTPUT_DIR / output_filename
    output_path.write_text(json.dumps(pages, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()