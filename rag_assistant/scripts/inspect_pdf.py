"""
Quick inspection tool for parsed PDF JSON files.
Shows a specific page or the longest page.

Usage:
    python scripts/inspect_pdf.py <parsed_json_filename> page <n>
    python scripts/inspect_pdf.py <parsed_json_filename> longest

Examples:
    python scripts/inspect_pdf.py 05_rag_parsed.json page 13
    python scripts/inspect_pdf.py 05_rag_parsed.json longest
"""

import json
import sys
from pathlib import Path


PARSED_DIR = Path(__file__).resolve().parent.parent / "parsed_pdfs"


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scripts/inspect_pdf.py <filename> page <n>")
        print("  python scripts/inspect_pdf.py <filename> longest")
        sys.exit(1)

    filename = sys.argv[1]
    mode = sys.argv[2]

    path = PARSED_DIR / filename
    if not path.exists():
        print(f"ERROR: not found: {path}")
        sys.exit(1)

    pages = json.loads(path.read_text(encoding="utf-8"))
    print(f"Loaded {len(pages)} pages from {filename}\n")

    if mode == "page":
        if len(sys.argv) != 4:
            print("Usage: python scripts/inspect_pdf.py <filename> page <n>")
            sys.exit(1)
        n = int(sys.argv[3])
        # Find the page with page_number == n
        matches = [p for p in pages if p["page_number"] == n]
        if not matches:
            print(f"No page with page_number={n}. Available range: "
                  f"{pages[0]['page_number']}-{pages[-1]['page_number']}")
            sys.exit(1)
        page = matches[0]
    elif mode == "longest":
        page = max(pages, key=lambda x: x["char_count"])
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    print(f"PAGE {page['page_number']} ({page['char_count']} chars, "
          f"{page['line_count']} lines)")
    print("-" * 60)
    print(page["content"])


if __name__ == "__main__":
    main()