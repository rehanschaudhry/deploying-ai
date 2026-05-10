"""
Step 3: Notebook parser.

Reads a single .ipynb file using nbformat, extracts cells (markdown + code),
strips outputs and noise, and saves a clean JSON of cells with metadata.

Usage:
    python scripts/parse_notebook.py <relative_path_from_repo_root>

Example:
    python scripts/parse_notebook.py 01_materials/labs/01_intro.ipynb
"""

import json
import sys
from pathlib import Path
import nbformat

#--- Configuration -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "parsed_notebooks"

#---Parser ----------------------------------------------------------------------
def parse_notebook(notebook_path: Path) -> dict:
    """
    Parse a Jupyter notebook into a list of clean cell dictionaries.

    Returns a list where each item represents one cell with:
      - cell_index, cell_type, content, char_count, line_count
    """
    # nbformat.read parses the .ipynb JSON and validates the structure.
    # as_version=4 normalizes to nbformat v4 (which is what Jupyter uses today).
    notebook = nbformat.read(notebook_path, as_version=4)

    parsed_cells = []
    for index, cell in enumerate(notebook.cells):
        # We only care about markdown and code cells. Ignore "raw" cells
        # (which are rare and usually contain LaTeX or HTML escapes).
        if cell.cell_type not in {"markdown", "code"}:
            continue

        # nbformat normalizes source to a string; strip trailing whitespace.
        content = cell.source.strip()

        # Skip empty cells after stripping.
        if not content:
            continue

        parsed_cells.append({
            "cell_index": index,
            "cell_type": cell.cell_type,
            "content": content,
            "char_count": len(content),
            "line_count": content.count("\n") + 1,
        })

    return parsed_cells

# --- Main ----------------------------------------------------------------------
def main():
    
    if len(sys.argv) != 2:
        print("Usage: python scripts/parse_notebook.py <relative_path_from_repo_root>")
        sys.exit(1)

    rel_path = sys.argv[1]
    notebook_path = REPO_ROOT / rel_path

    if not notebook_path.is_file():
        print(f"Error: {notebook_path} does not exist or is not a file.")
        sys.exit(1)

    if notebook_path.suffix.lower() != ".ipynb":
        print(f"Error: {notebook_path} is not a .ipynb file.")
        sys.exit(1)

    print(f"Parsing: {notebook_path}")
    cells = parse_notebook(notebook_path)

#---Summary ----------------------------------------------------------------------
    markdown_cells = [c for c in cells if c["cell_type"] == "markdown"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]

    print(f"\nTotal cells (after filtering empties): {len(cells)}")
    print(f"  markdown: {len(markdown_cells)}")
    print(f"  code: {len(code_cells)}")

    total_chars = sum(c["char_count"] for c in cells)
    print(f"  total chars: {total_chars:,}")

    # --- Save parsed output ---------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_filename = notebook_path.stem + "_parsed.json"
    output_path = OUTPUT_DIR / output_filename
    output_path.write_text(json.dumps(cells, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()    