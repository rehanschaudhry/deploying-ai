"""
Step 5: Parser for plain text files (.py and .md).

Reads a single file as plain UTF-8 text and saves a JSON record with
metadata. Whole file is one unit; the chunker in Step 6 will handle
splitting.

Usage:
    python scripts/parse_text_file.py <relative_path_from_repo_root>

Examples:
    python scripts/parse_text_file.py 05_src/course_chat/app.py
    python scripts/parse_text_file.py README.md
"""

import json
import sys
from pathlib import Path


# --- Configuration -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Two separate output directories so we can browse parsed files by type.
OUTPUT_DIRS = {
    "python":   Path(__file__).resolve().parent.parent / "parsed_python",
    "markdown": Path(__file__).resolve().parent.parent / "parsed_markdown",
}

# Map extensions to logical file types — same convention as walk_repo.py.
EXTENSION_TO_TYPE = {
    ".py": "python",
    ".md": "markdown",
}


# --- Parser --------------------------------------------------------------------
def parse_text_file(file_path: Path, file_type: str) -> dict:
    """
    Read a plain-text file and return a single-record dictionary
    representing the whole file as one unit.
    """
    # Read as UTF-8. Some course Python files have Unicode characters
    # (e.g. en-dashes in comments). errors='replace' substitutes a
    # placeholder for any undecodable bytes rather than crashing.
    content = file_path.read_text(encoding="utf-8", errors="replace").strip()

    return {
        "filename":   str(file_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "file_type":  file_type,
        "content":    content,
        "char_count": len(content),
        "line_count": content.count("\n") + 1 if content else 0,
    }


# --- Main ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/parse_text_file.py <relative_path_from_repo_root>")
        sys.exit(1)

    rel_path = sys.argv[1]
    file_path = REPO_ROOT / rel_path

    if not file_path.exists():
        print(f"ERROR: file not found: {file_path}")
        sys.exit(1)

    ext = file_path.suffix.lower()
    if ext not in EXTENSION_TO_TYPE:
        print(f"ERROR: unsupported extension '{ext}'. Supported: {list(EXTENSION_TO_TYPE)}")
        sys.exit(1)

    file_type = EXTENSION_TO_TYPE[ext]

    print(f"Parsing {file_type} file: {file_path}")
    record = parse_text_file(file_path, file_type)

    # --- Summary --------------------------------------------------------------
    print(f"\nFile: {record['filename']}")
    print(f"  type: {record['file_type']}")
    print(f"  chars: {record['char_count']:,}")
    print(f"  lines: {record['line_count']:,}")

    # --- Show the first ~200 chars as a sanity check --------------------------
    if record["content"]:
        preview = record["content"][:200]
        print(f"\nFirst ~200 chars:")
        print("  " + preview.replace("\n", "\n  "))

    # --- Save parsed output ---------------------------------------------------
    output_dir = OUTPUT_DIRS[file_type]
    output_dir.mkdir(exist_ok=True)

    # Flatten the relative path into a safe filename:
    #   05_src/course_chat/app.py  ->  05_src__course_chat__app_parsed.json
    safe_name = (
        record["filename"]
        .replace("/", "__")
        .replace(".", "_")
    )
    output_path = output_dir / f"{safe_name}_parsed.json"
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()