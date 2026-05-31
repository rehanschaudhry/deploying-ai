"""
Step 6: The chunker.

Reads a parsed JSON file (output of Steps 3, 4, or 5), splits its content
into chunks using LangChain's RecursiveCharacterTextSplitter, preserves
source metadata, and writes a chunks JSON file.

For notebooks and PDFs, we chunk by cell/page (Option B): each cell or
page becomes one or more chunks. Cells/pages smaller than MIN_UNIT_CHARS
get merged with the next one before splitting.

For text files (.py and .md), the whole file is one input unit.

Usage:
    python scripts/chunker.py <path_to_parsed_json>

Examples:
    python scripts/chunker.py parsed_notebooks/01_1_introduction_parsed.json
    python scripts/chunker.py parsed_pdfs/05_rag_parsed.json
    python scripts/chunker.py parsed_python/05_src__course_chat__app_py_parsed.json
"""

import json
import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Configuration -------------------------------------------------------------
CHUNK_SIZE     = 2000   # characters per chunk (roughly 500 tokens)
CHUNK_OVERLAP  = 100    # characters of overlap between adjacent chunks
MIN_UNIT_CHARS = 100    # minimum cell/page size before we merge with the next

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR  = PROJECT_DIR / "chunks"


# --- The splitter ---------------------------------------------------------------
# Built once, reused for every input. The splitter is stateless — calling
# .split_text() on it doesn't change anything about the splitter itself.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,            # count characters, not tokens
    separators=["\n\n", "\n", " ", ""],
)


# --- Helpers --------------------------------------------------------------------
def merge_short_units(units: list[dict], content_key: str) -> list[dict]:
    """
    Merge any unit whose content is shorter than MIN_UNIT_CHARS with the
    next unit. The merged unit keeps the metadata of the FIRST unit (so
    we can still cite back to where it started).

    `units` is a list of dicts (cells or pages); `content_key` is the
    name of the field that holds the actual text ("content" in both
    notebook and PDF parser output).
    """
    if not units:
        return []

    merged = []
    buffer = None  # the "carry-over" unit we haven't pushed yet

    for unit in units:
        if buffer is None:
            buffer = dict(unit)  # copy so we don't mutate the input
            continue

        if len(buffer[content_key]) < MIN_UNIT_CHARS:
            # Buffer is too short — merge this unit into it.
            buffer[content_key] = buffer[content_key] + "\n\n" + unit[content_key]
        else:
            # Buffer is long enough on its own — push it and start a new one.
            merged.append(buffer)
            buffer = dict(unit)

    # Push the final buffer.
    if buffer is not None:
        merged.append(buffer)

    return merged


def make_chunk(content: str, chunk_index: int, source_filename: str,
               file_type: str, extra_metadata: dict) -> dict:
    """
    Build one chunk dictionary.

    chunk_id is constructed from the source filename + position, so it's
    globally unique across the whole corpus.
    """
    safe_name = source_filename.replace("/", "__").replace(".", "_")
    return {
        "chunk_id":   f"{safe_name}_chunk_{chunk_index}",
        "content":    content,
        "char_count": len(content),
        "chunk_index": chunk_index,
        "source_metadata": {
            "source_file": source_filename,
            "file_type":   file_type,
            **extra_metadata,    # unpacks page_number, cell_index, etc.
        },
    }


# --- Chunking dispatch ----------------------------------------------------------
def chunk_notebook(parsed: list[dict], source_filename: str) -> list[dict]:
    """Notebooks come in as a list of cell dicts."""
    cells = merge_short_units(parsed, content_key="content")
    chunks = []
    for cell in cells:
        cell_chunks = splitter.split_text(cell["content"])
        for piece in cell_chunks:
            chunks.append(make_chunk(
                content=piece,
                chunk_index=len(chunks),
                source_filename=source_filename,
                file_type="notebook",
                extra_metadata={
                    "cell_index": cell["cell_index"],
                    "cell_type":  cell["cell_type"],
                },
            ))
    return chunks


def chunk_pdf(parsed: list[dict], source_filename: str) -> list[dict]:
    """PDFs come in as a list of page dicts."""
    pages = merge_short_units(parsed, content_key="content")
    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["content"])
        for piece in page_chunks:
            chunks.append(make_chunk(
                content=piece,
                chunk_index=len(chunks),
                source_filename=source_filename,
                file_type="pdf",
                extra_metadata={"page_number": page["page_number"]},
            ))
    return chunks


def chunk_text_file(parsed: dict, source_filename: str) -> list[dict]:
    """Text files (.py and .md) come in as a single record dict."""
    file_type = parsed.get("file_type", "unknown")
    pieces = splitter.split_text(parsed["content"])
    return [
        make_chunk(
            content=piece,
            chunk_index=i,
            source_filename=source_filename,
            file_type=file_type,
            extra_metadata={},
        )
        for i, piece in enumerate(pieces)
    ]


# --- Main ----------------------------------------------------------------------
def detect_and_chunk(parsed) -> tuple[list[dict], str]:
    """
    Detect what shape the parsed JSON is and dispatch to the right chunker.

    Returns (chunks, source_filename).
    """
    # Text-file parser output is a dict; notebook/PDF parsers output a list.
    if isinstance(parsed, dict):
        return (
            chunk_text_file(parsed, parsed["filename"]),
            parsed["filename"],
        )

    if isinstance(parsed, list) and len(parsed) > 0:
        first = parsed[0]
        if "cell_index" in first:
            # Notebooks have cell_index — we kept that as a metadata field.
            # Note: source filename isn't in the parsed JSON; we infer from CLI arg.
            return chunk_notebook(parsed, source_filename=None), None  # filled below
        if "page_number" in first:
            return chunk_pdf(parsed, source_filename=None), None  # filled below

    raise ValueError("Could not detect parsed file shape.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/chunker.py <path_to_parsed_json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        sys.exit(1)

    parsed = json.loads(input_path.read_text(encoding="utf-8"))

    # The text-file parser stores filename inside the JSON. The notebook
    # and PDF parsers don't (they were written before we needed it).
    # We infer the source filename from the input file path's stem.
    if isinstance(parsed, dict):
        source_filename = parsed["filename"]
        chunks = chunk_text_file(parsed, source_filename)
    elif isinstance(parsed, list) and len(parsed) > 0:
        # Derive a source filename from the input path's stem, stripping
        # the "_parsed" suffix our parsers add.
        stem = input_path.stem.removesuffix("_parsed")
        source_filename = stem
        first = parsed[0]
        if "cell_index" in first:
            chunks = chunk_notebook(parsed, source_filename)
        elif "page_number" in first:
            chunks = chunk_pdf(parsed, source_filename)
        else:
            print("ERROR: parsed JSON shape not recognized.")
            sys.exit(1)
    else:
        print("ERROR: parsed JSON is empty or malformed.")
        sys.exit(1)

    # --- Summary --------------------------------------------------------------
    print(f"Chunked: {source_filename}")
    print(f"  total chunks: {len(chunks)}")
    if chunks:
        char_counts = [c["char_count"] for c in chunks]
        print(f"  chunk sizes: min={min(char_counts)}, "
              f"max={max(char_counts)}, "
              f"avg={sum(char_counts) // len(char_counts)}")
        first_chunk = chunks[0]
        preview = first_chunk["content"][:200]
        print(f"\nFirst chunk ({first_chunk['chunk_id']}):")
        print("  " + preview.replace("\n", "\n  "))

    # --- Save ----------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_name = source_filename.replace("/", "__").replace(".", "_")
    output_path = OUTPUT_DIR / f"{safe_name}_chunks.json"
    output_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()