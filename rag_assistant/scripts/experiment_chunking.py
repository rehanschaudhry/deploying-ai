"""
Step 7: First MLflow experiment.

Runs the chunker on a single file with logged parameters and metrics.
The chunking logic itself is identical to scripts/chunker.py — what's
new is the MLflow tracking around it.

Usage:
    python scripts/experiment_chunking.py <path_to_parsed_json>

Example:
    python scripts/experiment_chunking.py parsed_notebooks/01_1_introduction_parsed.json
"""

import json
import sys
from pathlib import Path
import mlflow
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Configuration -------------------------------------------------------------
CHUNK_SIZE     = 2000
CHUNK_OVERLAP  = 100
MIN_UNIT_CHARS = 100
CHUNKING_STRATEGY = "cell_aware"   # describes our Option B approach

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR  = PROJECT_DIR / "chunks"

# --- MLflow setup --------------------------------------------------------------
MLFLOW_URI        = "http://localhost:5000"
EXPERIMENT_NAME   = "chunking_experiments"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# --- The splitter (same as scripts/chunker.py) ---------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


# --- Helpers (copied from chunker.py for clarity) -----------------------------
def merge_short_units(units, content_key):
    if not units:
        return []
    merged = []
    buffer = None
    for unit in units:
        if buffer is None:
            buffer = dict(unit)
            continue
        if len(buffer[content_key]) < MIN_UNIT_CHARS:
            buffer[content_key] = buffer[content_key] + "\n\n" + unit[content_key]
        else:
            merged.append(buffer)
            buffer = dict(unit)
    if buffer is not None:
        merged.append(buffer)
    return merged


def make_chunk(content, chunk_index, source_filename, file_type, extra_metadata):
    safe_name = source_filename.replace("/", "__").replace(".", "_")
    return {
        "chunk_id": f"{safe_name}_chunk_{chunk_index}",
        "content": content,
        "char_count": len(content),
        "chunk_index": chunk_index,
        "source_metadata": {
            "source_file": source_filename,
            "file_type": file_type,
            **extra_metadata,
        },
    }


def chunk_notebook(parsed, source_filename):
    cells = merge_short_units(parsed, content_key="content")
    chunks = []
    for cell in cells:
        for piece in splitter.split_text(cell["content"]):
            chunks.append(make_chunk(
                content=piece,
                chunk_index=len(chunks),
                source_filename=source_filename,
                file_type="notebook",
                extra_metadata={
                    "cell_index": cell["cell_index"],
                    "cell_type": cell["cell_type"],
                },
            ))
    return chunks


# --- Main experiment ----------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/experiment_chunking.py <path_to_parsed_json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        sys.exit(1)

    parsed = json.loads(input_path.read_text(encoding="utf-8"))
    source_filename = input_path.stem.removesuffix("_parsed")

    # Only handles notebooks for this first experiment — keeping scope small.
    if not (isinstance(parsed, list) and parsed and "cell_index" in parsed[0]):
        print("ERROR: this experiment script handles notebooks only for now.")
        sys.exit(1)

    # --- Begin MLflow run -----------------------------------------------------
    with mlflow.start_run(run_name=f"chunk_{source_filename}_size_{CHUNK_SIZE}") as run:
        print(f"Starting MLflow run: {run.info.run_id}")
        print(f"  Source: {source_filename}")

        # Log parameters (the choices we made for this run)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_param("min_unit_chars", MIN_UNIT_CHARS)
        mlflow.log_param("chunking_strategy", CHUNKING_STRATEGY)
        mlflow.log_param("source_filename", source_filename)
        mlflow.log_param("source_file_type", "notebook")

        # Log tags (categorical labels for organizing runs)
        mlflow.set_tag("phase", "phase_1")
        mlflow.set_tag("step", "step_7")
        mlflow.set_tag("notebook", source_filename)

        # Do the actual chunking
        chunks = chunk_notebook(parsed, source_filename)

        # Compute metrics
        char_counts = [c["char_count"] for c in chunks]
        total_chunks = len(chunks)
        total_chars = sum(char_counts)
        avg_chars = total_chars / total_chunks if total_chunks else 0
        min_chars = min(char_counts) if char_counts else 0
        max_chars = max(char_counts) if char_counts else 0

        # Log metrics (the things we measured)
        mlflow.log_metric("total_chunks", total_chunks)
        mlflow.log_metric("total_chars_chunked", total_chars)
        mlflow.log_metric("avg_chunk_chars", avg_chars)
        mlflow.log_metric("min_chunk_chars", min_chars)
        mlflow.log_metric("max_chunk_chars", max_chars)

        # Save chunks to disk and log as artifact
        OUTPUT_DIR.mkdir(exist_ok=True)
        safe_name = source_filename.replace("/", "__").replace(".", "_")
        output_path = OUTPUT_DIR / f"{safe_name}_chunks.json"
        output_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(output_path))

        # Summary printout
        print(f"\n  Chunks: {total_chunks}")
        print(f"  Avg chunk size: {avg_chars:.0f} chars")
        print(f"  Range: {min_chars}-{max_chars} chars")
        print(f"\n  Run URL: {MLFLOW_URI}/#/experiments/"
              f"{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
    