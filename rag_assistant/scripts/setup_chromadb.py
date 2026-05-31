"""
Step 8: ChromaDB setup.

Creates a ChromaDB persistent client, builds a collection, loads chunks
from one parsed-and-chunked notebook, and verifies retrieval with a
test query.

Scope: ONE file's chunks. Bulk loading happens in Step 9.

Usage:
    python scripts/setup_chromadb.py <path_to_chunks_json>

Example:
    python scripts/setup_chromadb.py chunks/01_1_introduction_chunks.json
"""

import json
import sys
from pathlib import Path
import chromadb


# --- Configuration -------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Where ChromaDB stores its data on disk. Already in .gitignore.
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# Name of the collection holding our course content.
COLLECTION_NAME = "course_content"

# The query we'll use to verify retrieval is working.
TEST_QUERY = "What is an API?"
TOP_K = 3


# --- ChromaDB setup ------------------------------------------------------------
# PersistentClient stores data in the given folder; survives across script runs.
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# get_or_create_collection is idempotent — safe to run multiple times.
# Using ChromaDB's default embedding function (all-MiniLM-L6-v2, runs locally).
collection = client.get_or_create_collection(name=COLLECTION_NAME)


# --- Helpers --------------------------------------------------------------------
def flatten_metadata(source_metadata: dict) -> dict:
    """
    ChromaDB metadata must be flat (no nested dicts) and contain only
    primitive types (str, int, float, bool). We flatten our nested
    source_metadata into a single-level dict.
    """
    flat = {}
    for k, v in source_metadata.items():
        # ChromaDB rejects None; convert to empty string.
        if v is None:
            flat[k] = ""
        else:
            flat[k] = v
    return flat


def load_chunks_into_collection(chunks: list[dict]) -> int:
    """
    Add a batch of chunks to the collection. Returns the number added.
    Skips chunks whose IDs are already in the collection (idempotent).
    """
    existing_ids = set(collection.get(ids=[c["chunk_id"] for c in chunks])["ids"])

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not new_chunks:
        print(f"  All {len(chunks)} chunks already in collection; nothing to add.")
        return 0

    collection.add(
        ids=[c["chunk_id"] for c in new_chunks],
        documents=[c["content"] for c in new_chunks],
        metadatas=[flatten_metadata(c["source_metadata"]) for c in new_chunks],
    )
    print(f"  Added {len(new_chunks)} new chunks "
          f"(skipped {len(chunks) - len(new_chunks)} already present).")
    return len(new_chunks)


def run_test_query(query: str, top_k: int):
    """Run a query against the collection and display the top results."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    # ChromaDB returns nested lists because query_texts can be a list of queries.
    # We only sent one query, so we always look at index [0].
    ids       = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nTop {len(ids)} results for query: {query!r}")
    print("=" * 70)
    for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1):
        preview = doc[:150].replace("\n", " ")
        print(f"\n[{rank}] {cid}")
        print(f"    distance: {dist:.4f}")
        print(f"    source: {meta.get('source_file')} "
              f"({meta.get('file_type')})")
        if "cell_index" in meta:
            print(f"    cell: {meta['cell_index']} ({meta.get('cell_type')})")
        elif "page_number" in meta:
            print(f"    page: {meta['page_number']}")
        print(f"    preview: {preview}...")


# --- Main ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/setup_chromadb.py <path_to_chunks_json>")
        sys.exit(1)

    chunks_path = Path(sys.argv[1])
    if not chunks_path.exists():
        print(f"ERROR: file not found: {chunks_path}")
        sys.exit(1)

    print(f"Loading chunks from: {chunks_path}")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"  {len(chunks)} chunks loaded from JSON.")

    print(f"\nCollection state before:")
    print(f"  total documents in collection: {collection.count()}")

    print(f"\nAdding chunks to collection '{COLLECTION_NAME}'...")
    added = load_chunks_into_collection(chunks)

    print(f"\nCollection state after:")
    print(f"  total documents in collection: {collection.count()}")

    # Run the test query regardless of whether we added new chunks.
    run_test_query(TEST_QUERY, TOP_K)


if __name__ == "__main__":
    main()