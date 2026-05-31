"""
Step 2: Repo file walker.

Walks the deploying-ai repo, classifies every file by type, and produces
an inventory. Output:
  - Console summary (counts by type, counts by folder)
  - file_inventory.json (full list of processable files with metadata)

Run from rag_assistant/ folder with venv active:
    python scripts/walk_repo.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


# --- Configuration -------------------------------------------------------------
# The root of your forked deploying-ai repo. We're inside rag_assistant/, so
# the repo root is one folder up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Where to save the inventory JSON.
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "file_inventory.json"

# Folders to skip entirely. We never recurse into these.
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".idea", ".vscode", "node_modules", ".mypy_cache",
    "mlruns", "mlartifacts", "chroma_db",
    "rag_assistant",  # don't index our own work
    "deploying-ai-env",  # another venv that lives in the repo
}

# Map file extensions to a logical "file_type" we'll use everywhere downstream.
# Anything not in this map gets classified as "other" and skipped from processing.
EXTENSION_MAP = {
    ".ipynb": "notebook",
    ".pdf":   "pdf",
    ".py":    "python",
    ".md":    "markdown",
    ".html":  "html",
    ".htm":   "html",
    ".txt":   "text",
}


# --- Walker --------------------------------------------------------------------
def classify(path: Path) -> str:
    """Return the logical file_type for a path, or 'other' if unsupported."""
    return EXTENSION_MAP.get(path.suffix.lower(), "other")


def is_excluded(path: Path, repo_root: Path) -> bool:
    """True if any part of the path (relative to repo root) is in the exclude list."""
    rel_parts = path.relative_to(repo_root).parts
    return any(part in EXCLUDED_DIRS for part in rel_parts)


def walk(repo_root: Path):
    """Walk the repo and yield (relative_path, file_type) for every file."""
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue  # skip directories themselves
        if is_excluded(path, repo_root):
            continue
        rel_path = path.relative_to(repo_root)
        yield rel_path, classify(path)


# --- Main ----------------------------------------------------------------------
def main():
    print(f"Walking repo: {REPO_ROOT}")
    print(f"Excluded dirs: {sorted(EXCLUDED_DIRS)}\n")

    inventory = []
    type_counts = Counter()
    folder_counts = defaultdict(Counter)

    for rel_path, file_type in walk(REPO_ROOT):
        type_counts[file_type] += 1
        top_folder = rel_path.parts[0] if len(rel_path.parts) > 1 else "(root)"
        folder_counts[top_folder][file_type] += 1

        # Only add processable files to the inventory we save.
        if file_type != "other":
            inventory.append({
                "relative_path": str(rel_path).replace("\\", "/"),
                "filename": rel_path.name,
                "folder": str(rel_path.parent).replace("\\", "/"),
                "file_type": file_type,
                "extension": rel_path.suffix.lower(),
            })

    # --- Print summary --------------------------------------------------------
    print("=" * 60)
    print("FILE TYPE COUNTS")
    print("=" * 60)
    for file_type in sorted(type_counts.keys()):
        print(f"  {file_type:<12} {type_counts[file_type]:>5}")
    print(f"  {'TOTAL':<12} {sum(type_counts.values()):>5}")
    print(f"  {'(processable)':<12} {len(inventory):>5}\n")

    print("=" * 60)
    print("BREAKDOWN BY TOP-LEVEL FOLDER")
    print("=" * 60)
    for folder in sorted(folder_counts.keys()):
        counts = folder_counts[folder]
        types_str = ", ".join(f"{t}={n}" for t, n in sorted(counts.items()))
        print(f"  {folder}")
        print(f"    {types_str}")

    # --- Save JSON inventory --------------------------------------------------
    OUTPUT_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"\nInventory saved: {OUTPUT_PATH}")
    print(f"Processable files: {len(inventory)}")


if __name__ == "__main__":
    main()