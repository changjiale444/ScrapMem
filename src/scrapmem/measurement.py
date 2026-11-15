"""File-based storage measurements with explicit source/memory accounting."""

from pathlib import Path


def tree_bytes(root):
    root = Path(root)
    if root.is_file():
        return root.stat().st_size
    if not root.is_dir():
        raise FileNotFoundError(root)
    # Count symlinked files once, at their resolved targets.
    seen = set()
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                total += path.stat().st_size
    return total


def measure_storage(store, raw_corpus=None):
    root = Path(store)
    pages = tree_bytes(root / "pages")
    index = (root / "memory.json").stat().st_size
    sources = tree_bytes(root / "sources")
    memory = pages + index
    out = {
        "scrapbook_bytes": pages,
        "index_bytes": index,
        "memory_bytes": memory,
        "source_metadata_bytes": sources,
        "all_store_bytes": tree_bytes(root),
        "scrapbook_MiB": pages / 2**20,
        "index_MiB": index / 2**20,
        "memory_MiB": memory / 2**20,
    }
    if raw_corpus:
        raw = tree_bytes(raw_corpus)
        if raw <= 0:
            raise ValueError("Raw corpus is empty")
        out.update(
            raw_corpus_bytes=raw,
            raw_corpus_MiB=raw / 2**20,
            storage_reduction=1 - memory / raw,
        )
    return out
