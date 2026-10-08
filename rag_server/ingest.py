"""One-time ingestion: docs -> chunks -> embeddings -> Chroma.

Owner: Member C

Run once (takes a few minutes the first time):
    python -m rag_server.ingest
Re-run with --reset after changing chunking settings or the docs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from rag_server import settings
from rag_server.chunking import load_and_chunk


def chunk_id(source: str, section: str, part: int, text: str) -> str:
    """Stable id, so re-running ingest updates instead of duplicating."""
    return hashlib.sha1(f"{source}|{section}|{part}|{text}".encode()).hexdigest()[:20]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Golu's documentation vector index")
    parser.add_argument("--source", type=Path, default=settings.DOCS_DIR, help="folder with .md/.mdx docs")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="only chunk and print stats")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        print(f"Docs folder not found: {args.source}\nSee README: 'Download the docs'.", file=sys.stderr)
        return 1

    chunks = load_and_chunk(args.source, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP,
                            exclude=settings.EXCLUDE)
    files = len({c.metadata["source"] for c in chunks})
    avg = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
    print(f"Loaded {files} files -> {len(chunks)} chunks (avg {avg:.0f} chars)")
    if args.dry_run:
        for c in chunks[:3]:
            print("-" * 60, "\n", c.embed_text[:500])
        return 0

    from rag_server.store import VectorStore

    store = VectorStore()
    if args.reset:
        store.reset()

    # Deduplicate ids (identical chunks) before inserting.
    seen: dict[str, object] = {}
    for c in chunks:
        seen.setdefault(chunk_id(c.metadata["source"], c.metadata["section"], c.metadata["part"], c.text), c)
    existing = set(store.collection.get(include=[])["ids"]) if store.count() else set()
    todo = [(i, c) for i, c in seen.items() if i not in existing]
    print(f"{len(existing)} chunks already indexed, embedding {len(todo)} new chunks with '{settings.EMBED_MODEL}'")

    start = time.perf_counter()
    for n in range(0, len(todo), args.batch_size):
        batch = todo[n:n + args.batch_size]
        store.add(
            ids=[i for i, _ in batch],
            texts=[c.embed_text for _, c in batch],
            metadatas=[c.metadata for _, c in batch],
        )
        done = min(n + args.batch_size, len(todo))
        print(f"  {done}/{len(todo)} embedded ({time.perf_counter() - start:.0f}s)", end="\r")
    print(f"\nDone. Index has {store.count()} chunks at {settings.CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
