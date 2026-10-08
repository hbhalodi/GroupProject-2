"""Vector store: ChromaDB (persisted on disk) + Ollama embeddings.

Owner: Member C

ingest.py writes to it once; server.py opens the same folder on every run,
so no re-embedding is needed between Golu sessions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_server import settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def default_embedder() -> Embedder:
    """Ollama embedding model (local, free). Requires `ollama pull nomic-embed-text`."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.EMBED_MODEL, base_url=settings.OLLAMA_BASE_URL)


@dataclass
class StoredDoc:
    id: str
    text: str  # chunk text with contextual header (what was embedded)
    metadata: dict


class VectorStore:
    def __init__(self, persist_dir: Path = settings.CHROMA_DIR,
                 collection: str = settings.COLLECTION, embedder: Embedder | None = None) -> None:
        # Chroma telemetry off (also keeps the MCP server's stdout clean).
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        import chromadb

        self.embedder = embedder or default_embedder()
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        # cosine distance: similarity = 1 - distance
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (used by ingest --reset)."""
        name, meta = self.collection.name, self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name, metadata=meta)

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        embeddings = self.embedder.embed_documents(texts)
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, text: str, k: int) -> list[tuple[StoredDoc, float]]:
        """Top-k by cosine similarity. Returns (doc, similarity) with similarity in [-1, 1]."""
        k = min(k, self.count())
        if k == 0:
            return []
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                         res["metadatas"][0], res["distances"][0]):
            out.append((StoredDoc(id_, doc, meta or {}), 1.0 - float(dist)))
        return out

    def all_docs(self) -> list[StoredDoc]:
        """Every stored chunk (used to build the BM25 index at server start)."""
        res = self.collection.get(include=["documents", "metadatas"])
        return [StoredDoc(i, d, m or {}) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
