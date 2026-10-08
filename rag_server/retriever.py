"""Retrievers: the vector-only baseline and Fusion Retrieval (our advanced RAG technique).

Owner: Member C

Fusion Retrieval (from NirDiamant/RAG_Techniques, "Fusion Retrieval"):
    1. Get candidates from vector search (semantic meaning)
       and from BM25 (exact keywords), CANDIDATE_POOL from each.
    2. Min-max normalize each score list to [0, 1] so they're comparable.
       A candidate missing from one list gets 0 for that score.
    3. combined = alpha * vector_score + (1 - alpha) * bm25_score
    4. Return the top k by combined score.

Why it helps for code docs: questions often mix concepts ("how do I give a
model tools") with exact identifiers ("bind_tools", "ToolMessage"). Vector
search handles the first, BM25 the second.

Both retrievers return the same Hit objects so evaluate.py can compare them.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_server.bm25 import BM25Index


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class VectorRetriever:
    """Baseline: plain semantic search."""

    name = "vector"

    def __init__(self, store) -> None:
        self.store = store

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return [Hit(d.id, d.text, d.metadata, s, vector_score=s)
                for d, s in self.store.query(query, k)]


class FusionRetriever:
    """Advanced: weighted fusion of vector and BM25 scores."""

    name = "fusion"

    def __init__(self, store, alpha: float = 0.5, pool: int = 40) -> None:
        self.store = store
        self.alpha = alpha
        self.pool = pool
        # BM25 is built in memory from the chunks already stored in Chroma,
        # so ingestion still happens only once.
        self.docs = store.all_docs()
        self.by_id = {d.id: d for d in self.docs}
        self.bm25 = BM25Index([d.text for d in self.docs])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        vector_raw = {d.id: s for d, s in self.store.query(query, self.pool)}
        bm25_raw = {self.docs[i].id: s for i, s in self.bm25.top_k(query, self.pool)}

        vec = _min_max(vector_raw)
        kw = _min_max(bm25_raw)
        candidates = set(vec) | set(kw)

        hits = []
        for doc_id in candidates:
            v, b = vec.get(doc_id, 0.0), kw.get(doc_id, 0.0)
            doc = self.by_id[doc_id]
            hits.append(Hit(doc_id, doc.text, doc.metadata,
                            score=self.alpha * v + (1 - self.alpha) * b,
                            vector_score=vector_raw.get(doc_id, 0.0),
                            bm25_score=bm25_raw.get(doc_id, 0.0)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


def make_retriever(store, mode: str, alpha: float, pool: int):
    if mode == "vector":
        return VectorRetriever(store)
    return FusionRetriever(store, alpha=alpha, pool=pool)
