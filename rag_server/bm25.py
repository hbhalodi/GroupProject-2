"""A small BM25 keyword index (Okapi BM25), written out so the math is visible.

Owner: Member C

BM25 scores a document higher when it contains the query terms, especially
rare terms, while not over-rewarding long documents. It complements vector
search: exact identifiers like `bind_tools` or `RecursiveCharacterTextSplitter`
match by keyword even when embeddings miss them.

    score(D, Q) = sum over terms t in Q of
        idf(t) * tf(t, D) * (k1 + 1) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "how",
    "i", "if", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
    "when", "with", "you", "your", "use", "using",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Identifiers like bind_tools are kept whole AND split
    into parts, and camelCase is split too, so 'ChatOllama' matches 'ollama'."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(text):
        word = raw.lower()
        if word in _STOPWORDS:
            continue
        tokens.append(word)
        parts = [p.lower() for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", raw) if p]
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in _STOPWORDS)
    return tokens


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(d)) for d in documents]
        self.doc_lens = [sum(c.values()) for c in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        n = len(documents)
        df: Counter = Counter()
        for counts in self.doc_tokens:
            df.update(counts.keys())
        # BM25+ style idf that never goes negative
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for the query."""
        terms = tokenize(query)
        out: list[float] = []
        for counts, length in zip(self.doc_tokens, self.doc_lens):
            score = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                score += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            out.append(score)
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """(document index, score) of the k best matches with score > 0."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda p: p[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
