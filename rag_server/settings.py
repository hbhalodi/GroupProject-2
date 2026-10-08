"""Settings for the RAG server, read from environment variables / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAG_HOME = Path(__file__).resolve().parent
load_dotenv(RAG_HOME.parent / ".env")

# Where the documentation lives (see README: "Download the docs").
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", RAG_HOME / "data" / "langchain-docs" / "src" / "oss" / "langchain"))
# Where Chroma persists the vectors. Created by ingest.py, reused by server.py.
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", RAG_HOME / "chroma_db"))
COLLECTION = os.getenv("RAG_COLLECTION", "langchain_docs")

# Embeddings run locally through Ollama (`ollama pull nomic-embed-text`).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Doc paths to skip (relative prefixes). The frontend/ pages are React/TypeScript.
EXCLUDE = tuple(e.strip() for e in os.getenv("RAG_EXCLUDE", "frontend/,changelog-js").split(",") if e.strip())

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))      # characters
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))  # characters

# "fusion" (advanced: BM25 + vector) or "vector" (baseline).
RETRIEVAL_MODE = os.getenv("RAG_MODE", "fusion")
# Weight of the vector score in fusion: 1.0 = vector only, 0.0 = BM25 only.
FUSION_ALPHA = float(os.getenv("RAG_FUSION_ALPHA", "0.5"))
# How many candidates each retriever contributes before fusing.
CANDIDATE_POOL = int(os.getenv("RAG_CANDIDATE_POOL", "40"))
