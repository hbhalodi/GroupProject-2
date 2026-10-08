"""Golu's custom RAG MCP server.

Owner: Member C

Pipeline:
    ingest.py     run ONCE: load docs -> clean -> chunk -> embed -> store in Chroma
    server.py     MCP server exposing `search_docs`; opens the existing Chroma DB
    retriever.py  vector-only baseline and the advanced Fusion Retrieval technique
    bm25.py       keyword (BM25) index used by fusion retrieval
    chunking.py   MDX/Markdown cleaning and heading-aware chunking
    store.py      thin wrapper over ChromaDB + Ollama embeddings
    evaluate.py   compare baseline vs. fusion retrieval for the report
"""
