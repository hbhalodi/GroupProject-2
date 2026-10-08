"""Golu's RAG MCP server: `search_docs` over the local LangChain docs index.

Owner: Member C

Golu starts this automatically (see mcp_config.json) and talks to it over
stdio. You can also test it on its own with the MCP Inspector:
    npx @modelcontextprotocol/inspector python -m rag_server.server

Important: an stdio MCP server must never print to stdout (stdout carries the
protocol messages). Use stderr for logs.
"""

from __future__ import annotations

import sys
import threading

from mcp.server.mcpserver import MCPServer

from rag_server import settings

mcp = MCPServer(
    "golu-docs",
    instructions="Search Golu's local index of the LangChain Python documentation.",
)

_retriever = None
_load_error: str | None = None
_lock = threading.Lock()


def _get_retriever():
    """Open Chroma and build the BM25 index on first use, so the server
    starts instantly and Golu's startup isn't slowed down."""
    global _retriever, _load_error
    with _lock:
        if _retriever is None and _load_error is None:
            try:
                from rag_server.retriever import make_retriever
                from rag_server.store import VectorStore

                store = VectorStore()
                if store.count() == 0:
                    _load_error = ("The documentation index is empty. "
                                   "Run `python -m rag_server.ingest` first.")
                else:
                    _retriever = make_retriever(store, settings.RETRIEVAL_MODE,
                                                settings.FUSION_ALPHA, settings.CANDIDATE_POOL)
                    print(f"[golu-docs] loaded {store.count()} chunks, mode={settings.RETRIEVAL_MODE}",
                          file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                _load_error = f"Could not open the documentation index: {type(exc).__name__}: {exc}"
        return _retriever


def format_hits(hits) -> str:
    lines = []
    for n, hit in enumerate(hits, 1):
        meta = hit.metadata
        body = hit.text.split("\n\n", 1)[-1] if hit.text.startswith("Document:") else hit.text
        lines.append(
            f"[{n}] {meta.get('section') or meta.get('title', '')}  (score {hit.score:.2f})\n"
            f"Source: {meta.get('url') or meta.get('source', '')}\n{body.strip()}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def search_docs(query: str, k: int = 5) -> str:
    """Search the local LangChain Python documentation (agents, models, tools,
    messages, streaming, structured output, RAG, MCP, middleware, memory).

    Use this BEFORE writing or changing code that uses LangChain, to get the
    current, correct API. Write the query as a short description of what you
    need, including any class or function names, e.g.
    "bind tools to a chat model" or "ToolMessage tool_call_id".

    Returns the k most relevant documentation chunks with their source URLs.
    """
    retriever = _get_retriever()
    if retriever is None:
        return _load_error or "Documentation index is not available."
    k = max(1, min(int(k), 10))
    hits = retriever.search(query, k)
    if not hits:
        return f"No documentation found for '{query}'."
    return format_hits(hits)


if __name__ == "__main__":
    mcp.run("stdio")
