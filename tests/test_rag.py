"""RAG pieces that run without Ollama or Chroma: cleaning, chunking, BM25, fusion."""

import asyncio
import os
import sys
from pathlib import Path

from rag_server.bm25 import BM25Index, tokenize
from rag_server.chunking import clean_mdx, load_and_chunk, pack_chunks, parse_frontmatter, split_sections
from rag_server.retriever import FusionRetriever, VectorRetriever
from rag_server.store import StoredDoc

SAMPLE = """---
title: Tools
description: Define tools
---

import Foo from '/snippets/foo.mdx';

Tools extend what [agents](/oss/agents) can do.

<Tip>
Use @[`@tool`] for simple tools.
</Tip>

## Create tools

:::python
```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    return query
```
:::

:::js
```ts
const search = tool(...)
```
:::

### Return values

Tools can return strings.
"""


def test_clean_mdx_keeps_python_drops_js_and_jsx():
    meta, body = parse_frontmatter(SAMPLE)
    assert meta["title"] == "Tools"
    cleaned = clean_mdx(body)
    assert "import Foo" not in cleaned
    assert "<Tip>" not in cleaned and "Use `@tool` for simple tools." in cleaned
    assert "Tools extend what agents can do." in cleaned
    assert "def search" in cleaned
    assert "const search" not in cleaned
    assert ":::" not in cleaned


def test_sections_and_chunk_packing():
    _, body = parse_frontmatter(SAMPLE)
    sections = split_sections(clean_mdx(body), "Tools")
    paths = [p for p, _ in sections]
    assert paths == ["Tools", "Tools > Create tools", "Tools > Create tools > Return values"]

    text = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(20))
    chunks = pack_chunks(text, size=400, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 400 for c in chunks)
    # overlap: the last paragraph of one chunk starts the next
    assert chunks[0].split("\n\n")[-1] == chunks[1].split("\n\n")[0]


def test_code_block_not_split_by_blank_lines():
    text = "Intro\n\n```python\na = 1\n\nb = 2\n```\n\nOutro"
    chunks = pack_chunks(text, size=30, overlap=0)
    assert any("a = 1\n\nb = 2" in c for c in chunks)


def test_load_and_chunk_metadata(tmp_path):
    (tmp_path / "tools.mdx").write_text(SAMPLE)
    chunks = load_and_chunk(tmp_path, size=500, overlap=50)
    assert chunks and all(c.metadata["title"] == "Tools" for c in chunks)
    assert chunks[0].embed_text.startswith("Document: Tools\nSection: ")


def test_tokenize_splits_identifiers():
    tokens = tokenize("Use ChatOllama.bind_tools now")
    assert {"chatollama", "chat", "ollama", "bind_tools", "bind", "tools"} <= set(tokens)
    assert "use" not in tokens


def test_bm25_prefers_exact_identifier():
    docs = [
        "Models can call tools after you bind them.",
        "Call bind_tools on the chat model to attach tool schemas.",
        "Streaming returns tokens as they are generated.",
    ]
    index = BM25Index(docs)
    best, _ = index.top_k("bind_tools", 1)[0]
    assert best == 1
    assert index.top_k("zebra", 3) == []


class FakeStore:
    """Vector store stand-in: 'semantic' similarity is given per query."""

    def __init__(self, docs, similarities):
        self.docs = [StoredDoc(str(i), t, {"source": f"doc{i}.mdx"}) for i, t in enumerate(docs)]
        self.similarities = similarities

    def all_docs(self):
        return self.docs

    def query(self, text, k):
        ranked = sorted(zip(self.docs, self.similarities), key=lambda p: p[1], reverse=True)
        return ranked[:k]


def test_fusion_combines_keyword_and_vector():
    docs = [
        "General overview of giving models the ability to act.",   # semantically close
        "Reference: bind_tools(tools) attaches schemas.",           # exact keyword
        "Unrelated page about deployment.",
    ]
    store = FakeStore(docs, similarities=[0.90, 0.60, 0.10])

    vector_top = VectorRetriever(store).search("bind_tools", k=1)[0]
    assert vector_top.metadata["source"] == "doc0.mdx"

    fusion = FusionRetriever(store, alpha=0.5, pool=3)
    top = fusion.search("bind_tools", k=3)
    assert top[0].metadata["source"] == "doc1.mdx"     # keyword match wins once fused
    assert top[0].bm25_score > 0 and top[0].vector_score == 0.60
    assert [h.score for h in top] == sorted((h.score for h in top), reverse=True)

    vector_only = FusionRetriever(store, alpha=1.0, pool=3).search("bind_tools", k=1)[0]
    assert vector_only.metadata["source"] == "doc0.mdx"


def test_rag_server_starts_and_reports_missing_index(tmp_path):
    """The RAG MCP server answers over stdio even before ingest has run."""
    from golu.config import ServerConfig
    from golu.mcp_client import MCPManager
    from golu.tools.registry import ToolRegistry

    async def scenario():
        registry = ToolRegistry()
        manager = MCPManager(tmp_path / "logs")
        cfg = ServerConfig(name="golu-docs", transport="stdio", command=sys.executable,
                           args=["-m", "rag_server.server"],
                           cwd=str(Path(__file__).resolve().parent.parent),
                           env={"RAG_CHROMA_DIR": str(tmp_path / "db")})
        try:
            status = await manager.connect(cfg, registry)
            assert status.connected, status.error
            assert status.tool_names == ["search_docs"]
            result = await registry.execute("search_docs", {"query": "tools"})
            assert "index" in result.text.lower()
        finally:
            await manager.close()

    asyncio.run(scenario())
