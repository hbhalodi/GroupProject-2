# Design Changes Since the Planning Phase

The original planning-phase architecture is kept unchanged in [architecture.md](architecture.md), as the report requires. This file records where the implementation differs and why, so the report's "updated architecture" section can be written from it.

| # | Planned (Oct 1) | Implemented | Why |
|---|---|---|---|
| 1 | MCP client via `langchain-mcp-adapters` | MCP client written directly on the official `mcp` Python SDK (`golu/mcp_client.py`) | The current `mcp` release (v2) renamed and changed APIs (e.g. `FastMCP` → `MCPServer`). Writing the client ourselves avoids version conflicts with a wrapper library and better meets the "build an MCP client" objective. |
| 2 | Tools as LangChain tool objects | Our own `ToolRegistry` + OpenAI-format JSON schemas passed to `bind_tools()` | One registry for local and MCP tools, with a per-tool `requires_confirmation` flag that drives confirm mode. LangChain is still used for the provider abstraction. |
| 3 | Connections opened as needed | One long-lived connection per server, each in its own asyncio task, connected in parallel with a 30 s timeout | Starting a new server process per tool call is slow. Per-task ownership was needed because the SDK's transports must be opened and closed in the same task, and it means an unreachable server is reported and skipped instead of hanging startup. |
| 4 | Context7 run through `npx` | Context7 connected over streamable HTTP (`https://mcp.context7.com/mcp`) | No local process needed. The client supports both transports, so a teammate can switch back to `npx -y @upstash/context7-mcp` in `mcp_config.json`. |
| 5 | Fusion Retrieval + reranking | Fusion Retrieval (BM25 + vector, min-max normalized, alpha-weighted) and contextual chunk headers; reranking not yet added | Fusion alone is the technique from RAG_Techniques; contextual headers were cheap to add at chunking time. Reranking stays optional if time allows. |
| 6 | LangChain loaders/splitters + `langchain-chroma` | Our own MDX cleaner and heading-aware chunker; `chromadb` used directly | The LangChain docs are MDX with JSX and `:::js` blocks that generic splitters keep as noise. A small custom chunker removes them, never splits code blocks, and records the section path for each chunk. |
| 7 | All LangChain docs indexed | `src/oss/langchain` only, excluding `frontend/` (React) pages | Golu is a Python assistant. 55 files → about 1,000 chunks. |

## Added beyond the plan

- `--stats-file` records steps, tool calls, errors, tokens and time per task, for the LLM comparison.
- `/model` switches provider/model mid-session; `a` at a confirmation prompt approves that tool for the rest of the session.
- `write_file` and `edit_file` calls show the new content or diff before running.
- `rag_server/evaluate.py` reports Hit@k and MRR for the vector baseline vs. fusion at several alpha values.
