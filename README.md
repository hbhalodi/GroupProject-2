# Golu: CLI AI Coding Assistant

Golu is a command-line coding agent. Give it a task in plain English and it works through it on its own: it explores your codebase, reads and edits files, runs commands and tests, and looks up documentation, then keeps going until the job is done.

```
golu ❯ fix the failing test in utils.py
⏺ list_directory(.) · filesystem
⏺ read_text_file(utils.py) · filesystem
⏺ write_file(utils.py) · filesystem      ← asks first in confirm mode
⏺ run_shell(python -m pytest) · local    ← asks first in confirm mode
● Fixed the bug in utils.py: add was subtracting. All tests pass.
```

## Features

| Requirement | Where |
|---|---|
| Agentic loop (reason → act → observe → repeat) | `golu/agent.py` |
| Tool calling: files, shell, code search, docs | `golu/tools/`, MCP servers |
| MCP client that loads tools dynamically from several servers | `golu/mcp_client.py`, `mcp_config.json` |
| Filesystem MCP server (`@modelcontextprotocol/server-filesystem`) | `mcp_config.json` |
| External resource MCP server (Context7 docs; Tavily optional) | `mcp_config.json` |
| Custom RAG MCP server with an advanced technique (Fusion Retrieval) | `rag_server/` |
| Provider abstraction: Ollama + Groq (OpenAI/Anthropic optional) | `golu/providers.py` |
| CLI: streaming, visible tool calls, confirm/auto modes | `golu/ui.py`, `golu/cli.py` |

## Setup

### 1. Prerequisites

- **Python 3.11+**
- **Node.js 18+** (the filesystem MCP server runs with `npx`)
- **Ollama**: https://ollama.com/download

```bash
ollama pull qwen2.5-coder:7b     # chat model with tool calling
ollama pull nomic-embed-text     # embeddings for the RAG server
```

### 2. Install

```bash
git clone https://github.com/hbhalodi/GroupProject-2.git
cd GroupProject-2
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Edit `.env` and add your `GROQ_API_KEY` (free at https://console.groq.com/keys) if you want the cloud model.

### 3. Build the documentation index (once)

Download the LangChain docs (only the folder we need):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/langchain-ai/docs rag_server/data/langchain-docs
git -C rag_server/data/langchain-docs sparse-checkout set src/oss/langchain
```

Chunk, embed and store them in Chroma (a few minutes the first time; Ollama must be running):

```bash
python -m rag_server.ingest
```

The index is saved in `rag_server/chroma_db/` and reused by every later session. Re-run with `--reset` only if you change the chunking settings.

### 4. Run

```bash
python -m golu                                   # work on the current folder
python -m golu --project path/to/some/repo       # work on another folder
python -m golu --provider groq                   # use Groq instead of Ollama
python -m golu --mode auto                       # don't ask before edits/commands
python -m golu -p "add docstrings to utils.py"   # one task, then exit
```

## Using Golu

**Modes**
- `confirm` (default): Golu asks before tools that change things (`write_file`, `edit_file`, `create_directory`, `move_file`, `run_shell`). Answer `y`, `n`, or `a` (always allow this tool for the session).
- `auto`: every tool runs without asking.

**Commands**

| Command | What it does |
|---|---|
| `/help` | list commands |
| `/tools` | show every loaded tool and which server it came from |
| `/mode confirm` / `/mode auto` | switch confirmation mode |
| `/model groq llama-3.3-70b-versatile` | switch provider/model mid-session |
| `/clear` | start a fresh conversation |
| `/exit` or Ctrl+D | quit |

Press **Ctrl+C** while Golu is working to stop the current task.

## How it works

```
User ─▶ CLI (golu/cli.py, golu/ui.py)
          │
          ▼
     Agentic loop (golu/agent.py) ◀──▶ LLM via provider abstraction (golu/providers.py)
          │ tool calls
          ▼
     Tool registry (golu/tools/registry.py)
          ├── local tools: run_shell, search_code
          └── MCP client (golu/mcp_client.py)
                ├── filesystem server   (npx, stdio)
                ├── golu-docs RAG server (python, stdio) ── Chroma + BM25
                └── context7           (HTTP)
```

1. The user's task is added to the conversation history.
2. The LLM gets the history plus the JSON schemas of every tool (local + all MCP servers).
3. If it answers with tool calls, Golu shows each one, asks for confirmation when needed, executes it, and adds the result to the history.
4. Repeat until the LLM replies without tool calls (the final answer), or the step limit is reached.

Full diagrams: [docs/architecture.md](docs/architecture.md). Changes since the planning phase: [docs/design-changes.md](docs/design-changes.md).

### MCP servers (`mcp_config.json`)

| Server | Transport | Tools | Notes |
|---|---|---|---|
| `filesystem` | stdio (`npx`) | read/write/edit/list/search files | limited to the project folder |
| `golu-docs` | stdio (Python) | `search_docs` | our RAG server |
| `context7` | HTTP | library docs lookup | works without a key; `CONTEXT7_API_KEY` raises the rate limit |
| `tavily` | HTTP | web search | disabled by default; set `TAVILY_API_KEY` and `"enabled": true` |

Add a server by adding an entry; its tools load automatically at startup. `${PROJECT_ROOT}`, `${GOLU_HOME}`, `${PYTHON}` and any environment variable can be used in values. A server that fails to start is shown in red in the banner and skipped; the others still work. Server logs are in `.golu/logs/`.

### RAG server and Fusion Retrieval

`python -m rag_server.ingest` cleans the MDX docs (drops JSX and JavaScript-only blocks), splits them by heading into ~1,200-character chunks without breaking code blocks, prepends a contextual header (document title + section) to each chunk, embeds them with `nomic-embed-text`, and stores them in Chroma.

At query time `search_docs` uses **Fusion Retrieval** ([RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)): it takes candidates from vector search and from a BM25 keyword index, normalizes both scores, and ranks by `alpha * vector + (1 - alpha) * bm25`. Vector search understands meaning; BM25 catches exact identifiers like `bind_tools` that embeddings often miss.

Compare it with the vector-only baseline:

```bash
python -m rag_server.evaluate --alphas 0.3 0.5 0.7 --show-misses
```

### Comparing LLMs

Run the same task with different models and log the numbers (steps, tool calls, errors, tokens, time):

```bash
python -m golu --provider ollama --mode auto -p "<task>" --stats-file results.jsonl
python -m golu --provider groq   --mode auto -p "<task>" --stats-file results.jsonl
```

## Project structure

```
golu/
  cli.py            entry point, REPL, slash commands
  ui.py             Rich rendering: banner, streaming, tool panels, confirmations
  agent.py          agentic loop
  prompts.py        system prompt
  providers.py      Ollama / Groq / OpenAI / Anthropic
  mcp_client.py     MCP client (official SDK, stdio + HTTP)
  config.py         settings and mcp_config.json loading
  tools/
    registry.py     unified tool registry
    local_tools.py  run_shell, search_code
rag_server/
  server.py         MCP server exposing search_docs
  ingest.py         one-time indexing
  chunking.py       MDX cleaning + heading-aware chunking
  bm25.py           keyword index
  retriever.py      vector baseline + Fusion Retrieval
  store.py          Chroma + Ollama embeddings
  evaluate.py       Hit@k / MRR comparison
  eval_questions.json
tests/              pytest suite (no LLM or network needed)
docs/               architecture, plan, diagrams
mcp_config.json     MCP servers
```

## Tests

```bash
pytest
```

The tests use a scripted fake model and small local MCP servers, so they need no API keys, Ollama, or Node.

## Troubleshooting

- **The filesystem server shows red in the banner:** check that `node --version` works and that `npx -y @modelcontextprotocol/server-filesystem .` starts. Details are in `.golu/logs/mcp-filesystem.log`.
- **`search_docs` says the index is empty:** run `python -m rag_server.ingest` (Ollama must be running).
- **The local model replies with JSON text instead of calling tools:** use a model with tool support (`qwen2.5-coder:7b`, `llama3.1:8b`, `qwen3:8b`), or try `--provider groq`.
- **Context7 errors / rate limits:** add a free `CONTEXT7_API_KEY` to `.env`.

## Team

| Member | Area |
|---|---|
| Member A | agentic loop, prompts, tool registry |
| Member B | providers, CLI/UI, LLM comparison |
| Member C | RAG server, Fusion Retrieval, RAG evaluation |
| Member D | MCP client and servers, local tools, setup docs |

Project plan: [docs/project-plan.md](docs/project-plan.md)
