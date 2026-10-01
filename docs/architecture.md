# Initial Architecture — Golu

> **Status:** Initial design (planning phase). This will change during development; any changes will be documented in the final report alongside this original version.

## Overview

Golu is a command-line AI coding assistant. The user types a natural-language task, and an agentic loop repeatedly asks an LLM what to do next, executes the tools it chooses (local tools and MCP server tools), feeds the results back, and stops when the model returns a final answer.

## Architecture Diagram

![Initial architecture](architecture.png)

```mermaid
---
title: Golu — Initial Architecture
---
flowchart TB
    User([Developer in terminal])

    subgraph CLI["CLI Layer"]
        REPL["REPL / Input<br/>(prompt_toolkit)"]
        Render["Renderer<br/>streaming text, tool-call panels,<br/>spinners (Rich)"]
        Confirm["Confirmation Gate<br/>confirm mode / auto mode"]
        Commands["Slash commands<br/>/model /mode /tools /clear /exit"]
    end

    subgraph Core["Agent Core"]
        Loop["Agentic Loop<br/>reason → act → observe → repeat"]
        History["Conversation State<br/>messages + tool results"]
        Prompt["System Prompt Builder<br/>project context, tool rules"]
        Registry["Tool Registry<br/>merges local + MCP tools"]
    end

    subgraph Providers["Provider Abstraction (LangChain)"]
        Factory["Model Factory<br/>config → chat model"]
        Ollama["Ollama<br/>(local, e.g. qwen2.5-coder)"]
        Cloud["Cloud provider<br/>(Groq; optional OpenAI/Anthropic)"]
    end

    subgraph Tools["Tools"]
        Local["Local tools<br/>run_shell, search_code (grep)"]
        MCPClient["MCP Client<br/>(langchain-mcp-adapters)<br/>loads tools dynamically"]
    end

    subgraph Servers["MCP Servers"]
        FS["Filesystem Server<br/>@modelcontextprotocol/server-filesystem<br/>read / write / edit / list"]
        Ext["External Resource Server<br/>Context7 (docs) or Tavily (web search)"]
        RAG["Custom RAG Server (ours)<br/>search_docs tool"]
    end

    subgraph RAGInternals["RAG Server internals"]
        Ingest["One-time ingest script<br/>load → chunk → embed"]
        VDB[("Chroma vector DB<br/>persisted on disk")]
        Adv["Advanced technique<br/>Fusion Retrieval (BM25 + vector)<br/>+ reranking"]
        Docs[/"Library docs<br/>(e.g. LangChain)"/]
    end

    User --> REPL
    REPL --> Commands
    REPL --> Loop
    Loop --> Prompt
    Loop <--> History
    Loop <-->|messages + tool schemas /<br/>text or tool calls| Factory
    Factory --> Ollama
    Factory --> Cloud
    Loop -->|tool call| Confirm
    Confirm -->|approved| Registry
    Registry --> Local
    Registry --> MCPClient
    MCPClient <-->|stdio| FS
    MCPClient <-->|stdio / HTTP| Ext
    MCPClient <-->|stdio| RAG
    RAG --> Adv
    Adv --> VDB
    Docs --> Ingest --> VDB
    Loop -->|tokens, tool events, results| Render
    Render --> User
```

## Workflow Diagrams (initial drafts)

Drafts of the state and sequence diagrams required for the final submission. They will be revised as the implementation takes shape. The two scenarios cover four distinct end-to-end operations: reading and editing a file, running a shell command, retrieving from our RAG server, and fetching external docs.

### State diagram — agentic loop

![State diagram](diagrams/state.png)

<details><summary>Mermaid source</summary>

```mermaid
---
title: Golu — Agent State Diagram (initial)
---
stateDiagram-v2
    [*] --> Startup
    Startup: Startup<br/>load config, pick provider,<br/>connect MCP servers, load tools
    Startup --> Idle

    Idle: Idle<br/>REPL waits for input
    Idle --> SlashCommand: input starts with "/"
    SlashCommand --> Idle: /model /mode /tools /clear
    Idle --> Exit: /exit or Ctrl+D
    Idle --> CallingLLM: user task added to history

    CallingLLM: Calling LLM<br/>history + tool schemas sent,<br/>response streamed
    CallingLLM --> ParsingResponse

    ParsingResponse --> FinalAnswer: no tool calls
    ParsingResponse --> ToolSelected: tool call(s) requested
    ParsingResponse --> MaxIterations: iteration limit reached

    ToolSelected: Tool selected<br/>name + args shown in CLI
    ToolSelected --> AwaitingConfirm: confirm mode AND tool writes/executes
    ToolSelected --> ExecutingTool: auto mode OR read-only tool

    AwaitingConfirm --> ExecutingTool: user approves
    AwaitingConfirm --> ToolDenied: user rejects

    ToolDenied: Denied<br/>"user rejected" added as tool result
    ToolDenied --> CallingLLM

    ExecutingTool: Executing tool<br/>local tool or MCP call<br/>(filesystem / external / RAG)
    ExecutingTool --> ObservingResult: success
    ExecutingTool --> ObservingResult: error (error text returned)

    ObservingResult: Observing result<br/>result shown (truncated),<br/>appended to history
    ObservingResult --> ToolSelected: more tool calls in same turn
    ObservingResult --> CallingLLM: all calls done

    FinalAnswer --> Idle
    MaxIterations: Stopped<br/>warn user, show progress
    MaxIterations --> Idle
    Exit --> [*]
```
</details>

### Sequence diagram — Scenario 1: fix a bug and run tests (confirm mode)

Shows filesystem MCP read/edit, confirmation prompts, and local shell execution.

![Scenario 1](diagrams/seq-edit-and-test.png)

<details><summary>Mermaid source</summary>

```mermaid
---
title: Scenario 1 — Fix a bug and run tests (confirm mode)
---
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as CLI Interface
    participant L as Agentic Loop
    participant LLM as LLM (Ollama / Groq)
    participant T as Tools / MCP Client
    participant FS as Filesystem MCP Server
    participant SH as Local Shell Tool

    U->>CLI: "Fix the failing test in utils.py"
    CLI->>L: submit task
    L->>LLM: history + tool schemas
    LLM-->>L: tool_call read_file(utils.py)
    L->>CLI: show tool call (read-only, no prompt)
    L->>T: read_file(utils.py)
    T->>FS: MCP call read_file
    FS-->>T: file contents
    T-->>L: result
    L->>LLM: history + file contents
    LLM-->>L: tool_call edit_file(utils.py, diff)
    L->>CLI: show tool call + diff
    CLI->>U: "Allow edit_file? [y/n]"
    U-->>CLI: y
    CLI-->>L: approved
    L->>T: edit_file(...)
    T->>FS: MCP call edit_file
    FS-->>T: success
    T-->>L: result
    L->>LLM: history + edit result
    LLM-->>L: tool_call run_shell("pytest")
    CLI->>U: "Allow run_shell: pytest? [y/n]"
    U-->>CLI: y
    L->>T: run_shell("pytest")
    T->>SH: execute with timeout
    SH-->>T: "5 passed"
    T-->>L: result
    L->>LLM: history + test output
    LLM-->>L: final answer (streamed)
    L->>CLI: stream tokens
    CLI->>U: "Fixed off-by-one in utils.py, all tests pass."
```
</details>

### Sequence diagram — Scenario 2: write code using library docs (auto mode)

Shows the custom RAG server (fusion retrieval + rerank), the external MCP server, and a file write without confirmation.

![Scenario 2](diagrams/seq-docs-lookup.png)

<details><summary>Mermaid source</summary>

```mermaid
---
title: Scenario 2 — Write code using library docs (auto mode)
---
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as CLI Interface
    participant L as Agentic Loop
    participant LLM as LLM (Ollama / Groq)
    participant T as Tools / MCP Client
    participant RAG as Custom RAG MCP Server
    participant VDB as Chroma Vector DB
    participant EXT as External MCP Server (Context7)
    participant FS as Filesystem MCP Server

    U->>CLI: "Add a LangChain retriever to search.py"
    CLI->>L: submit task (auto mode)
    L->>LLM: history + tool schemas
    LLM-->>L: tool_call search_docs("LangChain retriever")
    L->>CLI: show tool call + spinner
    L->>T: search_docs(query)
    T->>RAG: MCP call search_docs
    RAG->>VDB: vector search (top k)
    RAG->>RAG: BM25 keyword search
    RAG->>RAG: fuse scores + rerank
    RAG-->>T: top chunks + sources
    T-->>L: result
    L->>LLM: history + retrieved chunks
    LLM-->>L: tool_call get_library_docs("langchain", latest API)
    L->>T: get_library_docs(...)
    T->>EXT: MCP call
    EXT-->>T: current API docs
    T-->>L: result
    L->>LLM: history + external docs
    LLM-->>L: tool_call write_file(search.py, code)
    L->>CLI: show tool call (auto mode, no prompt)
    L->>T: write_file(...)
    T->>FS: MCP call write_file
    FS-->>T: success
    T-->>L: result
    L->>LLM: history + write result
    LLM-->>L: final answer (streamed)
    L->>CLI: stream tokens
    CLI->>U: summary of changes + doc sources used
```
</details>

## Components

| Component | Responsibility | Planned tech |
|---|---|---|
| **CLI / REPL** | Read user tasks, slash commands, show streamed output, tool-call panels and status spinners, branding/banner | Python, `rich`, `prompt_toolkit` |
| **Confirmation gate** | In *confirm* mode, ask before any tool that writes files or runs commands; in *auto* mode, execute directly. Read-only tools never prompt | Part of CLI layer |
| **Agentic loop** | Send history + tool schemas to LLM; if response has tool calls, execute them and append results; loop until a final answer or a max-iteration limit | Python, LangChain message types |
| **Conversation state** | Holds the message list for the session (optional extension: save/load to disk) | In-memory list |
| **Provider abstraction** | One interface for any model; chosen by config/CLI flag (`--provider ollama --model qwen2.5-coder`) | LangChain `init_chat_model` / `ChatOllama`, `ChatGroq` |
| **Tool registry** | Combines local tools and MCP-loaded tools into one list with names, descriptions, and JSON schemas | LangChain tools |
| **Local tools** | `run_shell` (with timeout and output limit), `search_code` (ripgrep/grep) | `subprocess` |
| **MCP client** | Starts/connects to each server from `mcp_config.json`, lists their tools at startup, routes calls | `mcp`, `langchain-mcp-adapters` |
| **Filesystem MCP server** | Read, write, edit, list files, scoped to the project directory | `@modelcontextprotocol/server-filesystem` (npx) |
| **External MCP server** | Up-to-date library docs or web search | Context7 (or Tavily) |
| **Custom RAG MCP server** | Exposes `search_docs(query)`; returns top chunks with sources | Python `mcp` SDK (FastMCP) |
| **Ingest pipeline** | Run once: load docs → split into chunks → embed → persist | LangChain loaders/splitters, `nomic-embed-text` via Ollama, Chroma |
| **Advanced RAG technique** | Fusion retrieval: combine BM25 keyword scores and vector similarity, then rerank top results | From NirDiamant/RAG_Techniques (final choice confirmed in Week 1) |

## Request Flow (summary)

1. User enters a task in the REPL.
2. The agentic loop adds it to history and calls the selected LLM with all tool schemas.
3. The LLM either answers in text (streamed to the screen) or requests one or more tool calls.
4. Each tool call is displayed; write/execute tools go through the confirmation gate.
5. The registry routes the call to a local tool or, through the MCP client, to the right server.
6. Results are shown (truncated) and appended to history.
7. Steps 2–6 repeat until the LLM gives a final answer or the iteration limit is hit.

## Key Design Decisions (initial)

- **Python + LangChain** for provider abstraction and MCP tool adapters, so switching between Ollama and a cloud model is a config change.
- **Groq as the cloud provider** because it has a free tier; OpenAI/Anthropic can be added later through the same factory.
- **Chroma persisted to disk** so the ingest script runs once and every later session just opens the existing collection.
- **Shell execution as a local tool** because the filesystem server cannot run commands; it always requires confirmation in confirm mode.
- **Iteration limit and output truncation** to keep small local models from looping or overflowing their context.

## Open Questions

- Context7 vs. Tavily for the external server (or both).
- Which library's docs to index for RAG (LangChain proposed).
- Final advanced RAG technique: Fusion Retrieval is the current pick; alternatives are Reranking alone, HyDE, or Contextual Chunk Headers.
- Which Ollama model handles tool calling reliably enough on team laptops.
