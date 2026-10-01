# Project Plan — Golu

> **Status:** Initial plan (planning phase). Dates and assignments will be adjusted as we go.
>
> **Deadline:** Thursday, Oct 15, 2026 at 12:00 PM. Internal target: everything submitted by **Wed Oct 14, 11:59 PM**, leaving the morning of Oct 15 as a buffer.
>
> **Assumption to confirm:** 4 members (A–D below; replace with names). With 3 members, Member D's work is split between A and C.

## Team Roles

Each member owns one area end to end (code, tests, and the matching part of the report). Everyone reviews PRs and helps where needed.

| Member | Primary ownership | Report / deliverable ownership |
|---|---|---|
| **Member A** | Agentic loop, conversation state, system prompt, tool registry | State diagram, system description |
| **Member B** | Provider abstraction (Ollama + Groq), CLI/REPL, rendering, confirmation modes, branding | LLM comparison section, demo video |
| **Member C** | Custom RAG MCP server: ingest pipeline, Chroma, advanced RAG technique | RAG evaluation section |
| **Member D** | MCP client, filesystem + external server config, local tools (`run_shell`, `search_code`), setup | Sequence diagrams, README + setup instructions |

## Schedule (2 weeks)

### Phase 0 — Planning (Thu Oct 1 – Fri Oct 2) ✅ this submission
- Create GitHub repo; PR-based workflow on `main`
- Commit `docs/architecture.md`, `docs/architecture.png`, and this plan **before any code**
- Decide: assistant name, docs to index for RAG, external MCP server (Context7 or Tavily), advanced RAG technique
- Everyone installs Python 3.11+, Node.js (for `npx`), Ollama, pulls a tool-calling model, and creates a Groq API key

### Phase 1 — Foundations (Sat Oct 3 – Mon Oct 5)
- **A:** Package skeleton, `requirements.txt`, config; minimal agentic loop (LLM → tool call → result → LLM)
- **B:** Model factory for Ollama and Groq with `--provider` / `--model` flags; basic REPL
- **C:** Ingest script: load docs → chunk → embed → persist to Chroma (run once)
- **D:** MCP client loading tools from the filesystem server via `mcp_config.json`
- **Checkpoint (Mon Oct 5):** agent reads and edits a project file through the filesystem MCP server

### Phase 2 — Core Features (Tue Oct 6 – Thu Oct 8)
- **A:** Multi-step loop (several tool calls per turn, iteration cap, error handling, output truncation)
- **B:** Streaming output, tool-call panels, spinners, confirm vs. auto mode
- **C:** RAG MCP server (FastMCP) exposing `search_docs`, baseline vector search first, then the advanced technique (Fusion Retrieval: BM25 + vector, plus reranking)
- **D:** `run_shell` and `search_code` tools; connect the external MCP server
- **Checkpoint (Thu Oct 8):** all three MCP servers invoked in a single session; RAG server reuses the existing vector DB

### Phase 3 — Integration + Evaluation (Fri Oct 9 – Sun Oct 11)
- **A + D:** Tune tool descriptions and system prompt; fix bugs found in end-to-end testing
- **B:** Slash commands (`/model`, `/mode`, `/tools`, `/clear`, `/exit`), banner/branding; run the same 2 coding tasks on at least two LLMs and record results (success, steps, time, errors)
- **C:** RAG evaluation: small question set, baseline vs. advanced retrieval
- **Checkpoint (Sun Oct 11):** feature freeze, no new features after this

### Phase 4 — Docs, Demo, Report (Mon Oct 12 – Wed Oct 14)
- **Mon:** state diagram (A), sequence diagrams with ≥2 scenarios and ≥3 end-to-end operations (D); code cleanup and comments
- **Tue:** record demo video (two non-trivial tasks, all three MCP servers visibly invoked); README + setup instructions tested on a fresh machine
- **Wed:** PDF report (design decisions, LLM comparison, RAG insights, reflection, original + updated architecture diagram); final repo check
- **Target:** submit by Wed Oct 14, 11:59 PM

### Buffer — Thu Oct 15 (before 12:00 PM)
- Only for emergencies and final submission

## Working Agreements

- **Git:** feature branches, small PRs, one reviewer, review turnaround within the same day
- **Syncs:** short daily check-in (15 min, or async in group chat) about progress and blockers
- **Tracking:** GitHub Issues + Project board, one issue per task above
- **Secrets:** API keys in `.env` (git-ignored); commit a `.env.example`
- **Scope:** optional extensions (session persistence, diff preview, undo) only if Phase 2 finishes early

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Two-week timeline leaves little slack | Daily checkpoints; feature freeze Oct 11; baseline versions first, then improvements |
| Small local models handle tool calling poorly | Test models in Phase 0–1; develop against Groq if needed, keep Ollama for the comparison |
| MCP setup differs across laptops (Node/npx, Windows paths) | Shared `mcp_config.json` and setup instructions, verified on every machine by Oct 5 |
| Advanced RAG takes longer than expected | Baseline `search_docs` ships first so integration isn't blocked |
| Free-tier rate limits (Groq, Tavily) | Short benchmark tasks; run evaluations early (Oct 10–11) |
| Agent makes destructive edits during testing | Confirm mode by default, filesystem server scoped to the project, test in a scratch repo |
