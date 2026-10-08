"""Golu: a command-line AI coding assistant.

Golu takes a natural-language task, asks an LLM what to do, runs the tools the
model picks (file edits, shell commands, documentation lookups through MCP
servers), feeds the results back, and repeats until the task is done.

Package layout:
    config.py       Settings from .env / CLI flags, MCP server config loading
    providers.py    Provider abstraction: Ollama, Groq (+ optional OpenAI/Anthropic)
    mcp_client.py   MCP client: connects to every server and loads tools dynamically
    tools/          Local tools (shell, code search) and the unified tool registry
    agent.py        The agentic loop: reason -> act -> observe -> repeat
    prompts.py      System prompt
    ui.py           Rich-based terminal rendering (streaming, tool panels, confirmations)
    cli.py          Entry point: argument parsing, REPL, slash commands
"""

__version__ = "0.1.0"
