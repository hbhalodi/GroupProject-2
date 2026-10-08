"""Configuration: runtime settings and MCP server definitions.

Owner: Member A (settings) / Member D (MCP server config)

Settings come from three places, in increasing priority:
    1. Defaults in this file
    2. Environment variables (loaded from a `.env` file in the Golu folder)
    3. Command-line flags (applied in cli.py)

MCP servers are defined in `mcp_config.json`. Values like ${PROJECT_ROOT}
are substituted at load time so the same config works on every teammate's
machine.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Folder that contains the golu/ package (the repo root). Used to find
# .env, mcp_config.json and the RAG server regardless of where Golu is launched.
GOLU_HOME = Path(__file__).resolve().parent.parent

load_dotenv(GOLU_HOME / ".env")

# Default model for each provider. Override with --model or *_MODEL in .env.
DEFAULT_MODELS = {
    "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
}


@dataclass
class Settings:
    """Everything the agent needs to know at runtime."""

    provider: str = os.getenv("GOLU_PROVIDER", "ollama")
    model: str | None = os.getenv("GOLU_MODEL") or None
    # "confirm": ask before tools that change files or run commands
    # "auto":    run every tool without asking
    mode: str = os.getenv("GOLU_MODE", "confirm")
    # The codebase Golu works on. The filesystem MCP server is limited to it.
    project_root: Path = field(default_factory=Path.cwd)
    mcp_config_path: Path = GOLU_HOME / "mcp_config.json"
    # Safety limits so small models can't loop forever or flood their context.
    max_steps: int = int(os.getenv("GOLU_MAX_STEPS", "25"))
    max_tool_output_chars: int = int(os.getenv("GOLU_MAX_TOOL_OUTPUT", "12000"))
    shell_timeout_seconds: int = int(os.getenv("GOLU_SHELL_TIMEOUT", "120"))
    temperature: float = float(os.getenv("GOLU_TEMPERATURE", "0"))
    log_dir: Path = GOLU_HOME / ".golu" / "logs"

    def resolved_model(self) -> str:
        """Model name to use: explicit --model, otherwise the provider default."""
        return self.model or DEFAULT_MODELS.get(self.provider, "")


@dataclass
class ServerConfig:
    """One MCP server entry from mcp_config.json."""

    name: str
    transport: str  # "stdio" or "http"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    description: str = ""
    # Tools from this server that change state and must be confirmed in confirm mode.
    confirm_tools: list[str] = field(default_factory=list)


_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _substitute(value: Any, variables: dict[str, str]) -> Any:
    """Replace ${VAR} placeholders in strings (recursively in lists/dicts).

    Lookup order: built-in variables (PROJECT_ROOT, GOLU_HOME, PYTHON), then
    environment variables. Unknown variables become an empty string.
    """
    if isinstance(value, str):
        return _VAR_PATTERN.sub(
            lambda m: variables.get(m.group(1), os.getenv(m.group(1), "")), value
        )
    if isinstance(value, list):
        return [_substitute(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, variables) for k, v in value.items()}
    return value


def load_server_configs(settings: Settings) -> list[ServerConfig]:
    """Read mcp_config.json and return the enabled servers.

    A server is skipped when `"enabled": false`, or when it lists
    `"requires_env"` variables that are not set (e.g. a missing API key),
    so one teammate's missing key doesn't break startup for everyone.
    """
    raw = json.loads(Path(settings.mcp_config_path).read_text(encoding="utf-8"))
    variables = {
        "PROJECT_ROOT": str(Path(settings.project_root).resolve()),
        "GOLU_HOME": str(GOLU_HOME),
        "PYTHON": sys.executable,  # same interpreter/venv that runs Golu
    }

    servers: list[ServerConfig] = []
    for name, entry in raw.get("servers", {}).items():
        if not entry.get("enabled", True):
            continue
        missing = [v for v in entry.get("requires_env", []) if not os.getenv(v)]
        if missing:
            print(
                f"[golu] skipping MCP server '{name}': missing env {', '.join(missing)}",
                file=sys.stderr,
            )
            continue
        entry = _substitute(entry, variables)
        servers.append(
            ServerConfig(
                name=name,
                transport=entry.get("transport", "stdio"),
                command=entry.get("command"),
                args=entry.get("args", []),
                env=entry.get("env", {}),
                cwd=entry.get("cwd"),
                url=entry.get("url"),
                # drop headers whose ${VAR} was not set (e.g. optional API keys)
                headers={k: v for k, v in entry.get("headers", {}).items() if v},
                description=entry.get("description", ""),
                confirm_tools=entry.get("confirm_tools", []),
            )
        )
    return servers
