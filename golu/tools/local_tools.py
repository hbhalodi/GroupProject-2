"""Local tools that MCP servers don't cover: running shell commands and searching code.

Owner: Member D

The filesystem MCP server can read/write files but cannot run programs, so
`run_shell` lives here. Both tools run inside the project root.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Any

from golu.tools.registry import ToolRegistry, ToolResult, ToolSpec

# Folders that are never worth searching.
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
              ".pytest_cache", "dist", "build", ".golu", "chroma_db"}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n... [{len(text) - limit} characters truncated] ...\n\n{text[-half:]}"


def make_run_shell(project_root: Path, timeout: int, output_limit: int):
    """Create the run_shell executor bound to a project directory."""

    async def run_shell(args: dict[str, Any]) -> ToolResult:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolResult("No command given.", True)
        cmd_timeout = int(args.get("timeout") or timeout)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(project_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=cmd_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(f"Command timed out after {cmd_timeout}s: {command}", True)

        text = out.decode("utf-8", errors="replace").strip() or "(no output)"
        text = _truncate(text, output_limit)
        return ToolResult(f"exit code {proc.returncode}\n{text}", proc.returncode != 0)

    return run_shell


def _python_search(root: Path, pattern: str, glob: str | None, max_results: int) -> list[str]:
    """Fallback code search when ripgrep isn't installed."""
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))
    hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if glob and not fnmatch.fnmatch(fname, glob):
                continue
            path = Path(dirpath) / fname
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            rel = path.relative_to(root)
                            hits.append(f"{rel}:{lineno}: {line.rstrip()[:200]}")
                            if len(hits) >= max_results:
                                return hits
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable file
    return hits


def make_search_code(project_root: Path, output_limit: int):
    """Create the search_code executor (ripgrep if available, else pure Python)."""

    async def search_code(args: dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return ToolResult("No pattern given.", True)
        sub = str(args.get("path") or ".")
        glob = args.get("glob") or None
        max_results = int(args.get("max_results") or 100)

        root = (project_root / sub).resolve()
        if not root.is_relative_to(project_root.resolve()):
            return ToolResult("Path must be inside the project.", True)

        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "--line-number", "--no-heading", "--color", "never",
                   "--max-count", "20", pattern, str(root)]
            if glob:
                cmd[1:1] = ["--glob", glob]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await proc.communicate()
            if proc.returncode not in (0, 1):  # 1 means "no matches"
                return ToolResult(err.decode(errors="replace"), True)
            prefix = str(project_root.resolve()) + os.sep
            lines = [l.replace(prefix, "", 1) for l in out.decode(errors="replace").splitlines()]
            hits = lines[:max_results]
        else:
            hits = await asyncio.to_thread(_python_search, root, pattern, glob, max_results)

        if not hits:
            return ToolResult(f"No matches for '{pattern}'.")
        return ToolResult(_truncate("\n".join(hits), output_limit))

    return search_code


def register_local_tools(registry: ToolRegistry, project_root: Path, settings) -> None:
    """Add run_shell and search_code to the registry."""
    registry.register(
        ToolSpec(
            name="run_shell",
            description=(
                "Run a shell command in the project root and return its output and exit "
                "code. Use it to run tests, scripts, linters, git, or package managers. "
                "Avoid interactive commands that wait for input."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout": {"type": "integer", "description": "Seconds before the command is stopped"},
                },
                "required": ["command"],
            },
            source="local",
            executor=make_run_shell(project_root, settings.shell_timeout_seconds,
                                    settings.max_tool_output_chars),
            requires_confirmation=True,
        )
    )
    registry.register(
        ToolSpec(
            name="search_code",
            description=(
                "Search the project's files for a regular expression and return matching "
                "lines as path:line: text. Use it to find where functions, classes or "
                "strings are defined or used."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex to search for"},
                    "path": {"type": "string", "description": "Sub-folder to search (default: whole project)"},
                    "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'"},
                },
                "required": ["pattern"],
            },
            source="local",
            executor=make_search_code(project_root, settings.max_tool_output_chars),
            requires_confirmation=False,
        )
    )
