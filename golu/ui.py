"""Terminal rendering for Golu: banner, streaming text, tool-call panels, confirmations.

Owner: Member B

Implements the AgentUI events from agent.py using Rich (colors, spinners,
panels) and prompt_toolkit (input with history). Swap the look freely; the
agent only depends on the method names.
"""

from __future__ import annotations

import json
from typing import Any

from prompt_toolkit import PromptSession
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from golu.agent import RunStats
from golu.tools.registry import ToolResult

# Brand colors
ACCENT = "#ff8c42"   # saffron orange
ACCENT2 = "#ffd166"  # marigold
MUTED = "grey58"
OK = "green3"
ERR = "red3"

BANNER = r"""
  ██████╗  ██████╗ ██╗     ██╗   ██╗
 ██╔════╝ ██╔═══██╗██║     ██║   ██║
 ██║  ███╗██║   ██║██║     ██║   ██║
 ██║   ██║██║   ██║██║     ██║   ██║
 ╚██████╔╝╚██████╔╝███████╗╚██████╔╝
  ╚═════╝  ╚═════╝ ╚══════╝ ╚═════╝ """

# Which argument to show as the one-line summary for well-known tools.
_SUMMARY_ARG = {
    "run_shell": "command",
    "search_code": "pattern",
    "search_docs": "query",
}


def _short(value: Any, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", "⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ConsoleUI:
    """Rich-based implementation of the AgentUI protocol."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(highlight=False)
        self._status = None
        self._streamed_text = False
        self._session_approved: set[str] = set()  # tools approved with "always"
        self._confirm_session: PromptSession | None = None
        self.project_root: str | None = None  # set by the app; used to shorten paths

    def _rel(self, text: str) -> str:
        """Show paths inside the project as relative paths (shorter, easier to read)."""
        if not self.project_root:
            return text
        root = self.project_root.rstrip("/\\")
        for sep in ("/", "\\"):
            text = text.replace(root + sep, "")
        return text.replace(root, ".")

    # -------------------------------------------------------------- banner
    def banner(self, provider: str, model: str, mode: str, project: str,
               servers: list[tuple[str, bool, int, str | None]]) -> None:
        """Startup screen. servers = [(name, connected, tool_count, error)]."""
        art = Text()
        lines = BANNER.strip("\n").splitlines()
        for i, line in enumerate(lines):
            art.append(line + "\n", style=ACCENT if i < len(lines) / 2 else ACCENT2)
        tagline = Text("your autonomous coding sidekick", style=f"italic {MUTED}")

        info = Table.grid(padding=(0, 2))
        info.add_column(style=MUTED, justify="right")
        info.add_column()
        info.add_row("model", f"[bold]{model}[/] [dim]via {provider}[/]")
        mode_style = OK if mode == "auto" else ACCENT2
        info.add_row("mode", f"[{mode_style}]{mode}[/]")
        info.add_row("project", project)
        for name, ok, count, error in servers:
            mark = f"[{OK}]●[/]" if ok else f"[{ERR}]●[/]"
            detail = f"{count} tool{'s' if count != 1 else ''}" if ok else f"[{ERR}]{_short(error or 'failed', 60)}[/]"
            info.add_row("mcp" if name == servers[0][0] else "", f"{mark} {name} [dim]{detail}[/]")

        self.console.print(Panel(Group(art, tagline, Text(""), info), border_style=ACCENT,
                                 padding=(0, 2), expand=False))
        self.console.print(f"[{MUTED}]Type a task, or /help for commands.[/]\n")

    # ------------------------------------------------------------ LLM events
    def on_llm_start(self, step: int) -> None:
        self._streamed_text = False
        label = "thinking" if step == 1 else f"thinking (step {step})"
        self._status = self.console.status(f"[{ACCENT}]{label}…[/]", spinner="dots")
        self._status.start()

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def on_text(self, text: str) -> None:
        if not self._streamed_text:
            self._stop_status()
            self.console.print(f"[{ACCENT}]●[/] ", end="")
            self._streamed_text = True
        self.console.print(text, end="", markup=False, highlight=False)

    def on_llm_end(self) -> None:
        self._stop_status()
        if self._streamed_text:
            self.console.print()

    # ----------------------------------------------------------- tool events
    def on_tool_call(self, name: str, args: dict[str, Any], source: str) -> None:
        key = _SUMMARY_ARG.get(name)
        if key and key in args:
            summary = _short(self._rel(str(args[key])), 100)
        elif "path" in args:
            summary = _short(self._rel(str(args["path"])), 100)
        else:
            summary = ", ".join(f"{k}={_short(self._rel(str(v)), 40)}" for k, v in args.items())
        self.console.print(
            f"[{ACCENT2}]⏺[/] [bold]{name}[/]([cyan]{summary}[/]) [dim]· {source}[/]",
            highlight=False,
        )
        self._show_payload(name, args)

    def _show_payload(self, name: str, args: dict[str, Any]) -> None:
        """For file writes/edits, show what will change before it happens."""
        if name == "write_file" and isinstance(args.get("content"), str):
            path = self._rel(str(args.get("path", "")))
            lexer = Syntax.guess_lexer(path, code=args["content"])
            self.console.print(Panel(
                Syntax(args["content"], lexer, line_numbers=True, word_wrap=True),
                title=f"new content · {path}", border_style=MUTED, expand=False,
            ))
        elif name == "edit_file" and isinstance(args.get("edits"), list):
            diff = []
            for edit in args["edits"]:
                diff += [f"- {l}" for l in str(edit.get("oldText", edit.get("old_text", ""))).splitlines()]
                diff += [f"+ {l}" for l in str(edit.get("newText", edit.get("new_text", ""))).splitlines()]
            self.console.print(Panel(Syntax("\n".join(diff), "diff", word_wrap=True),
                                     title=f"edit · {self._rel(str(args.get('path', '')))}",
                                     border_style=MUTED, expand=False))

    async def confirm(self, name: str, args: dict[str, Any]) -> bool:
        if name in self._session_approved:
            return True
        if self._confirm_session is None:
            self._confirm_session = PromptSession()
        while True:
            answer = (await self._confirm_session.prompt_async(
                f"  Allow {name}? [y]es / [n]o / [a]lways this session: "
            )).strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("a", "always"):
                self._session_approved.add(name)
                return True

    def on_tool_start(self, name: str) -> None:
        self._status = self.console.status(f"[{ACCENT2}]running {name}…[/]", spinner="dots")
        self._status.start()

    def on_tool_result(self, name: str, result: ToolResult) -> None:
        self._stop_status()
        lines = result.text.strip().splitlines() or ["(no output)"]
        shown = lines[:6]
        style = ERR if result.is_error else MUTED
        for i, line in enumerate(shown):
            prefix = "  ⎿ " if i == 0 else "    "
            self.console.print(Text(prefix + _short(self._rel(line), 160), style=style))
        if len(lines) > len(shown):
            self.console.print(Text(f"    … {len(lines) - len(shown)} more lines", style="dim"))

    def on_notice(self, message: str) -> None:
        self._stop_status()
        self.console.print(f"[{ACCENT2}]![/] {message}")

    # ----------------------------------------------------------------- misc
    def show_stats(self, stats: RunStats) -> None:
        tokens = stats.input_tokens + stats.output_tokens
        parts = [f"{stats.steps} steps", f"{stats.tool_calls} tool calls", f"{stats.seconds:.1f}s"]
        if tokens:
            parts.append(f"{tokens / 1000:.1f}k tokens")
        self.console.print(f"[dim]{' · '.join(parts)}[/]\n")

    def error(self, message: str) -> None:
        self._stop_status()
        self.console.print(f"[{ERR}]✗ {message}[/]")
