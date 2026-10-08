"""Golu entry point: parse flags, connect MCP servers, run the REPL.

Owner: Member B (REPL, slash commands) with Member D (startup wiring)

Usage:
    python -m golu                                  # interactive, confirm mode
    python -m golu --provider groq --mode auto      # cloud model, no confirmations
    python -m golu --project ../some-repo           # work on another folder
    python -m golu -p "add type hints to utils.py"  # one task, then exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.table import Table

from golu import __version__
from golu.agent import Agent
from golu.config import GOLU_HOME, Settings, load_server_configs
from golu.mcp_client import MCPManager
from golu.prompts import build_system_prompt
from golu.providers import SUPPORTED_PROVIDERS, ProviderError, create_chat_model
from golu.tools.local_tools import register_local_tools
from golu.tools.registry import ToolRegistry
from golu.ui import ACCENT, ConsoleUI

HELP = """\
[bold]Commands[/]
  /help                      show this help
  /tools                     list loaded tools by server
  /mode confirm|auto         ask before edits/commands, or run everything
  /model <provider> [model]  switch model, e.g. /model groq llama-3.3-70b-versatile
  /clear                     start a fresh conversation
  /exit                      quit (or Ctrl+D)

Press Ctrl+C while Golu is working to stop the current task."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="golu", description="Golu: autonomous CLI coding assistant")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="LLM provider")
    parser.add_argument("--model", help="model name (default depends on provider)")
    parser.add_argument("--mode", choices=("confirm", "auto"), help="confirmation mode")
    parser.add_argument("--project", type=Path, help="project folder to work in (default: current folder)")
    parser.add_argument("--config", type=Path, help="path to mcp_config.json")
    parser.add_argument("--max-steps", type=int, help="max LLM steps per task")
    parser.add_argument("-p", "--prompt", help="run a single task non-interactively and exit")
    parser.add_argument("--stats-file", type=Path,
                        help="append run stats as JSON lines (for the LLM comparison)")
    parser.add_argument("--version", action="version", version=f"golu {__version__}")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if args.provider:
        settings.provider = args.provider
    if args.model:
        settings.model = args.model
    if args.mode:
        settings.mode = args.mode
    if args.project:
        settings.project_root = args.project.resolve()
    if args.config:
        settings.mcp_config_path = args.config
    if args.max_steps:
        settings.max_steps = args.max_steps
    return settings


class GoluApp:
    """Holds the session state: settings, tools, MCP connections, agent."""

    def __init__(self, settings: Settings, ui: ConsoleUI) -> None:
        self.settings = settings
        self.ui = ui
        self.registry = ToolRegistry()
        self.mcp = MCPManager(settings.log_dir)
        self.agent: Agent | None = None

    async def start(self) -> None:
        s = self.settings
        if not s.project_root.is_dir():
            raise SystemExit(f"Project folder not found: {s.project_root}")

        register_local_tools(self.registry, s.project_root, s)
        self.ui.project_root = str(s.project_root.resolve())

        with self.ui.console.status(f"[{ACCENT}]connecting to MCP servers…[/]", spinner="dots"):
            await self.mcp.connect_all(load_server_configs(s), self.registry)

        model = create_chat_model(s)
        tool_sources: dict[str, list[str]] = {}
        for tool in self.registry.all():
            tool_sources.setdefault(tool.source, []).append(tool.name)

        self.agent = Agent(
            model=model,
            registry=self.registry,
            ui=self.ui,
            system_prompt=build_system_prompt(s.project_root, tool_sources),
            mode=s.mode,
            max_steps=s.max_steps,
            max_tool_output_chars=s.max_tool_output_chars,
        )

        servers = [("local", True, len(tool_sources.get("local", [])), None)] + [
            (st.name, st.connected, len(st.tool_names), st.error) for st in self.mcp.statuses
        ]
        self.ui.banner(s.provider, s.resolved_model(), s.mode, str(s.project_root), servers)

    async def run_task(self, task: str, stats_file: Path | None = None) -> None:
        assert self.agent is not None
        try:
            _, stats = await self.agent.run(task)
            self.ui.show_stats(stats)
            if stats_file:
                record = {"time": datetime.now().isoformat(timespec="seconds"),
                          "provider": self.settings.provider,
                          "model": self.settings.resolved_model(),
                          "task": task, **asdict(stats)}
                with stats_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        except asyncio.CancelledError:
            # Ctrl+C: stop this task but keep the session alive.
            task_obj = asyncio.current_task()
            if task_obj is not None:
                task_obj.uncancel()
            self.agent.repair_history()
            self.ui.on_notice("Task interrupted.")
        except Exception as exc:  # noqa: BLE001 - show provider/network errors, keep REPL alive
            self.agent.repair_history()
            self.ui.error(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------- commands
    def handle_command(self, line: str) -> bool:
        """Run a slash command. Returns False when the user wants to exit."""
        assert self.agent is not None
        parts = line.split()
        cmd, rest = parts[0].lower(), parts[1:]
        console = self.ui.console

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            console.print(HELP)
        elif cmd == "/clear":
            self.agent.reset()
            console.print("[dim]Conversation cleared.[/]")
        elif cmd == "/mode":
            if rest and rest[0] in ("confirm", "auto"):
                self.agent.mode = self.settings.mode = rest[0]
                console.print(f"Mode set to [bold]{rest[0]}[/].")
            else:
                console.print(f"Current mode: [bold]{self.agent.mode}[/]. Use /mode confirm or /mode auto.")
        elif cmd == "/model":
            if not rest:
                console.print(f"Current model: [bold]{self.settings.resolved_model()}[/] via {self.settings.provider}")
            else:
                old = (self.settings.provider, self.settings.model)
                self.settings.provider = rest[0]
                self.settings.model = rest[1] if len(rest) > 1 else None
                try:
                    self.agent.set_model(create_chat_model(self.settings))
                    console.print(f"Switched to [bold]{self.settings.resolved_model()}[/] via {self.settings.provider}.")
                except ProviderError as exc:
                    self.settings.provider, self.settings.model = old
                    self.ui.error(str(exc))
        elif cmd == "/tools":
            table = Table(title="Loaded tools", show_lines=False)
            table.add_column("source")
            table.add_column("tool")
            table.add_column("asks first", justify="center")
            for tool in sorted(self.registry.all(), key=lambda t: (t.source != "local", t.source, t.name)):
                table.add_row(tool.source, tool.name, "✓" if tool.requires_confirmation else "")
            console.print(table)
        else:
            console.print(f"Unknown command {cmd}. Type /help.")
        return True

    async def repl(self) -> None:
        history_file = GOLU_HOME / ".golu" / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        session: PromptSession = PromptSession(history=FileHistory(str(history_file)))

        while True:
            try:
                line = (await session.prompt_async([("fg:#ff8c42 bold", "golu ❯ ")])).strip()
            except KeyboardInterrupt:
                continue  # Ctrl+C at the prompt just clears the line
            except EOFError:
                break  # Ctrl+D
            if not line:
                continue
            if line.startswith("/"):
                if not self.handle_command(line):
                    break
                continue
            await self.run_task(line)

    async def close(self) -> None:
        await self.mcp.close()


async def amain(args: argparse.Namespace) -> int:
    ui = ConsoleUI()
    app = GoluApp(build_settings(args), ui)
    try:
        await app.start()
        if args.prompt:
            await app.run_task(args.prompt, args.stats_file)
        else:
            await app.repl()
    except ProviderError as exc:
        ui.error(str(exc))
        return 1
    finally:
        await app.close()
    ui.console.print("[dim]bye 👋[/]")
    return 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
