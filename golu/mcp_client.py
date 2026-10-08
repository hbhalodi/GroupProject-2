"""MCP client: connects to every configured MCP server and loads their tools.

Owner: Member D

Built directly on the official MCP Python SDK (`mcp` package, v2), not on a
wrapper library, so we control the whole flow:

    1. For each server in mcp_config.json, open a connection
       (stdio for local processes, streamable HTTP for remote servers).
    2. Run the MCP handshake (initialize) and call tools/list.
    3. Wrap each MCP tool as a ToolSpec and add it to the ToolRegistry.
    4. When the agent calls a tool, send tools/call to the right server.

Connections stay open for the whole session (one process per server), which
is much faster than starting a new process per tool call. Servers connect in
parallel with a timeout, so one unreachable server can't block startup.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from golu.config import ServerConfig
from golu.tools.registry import ToolRegistry, ToolResult, ToolSpec


@dataclass
class ServerStatus:
    """Connection result for one server, shown at startup and by /tools."""

    name: str
    connected: bool
    tool_names: list[str] = field(default_factory=list)
    error: str | None = None


def _attr(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
    """Read a field that is snake_case in MCP SDK v2 and camelCase in v1."""
    if hasattr(obj, snake):
        return getattr(obj, snake)
    return getattr(obj, camel, default)


def _result_to_text(result: Any) -> ToolResult:
    """Flatten an MCP CallToolResult into plain text for the LLM."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(block.text)
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            parts.append(getattr(resource, "text", None) or f"[resource {getattr(resource, 'uri', '')}]")
        elif kind == "image":
            parts.append(f"[image: {getattr(block, 'mime_type', None) or getattr(block, 'mimeType', '')}]")
        else:
            parts.append(str(block))
    if not parts:
        structured = _attr(result, "structured_content", "structuredContent")
        if structured:
            parts.append(str(structured))
    is_error = bool(_attr(result, "is_error", "isError", False))
    return ToolResult("\n".join(parts) or "(empty result)", is_error)


def _describe(exc: BaseException) -> str:
    """Readable error text, unwrapping anyio/asyncio exception groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timed out"
    return f"{type(exc).__name__}: {exc}"


class _Connection:
    """One live server connection, owned by its own asyncio task.

    The MCP SDK's transports use anyio task groups, which must be opened and
    closed by the same task. So each connection runs in a dedicated task that
    opens the transport, signals `ready`, waits for `stop`, then closes
    everything itself. This also lets us time out a server that never answers
    without breaking the others.
    """

    def __init__(self, cfg: ServerConfig, log_dir: Path) -> None:
        self.cfg = cfg
        self.log_dir = log_dir
        self.session: ClientSession | None = None
        self.ready: asyncio.Future = asyncio.get_running_loop().create_future()
        # Mark failures as handled even if nobody awaits `ready` anymore (after a timeout).
        self.ready.add_done_callback(lambda f: f.cancelled() or f.exception())
        self.stop = asyncio.Event()
        self.task = asyncio.create_task(self._run(), name=f"mcp-{cfg.name}")

    async def _open(self, stack: AsyncExitStack) -> ClientSession:
        cfg = self.cfg
        if cfg.transport == "stdio":
            # Resolve the executable so "npx" also works on Windows (npx.cmd).
            command = shutil.which(cfg.command or "") or cfg.command
            params = StdioServerParameters(
                command=command,
                args=cfg.args,
                env={**os.environ, **cfg.env},  # pass PATH etc. through to the server
                cwd=cfg.cwd or None,
            )
            # Server stderr goes to a log file instead of cluttering the terminal.
            self.log_dir.mkdir(parents=True, exist_ok=True)
            errlog = stack.enter_context(open(self.log_dir / f"mcp-{cfg.name}.log", "a", encoding="utf-8"))
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
        elif cfg.transport == "http":
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

            kwargs: dict[str, Any] = {}
            if cfg.headers:
                kwargs["http_client"] = await stack.enter_async_context(
                    create_mcp_http_client(headers=cfg.headers)
                )
            streams = await stack.enter_async_context(streamable_http_client(cfg.url, **kwargs))
            read, write = streams[0], streams[1]
        else:
            raise ValueError(f"unknown transport '{cfg.transport}'")
        return await stack.enter_async_context(ClientSession(read, write))

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                session = await self._open(stack)
                await session.initialize()
                listing = await session.list_tools()
                self.session = session
                self.ready.set_result(listing.tools)
                await self.stop.wait()
        except BaseException as exc:  # noqa: BLE001 - includes cancellation on timeout
            if not self.ready.done():
                self.ready.set_exception(exc if isinstance(exc, Exception) else TimeoutError())
        finally:
            self.session = None

    async def close(self, timeout: float = 5.0) -> None:
        self.stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout)
        except BaseException:  # noqa: BLE001
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


class MCPManager:
    """Owns the connections to all MCP servers for one Golu session."""

    def __init__(self, log_dir: Path, connect_timeout: float = 30.0, call_timeout: float = 120.0) -> None:
        self._log_dir = log_dir
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._connections: dict[str, _Connection] = {}
        self.statuses: list[ServerStatus] = []

    async def connect(self, cfg: ServerConfig, registry: ToolRegistry) -> ServerStatus:
        """Connect to one server and register its tools. Never raises:
        a broken or unreachable server is reported and skipped."""
        return self._register(cfg, *(await self._connect_quiet(cfg)), registry)

    async def connect_all(self, configs: list[ServerConfig], registry: ToolRegistry) -> list[ServerStatus]:
        """Connect to all servers in parallel; tools are registered in config order."""
        results = await asyncio.gather(*(self._connect_quiet(cfg) for cfg in configs))
        for cfg, result in zip(configs, results):
            self._register(cfg, *result, registry)
        return self.statuses

    async def _connect_quiet(self, cfg: ServerConfig):
        """Returns (connection, tools, None) on success or (None, None, error text)."""
        conn = _Connection(cfg, self._log_dir)
        try:
            tools = await asyncio.wait_for(asyncio.shield(conn.ready), self._connect_timeout)
            return conn, tools, None
        except BaseException as exc:  # noqa: BLE001
            await conn.close(timeout=2.0)
            return None, None, _describe(exc)

    def _register(self, cfg: ServerConfig, conn, tools, error, registry: ToolRegistry) -> ServerStatus:
        """Wrap each MCP tool as a ToolSpec and add it to the registry."""
        if error is not None:
            status = ServerStatus(cfg.name, False, error=error)
        else:
            self._connections[cfg.name] = conn
            names = [
                registry.register(ToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=_attr(tool, "input_schema", "inputSchema", {}) or {},
                    source=cfg.name,
                    executor=self._make_executor(cfg.name, tool.name),
                    requires_confirmation=tool.name in cfg.confirm_tools,
                ))
                for tool in tools
            ]
            status = ServerStatus(cfg.name, True, names)
        self.statuses.append(status)
        return status

    def _make_executor(self, server: str, tool_name: str):
        async def execute(args: dict[str, Any]) -> ToolResult:
            conn = self._connections.get(server)
            if conn is None or conn.session is None:
                return ToolResult(f"MCP server '{server}' is not connected.", True)
            result = await conn.session.call_tool(tool_name, args, read_timeout_seconds=self._call_timeout)
            return _result_to_text(result)

        return execute

    async def close(self) -> None:
        """Shut down every server process / HTTP session."""
        await asyncio.gather(*(c.close() for c in self._connections.values()), return_exceptions=True)
        self._connections.clear()
