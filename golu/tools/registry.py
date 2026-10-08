"""Tool registry: one list of tools, whether they're local or come from MCP servers.

Owner: Member A

The agent never cares where a tool lives. It asks the registry for the tool
schemas to send to the LLM, and asks the registry to execute a tool call.
The registry routes the call to a local Python function or to the right
MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ToolResult:
    """Outcome of running one tool."""

    text: str
    is_error: bool = False


# An executor receives the tool arguments and returns a ToolResult.
Executor = Callable[[dict[str, Any]], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    """Everything Golu knows about one tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    source: str  # "local" or the MCP server name, shown in the CLI
    executor: Executor
    requires_confirmation: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """Tool definition in the OpenAI function format.

        LangChain's bind_tools() accepts this format for every provider
        (Ollama, Groq, OpenAI, Anthropic), so we only need one schema shape.
        """
        params = dict(self.parameters or {})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or self.name,
                "parameters": params,
            },
        }


class ToolRegistry:
    """Holds all available tools and executes tool calls by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> str:
        """Add a tool. If the name is taken, prefix it with its source.

        Returns the name the tool was registered under.
        """
        name = spec.name
        if name in self._tools:
            name = f"{spec.source}__{spec.name}"
            spec.extra["original_name"] = spec.name
            spec.name = name
        self._tools[name] = spec
        return name

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def needs_confirmation(self, name: str) -> bool:
        spec = self._tools.get(name)
        return bool(spec and spec.requires_confirmation)

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Run a tool call. Errors are returned as results, not raised,
        so the LLM can read the error and try something else."""
        spec = self._tools.get(name)
        if spec is None:
            available = ", ".join(sorted(self._tools))
            return ToolResult(f"Unknown tool '{name}'. Available tools: {available}", True)
        try:
            return await spec.executor(args or {})
        except Exception as exc:  # noqa: BLE001 - surface any failure to the model
            return ToolResult(f"Tool '{name}' failed: {type(exc).__name__}: {exc}", True)
