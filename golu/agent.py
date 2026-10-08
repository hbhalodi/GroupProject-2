"""The agentic loop: reason -> act -> observe -> repeat.

Owner: Member A

One call to Agent.run(task):

    history += user task
    loop (at most max_steps times):
        stream the LLM's response (text is shown live)
        if the response has no tool calls  -> done, return the text
        for each tool call:
            show it; in confirm mode ask before state-changing tools
            execute it through the ToolRegistry (local or MCP)
            add the result to history as a ToolMessage
    (then the LLM sees the results and decides the next step)

The agent never prints anything itself. It reports events to a `ui` object
(see golu/ui.py), which keeps the loop testable and the display swappable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from golu.tools.registry import ToolRegistry, ToolResult


class AgentUI(Protocol):
    """Events the agent reports. Implemented by golu.ui.ConsoleUI (and by test fakes)."""

    def on_llm_start(self, step: int) -> None: ...
    def on_text(self, text: str) -> None: ...
    def on_llm_end(self) -> None: ...
    def on_tool_call(self, name: str, args: dict[str, Any], source: str) -> None: ...
    async def confirm(self, name: str, args: dict[str, Any]) -> bool: ...
    def on_tool_start(self, name: str) -> None: ...
    def on_tool_result(self, name: str, result: ToolResult) -> None: ...
    def on_notice(self, message: str) -> None: ...


@dataclass
class RunStats:
    """Numbers from one task, useful for the LLM comparison in the report."""

    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    rejected: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    hit_step_limit: bool = False


def _text_of(content: Any) -> str:
    """Message content is a string for most providers, a list of blocks for some."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
            if not isinstance(b, dict) or b.get("type") in (None, "text")
        )
    return str(content or "")


class Agent:
    def __init__(self, model, registry: ToolRegistry, ui: AgentUI, system_prompt: str,
                 mode: str = "confirm", max_steps: int = 25,
                 max_tool_output_chars: int = 12000) -> None:
        self.registry = registry
        self.ui = ui
        self.mode = mode
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars
        self.system_prompt = system_prompt
        self.history: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self.set_model(model)

    # ------------------------------------------------------------------ setup
    def set_model(self, model) -> None:
        """Swap the LLM (used by /model). Tool schemas are re-bound to it."""
        self.model = model
        self._bound = model.bind_tools(self.registry.openai_schemas())

    def reset(self) -> None:
        """Forget the conversation (used by /clear)."""
        self.history = [SystemMessage(content=self.system_prompt)]

    def repair_history(self) -> None:
        """After a Ctrl+C mid-task, add placeholder results for tool calls that
        never finished. Providers reject a history where a tool call has no result."""
        answered = {m.tool_call_id for m in self.history if isinstance(m, ToolMessage)}
        for message in list(self.history):
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    if call["id"] not in answered:
                        self.history.append(ToolMessage(
                            content="Cancelled by the user.", tool_call_id=call["id"],
                            name=call["name"], status="error"))
                        answered.add(call["id"])

    # ------------------------------------------------------------------- loop
    async def run(self, task: str) -> tuple[str, RunStats]:
        """Work on one user task until the model stops calling tools."""
        stats = RunStats()
        started = time.perf_counter()
        self.history.append(HumanMessage(content=task))
        final_text = ""

        for step in range(1, self.max_steps + 1):
            stats.steps = step
            ai = await self._call_llm(step, stats)
            self.history.append(ai)

            if not ai.tool_calls:
                final_text = _text_of(ai.content)
                break

            for call in ai.tool_calls:
                await self._handle_tool_call(call, stats)
        else:
            stats.hit_step_limit = True
            self.ui.on_notice(
                f"Stopped after {self.max_steps} steps. Send a follow-up message to continue."
            )

        stats.seconds = time.perf_counter() - started
        return final_text, stats

    async def _call_llm(self, step: int, stats: RunStats, allow_retry: bool = True) -> AIMessage:
        """Stream one LLM response, showing text as it arrives."""
        self.ui.on_llm_start(step)
        full = None
        try:
            async for chunk in self._bound.astream(self.history):
                text = _text_of(chunk.content)
                if text:
                    self.ui.on_text(text)
                full = chunk if full is None else full + chunk
        finally:
            self.ui.on_llm_end()

        if full is None:
            return AIMessage(content="")

        usage = getattr(full, "usage_metadata", None) or {}
        stats.input_tokens += usage.get("input_tokens", 0) or 0
        stats.output_tokens += usage.get("output_tokens", 0) or 0

        # Some local models return tool calls without ids; ToolMessages need one.
        tool_calls = []
        for call in full.tool_calls or []:
            tool_calls.append({**call, "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}"})

        message = AIMessage(content=full.content, tool_calls=tool_calls)

        # Tool calls whose arguments weren't valid JSON: tell the model so it can retry.
        invalid = getattr(full, "invalid_tool_calls", None) or []
        if invalid and not tool_calls and allow_retry:
            names = ", ".join(str(c.get("name")) for c in invalid)
            self.ui.on_notice(f"Model produced invalid tool arguments for: {names}")
            self.history.append(message)
            self.history.append(HumanMessage(
                content=f"Your last tool call ({names}) had invalid JSON arguments. "
                        "Call the tool again with valid JSON arguments."
            ))
            return await self._call_llm(step, stats, allow_retry=False)

        return message

    async def _handle_tool_call(self, call: dict[str, Any], stats: RunStats) -> None:
        name, args, call_id = call["name"], call.get("args") or {}, call["id"]
        spec = self.registry.get(name)
        self.ui.on_tool_call(name, args, spec.source if spec else "unknown")
        stats.tool_calls += 1
        stats.tools_used.append(name)

        if self.mode == "confirm" and self.registry.needs_confirmation(name):
            if not await self.ui.confirm(name, args):
                stats.rejected += 1
                result = ToolResult("The user rejected this tool call. Do not retry it.", True)
                self.ui.on_tool_result(name, result)
                self.history.append(ToolMessage(content=result.text, tool_call_id=call_id,
                                                name=name, status="error"))
                return

        self.ui.on_tool_start(name)
        result = await self.registry.execute(name, args)
        if result.is_error:
            stats.tool_errors += 1
        self.ui.on_tool_result(name, result)

        text = result.text
        if len(text) > self.max_tool_output_chars:
            text = text[: self.max_tool_output_chars] + (
                f"\n... [truncated {len(result.text) - self.max_tool_output_chars} characters]"
            )
        self.history.append(ToolMessage(content=text, tool_call_id=call_id, name=name,
                                        status="error" if result.is_error else "success"))

        # TODO(Member A): trim old tool results when history gets long, so small
        # local models don't overflow their context window in long sessions.
