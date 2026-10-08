"""Agentic loop tests with a scripted fake model (no Ollama/Groq needed)."""

import asyncio

from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.messages.tool import tool_call_chunk

from golu.agent import Agent
from golu.tools.registry import ToolRegistry, ToolResult, ToolSpec


class ScriptedModel:
    """Returns pre-written responses in order, streamed in two chunks each.
    A response is either text or (tool_name, json_args_string)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_histories = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def astream(self, messages):
        self.seen_histories.append(list(messages))
        reply = self.responses.pop(0)
        if isinstance(reply, str):
            half = len(reply) // 2
            yield AIMessageChunk(content=reply[:half])
            yield AIMessageChunk(content=reply[half:])
        else:
            name, args = reply
            # split the JSON args across chunks, like real streaming providers do
            yield AIMessageChunk(content="", tool_call_chunks=[
                tool_call_chunk(name=name, args=args[:3], id="call_1", index=0)])
            yield AIMessageChunk(content="", tool_call_chunks=[
                tool_call_chunk(name=None, args=args[3:], id=None, index=0)])


class RecordingUI:
    def __init__(self, approve=True):
        self.approve = approve
        self.events = []

    def on_llm_start(self, step): self.events.append(("llm", step))
    def on_text(self, text): self.events.append(("text", text))
    def on_llm_end(self): pass
    def on_tool_call(self, name, args, source): self.events.append(("call", name, args, source))
    def on_tool_start(self, name): pass
    def on_tool_result(self, name, result): self.events.append(("result", name, result.text))
    def on_notice(self, message): self.events.append(("notice", message))

    async def confirm(self, name, args):
        self.events.append(("confirm", name))
        return self.approve


def make_registry(log):
    registry = ToolRegistry()

    async def write_file(args):
        log.append(args)
        return ToolResult(f"wrote {args['path']}")

    async def read_file(args):
        return ToolResult("def add(a, b): return a - b")

    registry.register(ToolSpec("write_file", "", {}, "filesystem", write_file, requires_confirmation=True))
    registry.register(ToolSpec("read_file", "", {}, "filesystem", read_file))
    return registry


def test_loop_reads_writes_and_finishes():
    writes = []
    model = ScriptedModel([
        ("read_file", '{"path": "/p/utils.py"}'),
        ("write_file", '{"path": "/p/utils.py", "content": "fixed"}'),
        "Fixed the subtraction bug.",
    ])
    ui = RecordingUI(approve=True)
    agent = Agent(model, make_registry(writes), ui, "system", mode="confirm")

    text, stats = asyncio.run(agent.run("fix utils.py"))

    assert text == "Fixed the subtraction bug."
    assert stats.steps == 3 and stats.tool_calls == 2
    assert writes == [{"path": "/p/utils.py", "content": "fixed"}]
    assert ("confirm", "write_file") in ui.events          # write asked first
    assert ("confirm", "read_file") not in ui.events       # read did not
    assert len(model.bound_tools) == 2
    # the model saw the read result before deciding to write
    assert any(isinstance(m, ToolMessage) and "a - b" in m.content for m in model.seen_histories[1])


def test_rejected_tool_is_not_executed():
    writes = []
    model = ScriptedModel([("write_file", '{"path": "/p/x.py", "content": "x"}'), "Okay, I won't."])
    agent = Agent(model, make_registry(writes), RecordingUI(approve=False), "system", mode="confirm")
    _, stats = asyncio.run(agent.run("write x"))
    assert writes == [] and stats.rejected == 1
    last_tool_msg = [m for m in model.seen_histories[1] if isinstance(m, ToolMessage)][-1]
    assert "rejected" in last_tool_msg.content


def test_auto_mode_skips_confirmation():
    writes = []
    model = ScriptedModel([("write_file", '{"path": "/p/x.py", "content": "x"}'), "Done."])
    ui = RecordingUI(approve=False)
    agent = Agent(model, make_registry(writes), ui, "system", mode="auto")
    asyncio.run(agent.run("write x"))
    assert len(writes) == 1 and not any(e[0] == "confirm" for e in ui.events)


def test_step_limit():
    model = ScriptedModel([("read_file", '{"path": "/a"}')] * 3)
    ui = RecordingUI()
    agent = Agent(model, make_registry([]), ui, "system", max_steps=3)
    _, stats = asyncio.run(agent.run("loop forever"))
    assert stats.hit_step_limit and stats.steps == 3
    assert any(e[0] == "notice" for e in ui.events)


def test_repair_history_after_interrupt():
    model = ScriptedModel(["hi"])
    agent = Agent(model, make_registry([]), RecordingUI(), "system")
    from langchain_core.messages import AIMessage
    agent.history.append(AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {}, "id": "c9", "type": "tool_call"}]))
    agent.repair_history()
    assert isinstance(agent.history[-1], ToolMessage) and agent.history[-1].tool_call_id == "c9"
