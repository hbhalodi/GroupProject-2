import asyncio
import sys

from golu.config import Settings
from golu.tools.local_tools import register_local_tools
from golu.tools.registry import ToolRegistry, ToolResult, ToolSpec


def make_registry(tmp_path):
    settings = Settings(project_root=tmp_path, shell_timeout_seconds=5)
    registry = ToolRegistry()
    register_local_tools(registry, tmp_path, settings)
    return registry


def test_run_shell_success_and_failure(tmp_path):
    registry = make_registry(tmp_path)
    ok = asyncio.run(registry.execute("run_shell", {"command": f'"{sys.executable}" -c "print(42)"'}))
    assert not ok.is_error and "exit code 0" in ok.text and "42" in ok.text

    bad = asyncio.run(registry.execute("run_shell", {"command": f'"{sys.executable}" -c "import sys; sys.exit(3)"'}))
    assert bad.is_error and "exit code 3" in bad.text


def test_run_shell_timeout(tmp_path):
    registry = make_registry(tmp_path)
    res = asyncio.run(registry.execute(
        "run_shell", {"command": f'"{sys.executable}" -c "import time; time.sleep(5)"', "timeout": 1}))
    assert res.is_error and "timed out" in res.text


def test_run_shell_requires_confirmation(tmp_path):
    registry = make_registry(tmp_path)
    assert registry.needs_confirmation("run_shell")
    assert not registry.needs_confirmation("search_code")


def test_search_code(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "notes.txt").write_text("nothing here\n")
    registry = make_registry(tmp_path)

    res = asyncio.run(registry.execute("search_code", {"pattern": r"def add"}))
    assert "utils.py:1" in res.text

    res = asyncio.run(registry.execute("search_code", {"pattern": "add", "glob": "*.txt"}))
    assert "No matches" in res.text

    res = asyncio.run(registry.execute("search_code", {"pattern": "x", "path": "../.."}))
    assert res.is_error


def test_unknown_tool_and_name_collision():
    registry = ToolRegistry()

    async def ok(args):
        return ToolResult("ok")

    registry.register(ToolSpec("read", "a", {}, "local", ok))
    second = registry.register(ToolSpec("read", "b", {}, "server2", ok))
    assert second == "server2__read"
    assert {t.name for t in registry.all()} == {"read", "server2__read"}

    res = asyncio.run(registry.execute("missing", {}))
    assert res.is_error and "Unknown tool" in res.text


def test_executor_exception_becomes_error_result():
    registry = ToolRegistry()

    async def broken(args):
        raise RuntimeError("kaput")

    registry.register(ToolSpec("broken", "", {}, "local", broken))
    res = asyncio.run(registry.execute("broken", {}))
    assert res.is_error and "kaput" in res.text


def test_openai_schema_shape():
    async def ok(args):
        return ToolResult("ok")

    spec = ToolSpec("t", "desc", {"properties": {"x": {"type": "string"}}}, "local", ok)
    schema = spec.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["parameters"]["type"] == "object"
