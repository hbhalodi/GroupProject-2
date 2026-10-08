import asyncio
import json
import sys
from pathlib import Path

from golu.config import ServerConfig, Settings, load_server_configs
from golu.mcp_client import MCPManager
from golu.tools.registry import ToolRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "echo_server.py"


def test_config_substitution_and_filters(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_MISSING_KEY", raising=False)
    monkeypatch.setenv("MY_TOKEN", "abc")
    config = {
        "servers": {
            "fs": {"command": "npx", "args": ["${PROJECT_ROOT}"], "confirm_tools": ["write_file"]},
            "py": {"command": "${PYTHON}", "headers": {"X-Key": "${MY_TOKEN}", "X-Empty": "${NOPE}"}},
            "off": {"enabled": False, "command": "x"},
            "needs_key": {"command": "x", "requires_env": ["SOME_MISSING_KEY"]},
        }
    }
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))
    servers = load_server_configs(Settings(project_root=tmp_path, mcp_config_path=path))

    by_name = {s.name: s for s in servers}
    assert set(by_name) == {"fs", "py"}
    assert by_name["fs"].args == [str(tmp_path.resolve())]
    assert by_name["fs"].confirm_tools == ["write_file"]
    assert by_name["py"].command == sys.executable
    assert by_name["py"].headers == {"X-Key": "abc"}  # empty header dropped


def test_shipped_mcp_config_is_valid():
    servers = load_server_configs(Settings())
    names = {s.name for s in servers}
    assert {"filesystem", "golu-docs", "context7"} <= names


def test_mcp_client_loads_and_calls_tools(tmp_path):
    async def scenario():
        registry = ToolRegistry()
        manager = MCPManager(tmp_path / "logs")
        cfg = ServerConfig(name="echo", transport="stdio", command=sys.executable,
                           args=[str(FIXTURE)], confirm_tools=["write_note"])
        bad = ServerConfig(name="broken", transport="stdio", command="definitely-not-a-command-xyz")
        try:
            await manager.connect_all([cfg, bad], registry)
            statuses = {s.name: s for s in manager.statuses}
            assert statuses["echo"].connected
            assert set(statuses["echo"].tool_names) == {"add", "write_note", "fail"}
            assert not statuses["broken"].connected  # reported, not raised

            assert registry.get("add").source == "echo"
            assert registry.needs_confirmation("write_note")
            assert not registry.needs_confirmation("add")
            assert registry.get("add").parameters["properties"]["a"]["type"] == "integer"

            assert (await registry.execute("add", {"a": 2, "b": 3})).text == "5"
            err = await registry.execute("fail", {})
            assert err.is_error and "fail" in err.text  # MCP v2 hides exception details
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_unresponsive_server_times_out_without_blocking_others(tmp_path):
    async def scenario():
        registry = ToolRegistry()
        manager = MCPManager(tmp_path / "logs", connect_timeout=2)
        silent = ServerConfig(name="silent", transport="stdio", command=sys.executable,
                              args=["-c", "import time; time.sleep(30)"])
        echo = ServerConfig(name="echo", transport="stdio", command=sys.executable, args=[str(FIXTURE)])
        try:
            await manager.connect_all([silent, echo], registry)
            statuses = {s.name: s for s in manager.statuses}
            assert not statuses["silent"].connected and "timed out" in statuses["silent"].error
            assert statuses["echo"].connected
            assert (await registry.execute("add", {"a": 1, "b": 1})).text == "2"
        finally:
            await manager.close()

    asyncio.run(scenario())
