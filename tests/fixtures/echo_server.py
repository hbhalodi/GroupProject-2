"""Tiny MCP server used by the tests (no Node or network needed)."""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("echo")


@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@mcp.tool()
def write_note(text: str) -> str:
    """Pretend to write a note (a state-changing tool)."""
    return f"saved: {text}"


@mcp.tool()
def fail() -> str:
    """Always raises."""
    raise ValueError("boom")


if __name__ == "__main__":
    mcp.run("stdio")
