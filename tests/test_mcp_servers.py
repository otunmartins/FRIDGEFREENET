"""
Smoke tests for MCP servers.

Ensures all MCP servers load without decorator errors (e.g. @mcp.tool must be @mcp.tool()).
FastMCP raises TypeError if decorator is not called: "Use @tool() instead of @tool"
"""

import json
import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Single MCP server (insulin_ai_mcp_server.py); literature lives in insulin_ai.literature
MCP_SERVERS = [("insulin_ai_mcp_server", "insulin_ai_mcp_server.py")]


def _load_mcp_server(path: str):
    """Load MCP server module. Fails if @mcp.tool (no parens) is used."""
    full_path = os.path.join(ROOT, path)
    spec = importlib.util.spec_from_file_location("mcp_server", full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_insulin_ai_mcp_server_loads():
    """insulin-ai MCP server must load (run_autonomous_discovery, mine_literature, etc.)."""
    try:
        mod = _load_mcp_server("insulin_ai_mcp_server.py")
    except (ImportError, ModuleNotFoundError) as e:
        import pytest
        pytest.skip(f"MCP dependencies unavailable: {e}")
    assert hasattr(mod, "mcp")
    assert hasattr(mod.mcp, "run")


def test_all_mcp_servers_load():
    """All OpenCode-enabled MCP servers must load without @tool decorator errors."""
    try:
        _load_mcp_server("insulin_ai_mcp_server.py")
    except (ImportError, ModuleNotFoundError) as e:
        import pytest
        pytest.skip(f"MCP dependencies unavailable: {e}")
    for _name, path in MCP_SERVERS:
        _load_mcp_server(path)


def test_validate_psmiles_json_shape():
    """validate_psmiles returns JSON with valid; optional name_crosscheck when enabled."""
    try:
        mod = _load_mcp_server("insulin_ai_mcp_server.py")
    except (ImportError, ModuleNotFoundError) as e:
        import pytest

        pytest.skip(f"MCP dependencies unavailable: {e}")
    out = json.loads(mod.validate_psmiles("[*]OCC[*]", material_name="", crosscheck_web=False))
    assert "valid" in out
    assert out.get("valid") is True
    assert "name_crosscheck" not in out
    out2 = json.loads(
        mod.validate_psmiles("[*]OCC[*]", material_name="polyethylene glycol", crosscheck_web=True)
    )
    assert out2.get("valid") is True
    assert "name_crosscheck" in out2
    nc = out2["name_crosscheck"]
    assert nc.get("material_name") == "polyethylene glycol"
    assert "snippets" in nc
    assert "disclaimer" in nc


def test_no_mcp_tool_without_parens():
    """Fail if any MCP file uses @mcp.tool instead of @mcp.tool()."""
    import re

    bad_files = []
    for _name, path in MCP_SERVERS:
        full_path = os.path.join(ROOT, path)
        content = open(full_path, "r").read()
        # Match @mcp.tool at end of line or followed by non-open-paren
        if re.search(r"@mcp\.tool\s*(?!\()", content):
            bad_files.append(path)
    assert not bad_files, (
        f"Use @mcp.tool() not @mcp.tool in: {bad_files}. "
        "FastMCP requires parentheses: @mcp.tool()"
    )
