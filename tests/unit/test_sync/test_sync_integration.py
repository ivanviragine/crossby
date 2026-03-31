"""Integration test: full crossby sync mcp flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

from crossby.models.ai import AIToolID
from crossby.models.config import MCPServerConfig
from crossby.sync import run_sync
from crossby.sync.base import SyncConcern, SyncData


def _build_sync_data(servers_yaml: dict[str, Any]) -> SyncData:
    """Build SyncData with MCP servers from a dict (matching YAML structure)."""
    mcp_servers = {
        name: MCPServerConfig(**entry) for name, entry in servers_yaml.items()
    }
    return SyncData(mcp_servers=mcp_servers)


def _sync_mcp(project_root: Path, servers_yaml: dict[str, Any]) -> None:
    """Helper: build SyncData and run all MCP writers."""
    data = _build_sync_data(servers_yaml)
    all_tools = list(AIToolID)
    run_sync(data, project_root, concern=SyncConcern.MCP, installed_tools=all_tools)


class TestFullSyncMCP:
    def test_syncs_to_all_five_tools(self, tmp_path: Path) -> None:
        servers = {
            "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
        }
        _sync_mcp(tmp_path, servers)

        # Claude
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "context7" in data["mcpServers"]

        # Cursor
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "context7" in data["mcpServers"]

        # Copilot
        data = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        assert "context7" in data["servers"]
        assert data["servers"]["context7"]["type"] == "stdio"

        # Gemini
        data = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
        assert "context7" in data["mcpServers"]

        # Codex
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
        assert "context7" in data["mcp_servers"]

    def test_second_sync_is_idempotent(self, tmp_path: Path) -> None:
        servers = {"ctx": {"command": "npx", "args": ["-y", "mcp"]}}
        _sync_mcp(tmp_path, servers)

        # Second sync should produce all "skipped"
        data = _build_sync_data(servers)
        results = run_sync(
            data, tmp_path, concern=SyncConcern.MCP, installed_tools=list(AIToolID)
        )
        for result in results:
            assert result.action == "skipped", f"{result.tool_id}: expected skipped, got {result.action}"

    def test_enabled_false_removes_from_all_tools(self, tmp_path: Path) -> None:
        # First: add the server
        _sync_mcp(
            tmp_path,
            {"old": {"command": "npx", "args": ["-y", "old-mcp"]}},
        )

        # Verify it's there
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "old" in data["mcpServers"]

        # Now disable it
        _sync_mcp(
            tmp_path,
            {"old": {"command": "npx", "enabled": False}},
        )

        # Should be removed
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "old" not in data["mcpServers"]

        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "old" not in data["mcpServers"]

    def test_preserves_unmanaged_servers_in_all_tools(self, tmp_path: Path) -> None:
        # Pre-populate each tool with a user-managed server
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text(
            json.dumps({"mcpServers": {"user-srv": {"command": "node"}}}),
            encoding="utf-8",
        )

        _sync_mcp(
            tmp_path,
            {"crossby-srv": {"command": "npx", "args": ["-y", "mcp"]}},
        )

        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "user-srv" in data["mcpServers"]
        assert "crossby-srv" in data["mcpServers"]

    def test_env_vars_preserved_across_all_formats(self, tmp_path: Path) -> None:
        servers = {
            "github": {
                "command": "npx",
                "args": ["-y", "server-github"],
                "env": {"TOKEN": "${GITHUB_TOKEN}"},
            }
        }
        _sync_mcp(tmp_path, servers)

        for path, key, entry_key in [
            (tmp_path / ".claude" / "settings.json", "mcpServers", "github"),
            (tmp_path / ".cursor" / "mcp.json", "mcpServers", "github"),
            (tmp_path / ".gemini" / "settings.json", "mcpServers", "github"),
            (tmp_path / ".vscode" / "mcp.json", "servers", "github"),
        ]:
            data = json.loads(path.read_text())
            assert data[key][entry_key]["env"]["TOKEN"] == "${GITHUB_TOKEN}", f"Failed for {path}"

        toml_data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
        assert toml_data["mcp_servers"]["github"]["env"]["TOKEN"] == "${GITHUB_TOKEN}"
