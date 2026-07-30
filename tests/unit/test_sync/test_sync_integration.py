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
    mcp_servers = {name: MCPServerConfig(**entry) for name, entry in servers_yaml.items()}
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

        # Claude — servers land in .mcp.json (what Claude Code reads), not settings.json
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "context7" in data["mcpServers"]

        # Cursor
        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "context7" in data["mcpServers"]

        # Copilot
        data = json.loads((tmp_path / ".vscode" / "mcp.json").read_text())
        assert "context7" in data["servers"]
        assert data["servers"]["context7"]["type"] == "stdio"

        # Antigravity CLI
        data = json.loads((tmp_path / ".agents" / "mcp_config.json").read_text())
        assert "context7" in data["mcpServers"]

        # Codex
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
        assert "context7" in data["mcp_servers"]

    def test_second_sync_is_idempotent(self, tmp_path: Path) -> None:
        servers = {"ctx": {"command": "npx", "args": ["-y", "mcp"]}}
        _sync_mcp(tmp_path, servers)

        # Second sync should produce all "skipped"
        data = _build_sync_data(servers)
        results = run_sync(data, tmp_path, concern=SyncConcern.MCP, installed_tools=list(AIToolID))
        for result in results:
            assert result.action == "skipped", (
                f"{result.tool_id}: expected skipped, got {result.action}"
            )

    def test_enabled_false_removes_from_all_tools(self, tmp_path: Path) -> None:
        # First: add the server
        _sync_mcp(
            tmp_path,
            {"old": {"command": "npx", "args": ["-y", "old-mcp"]}},
        )

        # Verify it's there
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "old" in data["mcpServers"]

        # Now disable it
        _sync_mcp(
            tmp_path,
            {"old": {"command": "npx", "enabled": False}},
        )

        # Should be removed
        data = json.loads((tmp_path / ".mcp.json").read_text())
        assert "old" not in data["mcpServers"]

        data = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
        assert "old" not in data["mcpServers"]

    def test_preserves_unmanaged_servers_in_all_tools(self, tmp_path: Path) -> None:
        # A user server left in the legacy .claude/settings.json location must
        # be preserved untouched — crossby writes .mcp.json and never edits the
        # settings.json server table it no longer owns.
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text(
            json.dumps({"mcpServers": {"user-srv": {"command": "node"}}}),
            encoding="utf-8",
        )

        _sync_mcp(
            tmp_path,
            {"crossby-srv": {"command": "npx", "args": ["-y", "mcp"]}},
        )

        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "user-srv" in settings["mcpServers"]  # untouched
        assert json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"].keys() == {
            "crossby-srv"
        }

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
            (tmp_path / ".mcp.json", "mcpServers", "github"),
            (tmp_path / ".cursor" / "mcp.json", "mcpServers", "github"),
            (tmp_path / ".agents" / "mcp_config.json", "mcpServers", "github"),
            (tmp_path / ".vscode" / "mcp.json", "servers", "github"),
        ]:
            data = json.loads(path.read_text())
            assert data[key][entry_key]["env"]["TOKEN"] == "${GITHUB_TOKEN}", f"Failed for {path}"

        toml_data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text())
        assert toml_data["mcp_servers"]["github"]["env"]["TOKEN"] == "${GITHUB_TOKEN}"


class TestMCPScopeAndRoundTrip:
    def test_claude_mcp_round_trips_through_mcp_json(self, tmp_path: Path) -> None:
        # A server Claude already has in .mcp.json is read from and written back
        # to that one file; settings.json only ever carries the approval list.
        from crossby.sync.readers import build_sync_data

        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"ctx": {"command": "npx", "args": ["-y", "ctx"]}}}),
            encoding="utf-8",
        )
        data = build_sync_data(tmp_path, from_tool=AIToolID.CLAUDE)
        assert "ctx" in data.mcp_servers

        run_sync(
            data,
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.MCP,
            installed_tools=[AIToolID.CLAUDE],
        )
        assert "ctx" in json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
        settings = tmp_path / ".claude" / "settings.json"
        if settings.exists():
            assert "mcpServers" not in json.loads(settings.read_text())

    def test_user_scope_servers_stay_out_of_project_files_by_default(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from crossby.sync import mcp_discovery
        from crossby.sync.readers import build_sync_data

        home = tmp_path / "home" / ".claude.json"
        home.parent.mkdir()
        home.write_text(
            json.dumps({"mcpServers": {"personal": {"command": "npx", "env": {"S": "secret"}}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(mcp_discovery, "_GLOBAL_CLAUDE_JSON_PATH", home)

        # Default build: user-scope servers are never discovered...
        data = build_sync_data(tmp_path, from_tool=AIToolID.CLAUDE)
        assert data.mcp_servers == {}

        # ...and a full sync writes nothing into any committed project file.
        run_sync(data, tmp_path, concern=SyncConcern.MCP, installed_tools=list(AIToolID))
        for rel in (
            ".mcp.json",
            ".cursor/mcp.json",
            ".vscode/mcp.json",
            ".agents/mcp_config.json",
            ".codex/config.toml",
            ".claude/settings.json",
        ):
            assert not (tmp_path / rel).exists(), f"{rel} must not be created from user scope"

        # Opting in surfaces them.
        opted = build_sync_data(tmp_path, from_tool=AIToolID.CLAUDE, include_user_scope=True)
        assert "personal" in opted.mcp_servers
