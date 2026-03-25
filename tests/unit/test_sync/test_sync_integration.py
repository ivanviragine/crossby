"""Integration test: full crossby sync mcp flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def _load_config_and_sync(project_root: Path, servers_yaml: dict[str, Any]) -> None:
    """Helper: write .crossby.yml and run all MCP writers."""
    cfg = {"version": 1, "mcp_servers": servers_yaml}
    (project_root / ".crossby.yml").write_text(yaml.dump(cfg), encoding="utf-8")

    from crossby.config.loader import load_config
    from crossby.sync.mcp import MCP_WRITERS

    config = load_config(project_root)
    for writer in MCP_WRITERS.values():
        writer.write(config.mcp_servers, project_root)


class TestFullSyncMCP:
    def test_syncs_to_all_five_tools(self, tmp_path: Path) -> None:
        servers = {
            "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
        }
        _load_config_and_sync(tmp_path, servers)

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
        from crossby.config.loader import load_config
        from crossby.sync.mcp import MCP_WRITERS

        servers = {"ctx": {"command": "npx", "args": ["-y", "mcp"]}}
        _load_config_and_sync(tmp_path, servers)

        # Second sync should produce all "skipped"
        config = load_config(tmp_path)
        for writer in MCP_WRITERS.values():
            results = writer.write(config.mcp_servers, tmp_path)
            for r in results:
                assert r.action == "skipped", f"{writer.tool_id}: expected skipped, got {r.action}"

    def test_enabled_false_removes_from_all_tools(self, tmp_path: Path) -> None:
        # First: add the server
        _load_config_and_sync(
            tmp_path,
            {"old": {"command": "npx", "args": ["-y", "old-mcp"]}},
        )

        # Verify it's there
        data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "old" in data["mcpServers"]

        # Now disable it
        _load_config_and_sync(
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

        _load_config_and_sync(
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
        _load_config_and_sync(tmp_path, servers)

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
