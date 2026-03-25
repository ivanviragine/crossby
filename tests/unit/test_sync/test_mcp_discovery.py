"""Tests for MCP server discovery (crossby init scanning)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossby.sync.mcp_discovery import discover_mcp_servers


class TestDiscoverMCPServers:
    def test_empty_project_no_servers(self, tmp_path: Path) -> None:
        result = discover_mcp_servers(tmp_path)
        assert result.servers == {}
        assert result.conflicts == []

    def test_discovers_claude_servers(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "mcpServers": {
                "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
            }
        }), encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert "context7" in result.servers
        assert result.servers["context7"].source_tool == "claude"
        assert result.servers["context7"].data["command"] == "npx"

    def test_discovers_cursor_servers(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "mcpServers": {"myserver": {"command": "node", "args": ["server.js"]}}
        }), encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert "myserver" in result.servers
        assert result.servers["myserver"].source_tool == "cursor"

    def test_discovers_copilot_servers_normalizes_type(self, tmp_path: Path) -> None:
        path = tmp_path / ".vscode" / "mcp.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "servers": {
                "ctx": {"type": "stdio", "command": "npx", "args": ["-y", "mcp"]},
            }
        }), encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert "ctx" in result.servers
        assert result.servers["ctx"].source_tool == "copilot"
        # type field should not appear in canonical format (it's stdio default)
        assert "type" not in result.servers["ctx"].data

    def test_discovers_gemini_servers(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "hooks": [],
            "mcpServers": {"gemini-srv": {"command": "npx"}},
        }), encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert "gemini-srv" in result.servers
        assert result.servers["gemini-srv"].source_tool == "gemini"

    def test_discovers_codex_toml_servers(self, tmp_path: Path) -> None:
        tomli_w = pytest.importorskip("tomli_w")

        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text(
            tomli_w.dumps({"mcp_servers": {"codex-srv": {"command": "node"}}}),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert "codex-srv" in result.servers
        assert result.servers["codex-srv"].source_tool == "codex"

    def test_merges_union_first_seen_wins(self, tmp_path: Path) -> None:
        claude_path = tmp_path / ".claude" / "settings.json"
        claude_path.parent.mkdir()
        claude_path.write_text(json.dumps({
            "mcpServers": {"shared": {"command": "claude-version"}}
        }), encoding="utf-8")

        cursor_path = tmp_path / ".cursor" / "mcp.json"
        cursor_path.parent.mkdir()
        cursor_path.write_text(json.dumps({
            "mcpServers": {"shared": {"command": "cursor-version"}}
        }), encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        # claude comes first → first-seen wins
        assert result.servers["shared"].data["command"] == "claude-version"
        assert result.servers["shared"].source_tool == "claude"

    def test_conflict_reported(self, tmp_path: Path) -> None:
        for dirname, key in [(".claude", "mcpServers"), (".cursor", "mcpServers")]:
            path = tmp_path / dirname / ("settings.json" if "claude" in dirname else "mcp.json")
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({key: {"duplicate": {"command": "npx"}}}), encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict[0] == "duplicate"

    def test_ignores_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text("{bad json", encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert result.servers == {}

    def test_normalizes_http_server(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "mcpServers": {"api": {"url": "http://localhost/mcp", "transport": "http"}}
        }), encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert result.servers["api"].data["url"] == "http://localhost/mcp"

    def test_ignores_malformed_codex_toml(self, tmp_path: Path) -> None:
        pytest.importorskip("tomli_w")
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text("[[invalid toml\n", encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert result.servers == {}
