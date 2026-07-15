"""Tests for MCP server discovery (crossby init scanning)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossby.sync.mcp_discovery import discover_mcp_servers, report_oauth_configs


class TestDiscoverMCPServers:
    def test_empty_project_no_servers(self, tmp_path: Path) -> None:
        result = discover_mcp_servers(tmp_path)
        assert result.servers == {}
        assert result.conflicts == []

    def test_discovers_claude_servers(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
                    }
                }
            ),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert "context7" in result.servers
        assert result.servers["context7"].source_tool == "claude"
        assert result.servers["context7"].data["command"] == "npx"

    def test_discovers_project_mcp_json(self, tmp_path: Path) -> None:
        """.mcp.json is Claude's canonical project-scope MCP file."""
        path = tmp_path / ".mcp.json"
        path.write_text(
            json.dumps({"mcpServers": {"context7": {"command": "npx"}}}),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert "context7" in result.servers
        assert result.servers["context7"].source_tool == "claude"

    def test_discovers_global_claude_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """~/.claude.json is Claude's user-scope MCP file."""
        from crossby.sync import mcp_discovery

        fake_home_config = tmp_path / "home" / ".claude.json"
        fake_home_config.parent.mkdir()
        fake_home_config.write_text(
            json.dumps({"mcpServers": {"user-srv": {"command": "npx"}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(mcp_discovery, "_GLOBAL_CLAUDE_JSON_PATH", fake_home_config)

        result = discover_mcp_servers(tmp_path / "project")
        assert "user-srv" in result.servers
        assert result.servers["user-srv"].source_tool == "claude"

    def test_project_mcp_json_wins_over_claude_settings(self, tmp_path: Path) -> None:
        """Same server name in both Claude scopes: .mcp.json (most specific) wins,
        and the collision is not reported as a cross-tool conflict."""
        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "project-version"}}}),
            encoding="utf-8",
        )
        settings_json = tmp_path / ".claude" / "settings.json"
        settings_json.parent.mkdir()
        settings_json.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "settings-version"}}}),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert result.servers["shared"].data["command"] == "project-version"
        assert result.conflicts == []

    def test_discovers_cursor_servers(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps({"mcpServers": {"myserver": {"command": "node", "args": ["server.js"]}}}),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert "myserver" in result.servers
        assert result.servers["myserver"].source_tool == "cursor"

    def test_discovers_copilot_servers_normalizes_type(self, tmp_path: Path) -> None:
        path = tmp_path / ".vscode" / "mcp.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "servers": {
                        "ctx": {"type": "stdio", "command": "npx", "args": ["-y", "mcp"]},
                    }
                }
            ),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert "ctx" in result.servers
        assert result.servers["ctx"].source_tool == "copilot"
        # type field should not appear in canonical format (it's stdio default)
        assert "type" not in result.servers["ctx"].data

    def test_discovers_gemini_servers(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "hooks": [],
                    "mcpServers": {"gemini-srv": {"command": "npx"}},
                }
            ),
            encoding="utf-8",
        )

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
        claude_path.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "claude-version"}}}), encoding="utf-8"
        )

        cursor_path = tmp_path / ".cursor" / "mcp.json"
        cursor_path.parent.mkdir()
        cursor_path.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "cursor-version"}}}), encoding="utf-8"
        )

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
        path.write_text(
            json.dumps(
                {"mcpServers": {"api": {"url": "http://localhost/mcp", "transport": "http"}}}
            ),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert result.servers["api"].data["url"] == "http://localhost/mcp"

    def test_ignores_malformed_codex_toml(self, tmp_path: Path) -> None:
        pytest.importorskip("tomli_w")
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text("[[invalid toml\n", encoding="utf-8")

        result = discover_mcp_servers(tmp_path)
        assert result.servers == {}

    def test_records_oauth_server(self, tmp_path: Path) -> None:
        path = tmp_path / ".mcp.json"
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "secure-srv": {
                            "url": "https://example.com/mcp",
                            "oauth": {"callbackPort": 3000},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        result = discover_mcp_servers(tmp_path)
        assert result.oauth_servers == [("secure-srv", "claude")]

    def test_no_oauth_when_absent(self, tmp_path: Path) -> None:
        path = tmp_path / ".mcp.json"
        path.write_text(
            json.dumps({"mcpServers": {"plain-srv": {"command": "npx"}}}), encoding="utf-8"
        )

        result = discover_mcp_servers(tmp_path)
        assert result.oauth_servers == []


class TestReportOauthConfigs:
    def test_emits_manual_fix_row_for_oauth_server(self, tmp_path: Path) -> None:
        path = tmp_path / ".mcp.json"
        path.write_text(
            json.dumps(
                {"mcpServers": {"secure-srv": {"url": "https://x", "oauth": {"callbackPort": 1}}}}
            ),
            encoding="utf-8",
        )

        results = report_oauth_configs(tmp_path)
        assert len(results) == 1
        row = results[0]
        assert row.action == "skipped"
        assert row.file_path is None
        assert row.tool_id is None
        assert "secure-srv" in (row.message or "")
        assert "manual-fix" in (row.message or "").lower()

    def test_empty_when_no_oauth_servers(self, tmp_path: Path) -> None:
        assert report_oauth_configs(tmp_path) == []
