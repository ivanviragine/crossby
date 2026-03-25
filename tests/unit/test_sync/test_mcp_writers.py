"""Tests for MCP sync writers (all 5 tools)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

from crossby.models.config import CrossbyConfig, MCPServerConfig
from crossby.sync.mcp import (
    ClaudeMCPWriter,
    CopilotMCPWriter,
    CursorMCPWriter,
    GeminiMCPWriter,
    CodexMCPWriter,
    MCP_WRITERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STDIO_SERVER = MCPServerConfig(command="npx", args=["-y", "@upstash/context7-mcp"])
STDIO_WITH_ENV = MCPServerConfig(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"},
)
HTTP_SERVER = MCPServerConfig(transport="http", url="http://localhost:8080/mcp")
DISABLED_SERVER = MCPServerConfig(command="npx", args=["-y", "old-mcp"], enabled=False)


# ---------------------------------------------------------------------------
# Helpers shared across JSON-based writer tests
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg(servers: dict[str, MCPServerConfig]) -> CrossbyConfig:
    """Build a minimal CrossbyConfig with the given mcp_servers."""
    return CrossbyConfig(mcp_servers=servers)


# ---------------------------------------------------------------------------
# ClaudeMCPWriter
# ---------------------------------------------------------------------------


class TestClaudeMCPWriter:
    writer = ClaudeMCPWriter()

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".claude" / "settings.json"
        assert path.exists()
        data = _read_json(path)
        assert data["mcpServers"]["context7"] == {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]}

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "permissions": {"allow": ["Bash(git *)"]},
            "mcpServers": {"existing": {"command": "node", "args": ["server.js"]}},
        }), encoding="utf-8")

        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)

        data = _read_json(path)
        assert "existing" in data["mcpServers"]
        assert "context7" in data["mcpServers"]
        assert data["permissions"]["allow"] == ["Bash(git *)"]

    def test_preserves_unmanaged_servers(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        existing = {"mcpServers": {"user-server": {"command": "node"}}}
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg({"crossby-server": STDIO_SERVER}), tmp_path)

        data = _read_json(path)
        assert "user-server" in data["mcpServers"]
        assert "crossby-server" in data["mcpServers"]

    def test_idempotent_skipped(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "skipped"

    def test_removes_disabled_server(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"mcpServers": {"old": {"command": "npx"}}}), encoding="utf-8")

        servers = {"old": MCPServerConfig(command="npx", enabled=False)}
        self.writer.sync(_cfg(servers), tmp_path)

        data = _read_json(path)
        assert "old" not in data["mcpServers"]

    def test_disabled_server_not_added(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"never": DISABLED_SERVER}), tmp_path)
        # No enabled servers, nothing to write
        assert result.action == "skipped"

    def test_env_var_preserved(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"github": STDIO_WITH_ENV}), tmp_path)
        data = _read_json(tmp_path / ".claude" / "settings.json")
        assert data["mcpServers"]["github"]["env"] == {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}

    def test_http_server_entry(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"api": HTTP_SERVER}), tmp_path)
        data = _read_json(tmp_path / ".claude" / "settings.json")
        entry = data["mcpServers"]["api"]
        assert entry["url"] == "http://localhost:8080/mcp"
        assert "command" not in entry

    def test_dry_run_no_write(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".claude" / "settings.json").exists()

    def test_dry_run_no_change(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "skipped"

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text("{invalid json!!", encoding="utf-8")
        original_content = path.read_text()

        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        # File must not be truncated or overwritten
        assert path.read_text() == original_content

    def test_sorted_keys_output(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        raw = (tmp_path / ".claude" / "settings.json").read_text()
        # Verify keys are sorted (mcpServers should appear after any earlier key)
        data = json.loads(raw)
        assert list(data.keys()) == sorted(data.keys())

    def test_consistent_two_space_indent(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        raw = (tmp_path / ".claude" / "settings.json").read_text()
        assert '  "mcpServers"' in raw


# ---------------------------------------------------------------------------
# CursorMCPWriter
# ---------------------------------------------------------------------------


class TestCursorMCPWriter:
    writer = CursorMCPWriter()

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".cursor" / "mcp.json"
        assert path.exists()
        data = _read_json(path)
        assert "context7" in data["mcpServers"]

    def test_merges_preserving_existing(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"mcpServers": {"old": {"command": "node"}}}), encoding="utf-8")

        self.writer.sync(_cfg({"new": STDIO_SERVER}), tmp_path)
        data = _read_json(path)
        assert "old" in data["mcpServers"]
        assert "new" in data["mcpServers"]

    def test_idempotent(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "skipped"

    def test_removes_disabled(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"mcpServers": {"gone": {"command": "node"}}}), encoding="utf-8")

        self.writer.sync(_cfg({"gone": MCPServerConfig(command="node", enabled=False)}), tmp_path)
        data = _read_json(path)
        assert "gone" not in data["mcpServers"]

    def test_dry_run(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".cursor" / "mcp.json").exists()

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text("{bad", encoding="utf-8")
        original = path.read_text()
        result = self.writer.sync(_cfg({"x": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        assert path.read_text() == original

    def test_disabled_only_is_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"never": DISABLED_SERVER}), tmp_path)
        assert result.action == "skipped"
        assert not (tmp_path / ".cursor" / "mcp.json").exists()


# ---------------------------------------------------------------------------
# CopilotMCPWriter
# ---------------------------------------------------------------------------


class TestCopilotMCPWriter:
    writer = CopilotMCPWriter()

    def test_creates_servers_key(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = _read_json(tmp_path / ".vscode" / "mcp.json")
        assert "servers" in data
        assert "mcpServers" not in data

    def test_adds_type_field_stdio(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = _read_json(tmp_path / ".vscode" / "mcp.json")
        entry = data["servers"]["context7"]
        assert entry["type"] == "stdio"
        assert entry["command"] == "npx"

    def test_adds_type_field_http(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"api": HTTP_SERVER}), tmp_path)
        data = _read_json(tmp_path / ".vscode" / "mcp.json")
        entry = data["servers"]["api"]
        assert entry["type"] == "http"
        assert entry["url"] == "http://localhost:8080/mcp"

    def test_idempotent(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "skipped"

    def test_dry_run(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".vscode" / "mcp.json").exists()

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / ".vscode" / "mcp.json"
        path.parent.mkdir()
        path.write_text("[not-an-object]", encoding="utf-8")
        original = path.read_text()
        result = self.writer.sync(_cfg({"x": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        assert path.read_text() == original

    def test_disabled_only_is_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"never": DISABLED_SERVER}), tmp_path)
        assert result.action == "skipped"
        assert not (tmp_path / ".vscode" / "mcp.json").exists()


# ---------------------------------------------------------------------------
# GeminiMCPWriter
# ---------------------------------------------------------------------------


class TestGeminiMCPWriter:
    writer = GeminiMCPWriter()

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".gemini" / "settings.json")
        assert "context7" in data["mcpServers"]

    def test_preserves_other_gemini_settings(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        existing = {"hooks": [{"event": "pre_tool", "command": "echo hi"}], "mcpServers": {}}
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = _read_json(path)
        assert data["hooks"] == existing["hooks"]
        assert "context7" in data["mcpServers"]

    def test_idempotent(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "skipped"

    def test_dry_run(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".gemini" / "settings.json").exists()

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        path.write_text("{bad}", encoding="utf-8")
        original = path.read_text()
        result = self.writer.sync(_cfg({"x": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        assert path.read_text() == original

    def test_disabled_only_is_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"never": DISABLED_SERVER}), tmp_path)
        assert result.action == "skipped"
        assert not (tmp_path / ".gemini" / "settings.json").exists()


# ---------------------------------------------------------------------------
# CodexMCPWriter
# ---------------------------------------------------------------------------


class TestCodexMCPWriter:
    writer = CodexMCPWriter()

    def test_creates_new_toml_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".codex" / "config.toml"
        assert path.exists()
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert "context7" in data["mcp_servers"]
        assert data["mcp_servers"]["context7"]["command"] == "npx"

    def test_merges_into_existing_toml(self, tmp_path: Path) -> None:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        import tomli_w

        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text(tomli_w.dumps({"mcp_servers": {"old": {"command": "node"}}}), encoding="utf-8")

        self.writer.sync(_cfg({"new": STDIO_SERVER}), tmp_path)

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert "old" in data["mcp_servers"]
        assert "new" in data["mcp_servers"]

    def test_preserves_other_toml_keys(self, tmp_path: Path) -> None:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        import tomli_w

        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text(
            tomli_w.dumps({"model": "gpt-4o", "mcp_servers": {}}),
            encoding="utf-8",
        )

        self.writer.sync(_cfg({"ctx": STDIO_SERVER}), tmp_path)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert data["model"] == "gpt-4o"
        assert "ctx" in data["mcp_servers"]

    def test_idempotent(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "skipped"

    def test_removes_disabled_server(self, tmp_path: Path) -> None:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        import tomli_w

        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text(
            tomli_w.dumps({"mcp_servers": {"old": {"command": "npx"}}}),
            encoding="utf-8",
        )
        self.writer.sync(_cfg({"old": MCPServerConfig(command="npx", enabled=False)}), tmp_path)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert "old" not in data["mcp_servers"]

    def test_dry_run(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".codex" / "config.toml").exists()

    def test_malformed_toml_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text("[[invalid toml\n", encoding="utf-8")
        original = path.read_text()
        result = self.writer.sync(_cfg({"x": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        assert path.read_text() == original

    def test_missing_tomli_w_graceful_skip(self, tmp_path: Path) -> None:
        """CodexMCPWriter returns error if tomli-w is not installed."""
        with mock.patch.dict(sys.modules, {"tomli_w": None}):
            result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        assert not (tmp_path / ".codex" / "config.toml").exists()

    def test_disabled_only_is_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"never": DISABLED_SERVER}), tmp_path)
        assert result.action == "skipped"
        assert not (tmp_path / ".codex" / "config.toml").exists()

    def test_args_in_toml(self, tmp_path: Path) -> None:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))
        assert data["mcp_servers"]["context7"]["args"] == ["-y", "@upstash/context7-mcp"]


# ---------------------------------------------------------------------------
# MCP_WRITERS registry
# ---------------------------------------------------------------------------


class TestMCPWritersRegistry:
    def test_all_tools_registered(self) -> None:
        assert set(MCP_WRITERS.keys()) == {"claude", "cursor", "copilot", "gemini", "codex"}

    def test_writer_types(self) -> None:
        assert isinstance(MCP_WRITERS["claude"], ClaudeMCPWriter)
        assert isinstance(MCP_WRITERS["cursor"], CursorMCPWriter)
        assert isinstance(MCP_WRITERS["copilot"], CopilotMCPWriter)
        assert isinstance(MCP_WRITERS["gemini"], GeminiMCPWriter)
        assert isinstance(MCP_WRITERS["codex"], CodexMCPWriter)
