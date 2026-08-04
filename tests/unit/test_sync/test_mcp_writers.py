"""Tests for MCP sync writers (all 5 tools)."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.models.config import MCPServerConfig
from crossby.sync.base import SyncData
from crossby.sync.mcp import (
    AntigravityCLIMCPWriter,
    ClaudeMCPWriter,
    CodexMCPWriter,
    CopilotMCPWriter,
    CursorMCPWriter,
    report_dropped_default_fallbacks,
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


def _cfg(servers: dict[str, MCPServerConfig]) -> SyncData:
    """Build a minimal SyncData with the given mcp_servers."""
    return SyncData(mcp_servers=servers)


# ---------------------------------------------------------------------------
# ClaudeMCPWriter
# ---------------------------------------------------------------------------


class TestClaudeMCPWriter:
    writer = ClaudeMCPWriter()

    @staticmethod
    def _mcp(tmp_path: Path) -> Path:
        return tmp_path / ".mcp.json"

    @staticmethod
    def _settings(tmp_path: Path) -> Path:
        return tmp_path / ".claude" / "settings.json"

    def test_writes_to_mcp_json_not_settings(self, tmp_path: Path) -> None:
        # Regression for #84: Claude Code reads project MCP servers from
        # .mcp.json, not .claude/settings.json — the server table must land there.
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "created"
        mcp = self._mcp(tmp_path)
        assert mcp.exists()
        assert _read_json(mcp)["mcpServers"]["context7"] == {
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp"],
        }
        # settings.json holds only the narrow approval, never the server table.
        settings = _read_json(self._settings(tmp_path))
        assert "mcpServers" not in settings

    def test_approves_written_names_narrowly(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER, "api": HTTP_SERVER}), tmp_path)
        settings = _read_json(self._settings(tmp_path))
        assert settings["enabledMcpjsonServers"] == ["api", "context7"]
        # Never the blanket approval that would bless servers crossby didn't write.
        assert "enableAllProjectMcpServers" not in settings

    def test_approval_note_counts_only_newly_approved(self, tmp_path: Path) -> None:
        # "a" is already approved; adding "b" must report "approved 1", not 2 —
        # the note reflects the delta, not the total enabled count.
        self.writer.sync(_cfg({"a": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"a": STDIO_SERVER, "b": STDIO_WITH_ENV}), tmp_path)
        assert result.message is not None
        assert "approved 1 server(s)" in result.message
        assert set(_read_json(self._settings(tmp_path))["enabledMcpjsonServers"]) == {"a", "b"}

    def test_revoke_only_note_reports_a_revocation(self, tmp_path: Path) -> None:
        # Disabling one of two approved servers is a revocation, not an approval —
        # the note must say so rather than reporting a bogus approval count.
        # ``b`` is only removable because it is in ``mcp_remove`` (the
        # ledger-bounded set run_sync computes for a crossby-owned server).
        self.writer.sync(_cfg({"a": STDIO_SERVER, "b": STDIO_WITH_ENV}), tmp_path)
        result = self.writer.sync(
            SyncData(
                mcp_servers={"a": STDIO_SERVER, "b": MCPServerConfig(command="npx", enabled=False)},
                mcp_remove=frozenset({"b"}),
            ),
            tmp_path,
        )
        assert result.message is not None
        assert "revoked 1 server(s)" in result.message
        assert "approved" not in result.message

    def test_approval_only_change_reports_settings_as_the_file(self, tmp_path: Path) -> None:
        # .mcp.json already has the server, but the approval was removed. The
        # re-sync changes only settings.json, so that's the file reported —
        # not the byte-for-byte unchanged .mcp.json.
        self.writer.sync(_cfg({"ctx": STDIO_SERVER}), tmp_path)
        settings = self._settings(tmp_path)
        settings.write_text(json.dumps({}), encoding="utf-8")  # drop the approval
        mcp_before = self._mcp(tmp_path).read_text()

        result = self.writer.sync(_cfg({"ctx": STDIO_SERVER}), tmp_path)
        assert result.action == "updated"
        assert result.file_path == settings
        assert self._mcp(tmp_path).read_text() == mcp_before  # .mcp.json untouched
        assert "ctx" in _read_json(settings)["enabledMcpjsonServers"]

    def test_disabling_revokes_the_approval(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"old": STDIO_SERVER}), tmp_path)
        assert "old" in _read_json(self._settings(tmp_path))["enabledMcpjsonServers"]
        # crossby owns "old", so run_sync would pass it in mcp_remove.
        self.writer.sync(
            SyncData(
                mcp_servers={"old": MCPServerConfig(command="npx", enabled=False)},
                mcp_remove=frozenset({"old"}),
            ),
            tmp_path,
        )
        assert "old" not in _read_json(self._settings(tmp_path))["enabledMcpjsonServers"]
        assert "old" not in _read_json(self._mcp(tmp_path)).get("mcpServers", {})

    def test_merges_into_existing_mcp_json(self, tmp_path: Path) -> None:
        mcp = self._mcp(tmp_path)
        mcp.write_text(
            json.dumps({"mcpServers": {"existing": {"command": "node", "args": ["server.js"]}}}),
            encoding="utf-8",
        )
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = _read_json(mcp)
        assert "existing" in data["mcpServers"]
        assert "context7" in data["mcpServers"]

    def test_preserves_unmanaged_servers(self, tmp_path: Path) -> None:
        mcp = self._mcp(tmp_path)
        mcp.write_text(
            json.dumps({"mcpServers": {"user-server": {"command": "node"}}}), encoding="utf-8"
        )
        self.writer.sync(_cfg({"crossby-server": STDIO_SERVER}), tmp_path)
        data = _read_json(mcp)
        assert "user-server" in data["mcpServers"]
        assert "crossby-server" in data["mcpServers"]

    def test_idempotent_skipped(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "skipped"

    def test_update_existing_server_with_changed_args(self, tmp_path: Path) -> None:
        """Updating a server with different args triggers an update."""
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        updated = MCPServerConfig(command="npx", args=["-y", "@upstash/context7-mcp", "--new-flag"])
        result = self.writer.sync(_cfg({"context7": updated}), tmp_path)
        assert result.action == "updated"
        data = _read_json(self._mcp(tmp_path))
        assert "--new-flag" in data["mcpServers"]["context7"]["args"]

    def test_removes_disabled_server(self, tmp_path: Path) -> None:
        mcp = self._mcp(tmp_path)
        mcp.write_text(json.dumps({"mcpServers": {"old": {"command": "npx"}}}), encoding="utf-8")
        # Only removed because crossby owns "old" (mcp_remove is ledger-bounded).
        data = SyncData(
            mcp_servers={"old": MCPServerConfig(command="npx", enabled=False)},
            mcp_remove=frozenset({"old"}),
        )
        self.writer.sync(data, tmp_path)
        assert "old" not in _read_json(mcp)["mcpServers"]

    def test_disabled_unowned_server_survives(self, tmp_path: Path) -> None:
        # A disabled source server whose name crossby never wrote (absent from
        # mcp_remove) must not delete a same-named hand-authored target entry.
        mcp = self._mcp(tmp_path)
        mcp.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "hand-written"}}}), encoding="utf-8"
        )
        self.writer.sync(_cfg({"shared": MCPServerConfig(command="npx", enabled=False)}), tmp_path)
        assert _read_json(mcp)["mcpServers"]["shared"]["command"] == "hand-written"

    def test_overwriting_existing_server_is_not_claimed_as_owned(self, tmp_path: Path) -> None:
        # Ownership (``created``) is creation-only: overwriting a same-named
        # server applies the change but never claims it, so a hand-authored
        # server crossby merely overwrote is never later revoked. Only a
        # genuinely new server enters ``created``.
        mcp = self._mcp(tmp_path)
        mcp.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "hand-written"}}}), encoding="utf-8"
        )
        result = self.writer.sync(_cfg({"shared": STDIO_SERVER, "fresh": HTTP_SERVER}), tmp_path)
        assert set(result.created) == {"fresh"}
        # The overwrite still applied on disk — creation-only ownership does not
        # suppress the merge itself.
        assert _read_json(mcp)["mcpServers"]["shared"]["command"] == "npx"

    def test_disabled_server_not_added(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"never": DISABLED_SERVER}), tmp_path)
        # No enabled servers, nothing to write — and no empty settings.json created.
        assert result.action == "skipped"
        assert not self._mcp(tmp_path).exists()
        assert not self._settings(tmp_path).exists()

    def test_env_var_preserved(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"github": STDIO_WITH_ENV}), tmp_path)
        data = _read_json(self._mcp(tmp_path))
        assert data["mcpServers"]["github"]["env"] == {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}",
        }

    def test_http_server_entry_has_type_not_transport(self, tmp_path: Path) -> None:
        # Regression for #84: a remote entry with a url but no `type` is a
        # config error Claude Code rejects. It must carry `type`, never `transport`.
        self.writer.sync(_cfg({"api": HTTP_SERVER}), tmp_path)
        entry = _read_json(self._mcp(tmp_path))["mcpServers"]["api"]
        assert entry["url"] == "http://localhost:8080/mcp"
        assert entry["type"] == "http"
        assert "transport" not in entry
        assert "command" not in entry

    def test_dry_run_no_write(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not self._mcp(tmp_path).exists()
        assert not self._settings(tmp_path).exists()

    def test_dry_run_no_change(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "skipped"

    def test_malformed_mcp_json_skipped(self, tmp_path: Path) -> None:
        mcp = self._mcp(tmp_path)
        mcp.write_text("{invalid json!!", encoding="utf-8")
        original_content = mcp.read_text()
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        # File must not be truncated or overwritten
        assert mcp.read_text() == original_content

    def test_malformed_settings_leaves_it_intact(self, tmp_path: Path) -> None:
        # A broken settings.json can't take the approval, but the servers still
        # reach .mcp.json and the malformed file is left byte-for-byte intact.
        settings = self._settings(tmp_path)
        settings.parent.mkdir()
        settings.write_text("{broken", encoding="utf-8")
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "created"
        assert "context7" in _read_json(self._mcp(tmp_path))["mcpServers"]
        assert settings.read_text() == "{broken"

    def test_sorted_keys_output(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = json.loads(self._mcp(tmp_path).read_text())
        assert list(data.keys()) == sorted(data.keys())

    def test_consistent_two_space_indent(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        raw = self._mcp(tmp_path).read_text()
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

        self.writer.sync(
            SyncData(
                mcp_servers={"gone": MCPServerConfig(command="node", enabled=False)},
                mcp_remove=frozenset({"gone"}),
            ),
            tmp_path,
        )
        data = _read_json(path)
        assert "gone" not in data["mcpServers"]

    def test_disabled_unowned_server_survives(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "mcp.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "hand-written"}}}), encoding="utf-8"
        )
        self.writer.sync(_cfg({"shared": MCPServerConfig(command="node", enabled=False)}), tmp_path)
        assert _read_json(path)["mcpServers"]["shared"]["command"] == "hand-written"

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
# AntigravityCLIMCPWriter
# ---------------------------------------------------------------------------


class TestAntigravityCLIMCPWriter:
    writer = AntigravityCLIMCPWriter()

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".agents" / "mcp_config.json")
        assert "context7" in data["mcpServers"]

    def test_preserves_other_settings(self, tmp_path: Path) -> None:
        path = tmp_path / ".agents" / "mcp_config.json"
        path.parent.mkdir()
        existing = {"other": "value", "mcpServers": {}}
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = _read_json(path)
        assert data["other"] == existing["other"]
        assert "context7" in data["mcpServers"]

    def test_idempotent(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        assert result.action == "skipped"

    def test_dry_run(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".agents" / "mcp_config.json").exists()

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / ".agents" / "mcp_config.json"
        path.parent.mkdir()
        path.write_text("{bad}", encoding="utf-8")
        original = path.read_text()
        result = self.writer.sync(_cfg({"x": STDIO_SERVER}), tmp_path)
        assert result.action == "error"
        assert path.read_text() == original

    def test_disabled_only_is_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg({"never": DISABLED_SERVER}), tmp_path)
        assert result.action == "skipped"
        assert not (tmp_path / ".agents" / "mcp_config.json").exists()

    def test_stdio_server_entry_unchanged(self, tmp_path: Path) -> None:
        """Stdio servers use the standard shape — no ``serverUrl`` rewrite."""
        self.writer.sync(_cfg({"context7": STDIO_SERVER}), tmp_path)
        data = _read_json(tmp_path / ".agents" / "mcp_config.json")
        entry = data["mcpServers"]["context7"]
        assert entry == {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]}
        assert "serverUrl" not in entry

    def test_remote_server_uses_server_url_key(self, tmp_path: Path) -> None:
        """Remote (http/sse) servers use ``serverUrl`` instead of ``url`` —

        the one documented deviation from the standard Claude/Cursor JSON
        shape (see ``_to_antigravity_cli_entry``).
        """
        result = self.writer.sync(_cfg({"api": HTTP_SERVER}), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".agents" / "mcp_config.json")
        entry = data["mcpServers"]["api"]
        assert entry["serverUrl"] == "http://localhost:8080/mcp"
        assert "url" not in entry

    def test_remote_server_with_env_and_headers(self, tmp_path: Path) -> None:
        server = MCPServerConfig(
            transport="http",
            url="https://api.example.com/mcp",
            env={"API_TOKEN": "${API_TOKEN}"},
            headers={"X-Tenant": "northstar"},
        )
        self.writer.sync(_cfg({"svc": server}), tmp_path)
        data = _read_json(tmp_path / ".agents" / "mcp_config.json")
        entry = data["mcpServers"]["svc"]
        assert entry["serverUrl"] == "https://api.example.com/mcp"
        assert entry["env"] == {"API_TOKEN": "${API_TOKEN}"}
        assert entry["headers"] == {"X-Tenant": "northstar"}


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
        path.write_text(
            tomli_w.dumps({"mcp_servers": {"old": {"command": "node"}}}),
            encoding="utf-8",
        )

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
        # Only removed because crossby owns "old" (mcp_remove is ledger-bounded).
        self.writer.sync(
            SyncData(
                mcp_servers={"old": MCPServerConfig(command="npx", enabled=False)},
                mcp_remove=frozenset({"old"}),
            ),
            tmp_path,
        )
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        # Removing the last server drops the table entirely rather than leaving
        # an empty `[mcp_servers]` behind.
        assert "old" not in data.get("mcp_servers", {})

    def test_disabled_unowned_server_survives(self, tmp_path: Path) -> None:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        import tomli_w

        path = tmp_path / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text(
            tomli_w.dumps({"mcp_servers": {"shared": {"command": "hand-written"}}}),
            encoding="utf-8",
        )
        self.writer.sync(_cfg({"shared": MCPServerConfig(command="npx", enabled=False)}), tmp_path)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert data["mcp_servers"]["shared"]["command"] == "hand-written"

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

    def test_http_server_includes_transport_in_toml(self, tmp_path: Path) -> None:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        self.writer.sync(_cfg({"api": HTTP_SERVER}), tmp_path)
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))
        entry = data["mcp_servers"]["api"]
        assert entry["url"] == "http://localhost:8080/mcp"
        assert entry["transport"] == "http"

    def test_bearer_authorization_rewritten(self, tmp_path: Path) -> None:
        """Authorization: Bearer ${TOKEN} → bearer_token_env_var = "TOKEN"."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        server = MCPServerConfig(
            transport="http",
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer ${API_TOKEN}"},
        )
        self.writer.sync(_cfg({"svc": server}), tmp_path)
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))
        entry = data["mcp_servers"]["svc"]
        assert entry["bearer_token_env_var"] == "API_TOKEN"
        # No raw Authorization header should be left.
        assert "http_headers" not in entry or "Authorization" not in entry.get("http_headers", {})

    def test_env_var_headers_rewritten(self, tmp_path: Path) -> None:
        """${VAR} headers → env_http_headers; literal headers stay."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        server = MCPServerConfig(
            transport="http",
            url="https://api.example.com/mcp",
            headers={
                "X-Tenant": "${TENANT_ID}",
                "X-Project": "northstar",
            },
        )
        self.writer.sync(_cfg({"svc": server}), tmp_path)
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))
        entry = data["mcp_servers"]["svc"]
        assert entry["env_http_headers"] == {"X-Tenant": "TENANT_ID"}
        assert entry["http_headers"] == {"X-Project": "northstar"}

    def test_env_self_reference_rewritten_to_env_vars(self, tmp_path: Path) -> None:
        """env: {KEY: "${KEY}"} → env_vars = ["KEY"]."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        server = MCPServerConfig(
            command="npx",
            args=["-y", "@example/mcp"],
            env={"API_TOKEN": "${API_TOKEN}", "DEBUG": "true"},
        )
        self.writer.sync(_cfg({"svc": server}), tmp_path)
        data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))
        entry = data["mcp_servers"]["svc"]
        assert entry["env_vars"] == ["API_TOKEN"]
        assert entry["env"] == {"DEBUG": "true"}


# ---------------------------------------------------------------------------
# report_dropped_default_fallbacks — the manual-fix note for ${VAR:-default}
# ---------------------------------------------------------------------------


class TestReportDroppedDefaultFallbacks:
    def test_default_fallback_header_produces_a_note(self) -> None:
        servers = {
            "svc": MCPServerConfig(
                transport="http",
                url="https://api.example.com/mcp",
                headers={"Authorization": "Bearer ${TOKEN:-fallback123}"},
            )
        }
        rows = report_dropped_default_fallbacks(servers)
        assert len(rows) == 1
        row = rows[0]
        assert row.file_path is None  # detect-only row, counts as Not Added
        assert "svc" in row.message and "Authorization" in row.message
        assert "${VAR:-default}" in row.message

    def test_note_is_tallied_as_a_manual_fix(self) -> None:
        # Parity with report_oauth_configs: the row must land in the doctor's
        # manual-fix summary, not just render as "Not Added" in the table.
        from crossby.sync.plan import summarize_plan

        servers = {
            "svc": MCPServerConfig(
                transport="http",
                url="https://api.example.com/mcp",
                headers={"Authorization": "Bearer ${TOKEN:-fallback123}"},
            )
        }
        rows = report_dropped_default_fallbacks(servers)
        summary = summarize_plan(rows)
        assert len(summary.manual_fix_results) == 1

    def test_env_default_fallback_produces_a_note(self) -> None:
        servers = {"svc": MCPServerConfig(command="npx", env={"KEY": "${KEY:-dev}"})}
        rows = report_dropped_default_fallbacks(servers)
        assert len(rows) == 1
        assert "env var" in rows[0].message and "KEY" in rows[0].message

    def test_no_note_without_a_default(self) -> None:
        servers = {
            "svc": MCPServerConfig(
                command="npx",
                env={"KEY": "${KEY}"},  # plain indirection, no default to drop
            )
        }
        assert report_dropped_default_fallbacks(servers) == []

    def test_disabled_server_is_ignored(self) -> None:
        servers = {"svc": MCPServerConfig(command="npx", enabled=False, env={"KEY": "${KEY:-dev}"})}
        assert report_dropped_default_fallbacks(servers) == []
