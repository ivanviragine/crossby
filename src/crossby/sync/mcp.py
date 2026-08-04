"""MCP server sync writers — one per AI tool.

Each writer merges .crossby.yml mcp_servers into the tool's native config
format using a non-destructive merge strategy:
- Enabled servers are added or updated.
- Servers with enabled=False are removed from the target (if present).
- Servers in the target but NOT in .crossby.yml are preserved.
- Idempotent: identical existing definitions produce action="skipped".
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any

from crossby.models.ai import AIToolID
from crossby.models.config import MCPServerConfig
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult
from crossby.sync.json_utils import (
    SyncAction,
    atomic_write_text,
    read_json_file,
    read_merge_write_json,
    write_json_file,
)
from crossby.sync.toml_edit import remove_table, splice_or_none, upsert_table


def _split_servers(
    servers: dict[str, MCPServerConfig],
) -> tuple[dict[str, MCPServerConfig], set[str]]:
    """Split servers into (enabled, disabled_names)."""
    enabled = {name: s for name, s in servers.items() if s.enabled}
    disabled = {name for name, s in servers.items() if not s.enabled}
    return enabled, disabled


def _to_stdio_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a stdio server to the standard JSON entry (Claude/Cursor)."""
    entry: dict[str, Any] = {"command": server.command}
    if server.args:
        entry["args"] = server.args
    if server.env:
        entry["env"] = server.env
    return entry


def _to_http_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert an http/sse server to the standard JSON entry.

    The transport goes in ``type``, not ``transport``: Claude Code reads a
    remote entry with a ``url`` but no ``type`` as a stdio server, fails to
    launch it, and reports ``has a "url" but no "type"``. Cursor reads the same
    key, so both tools share this shape.
    """
    entry: dict[str, Any] = {"url": server.url, "type": server.transport}
    if server.env:
        entry["env"] = server.env
    if server.headers:
        entry["headers"] = server.headers
    return entry


def _to_json_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a server to the standard JSON entry (Claude/Cursor format)."""
    if server.command is not None:
        return _to_stdio_entry(server)
    return _to_http_entry(server)


def _to_antigravity_cli_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a server to Antigravity CLI's JSON entry.

    Remote (http/sse) servers use ``serverUrl`` instead of ``url`` — the one
    documented deviation from the standard Claude/Cursor JSON shape. Note a
    project-local ``.antigravitycli/mcp_config.json`` is discovered by the
    CLI but its ``mcpServers`` is silently ignored, so this must land in
    ``.agents/mcp_config.json``.
    """
    if server.command is not None:
        return _to_stdio_entry(server)
    entry: dict[str, Any] = {"serverUrl": server.url}
    if server.env:
        entry["env"] = server.env
    if server.headers:
        entry["headers"] = server.headers
    return entry


def _to_copilot_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a server to Copilot's JSON entry (includes explicit 'type' field)."""
    entry: dict[str, Any] = {"type": server.transport}
    if server.command is not None:
        entry["command"] = server.command
        if server.args:
            entry["args"] = server.args
    if server.url is not None:
        entry["url"] = server.url
    if server.env:
        entry["env"] = server.env
    if server.headers:
        entry["headers"] = server.headers
    return entry


def _to_toml_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a server to the TOML entry format (Codex).

    Applies Codex-specific transport rewrites: ``Authorization: Bearer
    ${VAR}`` collapses into ``bearer_token_env_var``; ``${VAR}`` headers
    move to ``env_http_headers``; ``KEY = "${KEY}"`` env entries move to
    ``env_vars``. Literal headers/env entries stay verbatim.
    """
    from crossby.sync.mcp_transports import (
        rewrite_env_for_codex,
        rewrite_headers_for_codex,
    )

    entry: dict[str, Any] = {}
    if server.command is not None:
        entry["command"] = server.command
        if server.args:
            entry["args"] = server.args
    if server.url is not None:
        entry["url"] = server.url
        entry["transport"] = server.transport

    if server.env:
        env_rewrite = rewrite_env_for_codex(server.env)
        if env_rewrite.env:
            entry["env"] = env_rewrite.env
        if env_rewrite.env_vars:
            entry["env_vars"] = env_rewrite.env_vars

    if server.headers:
        header_rewrite = rewrite_headers_for_codex(server.headers)
        if header_rewrite.bearer_token_env_var is not None:
            entry["bearer_token_env_var"] = header_rewrite.bearer_token_env_var
        if header_rewrite.http_headers:
            entry["http_headers"] = header_rewrite.http_headers
        if header_rewrite.env_http_headers:
            entry["env_http_headers"] = header_rewrite.env_http_headers

    return entry


class _JsonMCPWriter(AbstractSyncWriter):
    """Base class for JSON-based MCP writers (Claude, Cursor, Copilot, Antigravity CLI)."""

    concern = SyncConcern.MCP

    @property
    @abstractmethod
    def _config_path_parts(self) -> tuple[str, str]:
        """Return (directory, filename) relative to project_root."""

    @property
    @abstractmethod
    def _mcp_key(self) -> str:
        """Return the top-level JSON key for MCP servers."""

    def _to_entry(self, server: MCPServerConfig) -> dict[str, Any]:
        """Convert a server to the tool's JSON entry format."""
        return _to_json_entry(server)

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        dirname, filename = self._config_path_parts
        path = project_root / dirname / filename
        enabled, _disabled = _split_servers(data.mcp_servers)
        updates = {name: self._to_entry(s) for name, s in enabled.items()}
        # Only ledger-owned names may be removed (``data.mcp_remove``), so a
        # same-named server crossby never wrote survives even when a source
        # server of that name is disabled.
        action, message, written, created, removed = read_merge_write_json(
            path, self._mcp_key, updates, set(data.mcp_remove), dry_run
        )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=message or None,
            added=len(written),
            revoked=removed,
            # Ownership = only freshly-added servers, never overwritten ones.
            created=tuple(created),
        )


def _approve_mcp_json_servers(
    settings_path: Path,
    approve: set[str],
    revoke: set[str],
    *,
    dry_run: bool,
) -> tuple[bool, int, int]:
    """Record ``approve`` in (and drop ``revoke`` from) ``enabledMcpjsonServers``.

    This is the one legitimate MCP use of ``.claude/settings.json``: a *narrow*
    approval of exactly the servers crossby wrote to ``.mcp.json``, rather than
    ``enableAllProjectMcpServers`` which would bless servers crossby never
    touched. Returns ``(changed, added, removed)`` — whether the list changed at
    all, and how many names were newly added / revoked, so the caller's message
    reflects the real delta rather than the total enabled count.

    A missing settings.json is created only when there's something to approve;
    a malformed one is left byte-for-byte intact and the approval is skipped
    with a warning (the servers are already safely in ``.mcp.json``).
    """
    if not approve and not revoke:
        return False, 0, 0
    data, error, was_new = read_json_file(settings_path)
    if error is not None:
        warnings.warn(
            f"{settings_path} {error} — servers were written to .mcp.json but "
            "could not be recorded in enabledMcpjsonServers. Fix the file manually.",
            stacklevel=3,
        )
        return False, 0, 0
    existing = data or {}
    current = existing.get("enabledMcpjsonServers")
    names = [n for n in current if isinstance(n, str)] if isinstance(current, list) else []
    before = set(names)
    result = list(names)
    for name in sorted(approve):
        if name not in result:
            result.append(name)
    result = [n for n in result if n not in revoke]
    if result == names:
        return False, 0, 0
    if was_new and not result:
        return False, 0, 0  # nothing to record — don't create an empty settings.json
    added = sum(1 for n in result if n not in before)
    removed = sum(1 for n in before if n in revoke)
    if not dry_run:
        existing["enabledMcpjsonServers"] = result
        write_json_file(settings_path, existing)
    return True, added, removed


class ClaudeMCPWriter(_JsonMCPWriter):
    """Merges MCP servers into ``.mcp.json`` → ``mcpServers`` (what Claude reads).

    Claude Code loads project MCP servers from ``.mcp.json``, *not* from
    ``.claude/settings.json`` — that file holds only MCP *policy* keys. Older
    crossby wrote servers into ``settings.json`` where Claude never read them,
    so every such sync was inert. Servers now land in ``.mcp.json``; discovery
    still reads ``.claude/settings.json`` too, so whatever a user or an older
    crossby left there keeps working.

    After the write, the names crossby wrote are approved narrowly via
    ``enabledMcpjsonServers`` in ``.claude/settings.json``. Note: as of Claude
    Code v2.1.196 these approval keys are ignored in an *untrusted* folder, so a
    freshly cloned repo still shows the trust dialog on first open.
    """

    tool_id = AIToolID.CLAUDE

    @property
    def _config_path_parts(self) -> tuple[str, str]:
        return "", ".mcp.json"

    @property
    def _mcp_key(self) -> str:
        return "mcpServers"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        mcp_path = project_root / ".mcp.json"
        settings_path = project_root / ".claude" / "settings.json"
        enabled, _disabled = _split_servers(data.mcp_servers)
        updates = {name: self._to_entry(s) for name, s in enabled.items()}
        # Only ledger-owned names may be removed / de-approved.
        revoke = set(data.mcp_remove)

        action, message, written, created, removed = read_merge_write_json(
            mcp_path, self._mcp_key, updates, revoke, dry_run
        )
        if action == "error":
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=mcp_path,
                message=message or None,
            )

        approval_changed, approved_n, revoked_n = _approve_mcp_json_servers(
            settings_path, set(enabled), revoke, dry_run=dry_run
        )
        # When only the approval changed, settings.json is the file that was
        # touched — report it as the artifact rather than the unchanged .mcp.json.
        changed_file = mcp_path
        if action == "skipped" and approval_changed:
            action = "updated"
            changed_file = settings_path

        note: str | None = message or None
        if approval_changed:
            parts = []
            if approved_n:
                parts.append(f"approved {approved_n} server(s)")
            if revoked_n:
                parts.append(f"revoked {revoked_n} server(s)")
            if parts:
                note = f"{' and '.join(parts)} in {settings_path.name} (enabledMcpjsonServers)"
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=changed_file,
            message=note,
            added=len(written) + approved_n,
            revoked=removed + revoked_n,
            # Ownership = only server names crossby freshly added to .mcp.json —
            # never one it overwrote, so a hand-authored server is never claimed.
            created=tuple(created),
        )


class CursorMCPWriter(_JsonMCPWriter):
    """Merges MCP servers into .cursor/mcp.json → mcpServers."""

    tool_id = AIToolID.CURSOR

    @property
    def _config_path_parts(self) -> tuple[str, str]:
        return ".cursor", "mcp.json"

    @property
    def _mcp_key(self) -> str:
        return "mcpServers"


class CopilotMCPWriter(_JsonMCPWriter):
    """Merges MCP servers into .vscode/mcp.json → servers (Copilot format)."""

    tool_id = AIToolID.COPILOT

    @property
    def _config_path_parts(self) -> tuple[str, str]:
        return ".vscode", "mcp.json"

    @property
    def _mcp_key(self) -> str:
        return "servers"

    def _to_entry(self, server: MCPServerConfig) -> dict[str, Any]:
        return _to_copilot_entry(server)


class AntigravityCLIMCPWriter(_JsonMCPWriter):
    """Merges MCP servers into .agents/mcp_config.json → mcpServers.

    Remote servers use ``serverUrl`` rather than ``url``/``httpUrl`` — see
    :func:`_to_antigravity_cli_entry`.
    """

    tool_id = AIToolID.ANTIGRAVITY_CLI

    @property
    def _config_path_parts(self) -> tuple[str, str]:
        return ".agents", "mcp_config.json"

    @property
    def _mcp_key(self) -> str:
        return "mcpServers"

    def _to_entry(self, server: MCPServerConfig) -> dict[str, Any]:
        return _to_antigravity_cli_entry(server)


class CodexMCPWriter(AbstractSyncWriter):
    """Merges MCP servers into .codex/config.toml → [mcp_servers.<name>]."""

    tool_id = AIToolID.CODEX
    concern = SyncConcern.MCP

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        path = project_root / ".codex" / "config.toml"
        enabled, _disabled = _split_servers(data.mcp_servers)
        # Only ledger-owned names may be removed (``data.mcp_remove``).
        action, message, written, created, removed = self._write_toml(
            path, enabled, set(data.mcp_remove), dry_run
        )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=message or None,
            added=len(written),
            revoked=removed,
            # Ownership = only freshly-added tables, never overwritten ones.
            created=tuple(created),
        )

    def _write_toml(
        self,
        path: Path,
        enabled: dict[str, MCPServerConfig],
        revoke: set[str],
        dry_run: bool,
    ) -> tuple[SyncAction, str, list[str], list[str], int]:
        import tomllib

        import tomli_w

        was_new = not path.exists()
        existing: dict[str, Any] = {}
        original = ""
        if not was_new:
            try:
                original = path.read_text(encoding="utf-8")
                existing = tomllib.loads(original)
            except Exception as e:
                msg = (
                    f"{path} contains invalid TOML — skipping MCP sync for Codex. "
                    f"Fix the file manually or delete it. ({e})"
                )
                warnings.warn(msg, stacklevel=3)
                return "error", msg, [], [], 0

        mcp_section: dict[str, Any] = existing.get("mcp_servers", {})
        if not isinstance(mcp_section, dict):
            mcp_section = {}

        written: list[str] = []
        created: list[str] = []
        removed: list[str] = []
        for name, server in enabled.items():
            entry = _to_toml_entry(server)
            if mcp_section.get(name) != entry:
                # Overwriting a same-named table applies the change but is never
                # claimed as owned — only genuinely new tables enter ``created``.
                if name not in mcp_section:
                    created.append(name)
                mcp_section[name] = entry
                written.append(name)

        for name in revoke:
            if name in mcp_section:
                del mcp_section[name]
                removed.append(name)

        if not written and not removed:
            return "skipped", "", [], [], 0

        if dry_run:
            return ("created" if was_new else "updated"), "", written, created, len(removed)

        if mcp_section:
            existing["mcp_servers"] = mcp_section
        else:
            # Removing the last server drops the table rather than leaving an
            # empty `[mcp_servers]` behind — and keeps the textual splice
            # matching, so the user's comments survive that case too.
            existing.pop("mcp_servers", None)

        new_text = self._splice(original, existing, written, removed)
        if new_text is None:
            # Couldn't edit in place — fall back to a full dump, which is
            # correct but drops the user's comments and key ordering.
            new_text = tomli_w.dumps(existing)
        atomic_write_text(path, new_text)
        return ("created" if was_new else "updated"), "", written, created, len(removed)

    @staticmethod
    def _splice(
        original: str,
        expected: dict[str, Any],
        written: list[str],
        removed: list[str],
    ) -> str | None:
        """Apply the server add/remove edits to *original* textually.

        ``.codex/config.toml`` belongs to the user — their model, sandbox and
        profile settings live beside our ``[mcp_servers.*]`` tables — so a
        ``tomllib`` → ``tomli_w`` round-trip that discards every comment in the
        file is too blunt for adding one server. Returns ``None`` when the edit
        can't be applied safely, leaving the caller to round-trip instead.
        """
        import tomli_w

        text = original
        for name in removed:
            spliced = remove_table(text, ("mcp_servers", name))
            if spliced is None:
                return None
            text = spliced
        for name in written:
            rendered = tomli_w.dumps({"mcp_servers": {name: expected["mcp_servers"][name]}})
            spliced = upsert_table(text, ("mcp_servers", name), rendered)
            if spliced is None:
                return None
            text = spliced
        return splice_or_none(text, expected)


def report_dropped_default_fallbacks(
    servers: dict[str, MCPServerConfig],
) -> list[SyncResult]:
    """Manual-fix rows for ``${VAR:-default}`` defaults Codex can't represent.

    :mod:`crossby.sync.mcp_transports` rewrites ``${VAR}`` indirection into
    Codex's env-var fields but drops any ``${VAR:-default}`` fallback — its
    module contract says the literal is written and "a manual-fix note is
    emitted by the caller". Nothing read those ``dropped_default_fallbacks``
    fields until now; this is that caller. One row per (server, field) so the
    default can be restored directly in ``.codex/config.toml``. Same detect-only
    shape as :func:`crossby.sync.mcp_discovery.report_oauth_configs`.
    """
    from crossby.sync.mcp_transports import rewrite_env_for_codex, rewrite_headers_for_codex

    rows: list[SyncResult] = []
    for name, server in servers.items():
        if not server.enabled:
            continue
        dropped: list[tuple[str, str]] = []
        if server.headers:
            dropped += [
                ("header", h)
                for h in rewrite_headers_for_codex(server.headers).dropped_default_fallbacks
            ]
        if server.env:
            dropped += [
                ("env var", e) for e in rewrite_env_for_codex(server.env).dropped_default_fallbacks
            ]
        for kind, field_name in dropped:
            rows.append(
                SyncResult(
                    tool_id=None,
                    concern=SyncConcern.MCP,
                    action="skipped",
                    file_path=None,
                    message=(
                        f"MCP server `{name}` {kind} `{field_name}` uses a "
                        "`${VAR:-default}` default that Codex's config can't represent; "
                        "the default was dropped. This is a manual-fix — set it "
                        "directly in `.codex/config.toml`."
                    ),
                )
            )
    return rows
