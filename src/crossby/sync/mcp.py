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
from pathlib import Path
from typing import Any

from crossby.models.config import MCPServerConfig
from crossby.sync.base import AbstractSyncWriter, SyncResult
from crossby.sync.json_utils import read_merge_write_json


def _split_servers(
    servers: dict[str, MCPServerConfig],
) -> tuple[dict[str, MCPServerConfig], set[str]]:
    """Split servers into (enabled, disabled_names)."""
    enabled = {name: s for name, s in servers.items() if s.enabled}
    disabled = {name for name, s in servers.items() if not s.enabled}
    return enabled, disabled


def _to_stdio_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a stdio server to the standard JSON entry (Claude/Cursor/Gemini)."""
    entry: dict[str, Any] = {"command": server.command}
    if server.args:
        entry["args"] = server.args
    if server.env:
        entry["env"] = server.env
    return entry


def _to_http_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert an http/sse server to the standard JSON entry."""
    entry: dict[str, Any] = {"url": server.url, "transport": server.transport}
    if server.env:
        entry["env"] = server.env
    return entry


def _to_json_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a server to the standard JSON entry (Claude/Cursor/Gemini format)."""
    if server.command is not None:
        return _to_stdio_entry(server)
    return _to_http_entry(server)


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
    return entry


def _to_toml_entry(server: MCPServerConfig) -> dict[str, Any]:
    """Convert a server to the TOML entry format (Codex)."""
    entry: dict[str, Any] = {}
    if server.command is not None:
        entry["command"] = server.command
        if server.args:
            entry["args"] = server.args
    if server.url is not None:
        entry["url"] = server.url
    if server.env:
        entry["env"] = dict(server.env)
    return entry


class ClaudeMCPWriter(AbstractSyncWriter):
    """Merges MCP servers into .claude/settings.json → mcpServers."""

    @property
    def tool_id(self) -> str:
        return "claude"

    def write(
        self,
        servers: dict[str, MCPServerConfig],
        project_root: Path,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        path = project_root / ".claude" / "settings.json"
        enabled, disabled = _split_servers(servers)
        updates = {name: _to_json_entry(s) for name, s in enabled.items()}
        action, message = read_merge_write_json(path, "mcpServers", updates, disabled, dry_run)
        return [SyncResult(tool=self.tool_id, path=path, action=action, message=message, dry_run=dry_run)]  # type: ignore[arg-type]


class CursorMCPWriter(AbstractSyncWriter):
    """Merges MCP servers into .cursor/mcp.json → mcpServers."""

    @property
    def tool_id(self) -> str:
        return "cursor"

    def write(
        self,
        servers: dict[str, MCPServerConfig],
        project_root: Path,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        path = project_root / ".cursor" / "mcp.json"
        enabled, disabled = _split_servers(servers)
        updates = {name: _to_json_entry(s) for name, s in enabled.items()}
        action, message = read_merge_write_json(path, "mcpServers", updates, disabled, dry_run)
        return [SyncResult(tool=self.tool_id, path=path, action=action, message=message, dry_run=dry_run)]  # type: ignore[arg-type]


class CopilotMCPWriter(AbstractSyncWriter):
    """Merges MCP servers into .vscode/mcp.json → servers (Copilot format)."""

    @property
    def tool_id(self) -> str:
        return "copilot"

    def write(
        self,
        servers: dict[str, MCPServerConfig],
        project_root: Path,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        path = project_root / ".vscode" / "mcp.json"
        enabled, disabled = _split_servers(servers)
        updates = {name: _to_copilot_entry(s) for name, s in enabled.items()}
        action, message = read_merge_write_json(path, "servers", updates, disabled, dry_run)
        return [SyncResult(tool=self.tool_id, path=path, action=action, message=message, dry_run=dry_run)]  # type: ignore[arg-type]


class GeminiMCPWriter(AbstractSyncWriter):
    """Merges MCP servers into .gemini/settings.json → mcpServers."""

    @property
    def tool_id(self) -> str:
        return "gemini"

    def write(
        self,
        servers: dict[str, MCPServerConfig],
        project_root: Path,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        path = project_root / ".gemini" / "settings.json"
        enabled, disabled = _split_servers(servers)
        updates = {name: _to_json_entry(s) for name, s in enabled.items()}
        action, message = read_merge_write_json(path, "mcpServers", updates, disabled, dry_run)
        return [SyncResult(tool=self.tool_id, path=path, action=action, message=message, dry_run=dry_run)]  # type: ignore[arg-type]


class CodexMCPWriter(AbstractSyncWriter):
    """Merges MCP servers into .codex/config.toml → [mcp_servers.<name>]."""

    @property
    def tool_id(self) -> str:
        return "codex"

    def write(
        self,
        servers: dict[str, MCPServerConfig],
        project_root: Path,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        path = project_root / ".codex" / "config.toml"
        enabled, disabled = _split_servers(servers)
        action, message = self._write_toml(path, enabled, disabled, dry_run)
        return [SyncResult(tool=self.tool_id, path=path, action=action, message=message, dry_run=dry_run)]  # type: ignore[arg-type]

    def _write_toml(
        self,
        path: Path,
        enabled: dict[str, MCPServerConfig],
        disabled: set[str],
        dry_run: bool,
    ) -> tuple[str, str]:
        try:
            import tomli_w

            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            msg = (
                "tomli-w is not installed — skipping Codex MCP sync. "
                "Install it with: pip install tomli-w"
            )
            warnings.warn(msg, stacklevel=3)
            return "error", msg

        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = tomllib.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                msg = (
                    f"{path} contains invalid TOML — skipping MCP sync for Codex. "
                    f"Fix the file manually or delete it. ({e})"
                )
                warnings.warn(msg, stacklevel=3)
                return "error", msg

        was_new = not path.exists()
        mcp_section: dict[str, Any] = existing.get("mcp_servers", {})
        if not isinstance(mcp_section, dict):
            mcp_section = {}

        changed = False
        for name, server in enabled.items():
            entry = _to_toml_entry(server)
            if mcp_section.get(name) != entry:
                mcp_section[name] = entry
                changed = True

        for name in disabled:
            if name in mcp_section:
                del mcp_section[name]
                changed = True

        if not changed:
            return "skipped", ""

        if dry_run:
            return ("created" if was_new else "updated"), ""

        existing["mcp_servers"] = mcp_section
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(existing), encoding="utf-8")
        return ("created" if was_new else "updated"), ""


# Registry: all available MCP writers, keyed by tool_id
MCP_WRITERS: dict[str, AbstractSyncWriter] = {
    w.tool_id: w
    for w in [
        ClaudeMCPWriter(),
        CursorMCPWriter(),
        CopilotMCPWriter(),
        GeminiMCPWriter(),
        CodexMCPWriter(),
    ]
}
