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
from crossby.models.config import CrossbyConfig, MCPServerConfig
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncResult
from crossby.sync.json_utils import SyncAction, read_merge_write_json


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


class _JsonMCPWriter(AbstractSyncWriter):
    """Base class for JSON-based MCP writers (Claude, Cursor, Gemini, Copilot)."""

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
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        dirname, filename = self._config_path_parts
        path = project_root / dirname / filename
        enabled, disabled = _split_servers(config.mcp_servers)
        updates = {name: self._to_entry(s) for name, s in enabled.items()}
        action, message = read_merge_write_json(path, self._mcp_key, updates, disabled, dry_run)
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=message or None,
        )


class ClaudeMCPWriter(_JsonMCPWriter):
    """Merges MCP servers into .claude/settings.json → mcpServers."""

    tool_id = AIToolID.CLAUDE

    @property
    def _config_path_parts(self) -> tuple[str, str]:
        return ".claude", "settings.json"

    @property
    def _mcp_key(self) -> str:
        return "mcpServers"


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


class GeminiMCPWriter(_JsonMCPWriter):
    """Merges MCP servers into .gemini/settings.json → mcpServers."""

    tool_id = AIToolID.GEMINI

    @property
    def _config_path_parts(self) -> tuple[str, str]:
        return ".gemini", "settings.json"

    @property
    def _mcp_key(self) -> str:
        return "mcpServers"


class CodexMCPWriter(AbstractSyncWriter):
    """Merges MCP servers into .codex/config.toml → [mcp_servers.<name>]."""

    tool_id = AIToolID.CODEX
    concern = SyncConcern.MCP

    def sync(
        self,
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        path = project_root / ".codex" / "config.toml"
        enabled, disabled = _split_servers(config.mcp_servers)
        action, message = self._write_toml(path, enabled, disabled, dry_run)
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=message or None,
        )

    def _write_toml(
        self,
        path: Path,
        enabled: dict[str, MCPServerConfig],
        disabled: set[str],
        dry_run: bool,
    ) -> tuple[SyncAction, str]:
        try:
            import tomli_w
        except ImportError:
            msg = (
                "tomli-w is not installed — skipping Codex MCP sync. "
                "Install it with: pip install tomli-w"
            )
            warnings.warn(msg, stacklevel=3)
            return "error", msg

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                msg = (
                    "Neither tomllib (Python 3.11+) nor tomli is available — "
                    "skipping Codex MCP sync. Install tomli: pip install tomli"
                )
                warnings.warn(msg, stacklevel=3)
                return "error", msg

        was_new = not path.exists()
        existing: dict[str, Any] = {}
        if not was_new:
            try:
                existing = tomllib.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                msg = (
                    f"{path} contains invalid TOML — skipping MCP sync for Codex. "
                    f"Fix the file manually or delete it. ({e})"
                )
                warnings.warn(msg, stacklevel=3)
                return "error", msg

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


# Registry: all available MCP writers, keyed by tool_id string
MCP_WRITERS: dict[str, AbstractSyncWriter] = {
    str(w.tool_id): w
    for w in [
        ClaudeMCPWriter(),
        CursorMCPWriter(),
        CopilotMCPWriter(),
        GeminiMCPWriter(),
        CodexMCPWriter(),
    ]
}
