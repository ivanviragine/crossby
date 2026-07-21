"""MCP server discovery — scan existing tool configs for mcp_servers entries.

Used by `crossby init` to propose a unified mcp_servers section from
servers already configured in any of the supported tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crossby.sync.base import SyncConcern, SyncResult


@dataclass
class DiscoveredServer:
    """A single MCP server entry found in a tool config."""

    name: str
    source_tool: str
    data: dict[str, Any]


@dataclass
class DiscoveryResult:
    """Result of scanning all tool configs for MCP servers."""

    servers: dict[str, DiscoveredServer] = field(default_factory=dict)
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)  # (name, tool1, tool2)
    # (name, source_tool) for every kept server whose raw entry declared an
    # ``oauth`` block. No ``MCPServerConfig`` field represents OAuth config
    # (callbackPort, clientId, authServerMetadataUrl, ...), so every writer
    # silently drops it today; this list lets callers surface that instead
    # of losing it with no trace. See :func:`report_oauth_configs`.
    oauth_servers: list[tuple[str, str]] = field(default_factory=list)


# Claude's user-scope config file (``claude mcp add --scope user``). Module-level
# so tests can monkeypatch it, matching the pattern used for Cursor's global
# config path in sync/permissions.py.
_GLOBAL_CLAUDE_JSON_PATH = Path.home() / ".claude.json"


def _read_json_section(path: Path, key: str) -> dict[str, Any] | None:
    """Read a JSON file and return the value at ``key``, or None on error."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    section = raw.get(key)
    if not isinstance(section, dict):
        return None
    return section


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool-specific entry to the canonical crossby format."""
    result: dict[str, Any] = {}

    # Copilot adds a "type" field for transport — translate back
    transport = entry.get("type") or entry.get("transport")
    if transport and transport != "stdio":
        result["transport"] = transport

    if "command" in entry:
        result["command"] = entry["command"]
    if entry.get("args"):
        result["args"] = entry["args"]
    if entry.get("env"):
        result["env"] = entry["env"]
    if entry.get("headers"):
        result["headers"] = entry["headers"]
    # Antigravity CLI uses "serverUrl" instead of "url" for remote servers.
    if "url" in entry:
        result["url"] = entry["url"]
    elif "serverUrl" in entry:
        result["url"] = entry["serverUrl"]

    return result


def discover_mcp_servers(project_root: Path) -> DiscoveryResult:
    """Scan all tool config files for MCP server definitions.

    Scans:
    - .mcp.json → mcpServers (Claude's project-scope file, e.g. ``claude mcp
      add --scope project``; this is the canonical location for most real
      projects, checked into version control)
    - .claude/settings.json → mcpServers (legacy/manual location)
    - ~/.claude.json → mcpServers (Claude's user-scope file)
    - .cursor/mcp.json → mcpServers
    - .vscode/mcp.json → servers (Copilot format)
    - .agents/mcp_config.json → mcpServers (Antigravity CLI)
    - .codex/config.toml → mcp_servers

    Claude sources are scanned most-specific-first (project .mcp.json, then
    .claude/settings.json, then the user-scope ~/.claude.json) so the
    first-seen-wins merge below prefers the project-scoped definition when
    the same server name appears in more than one Claude source.

    Returns:
        DiscoveryResult with merged servers (first-seen wins) and conflicts.
    """
    result = DiscoveryResult()

    sources: list[tuple[str, Path, str]] = [
        ("claude", project_root / ".mcp.json", "mcpServers"),
        ("claude", project_root / ".claude" / "settings.json", "mcpServers"),
        ("claude", _GLOBAL_CLAUDE_JSON_PATH, "mcpServers"),
        ("cursor", project_root / ".cursor" / "mcp.json", "mcpServers"),
        ("copilot", project_root / ".vscode" / "mcp.json", "servers"),
        ("antigravity-cli", project_root / ".agents" / "mcp_config.json", "mcpServers"),
    ]

    for tool, path, key in sources:
        section = _read_json_section(path, key)
        if section is None:
            continue
        for name, entry in section.items():
            if not isinstance(entry, dict):
                continue
            normalized = _normalize_entry(entry)
            if name in result.servers:
                existing_tool = result.servers[name].source_tool
                # Multiple Claude scopes (.mcp.json, .claude/settings.json,
                # ~/.claude.json) share the "claude" tool label; a name
                # collision between them is scope precedence, not a
                # cross-tool conflict, so it's resolved silently
                # (first-seen — i.e. most specific scope — wins).
                if existing_tool != tool:
                    result.conflicts.append((name, existing_tool, tool))
            else:
                result.servers[name] = DiscoveredServer(
                    name=name, source_tool=tool, data=normalized
                )
                if isinstance(entry.get("oauth"), dict):
                    result.oauth_servers.append((name, tool))

    # Codex TOML
    codex_path = project_root / ".codex" / "config.toml"
    if codex_path.exists():
        toml_section = _read_codex_mcp(codex_path)
        if toml_section is not None:
            for name, entry in toml_section.items():
                if not isinstance(entry, dict):
                    continue
                if name in result.servers:
                    result.conflicts.append((name, result.servers[name].source_tool, "codex"))
                else:
                    result.servers[name] = DiscoveredServer(
                        name=name, source_tool="codex", data=dict(entry)
                    )

    return result


def _read_codex_mcp(path: Path) -> dict[str, Any] | None:
    """Read mcp_servers from a Codex TOML config file."""
    try:
        import tomllib

        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        section = raw.get("mcp_servers")
        if isinstance(section, dict):
            return section
    except Exception:
        pass
    return None


def report_oauth_configs(project_root: Path) -> list[SyncResult]:
    """Report MCP servers whose source entry has an ``oauth`` block.

    No writer in :mod:`crossby.sync.mcp` ports OAuth config (``callbackPort``,
    ``clientId``, ``authServerMetadataUrl``, ...) across tools — it's dropped
    silently during discovery/normalization today because
    :class:`crossby.models.config.MCPServerConfig` has no field for it. This
    turns that silent drop into a manual-fix report row instead, mirroring
    :func:`crossby.sync.plugins.report_plugins`'s detect-only pattern.

    ``file_path=None`` is deliberate (there is no per-target artifact this
    row is about) — it's what makes :func:`crossby.sync.report.classify_status`
    read the row as ``Not Added`` and :func:`crossby.sync.plan.summarize_plan`
    count it toward the doctor readiness score.
    """
    discovery = discover_mcp_servers(project_root)
    return [
        SyncResult(
            tool_id=None,
            concern=SyncConcern.MCP,
            action="skipped",
            file_path=None,
            message=(
                f"MCP server `{name}` (from {source_tool}) has an `oauth` block Crossby "
                "does not port across tools; this is a manual-fix — configure OAuth "
                "directly in each target tool's native config."
            ),
        )
        for name, source_tool in discovery.oauth_servers
    ]
