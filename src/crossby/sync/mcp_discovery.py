"""MCP server discovery — scan existing tool configs for mcp_servers entries.

Used by `crossby init` to propose a unified mcp_servers section from
servers already configured in any of the supported tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


def _read_json_section(path: Path, key: str) -> dict[str, Any] | None:
    """Read a JSON file and return the value at ``key``, or None on error."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
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
    if "args" in entry and entry["args"]:
        result["args"] = entry["args"]
    if "env" in entry and entry["env"]:
        result["env"] = entry["env"]
    if "url" in entry:
        result["url"] = entry["url"]

    return result


def discover_mcp_servers(project_root: Path) -> DiscoveryResult:
    """Scan all tool config files for MCP server definitions.

    Scans:
    - .claude/settings.json → mcpServers
    - .cursor/mcp.json → mcpServers
    - .vscode/mcp.json → servers (Copilot format)
    - .gemini/settings.json → mcpServers
    - .codex/config.toml → mcp_servers

    Returns:
        DiscoveryResult with merged servers (first-seen wins) and conflicts.
    """
    result = DiscoveryResult()

    sources: list[tuple[str, Path, str]] = [
        ("claude", project_root / ".claude" / "settings.json", "mcpServers"),
        ("cursor", project_root / ".cursor" / "mcp.json", "mcpServers"),
        ("copilot", project_root / ".vscode" / "mcp.json", "servers"),
        ("gemini", project_root / ".gemini" / "settings.json", "mcpServers"),
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
                result.conflicts.append((name, result.servers[name].source_tool, tool))
            else:
                result.servers[name] = DiscoveredServer(
                    name=name, source_tool=tool, data=normalized
                )

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
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                return None
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        section = raw.get("mcp_servers")
        if isinstance(section, dict):
            return section  # type: ignore[return-value]
    except Exception:
        pass
    return None
