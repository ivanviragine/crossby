"""Sync readers — extract sync data directly from tool config files.

Each reader scans the filesystem for a specific concern (rules, agents,
MCP, permissions, hooks) and returns what it found.  The results feed into
:class:`SyncData` so that ``crossby sync`` works without a ``.crossby.yml``.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry, MCPServerConfig
from crossby.sync.base import SyncData


# ---------------------------------------------------------------------------
# Rules reader
# ---------------------------------------------------------------------------

# Tool → relative instruction file path (same as rules.TOOL_TARGETS)
_INSTRUCTION_FILES: dict[AIToolID, str] = {
    AIToolID.CLAUDE: "CLAUDE.md",
    AIToolID.CURSOR: ".cursorrules",
    AIToolID.COPILOT: ".github/copilot-instructions.md",
    AIToolID.GEMINI: "GEMINI.md",
    AIToolID.CODEX: "AGENTS.md",
}

# Priority order when multiple instruction files exist
_RULES_PRIORITY: list[AIToolID] = [
    AIToolID.CODEX,  # AGENTS.md — most generic name
    AIToolID.CLAUDE,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.GEMINI,
]


def detect_rules(project_root: Path) -> dict[AIToolID, str]:
    """Find existing instruction files across all tools.

    Returns a dict of tool → relative file path for each tool that has one.
    Broken symlinks are excluded — only files that actually resolve are returned.
    """
    found: dict[AIToolID, str] = {}
    for tool_id, rel_path in _INSTRUCTION_FILES.items():
        path = project_root / rel_path
        if path.exists():
            found[tool_id] = rel_path
    return found


def suggest_rules_source(found: dict[AIToolID, str]) -> AIToolID | None:
    """Suggest which tool's instruction file should be the canonical source.

    Returns None if no instruction files were found.
    """
    if not found:
        return None
    for tool_id in _RULES_PRIORITY:
        if tool_id in found:
            return tool_id
    return next(iter(found))


# ---------------------------------------------------------------------------
# Agents reader
# ---------------------------------------------------------------------------

# Tool → relative agents directory (same as agents._AGENT_TARGET_PATHS)
_AGENT_DIRS: dict[AIToolID, str] = {
    AIToolID.CLAUDE: ".claude/agents",
    AIToolID.COPILOT: ".github/agents",
    AIToolID.CURSOR: ".cursor/agents",
    AIToolID.GEMINI: ".gemini/agents",
    AIToolID.CODEX: ".agents",
}

_AGENTS_PRIORITY: list[AIToolID] = [
    AIToolID.CLAUDE,
    AIToolID.CODEX,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.GEMINI,
]


def detect_agents(project_root: Path) -> dict[AIToolID, str]:
    """Find existing agent directories across all tools.

    Returns a dict of tool → relative directory path for each tool that has one.
    """
    found: dict[AIToolID, str] = {}
    for tool_id, rel_path in _AGENT_DIRS.items():
        path = project_root / rel_path
        if path.is_dir():
            found[tool_id] = rel_path
    return found


def suggest_agents_source(found: dict[AIToolID, str]) -> AIToolID | None:
    """Suggest which tool's agents directory should be the canonical source."""
    if not found:
        return None
    for tool_id in _AGENTS_PRIORITY:
        if tool_id in found:
            return tool_id
    return next(iter(found))


# ---------------------------------------------------------------------------
# MCP reader
# ---------------------------------------------------------------------------


def discover_mcp(
    project_root: Path, from_tool: AIToolID | None = None
) -> dict[str, MCPServerConfig]:
    """Discover MCP servers from tool config files.

    Uses the existing ``mcp_discovery`` module to scan all tool configs.
    When *from_tool* is set, only that tool's config is scanned.

    Returns server name → MCPServerConfig (validated).
    """
    from crossby.sync.mcp_discovery import discover_mcp_servers

    discovery = discover_mcp_servers(project_root)
    servers: dict[str, MCPServerConfig] = {}
    for name, discovered in discovery.servers.items():
        if from_tool is not None and discovered.source_tool != str(from_tool):
            continue
        try:
            servers[name] = MCPServerConfig(**discovered.data)
        except Exception:
            continue
    return servers


# ---------------------------------------------------------------------------
# Permissions reader
# ---------------------------------------------------------------------------


def _read_claude_allowlist(project_root: Path) -> list[str]:
    """Read Claude allowlist → canonical patterns."""
    settings_path = project_root / ".claude" / "settings.json"
    if not settings_path.is_file():
        return []
    with contextlib.suppress(json.JSONDecodeError, OSError):
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            perms = raw.get("permissions")
            allow = perms.get("allow", []) if isinstance(perms, dict) else []
            if isinstance(allow, list):
                return [
                    p[5:-1]
                    for p in allow
                    if isinstance(p, str) and p.startswith("Bash(") and p.endswith(")")
                ]
    return []


def _read_cursor_allowlist(project_root: Path) -> list[str]:
    """Read Cursor allowlist → canonical patterns."""
    config_file = project_root / ".cursor" / "cli.json"
    if not config_file.is_file():
        return []
    with contextlib.suppress(json.JSONDecodeError, OSError):
        raw = json.loads(config_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            perms = raw.get("permissions")
            allow = perms.get("allow", []) if isinstance(perms, dict) else []
            if isinstance(allow, list):
                return [
                    p[6:-1]
                    for p in allow
                    if isinstance(p, str) and p.startswith("Shell(") and p.endswith(")")
                ]
    return []


def _read_gemini_permissions(project_root: Path) -> list[str]:
    """Read Gemini policy file → canonical patterns."""
    policy_file = project_root / ".gemini" / "policies" / "crossby.toml"
    if not policy_file.is_file():
        return []
    with contextlib.suppress(OSError):
        content = policy_file.read_text(encoding="utf-8")
        result: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("commandPrefix"):
                # Parse: commandPrefix = "binary"
                _, _, value = line.partition("=")
                value = value.strip().strip('"')
                if value:
                    result.append(value)
        return result
    return []


_PERMISSION_READERS: dict[AIToolID, Any] = {
    AIToolID.CLAUDE: _read_claude_allowlist,
    AIToolID.CURSOR: _read_cursor_allowlist,
    AIToolID.GEMINI: _read_gemini_permissions,
}


def discover_permissions(
    project_root: Path, from_tool: AIToolID | None = None
) -> list[str]:
    """Read allowlist patterns from tool configs.

    Scans Claude, Cursor, and Gemini configs for persistent allowlists.
    Returns deduplicated canonical patterns.
    """
    if from_tool is not None and from_tool not in _PERMISSION_READERS:
        return []
    seen: set[str] = set()
    result: list[str] = []
    readers = (
        {from_tool: _PERMISSION_READERS[from_tool]}
        if from_tool and from_tool in _PERMISSION_READERS
        else _PERMISSION_READERS
    )
    for reader_fn in readers.values():
        for pattern in reader_fn(project_root):
            if pattern not in seen:
                seen.add(pattern)
                result.append(pattern)
    return result


# ---------------------------------------------------------------------------
# Hooks reader
# ---------------------------------------------------------------------------

# Reverse event name maps (tool-specific → canonical)
_REVERSE_EVENTS: dict[str, str] = {
    "PreToolUse": "pre_tool_use",
    "preToolUse": "pre_tool_use",
    "BeforeTool": "pre_tool_use",
}

# Reverse tool name maps (tool-specific → canonical)
_REVERSE_TOOLS: dict[str, str] = {
    "Shell": "Bash",
    "shell": "Bash",
    "edit": "Edit",
    "write": "Write",
    "read": "Read",
    "search": "Grep",
    "glob": "Glob",
    "web_search": "WebSearch",
    "web_fetch": "WebFetch",
}


def _reverse_tool_name(name: str) -> str:
    return _REVERSE_TOOLS.get(name, name)


def _reverse_event_name(name: str) -> str:
    return _REVERSE_EVENTS.get(name, name)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with contextlib.suppress(json.JSONDecodeError, OSError):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    return None


def _read_claude_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .claude/settings.json."""
    data = _read_json(project_root / ".claude" / "settings.json")
    if not data:
        return []
    hooks_section = data.get("hooks")
    if not isinstance(hooks_section, dict):
        return []
    result: list[HookEntry] = []
    for event_name, entries in hooks_section.items():
        canonical_event = _reverse_event_name(event_name)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            matcher = entry.get("matcher", "")
            tools = matcher.split("|") if matcher and matcher != ".*" else []
            inner_hooks = entry.get("hooks", [])
            if not isinstance(inner_hooks, list):
                continue
            for inner in inner_hooks:
                if isinstance(inner, dict) and "command" in inner:
                    result.append(HookEntry(
                        event=canonical_event,
                        command=inner["command"],
                        tools=tools,
                    ))
    return result


def _read_cursor_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .cursor/hooks.json."""
    data = _read_json(project_root / ".cursor" / "hooks.json")
    if not data:
        return []
    result: list[HookEntry] = []
    for event_name, entries in data.items():
        if not isinstance(entries, list):
            continue
        canonical_event = _reverse_event_name(event_name)
        for entry in entries:
            if not isinstance(entry, dict) or "command" not in entry:
                continue
            tools_raw = entry.get("tools", [])
            tools = [_reverse_tool_name(t) for t in tools_raw] if isinstance(tools_raw, list) else []
            result.append(HookEntry(
                event=canonical_event,
                command=entry["command"],
                tools=tools,
            ))
    return result


def _read_copilot_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .github/hooks/hooks.json."""
    data = _read_json(project_root / ".github" / "hooks" / "hooks.json")
    if not data:
        return []
    hooks_section = data.get("hooks")
    if not isinstance(hooks_section, dict):
        return []
    result: list[HookEntry] = []
    for event_name, entries in hooks_section.items():
        if not isinstance(entries, list):
            continue
        canonical_event = _reverse_event_name(event_name)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            command = entry.get("bash")
            if not command:
                continue
            result.append(HookEntry(
                event=canonical_event,
                command=command,
                tools=[],
                description=entry.get("comment", ""),
            ))
    return result


def _read_gemini_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .gemini/settings.json.

    Supports both the current nested format (``hooks`` is a dict keyed by event
    name) and the legacy flat-array format (``hooks`` is a list) for backward
    compatibility.
    """
    data = _read_json(project_root / ".gemini" / "settings.json")
    if not data:
        return []
    hooks_raw = data.get("hooks")
    if isinstance(hooks_raw, dict):
        return _read_gemini_hooks_nested(hooks_raw)
    if isinstance(hooks_raw, list):
        return _read_gemini_hooks_flat(hooks_raw)
    return []


def _read_gemini_hooks_nested(hooks_section: dict[str, Any]) -> list[HookEntry]:
    """Parse the nested object-keyed Gemini hooks format."""
    result: list[HookEntry] = []
    for event_name, entries in hooks_section.items():
        canonical_event = _reverse_event_name(event_name)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            matcher = entry.get("matcher", "")
            tools = matcher.split("|") if matcher and matcher != ".*" else []
            inner_hooks = entry.get("hooks", [])
            if not isinstance(inner_hooks, list):
                continue
            for inner in inner_hooks:
                if isinstance(inner, dict) and "command" in inner:
                    result.append(HookEntry(
                        event=canonical_event,
                        command=inner["command"],
                        tools=tools,
                    ))
    return result


def _read_gemini_hooks_flat(hooks_list: list[Any]) -> list[HookEntry]:
    """Parse the legacy flat-array Gemini hooks format."""
    result: list[HookEntry] = []
    for entry in hooks_list:
        if not isinstance(entry, dict) or "command" not in entry:
            continue
        event_name = entry.get("event", "")
        canonical_event = _reverse_event_name(event_name)
        tools_raw = entry.get("tools", [])
        tools = (
            [] if tools_raw == [".*"] else
            [_reverse_tool_name(t) for t in tools_raw] if isinstance(tools_raw, list) else
            []
        )
        result.append(HookEntry(
            event=canonical_event,
            command=entry["command"],
            tools=tools,
        ))
    return result


_HOOK_READERS: dict[AIToolID, Any] = {
    AIToolID.CLAUDE: _read_claude_hooks,
    AIToolID.CURSOR: _read_cursor_hooks,
    AIToolID.COPILOT: _read_copilot_hooks,
    AIToolID.GEMINI: _read_gemini_hooks,
}


def discover_hooks(
    project_root: Path, from_tool: AIToolID | None = None
) -> list[HookEntry]:
    """Read hooks from tool configs.

    Returns deduplicated canonical hook entries (dedup by event+command).
    """
    if from_tool is not None and from_tool not in _HOOK_READERS:
        return []
    seen: set[tuple[str, str]] = set()
    result: list[HookEntry] = []
    readers = (
        {from_tool: _HOOK_READERS[from_tool]}
        if from_tool and from_tool in _HOOK_READERS
        else _HOOK_READERS
    )
    for reader_fn in readers.values():
        for hook in reader_fn(project_root):
            key = (hook.event, hook.command)
            if key not in seen:
                seen.add(key)
                result.append(hook)
    return result


# ---------------------------------------------------------------------------
# Project scan (for wizard display)
# ---------------------------------------------------------------------------


@dataclass
class ConcernScan:
    """Scan result for a single concern — what was found and where."""

    found: dict[AIToolID, Any]  # tool → concern-specific data
    summary: str  # human-readable summary for wizard display


@dataclass
class ProjectScan:
    """Full project scan result used by the sync wizard."""

    installed_tools: list[AIToolID]
    rules: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    agents: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    mcp: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    permissions: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    hooks: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))


def scan_project(project_root: Path, installed_tools: list[AIToolID]) -> ProjectScan:
    """Scan the project for all sync-relevant data across all tools.

    Returns a :class:`ProjectScan` with per-concern findings, used by the
    interactive wizard to display what was found and ask for confirmation.
    """
    # Rules
    rules_found = detect_rules(project_root)
    rules_summary = ", ".join(
        f"{rel_path} ({tool_id})" for tool_id, rel_path in rules_found.items()
    ) if rules_found else "none found"

    # Agents
    agents_found = detect_agents(project_root)
    agents_details: dict[AIToolID, str] = {}
    for tool_id, rel_path in agents_found.items():
        dir_path = project_root / rel_path
        count = sum(1 for _ in dir_path.glob("*.md")) if dir_path.is_dir() else 0
        agents_details[tool_id] = f"{rel_path} ({count} file{'s' if count != 1 else ''})"
    agents_summary = ", ".join(agents_details.values()) if agents_details else "none found"

    # MCP — scan per tool
    from crossby.sync.mcp_discovery import discover_mcp_servers
    discovery = discover_mcp_servers(project_root)
    mcp_by_tool: dict[AIToolID, list[str]] = {}
    for name, ds in discovery.servers.items():
        try:
            tid = AIToolID(ds.source_tool)
        except (ValueError, TypeError):
            continue
        mcp_by_tool.setdefault(tid, []).append(name)
    mcp_summary = (
        f"{len(discovery.servers)} server(s) from "
        + ", ".join(str(t) for t in mcp_by_tool)
        if discovery.servers else "none found"
    )

    # Permissions — scan per tool
    perm_by_tool: dict[AIToolID, list[str]] = {}
    for tool_id, reader_fn in _PERMISSION_READERS.items():
        patterns = reader_fn(project_root)
        if patterns:
            perm_by_tool[tool_id] = patterns
    total_perms = sum(len(v) for v in perm_by_tool.values())
    perm_summary = (
        f"{total_perms} pattern(s) from "
        + ", ".join(str(t) for t in perm_by_tool)
        if perm_by_tool else "none found"
    )

    # Hooks — scan per tool
    hooks_by_tool: dict[AIToolID, list[HookEntry]] = {}
    for tool_id, reader_fn in _HOOK_READERS.items():
        hooks = reader_fn(project_root)
        if hooks:
            hooks_by_tool[tool_id] = hooks
    total_hooks = sum(len(v) for v in hooks_by_tool.values())
    hooks_summary = (
        f"{total_hooks} hook(s) from "
        + ", ".join(str(t) for t in hooks_by_tool)
        if hooks_by_tool else "none found"
    )

    return ProjectScan(
        installed_tools=installed_tools,
        rules=ConcernScan(found=rules_found, summary=rules_summary),
        agents=ConcernScan(found=dict(agents_found), summary=agents_summary),
        mcp=ConcernScan(found=dict(mcp_by_tool), summary=mcp_summary),
        permissions=ConcernScan(found=dict(perm_by_tool), summary=perm_summary),
        hooks=ConcernScan(found=dict(hooks_by_tool), summary=hooks_summary),
    )


# ---------------------------------------------------------------------------
# SyncData builder
# ---------------------------------------------------------------------------


def build_sync_data(
    project_root: Path,
    from_tool: AIToolID | None = None,
) -> SyncData:
    """Build :class:`SyncData` by reading directly from tool configs.

    When *from_tool* is specified, only that tool's configs are read.
    Otherwise all tool configs are scanned and auto-resolved.
    """
    # Rules
    rules_found = detect_rules(project_root)
    rules_source: str | None = None
    if from_tool and from_tool in rules_found:
        rules_source = rules_found[from_tool]
    elif not from_tool and rules_found:
        source_tool = suggest_rules_source(rules_found)
        if source_tool:
            rules_source = rules_found[source_tool]

    # Agents
    agents_found = detect_agents(project_root)
    agents_source: str | None = None
    if from_tool and from_tool in agents_found:
        agents_source = agents_found[from_tool]
    elif not from_tool and agents_found:
        source_tool = suggest_agents_source(agents_found)
        if source_tool:
            agents_source = agents_found[source_tool]

    # MCP, permissions, hooks
    mcp_servers = discover_mcp(project_root, from_tool=from_tool)
    allowed_commands = discover_permissions(project_root, from_tool=from_tool)
    hooks = discover_hooks(project_root, from_tool=from_tool)

    return SyncData(
        rules_source=rules_source,
        agents_source=agents_source,
        mcp_servers=mcp_servers,
        allowed_commands=allowed_commands,
        hooks=hooks,
    )
