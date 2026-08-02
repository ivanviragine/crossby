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

import structlog

from crossby.config.skills import _SCAN_ORDER as _SKILLS_SCAN_ORDER
from crossby.config.skills import SKILLS_DIR, count_skills
from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry, MCPServerConfig
from crossby.sync.base import SyncData

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Rules reader
# ---------------------------------------------------------------------------

# Tool → relative instruction file path (same as rules.TOOL_TARGETS)
_INSTRUCTION_FILES: dict[AIToolID, str] = {
    AIToolID.CLAUDE: "CLAUDE.md",
    AIToolID.CURSOR: ".cursorrules",
    AIToolID.COPILOT: ".github/copilot-instructions.md",
    AIToolID.CODEX: "AGENTS.md",
    AIToolID.ANTIGRAVITY_CLI: "AGENTS.md",
}

# Priority order when multiple instruction files exist
_RULES_PRIORITY: list[AIToolID] = [
    AIToolID.CODEX,  # AGENTS.md — most generic name
    AIToolID.CLAUDE,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.ANTIGRAVITY_CLI,
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
    AIToolID.CODEX: ".codex/agents",
    AIToolID.ANTIGRAVITY_CLI: ".agents/agents",
}

_AGENTS_PRIORITY: list[AIToolID] = [
    AIToolID.CLAUDE,
    AIToolID.CODEX,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.ANTIGRAVITY_CLI,
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
# Skills reader
# ---------------------------------------------------------------------------

# _SKILLS_SCAN_ORDER is imported from config.skills._SCAN_ORDER.
# The order matters: CLAUDE → CODEX → ANTIGRAVITY_CLI → CURSOR → COPILOT.
# CLAUDE is preferred as the canonical source when multiple tools have skills dirs.
# CURSOR and COPILOT are last since they are usually the symlink targets, not sources.


def detect_skills(project_root: Path) -> dict[AIToolID, str]:
    """Find existing skills directories across all tools.

    Returns a dict of tool → relative directory path for each tool that has one.
    Distinct from config.skills.detect_skills_source(), which returns a single
    absolute Path | None for the non-symlinked source.
    """
    found: dict[AIToolID, str] = {}
    for tool_id, rel_path in SKILLS_DIR.items():
        path = project_root / rel_path
        if path.is_dir():
            found[tool_id] = rel_path
    return found


def suggest_skills_source(found: dict[AIToolID, str]) -> AIToolID | None:
    """Suggest which tool's skills directory should be the canonical source."""
    if not found:
        return None
    for tool_id in _SKILLS_SCAN_ORDER:
        if tool_id in found:
            return tool_id
    return next(iter(found))


# ---------------------------------------------------------------------------
# MCP reader
# ---------------------------------------------------------------------------


def discover_mcp(
    project_root: Path,
    from_tool: AIToolID | None = None,
    *,
    include_user_scope: bool = False,
) -> dict[str, MCPServerConfig]:
    """Discover MCP servers from tool config files.

    Uses the existing ``mcp_discovery`` module to scan all tool configs.
    When *from_tool* is set, only that tool's config is scanned. User-scope
    ``~/.claude.json`` is only read when *include_user_scope* is set.

    Returns server name → MCPServerConfig (validated).
    """
    from crossby.sync.mcp_discovery import discover_mcp_servers

    discovery = discover_mcp_servers(project_root, include_user_scope=include_user_scope)
    for name, kept_from, ignored_from in discovery.conflicts:
        logger.warning(
            "mcp.conflict",
            name=name,
            kept_from=kept_from,
            ignored_from=ignored_from,
            hint=(
                f"MCP server {name!r} is defined in both {kept_from!r} and "
                f"{ignored_from!r}; keeping the {kept_from!r} definition"
            ),
        )
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


_PERMISSION_READERS: dict[AIToolID, Any] = {
    AIToolID.CLAUDE: _read_claude_allowlist,
    AIToolID.CURSOR: _read_cursor_allowlist,
}


def discover_permissions(project_root: Path, from_tool: AIToolID | None = None) -> list[str]:
    """Read allowlist patterns from tool configs.

    Scans Claude and Cursor configs for persistent allowlists.
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
    # pre_tool_use
    "PreToolUse": "pre_tool_use",
    "preToolUse": "pre_tool_use",
    # post_tool_use
    "PostToolUse": "post_tool_use",
    "postToolUse": "post_tool_use",
    # session_start
    "SessionStart": "session_start",
    "sessionStart": "session_start",
    # user_prompt_submit
    "UserPromptSubmit": "user_prompt_submit",
    "userPromptSubmit": "user_prompt_submit",
    "beforeSubmitPrompt": "user_prompt_submit",
    # stop
    "Stop": "stop",
    "stop": "stop",
    "agentStop": "stop",  # Copilot names its turn-complete event agentStop
    # notification
    "Notification": "notification",
    "notification": "notification",
    # Cursor's dedicated shell event. CursorHooksWriter fans a shell-scoped
    # pre_tool_use hook out to it, so it must fold back into pre_tool_use or a
    # read → write cycle grows a spurious extra hook every time.
    "beforeShellExecution": "pre_tool_use",
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
    # Antigravity CLI natives (see _TOOL_NAME_MAP[ANTIGRAVITY_CLI]).
    "write_to_file": "Write",
    "replace_file_content": "Edit",
    "multi_replace_file_content": "MultiEdit",
    "run_command": "Bash",
    "view_file": "Read",
    "grep_search": "Grep",
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
                    result.append(
                        HookEntry(
                            event=canonical_event,
                            command=inner["command"],
                            tools=tools,
                        )
                    )
    return result


def _read_cursor_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .cursor/hooks.json.

    Reads the real Cursor shape — ``{"version": 1, "hooks": {"<event>": [...]}}``
    with a ``matcher`` regex — and still accepts the flat top-level layout
    crossby wrote before 0.13 so an un-migrated file round-trips.

    ``beforeShellExecution`` maps back to ``pre_tool_use``, because that is what
    ``CursorHooksWriter`` fans a shell-scoped pre-tool hook out to. Entries are
    deduped by ``(event, command)`` so the fanned-out pair collapses back into
    the single :class:`HookEntry` it came from instead of reappearing as a
    spurious second hook on every read → write cycle.

    Note that ``fail_closed`` and ``timeout`` are deliberately not read back
    here; the readers drop both for every tool today (see ``_read_copilot_hooks``
    too). That is a known round-trip gap tracked in #88 §5, not an oversight of
    this function — the writers are upgrade-safe and never downgrade an existing
    ``failClosed``, which bounds the impact.
    """
    data = _read_json(project_root / ".cursor" / "hooks.json")
    if not data:
        return []
    hooks_section = data.get("hooks")
    # Pre-0.13 crossby wrote event arrays at the top level with no wrapper.
    source = hooks_section if isinstance(hooks_section, dict) else data

    merged: dict[tuple[str, str], HookEntry] = {}
    for event_name, entries in source.items():
        if not isinstance(entries, list):
            continue
        canonical_event = _reverse_event_name(event_name)
        for entry in entries:
            if not isinstance(entry, dict) or "command" not in entry:
                continue
            command = entry["command"]
            key = (canonical_event, command)
            tools = _cursor_entry_tools(entry)
            previous = merged.get(key)
            if previous is None:
                merged[key] = HookEntry(event=canonical_event, command=command, tools=tools)
                continue
            # The fan-out pair collapses here. The `beforeShellExecution` half is
            # written unscoped, so whichever half is read second must not erase
            # the other's tool scope — keep the scoped one regardless of the key
            # order the JSON happened to be in.
            if tools and not previous.tools:
                merged[key] = HookEntry(event=canonical_event, command=command, tools=tools)
    return list(merged.values())


def _cursor_entry_tools(entry: dict[str, Any]) -> list[str]:
    """Recover canonical tool names from a Cursor entry's scope.

    Prefers the ``matcher`` regex Cursor actually uses (alternation of tool
    names, e.g. ``Write|Shell``); falls back to the ``tools`` array that only
    ever existed in crossby's own pre-0.13 output. A catch-all matcher means
    "all tools", which is the same as no scope.
    """
    matcher = entry.get("matcher")
    if isinstance(matcher, str) and matcher and matcher not in (".*", "*"):
        return [_reverse_tool_name(t) for t in matcher.split("|") if t]
    tools_raw = entry.get("tools")
    if isinstance(tools_raw, list):
        return [_reverse_tool_name(t) for t in tools_raw]
    return []


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
            result.append(
                HookEntry(
                    event=canonical_event,
                    command=command,
                    tools=[],
                    description=entry.get("comment", ""),
                )
            )
    return result


_HOOK_READERS: dict[AIToolID, Any] = {
    AIToolID.CLAUDE: _read_claude_hooks,
    AIToolID.CURSOR: _read_cursor_hooks,
    AIToolID.COPILOT: _read_copilot_hooks,
}


def discover_hooks(project_root: Path, from_tool: AIToolID | None = None) -> list[HookEntry]:
    """Read hooks from tool configs.

    When the same ``(event, command)`` is defined by multiple tools, tool
    scopes are unioned — an empty ``tools`` list means "all tools" and wins.
    """
    if from_tool is not None and from_tool not in _HOOK_READERS:
        return []
    merged: dict[tuple[str, str], HookEntry] = {}
    readers = (
        {from_tool: _HOOK_READERS[from_tool]}
        if from_tool and from_tool in _HOOK_READERS
        else _HOOK_READERS
    )
    for reader_fn in readers.values():
        for hook in reader_fn(project_root):
            key = (hook.event, hook.command)
            existing = merged.get(key)
            if existing is None:
                merged[key] = hook
                continue
            if not existing.tools or not hook.tools:
                unioned: list[str] = []
            else:
                unioned = list(dict.fromkeys([*existing.tools, *hook.tools]))
            merged[key] = HookEntry(
                event=existing.event,
                command=existing.command,
                tools=unioned,
                description=existing.description or hook.description,
            )
    return list(merged.values())


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
    skills: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    mcp: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    permissions: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    hooks: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))
    plugins: ConcernScan = field(default_factory=lambda: ConcernScan({}, ""))


def scan_project(
    project_root: Path,
    installed_tools: list[AIToolID],
    *,
    include_user_scope: bool = False,
) -> ProjectScan:
    """Scan the project for all sync-relevant data across all tools.

    Returns a :class:`ProjectScan` with per-concern findings, used by the
    interactive wizard to display what was found and ask for confirmation.
    When *include_user_scope* is set, user-scope ``~/.claude.json`` MCP servers
    are included in the MCP scan and labelled ``user``.
    """
    # Rules
    rules_found = detect_rules(project_root)
    rules_summary = (
        ", ".join(f"{rel_path} ({tool_id})" for tool_id, rel_path in rules_found.items())
        if rules_found
        else "none found"
    )

    # Agents
    agents_found = detect_agents(project_root)
    agents_details: dict[AIToolID, str] = {}
    for tool_id, rel_path in agents_found.items():
        dir_path = project_root / rel_path
        count = sum(1 for _ in dir_path.glob("*.md")) if dir_path.is_dir() else 0
        agents_details[tool_id] = f"{rel_path} ({count} file{'s' if count != 1 else ''})"
    agents_summary = ", ".join(agents_details.values()) if agents_details else "none found"

    # Skills — count_skills reuses the same SKILL.md-bearing-subdir logic as detection.py
    skills_found = detect_skills(project_root)
    skills_details: dict[AIToolID, str] = {}
    for tool_id, rel_path in skills_found.items():
        dir_path = project_root / rel_path
        n = count_skills(dir_path)
        skills_details[tool_id] = f"{rel_path} ({n} skill{'s' if n != 1 else ''})"
    skills_summary = ", ".join(skills_details.values()) if skills_details else "none found"

    # MCP — scan per tool
    from crossby.sync.mcp_discovery import discover_mcp_servers

    discovery = discover_mcp_servers(project_root, include_user_scope=include_user_scope)
    mcp_by_tool: dict[AIToolID, list[str]] = {}
    for name, ds in discovery.servers.items():
        try:
            tid = AIToolID(ds.source_tool)
        except (ValueError, TypeError):
            continue
        mcp_by_tool.setdefault(tid, []).append(name)
    if discovery.servers:
        n_user = sum(1 for ds in discovery.servers.values() if ds.scope == "user")
        n_project = len(discovery.servers) - n_user
        scope_note = f" ({n_project} project, {n_user} user)" if n_user else ""
        mcp_summary = (
            f"{len(discovery.servers)} server(s) from "
            + ", ".join(str(t) for t in mcp_by_tool)
            + scope_note
        )
    else:
        mcp_summary = "none found"
        if not include_user_scope:
            mcp_summary += " (pass --include-user-scope to also read ~/.claude.json)"

    # Permissions — scan per tool
    perm_by_tool: dict[AIToolID, list[str]] = {}
    for tool_id, reader_fn in _PERMISSION_READERS.items():
        patterns = reader_fn(project_root)
        if patterns:
            perm_by_tool[tool_id] = patterns
    total_perms = sum(len(v) for v in perm_by_tool.values())
    perm_summary = (
        f"{total_perms} pattern(s) from " + ", ".join(str(t) for t in perm_by_tool)
        if perm_by_tool
        else "none found"
    )

    # Hooks — scan per tool
    hooks_by_tool: dict[AIToolID, list[HookEntry]] = {}
    for tool_id, reader_fn in _HOOK_READERS.items():
        hooks = reader_fn(project_root)
        if hooks:
            hooks_by_tool[tool_id] = hooks
    total_hooks = sum(len(v) for v in hooks_by_tool.values())
    hooks_summary = (
        f"{total_hooks} hook(s) from " + ", ".join(str(t) for t in hooks_by_tool)
        if hooks_by_tool
        else "none found"
    )

    # Plugins — detect-only; no per-tool dimension since plugins live on
    # the source side. We pack the labelled list under a single AIToolID.CLAUDE
    # key so the wizard's "found?" check still works without a special case.
    from crossby.sync.plugins import discover_plugins

    plugin_findings = discover_plugins(project_root)
    plugins_summary = (
        f"{len(plugin_findings)} item(s) — manual setup required"
        if plugin_findings
        else "none found"
    )
    plugins_found: dict[AIToolID, list[str]] = (
        {AIToolID.CLAUDE: [f.label for f in plugin_findings]} if plugin_findings else {}
    )

    return ProjectScan(
        installed_tools=installed_tools,
        rules=ConcernScan(found=rules_found, summary=rules_summary),
        agents=ConcernScan(found=dict(agents_found), summary=agents_summary),
        skills=ConcernScan(found=dict(skills_found), summary=skills_summary),
        mcp=ConcernScan(found=dict(mcp_by_tool), summary=mcp_summary),
        permissions=ConcernScan(found=dict(perm_by_tool), summary=perm_summary),
        hooks=ConcernScan(found=dict(hooks_by_tool), summary=hooks_summary),
        plugins=ConcernScan(found=dict(plugins_found), summary=plugins_summary),
    )


# ---------------------------------------------------------------------------
# SyncData builder
# ---------------------------------------------------------------------------


def build_sync_data(
    project_root: Path,
    from_tool: AIToolID | None = None,
    *,
    include_user_scope: bool = False,
) -> SyncData:
    """Build :class:`SyncData` by reading directly from tool configs.

    When *from_tool* is specified, only that tool's configs are read.
    Otherwise all tool configs are scanned and auto-resolved. User-scope
    ``~/.claude.json`` MCP servers are included only when *include_user_scope*
    is set — by default they stay out of the committed project files.
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

    # Skills
    skills_found = detect_skills(project_root)
    skills_source: str | None = None
    if from_tool and from_tool in skills_found:
        skills_source = skills_found[from_tool]
    elif not from_tool and skills_found:
        source_tool = suggest_skills_source(skills_found)
        if source_tool:
            skills_source = skills_found[source_tool]

    # MCP, permissions, hooks
    mcp_servers = discover_mcp(
        project_root, from_tool=from_tool, include_user_scope=include_user_scope
    )
    allowed_commands = discover_permissions(project_root, from_tool=from_tool)
    hooks = discover_hooks(project_root, from_tool=from_tool)

    return SyncData(
        rules_source=rules_source,
        agents_source=agents_source,
        skills_source=skills_source,
        mcp_servers=mcp_servers,
        allowed_commands=allowed_commands,
        hooks=hooks,
    )
