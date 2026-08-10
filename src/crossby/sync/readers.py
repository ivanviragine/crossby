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
from crossby.sync.base import SyncConcern, SyncData
from crossby.sync.hooks import (
    _ANTIGRAVITY_CLI_SUPPORTED_EVENTS,
    _CURSOR_SHELL_EVENT,
    _PLAIN_ALTERNATION,
)

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


# Tools discover_mcp_servers (sync/mcp_discovery.py) actually scans a config
# file for — kept in sync with its ``sources`` list plus the Codex TOML branch.
_MCP_SOURCE_TOOLS: frozenset[AIToolID] = frozenset(
    {AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.COPILOT, AIToolID.ANTIGRAVITY_CLI, AIToolID.CODEX}
)


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


def _read_claude_shape_hooks(path: Path) -> list[HookEntry]:
    """Parse the Claude-shaped hooks layout at *path* into canonical entries.

    Both Claude (``.claude/settings.json``) and Codex (``.codex/hooks.json``)
    store hooks as ``{"hooks": {"<EventName>": [{"matcher", "hooks":
    [{"type", "command"}]}]}}``, so both readers share this body rather than
    diverging by copy-paste.

    ``matcher`` becomes ``tools`` via :func:`_matcher_tools`, so only a plain
    ``|`` alternation of tool names is recovered. A catch-all ``.*`` or exotic
    regex (``Write.*``) yields unscoped ``[]`` rather than a bogus token like
    ``Write.*``: carried into a cross-tool sync that token translates to nothing
    the target tool ever emits, silently rendering the guard inert (whereas an
    unscoped hook re-scopes to ``.*`` and still fires). A handler whose
    ``command`` is missing or not a non-empty ``str`` is skipped rather than
    handed to Pydantic (data-type hardening).
    """
    data = _read_json(path)
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
            tools = _matcher_tools(entry.get("matcher"))
            inner_hooks = entry.get("hooks", [])
            if not isinstance(inner_hooks, list):
                continue
            for inner in inner_hooks:
                if not isinstance(inner, dict):
                    continue
                command = inner.get("command")
                if not isinstance(command, str) or not command:
                    continue
                result.append(
                    HookEntry(
                        event=canonical_event,
                        command=command,
                        tools=tools,
                    )
                )
    return result


def _read_claude_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .claude/settings.json (Claude-shaped)."""
    return _read_claude_shape_hooks(project_root / ".claude" / "settings.json")


def _read_codex_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .codex/hooks.json.

    Codex writes the Claude-shaped nested-handler JSON (the same structure
    ``CodexHooksWriter`` emits and ``_read_claude_hooks`` parses), so this is a
    thin wrapper over :func:`_read_claude_shape_hooks`. Codex's event names
    (``SessionStart``/``UserPromptSubmit``/``Stop``/``PreToolUse``/``PostToolUse``)
    all reverse-map via ``_REVERSE_EVENTS``, so no map changes are needed.
    """
    return _read_claude_shape_hooks(project_root / ".codex" / "hooks.json")


def _read_cursor_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .cursor/hooks.json.

    Reads the real Cursor shape — ``{"version": 1, "hooks": {"<event>": [...]}}``
    with a ``matcher`` regex — and still accepts the flat top-level layout
    crossby wrote before 0.13 so an un-migrated file round-trips.

    ``beforeShellExecution`` maps back to ``pre_tool_use``, because that is what
    ``CursorHooksWriter`` fans a shell-scoped pre-tool hook out to. Entries are
    deduped by ``(event, command)`` so the fanned-out pair collapses back into
    the single :class:`HookEntry` it came from instead of reappearing as a
    spurious second hook on every read → write cycle. When two duplicates carry
    *distinct* non-empty scopes (a hand-authored ``Write`` guard and a ``Shell``
    guard on the same command), their tool lists are unioned so neither guard is
    dropped; an unscoped (``[]``) side loses to a scoped one so the fan-out
    mirror never erases its partner's scope.

    ``failClosed`` is read back (bool-typed only, else ``False``) onto
    ``HookEntry.fail_closed``; a ``True`` from either half of a same-``(event,
    command)`` duplicate wins, so the flag survives the local dedupe here (and
    then the cross-tool merge in :func:`discover_hooks`). ``timeout`` is still
    dropped for every tool today — a bounded round-trip gap (#88 §5), since the
    writers never downgrade an existing timeout.

    An unscoped entry only yields to a scoped duplicate when it came from the
    ``beforeShellExecution`` key specifically — the one key the writer ever
    fans a hook out to unscoped (see :func:`_merge_cursor_tool_scopes`). A
    genuinely unscoped ``preToolUse`` entry (no matcher/tools) means "all
    tools" and must never be narrowed by a same-command scoped duplicate.

    Data-type hardening: a ``command`` that is not a non-empty ``str`` is skipped
    (a non-``str`` also breaks the ``(event, command)`` dedupe key before it ever
    reaches Pydantic). An entry whose ``tools``/``matcher`` scope is *present but
    yields no valid tool* is skipped rather than emitted unscoped — an empty
    ``HookEntry.tools`` means "all tools", so silently broadening a scoped guard
    to everything is worse than dropping it (see :func:`_cursor_entry_tools`).
    """
    data = _read_json(project_root / ".cursor" / "hooks.json")
    if not data:
        return []
    hooks_section = data.get("hooks")
    # Pre-0.13 crossby wrote event arrays at the top level with no wrapper.
    source = hooks_section if isinstance(hooks_section, dict) else data

    merged: dict[tuple[str, str], HookEntry] = {}
    # Tracks, per key, whether the merged-so-far unscoped ([]) state (if any)
    # is a genuine "all tools" declaration rather than a beforeShellExecution
    # fan-out artifact — see _merge_cursor_tool_scopes.
    real_unscoped: dict[tuple[str, str], bool] = {}
    for event_name, entries in source.items():
        if not isinstance(entries, list):
            continue
        canonical_event = _reverse_event_name(event_name)
        is_fanout_event = event_name == _CURSOR_SHELL_EVENT
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            command = entry.get("command")
            if not isinstance(command, str) or not command:
                continue
            tools = _cursor_entry_tools(entry)
            if tools is None:
                # Scope was present but yielded no valid tool — skip rather than
                # emit unscoped (which would broaden a scoped hook to all tools).
                continue
            entry_real_unscoped = not tools and not is_fanout_event
            fail_closed = entry.get("failClosed") is True
            key = (canonical_event, command)
            previous = merged.get(key)
            if previous is None:
                merged[key] = HookEntry(
                    event=canonical_event,
                    command=command,
                    tools=tools,
                    fail_closed=fail_closed,
                )
                real_unscoped[key] = entry_real_unscoped
                continue
            # The fan-out pair (and any same-file duplicate) collapses here.
            # Three cases must all survive:
            #   * The fan-out pair — one half is the unscoped `beforeShellExecution`
            #     mirror. Whichever half is read second must not erase the other's
            #     tool scope, so a scoped side always beats that artifact,
            #     regardless of JSON key order.
            #   * A genuinely unscoped `preToolUse` entry (no matcher/tools) sharing
            #     a command with a scoped duplicate — unscoped means "all tools", so
            #     it must dominate rather than being silently narrowed to the
            #     scoped duplicate's tool list.
            #   * Two genuinely-distinct scoped entries sharing the same
            #     `(event, command)` (a hand-authored `Write` guard and a `Shell`
            #     guard on the same command) — union their scopes so neither guard
            #     is silently dropped.
            # A `failClosed: true` from any duplicate wins so the flag is never
            # dropped by dedupe.
            merged_tools, merged_real_unscoped = _merge_cursor_tool_scopes(
                previous.tools, real_unscoped.get(key, False), tools, entry_real_unscoped
            )
            merged[key] = HookEntry(
                event=canonical_event,
                command=command,
                tools=merged_tools,
                fail_closed=previous.fail_closed or fail_closed,
            )
            real_unscoped[key] = merged_real_unscoped
    return list(merged.values())


def _merge_cursor_tool_scopes(
    a_tools: list[str], a_real_unscoped: bool, b_tools: list[str], b_real_unscoped: bool
) -> tuple[list[str], bool]:
    """Merge two same-``(event, command)`` Cursor tool scopes into one.

    ``a``/``b`` are each either scoped (non-empty ``tools``) or unscoped
    (``[]``) — and an unscoped side is flagged *real* when it did not come
    from the ``beforeShellExecution`` fan-out mirror. A real-unscoped side
    always wins (it means "all tools" and must not be narrowed); a
    fan-out-artifact unscoped side always yields to a scoped duplicate,
    order-independent either way. Two scoped sides union their tool lists.
    Returns ``(merged_tools, merged_real_unscoped)``.
    """
    if a_real_unscoped or b_real_unscoped:
        return [], True
    if a_tools and b_tools:
        return list(dict.fromkeys([*a_tools, *b_tools])), False
    return (a_tools or b_tools), False


def _matcher_tools(matcher: Any) -> list[str]:
    """Recover canonical tool names from a hook ``matcher`` regex.

    ``matcher`` is a regex, not a tool list, so only a plain ``|`` alternation of
    literal tool names (``Write|Shell``) is recoverable — the sole matcher shape
    that is also a tool list (see ``_PLAIN_ALTERNATION`` in ``sync/hooks.py``).
    Each token is reverse-mapped to its canonical crossby name; a native name
    crossby does not recognize passes through unchanged.

    A catch-all ``.*`` or any fancier regex (``Write.*``, ``(Write|Shell)``)
    means "all tools" / is unrepresentable, so it yields ``[]`` (unscoped) rather
    than fragmenting into bogus tool tokens that would be unioned straight back
    into the matcher on the next write. The paired ``_widen_matcher`` guard then
    keeps such a hand-authored matcher from being broadened to ``.*`` on re-sync.
    """
    if isinstance(matcher, str) and _PLAIN_ALTERNATION.fullmatch(matcher):
        return [_reverse_tool_name(t) for t in matcher.split("|") if t]
    return []


def _cursor_entry_tools(entry: dict[str, Any]) -> list[str] | None:
    """Recover canonical tool names from a Cursor entry's scope.

    Prefers the ``matcher`` regex Cursor actually uses (alternation of tool
    names, e.g. ``Write|Shell``); falls back to the ``tools`` array that only
    ever existed in crossby's own pre-0.13 output. A catch-all matcher means
    "all tools", which is the same as no scope.

    ``matcher`` is a regex, not a tool list, so only a plain ``|`` alternation
    is split (via :func:`_matcher_tools`). A hand-authored matcher like
    ``Write.*`` or ``(Write|Shell)`` would otherwise yield fragments (``(Write``,
    ``Shell)``) that become ``HookEntry.tools`` and get unioned straight back
    into the matcher on the next write, corrupting ``.cursor/hooks.json``. Such
    an unrepresentable matcher maps to unscoped ``[]`` (the paired
    ``_widen_matcher`` guard preserves the original matcher on re-sync), so a
    matcher never skips the entry.

    Returns ``None`` (a skip sentinel) when a ``tools`` array is *present but
    yields no valid string tool* — a wrongly-typed list like ``[1, 2]`` was a
    scoping intent, and emitting it as unscoped ``[]`` would silently broaden the
    hook to **all** tools. The caller drops such an entry instead. A mix of valid
    and invalid entries keeps the valid ones (scope preserved). A genuinely
    absent scope, or an explicit empty ``tools: []``, maps to ``[]`` (unscoped).

    Also returns ``None`` when ``matcher`` is *present but not a string*
    (e.g. ``matcher: 123``) — ``_matcher_tools`` silently treats any non-str
    input the same as "no plain alternation found" (``[]``), so without this
    check a malformed matcher would fall through to the absent-scope case and
    get emitted as unscoped, broadening the hook to all tools instead of
    being dropped.
    """
    matcher = entry.get("matcher")
    if matcher is not None and not isinstance(matcher, str):
        return None
    from_matcher = _matcher_tools(matcher)
    if from_matcher:
        return from_matcher
    tools_raw = entry.get("tools")
    if tools_raw is None:
        return []
    if not isinstance(tools_raw, list):
        # A present-but-malformed scope (e.g. ``tools: "Write"``) — skip.
        return None
    if not tools_raw:
        return []
    valid = [_reverse_tool_name(t) for t in tools_raw if isinstance(t, str) and t]
    return valid if valid else None


def _read_copilot_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .github/hooks/hooks.json.

    Data-type hardening: a handler whose ``bash`` is not a non-empty ``str`` is
    skipped rather than handed to Pydantic (which rejects it with a
    ``ValidationError`` rather than coercing), and a non-``str`` ``comment`` is
    coerced to ``""`` so a wrongly-typed sibling field can't crash the read. A
    Copilot hook has no per-tool scope, so ``tools`` is always ``[]``.
    """
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
            if not isinstance(command, str) or not command:
                continue
            comment = entry.get("comment", "")
            if not isinstance(comment, str):
                comment = ""
            result.append(
                HookEntry(
                    event=canonical_event,
                    command=command,
                    tools=[],
                    description=comment,
                )
            )
    return result


def _read_agy_hooks(project_root: Path) -> list[HookEntry]:
    """Read hooks from .agents/hooks.json (Antigravity CLI's container map).

    agy's file maps arbitrary *container* names to ``{agy_event: [entries]}``.
    Tool-execution events (PreToolUse/PostToolUse) wrap handlers in a
    ``{"matcher", "hooks": [...]}`` object; ``Stop`` lists handlers
    (``{"type", "command"}``) directly with no matcher. Both shapes are handled
    here, mirroring the traversal in ``sync/hooks.py``'s ``_agy_command_present``.

    Only events that reverse-map to an agy-writer-supported canonical event
    (``pre_tool_use``, ``post_tool_use``, ``stop`` — see
    ``_ANTIGRAVITY_CLI_SUPPORTED_EVENTS``) are emitted. agy also exposes
    ``PreInvocation``/``PostInvocation``, which have no canonical crossby event;
    those keys are skipped rather than emitted with a passthrough event string
    the writer would then drop on the next sync.

    Tool scope is recovered from the ``matcher`` via :func:`_matcher_tools`, so
    agy's native names (``write_to_file``/``run_command``/…) reverse cleanly to
    canonical tools while a catch-all or exotic regex yields unscoped ``[]``.

    Round-trip gaps (deliberate): the per-hook ``description`` is left empty
    because agy encodes no per-hook comment — only the lossy container *name*
    slug — and ``fail_closed``/``timeout`` are dropped for every tool today
    (#88 §5).
    """
    data = _read_json(project_root / ".agents" / "hooks.json")
    if not data:
        return []
    result: list[HookEntry] = []
    for container in data.values():
        if not isinstance(container, dict):
            continue
        for event_name, entries in container.items():
            canonical_event = _reverse_event_name(event_name)
            if canonical_event not in _ANTIGRAVITY_CLI_SUPPORTED_EVENTS:
                continue
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                # Bare Stop handler: {"type", "command"} directly in the list.
                bare_command = entry.get("command")
                if isinstance(bare_command, str) and bare_command:
                    result.append(HookEntry(event=canonical_event, command=bare_command, tools=[]))
                    continue
                # Matcher-wrapped entry: {"matcher", "hooks": [{"type","command"}]}.
                inner_hooks = entry.get("hooks")
                if not isinstance(inner_hooks, list):
                    continue
                tools = _matcher_tools(entry.get("matcher"))
                for inner in inner_hooks:
                    if not isinstance(inner, dict):
                        continue
                    command = inner.get("command")
                    if not isinstance(command, str) or not command:
                        continue
                    result.append(HookEntry(event=canonical_event, command=command, tools=tools))
    return result


_HOOK_READERS: dict[AIToolID, Any] = {
    AIToolID.CLAUDE: _read_claude_hooks,
    AIToolID.CURSOR: _read_cursor_hooks,
    AIToolID.COPILOT: _read_copilot_hooks,
    AIToolID.CODEX: _read_codex_hooks,
    AIToolID.ANTIGRAVITY_CLI: _read_agy_hooks,
}


# Concern → tools whose configs a reader can extract that concern from. Every
# concern that reads from a per-tool source mapping is gated here — a tool
# absent from the mapping has no code path that could ever populate it (not
# just "found nothing this run"), so a ``--from`` tool outside this set would
# otherwise sync nothing with no explanation. PLUGINS is intentionally absent:
# discover_plugins() has no per-tool ``--from`` dimension at all (see
# scan_project's plugins comment), so there is no "unsupported tool" case to
# gate.
_READER_TOOLS: dict[SyncConcern, frozenset[AIToolID]] = {
    SyncConcern.PERMISSIONS: frozenset(_PERMISSION_READERS),
    SyncConcern.HOOKS: frozenset(_HOOK_READERS),
    SyncConcern.RULES: frozenset(_INSTRUCTION_FILES),
    SyncConcern.AGENTS: frozenset(_AGENT_DIRS),
    SyncConcern.SKILLS: frozenset(SKILLS_DIR),
    SyncConcern.MCP: _MCP_SOURCE_TOOLS,
}


def reader_available(tool: AIToolID, concern: SyncConcern) -> bool:
    """Report whether a reader can extract *concern* from *tool*'s configs.

    Returns ``False`` for any registry-gated concern (permissions, hooks,
    rules, agents, skills, MCP) whose source mapping omits *tool* — the case
    where ``crossby sync <concern> --from <tool>`` would otherwise sync
    nothing with no explanation. Callers use this at the CLI layer to warn
    instead of failing silently. PLUGINS has no per-tool source mapping (it
    doesn't take a ``--from`` tool), so it is always considered available.
    """
    tools = _READER_TOOLS.get(concern)
    return True if tools is None else tool in tools


def discover_hooks(project_root: Path, from_tool: AIToolID | None = None) -> list[HookEntry]:
    """Read hooks from tool configs.

    When the same ``(event, command)`` is defined by multiple tools, tool
    scopes are unioned — an empty ``tools`` list means "all tools" and wins.
    ``fail_closed`` is ORed across the duplicates so a ``True`` on any side
    survives the cross-tool merge (mirroring the local dedupe each reader runs
    first); the guard is never silently downgraded to fail-open.
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
                fail_closed=existing.fail_closed or hook.fail_closed,
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
