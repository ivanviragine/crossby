"""Hooks sync writers — one per AI tool.

Each writer merges hooks from :class:`~crossby.sync.base.SyncData` (populated
by readers or the sync wizard) into the tool's native config format using a
non-destructive merge strategy (dedup by (event, command)):
- New hooks are appended to the tool's hooks list.
- Hooks with the same (event, command) are merged — the matcher/tools list is
  widened if the desired coverage has grown (upgrade-safe).
- Hooks in the target but NOT in SyncData are preserved.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult
from crossby.sync.json_utils import atomic_write_text, read_json_file, write_json_file
from crossby.sync.manual_fix import ManualFixNote
from crossby.sync.toml_edit import set_scalar, splice_or_none

_HookAction = Literal["created", "updated", "skipped", "error"]


# ---------------------------------------------------------------------------
# Event name translation
# ---------------------------------------------------------------------------

_EVENT_NAMES: dict[AIToolID, dict[str, str]] = {
    AIToolID.CLAUDE: {
        "pre_tool_use": "PreToolUse",
        "post_tool_use": "PostToolUse",
        "session_start": "SessionStart",
        "user_prompt_submit": "UserPromptSubmit",
        "stop": "Stop",
        "notification": "Notification",
    },
    AIToolID.CURSOR: {
        "pre_tool_use": "preToolUse",
        "post_tool_use": "postToolUse",
        "session_start": "sessionStart",
        "user_prompt_submit": "beforeSubmitPrompt",
        "stop": "stop",
    },
    AIToolID.COPILOT: {
        "pre_tool_use": "preToolUse",
        "post_tool_use": "postToolUse",
        "session_start": "sessionStart",
        # Copilot names the turn-complete event `agentStop`, not `stop`. (Its
        # prompt event is likewise `userPromptSubmitted`, not `userPromptSubmit`
        # — not mapped here because crossby does not yet register prompt hooks
        # for Copilot.)
        "stop": "agentStop",
    },
    AIToolID.CODEX: {
        "pre_tool_use": "PreToolUse",
        "post_tool_use": "PostToolUse",
        "session_start": "SessionStart",
        "user_prompt_submit": "UserPromptSubmit",
        "stop": "Stop",
    },
    AIToolID.ANTIGRAVITY_CLI: {
        "pre_tool_use": "PreToolUse",
        "post_tool_use": "PostToolUse",
        "stop": "Stop",
    },
}


def _translate_event(event: str, tool_id: AIToolID) -> str:
    """Translate canonical event name to tool-specific name."""
    return _EVENT_NAMES.get(tool_id, {}).get(event, event)


# ---------------------------------------------------------------------------
# Tool name translation
# ---------------------------------------------------------------------------

_TOOL_NAME_MAP: dict[AIToolID, dict[str, str]] = {
    # Cursor has no `Edit` tool — editing collapses into `Write` (this mirrors
    # cursor-agent's own Claude-config importer, which maps Edit → Write).
    AIToolID.CURSOR: {"Bash": "Shell", "Edit": "Write", "MultiEdit": "Write"},
    AIToolID.COPILOT: {"Edit": "edit", "Write": "write", "Bash": "shell"},
    # agy's native tool-call names, live-captured from `toolCall.name` on a
    # running agy v1.1.9. Without this map a canonical guard registered against
    # `Write|Edit` compiles to a matcher that matches none of agy's real tool
    # names, so the guard installs and then never fires.
    AIToolID.ANTIGRAVITY_CLI: {
        "Write": "write_to_file",
        "Edit": "replace_file_content",
        "MultiEdit": "multi_replace_file_content",
        "Bash": "run_command",
        "Read": "view_file",
        "Grep": "grep_search",
    },
}


def _translate_tools(tools: list[str], tool_id: AIToolID) -> list[str]:
    """Translate canonical tool names to tool-specific names, preserving order.

    Deduplicates: several canonical names can collapse onto one native name
    (Cursor maps both ``Edit`` and ``MultiEdit`` to ``Write``), and a matcher
    listing the same alternative twice is noise.
    """
    mapping = _TOOL_NAME_MAP.get(tool_id, {})
    seen: list[str] = []
    for tool in tools:
        native = mapping.get(tool, tool)
        if native not in seen:
            seen.append(native)
    return seen


def _command_handler(hook: HookEntry) -> dict[str, Any]:
    """Build the ``{"type": "command", ...}`` handler Claude/Codex/agy all use.

    These three share the Claude-shaped nested handler and all spell the hook
    timeout ``timeout`` in seconds (Copilot is the odd one out with
    ``timeoutSec``, so its writer builds its own entry).
    """
    handler: dict[str, Any] = {"type": "command", "command": hook.command}
    if hook.timeout is not None:
        handler["timeout"] = hook.timeout
    return handler


def _tools_to_matcher(tools: list[str]) -> str:
    """Convert tools list to Claude regex matcher string."""
    if not tools or tools == ["*"]:
        return ".*"
    return "|".join(tools)


def _widen_matcher(existing: str | None, desired_tools: list[str]) -> str:
    """Return a regex matcher that covers both existing and desired tool sets.

    Used by Claude/Copilot hook merge to make repeat syncs additive
    instead of destructive — replacing a broader existing matcher (``.*``,
    ``Edit|Write``) with a narrower desired one (``Edit``) would silently
    drop coverage.

    Catch-all (``.*``) wins on either side. Otherwise the union of pipe-
    separated tokens is returned, preserving the order of existing tokens
    and appending any new desired ones.
    """
    desired_matcher = _tools_to_matcher(desired_tools)
    if not existing or not isinstance(existing, str) or existing.strip() == "":
        return desired_matcher
    if existing == ".*" or desired_matcher == ".*":
        return ".*"
    existing_tokens = [t for t in existing.split("|") if t]
    new_tokens = list(existing_tokens)
    for token in desired_matcher.split("|"):
        if token and token not in new_tokens:
            new_tokens.append(token)
    return "|".join(new_tokens)


# ---------------------------------------------------------------------------
# Shared filtering / messaging
# ---------------------------------------------------------------------------


def _filter_supported_hooks(
    hooks: Sequence[HookEntry],
    supported: frozenset[str],
) -> tuple[list[HookEntry], list[ManualFixNote]]:
    """Split incoming hooks into supported-event entries plus drop notes.

    A note is emitted once per distinct unsupported event so a source file
    with three ``Notification`` hooks produces one row instead of three.
    """
    kept: list[HookEntry] = []
    notes: list[ManualFixNote] = []
    seen: set[str] = set()
    for hook in hooks:
        if hook.event in supported:
            kept.append(hook)
            continue
        if hook.event in seen:
            continue
        seen.add(hook.event)
        notes.append(
            ManualFixNote(
                category=f"hooks.{hook.event}",
                message=(
                    f"Source has a `{hook.event}` hook that the target tool "
                    "does not support; translate or remove it manually."
                ),
            )
        )
    return kept, notes


def _message_with_notes(
    base: str | None,
    notes: Sequence[ManualFixNote],
) -> str | None:
    """Combine an optional base message with manual-fix note summaries.

    Always includes the literal substring ``manual_fix`` when notes exist so
    :func:`crossby.sync.report.classify_status` flips the row to
    ``Check before using``.
    """
    if not notes:
        return base
    summary = "; ".join(note.category or note.message for note in notes)
    suffix = f"manual_fix: {summary}"
    if not base:
        return suffix
    return f"{base}; {suffix}"


# ---------------------------------------------------------------------------
# ClaudeHooksWriter
# ---------------------------------------------------------------------------


_CLAUDE_SUPPORTED_EVENTS: frozenset[str] = frozenset(
    {
        "pre_tool_use",
        "post_tool_use",
        "session_start",
        "user_prompt_submit",
        "stop",
        "notification",
    }
)


class ClaudeHooksWriter(AbstractSyncWriter):
    """Merges hooks into .claude/settings.json → hooks.<EventName>[].

    Format::

        {
          "hooks": {
            "PreToolUse": [
              {
                "matcher": "Edit|Write",
                "hooks": [{"type": "command", "command": "..."}]
              }
            ]
          }
        }

    Merge key: command value within any entry's inner ``hooks[]``.
    When the command matches, the ``matcher`` is widened if the desired
    tool coverage has grown (upgrade-safe).
    """

    tool_id = AIToolID.CLAUDE
    concern = SyncConcern.HOOKS

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not data.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".claude" / "settings.json"
        file_data, error, was_new = read_json_file(path)
        if error is not None:
            msg = f"{path} {error} — skipping hooks sync. Fix the file manually or delete it."
            warnings.warn(msg, stacklevel=2)
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                file_path=path,
                message=msg,
            )

        kept, notes = _filter_supported_hooks(data.hooks, _CLAUDE_SUPPORTED_EVENTS)
        existing = file_data or {}
        hooks_section: dict[str, Any] = existing.get("hooks", {})
        if not isinstance(hooks_section, dict):
            hooks_section = {}

        changed = False
        for hook in kept:
            event_name = _translate_event(hook.event, self.tool_id)
            event_list: list[Any] = hooks_section.get(event_name, [])
            if not isinstance(event_list, list):
                event_list = []

            # Dedup by command; widen matcher if tool coverage has grown.
            # The widen — not replace — semantic protects existing broader
            # coverage (e.g. ``.*`` or ``Edit|Write|Bash``) from being
            # silently narrowed when the desired hook only names a subset.
            command = hook.command
            desired_tools = hook.tools or []
            already_exists = False
            for entry in event_list:
                if not isinstance(entry, dict):
                    continue
                inner_hooks = entry.get("hooks")
                if not isinstance(inner_hooks, list):
                    continue
                found_in_entry = any(
                    (isinstance(inner, dict) and inner.get("command") == command)
                    or (isinstance(inner, str) and inner == command)
                    for inner in inner_hooks
                )
                if found_in_entry:
                    already_exists = True
                    existing_matcher = entry.get("matcher")
                    widened = _widen_matcher(
                        existing_matcher if isinstance(existing_matcher, str) else None,
                        desired_tools,
                    )
                    if widened != existing_matcher:
                        entry["matcher"] = widened
                        changed = True
                    break

            if not already_exists:
                new_entry: dict[str, Any] = {
                    "matcher": _tools_to_matcher(desired_tools),
                    "hooks": [_command_handler(hook)],
                }
                event_list.append(new_entry)
                hooks_section[event_name] = event_list
                changed = True

        if not changed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=path,
                message=_message_with_notes(None, notes),
            )

        action: _HookAction = "created" if was_new else "updated"
        if not dry_run:
            existing["hooks"] = hooks_section
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=_message_with_notes(None, notes),
        )


# ---------------------------------------------------------------------------
# CursorHooksWriter
# ---------------------------------------------------------------------------


_CURSOR_SUPPORTED_EVENTS: frozenset[str] = frozenset(
    {"pre_tool_use", "post_tool_use", "session_start", "user_prompt_submit", "stop"}
)
# Cursor honours per-tool scoping only on its tool-execution events, where the
# matcher is tested against `tool_name`. On every other event the matcher is
# tested against something else entirely (a literal like "Stop", or the command
# string), so a tool scope there would silently match nothing.
_CURSOR_TOOL_SCOPE_EVENTS: frozenset[str] = frozenset({"pre_tool_use", "post_tool_use"})

# Cursor's dedicated shell event. Not a canonical crossby event: a `pre_tool_use`
# hook scoped to a shell tool fans out to this as a second entry (see
# CursorHooksWriter), so callers register once and get shell coverage on all
# five tools.
_CURSOR_SHELL_EVENT = "beforeShellExecution"
# Native (post-translation) Cursor tool names that mean "run a shell command".
_CURSOR_SHELL_TOOLS: frozenset[str] = frozenset({"Shell"})

# Every event name Cursor's config validator accepts. Anything else in the
# `hooks` object makes Cursor reject the WHOLE file with "Unknown hook type",
# so the legacy-shape migration below must not promote stray keys into it.
_CURSOR_KNOWN_EVENTS: frozenset[str] = frozenset(
    {
        "beforeShellExecution",
        "beforeMCPExecution",
        "afterShellExecution",
        "afterMCPExecution",
        "beforeReadFile",
        "afterFileEdit",
        "beforeTabFileRead",
        "afterTabFileEdit",
        "stop",
        "beforeSubmitPrompt",
        "afterAgentResponse",
        "afterAgentThought",
        "sessionStart",
        "sessionEnd",
        "preCompact",
        "subagentStart",
        "subagentStop",
        "preToolUse",
        "postToolUse",
        "postToolUseFailure",
    }
)


def _migrate_legacy_cursor_config(existing: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Lift a pre-0.13 flat Cursor hooks file into the ``{version, hooks}`` shape.

    crossby <=0.12 wrote event arrays at the *top level* with no wrapper, e.g.
    ``{"preToolUse": [{"event": …, "command": …, "tools": [...]}]}``. Cursor
    rejects that outright — its loader requires a top-level ``hooks`` object and
    reports "missing 'hooks' property", so every hook crossby ever wrote for
    Cursor was silently inert. Migrate those keys in place rather than leaving a
    broken file behind, and convert each entry's ``tools`` list to the ``matcher``
    regex Cursor actually reads (there is no ``tools`` array in its schema).

    Returns the (possibly rewritten) config and whether anything moved.
    """
    hooks_section = existing.get("hooks")
    legacy_keys = [
        key
        for key, value in existing.items()
        if key in _CURSOR_KNOWN_EVENTS and isinstance(value, list)
    ]
    if not legacy_keys and isinstance(hooks_section, dict):
        return existing, False

    migrated: dict[str, Any] = dict(hooks_section) if isinstance(hooks_section, dict) else {}
    for key in legacy_keys:
        entries = existing.pop(key)
        target: list[Any] = migrated.get(key, [])
        if not isinstance(target, list):
            target = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("command"):
                continue
            converted = {k: v for k, v in entry.items() if k not in ("event", "tools")}
            converted.setdefault("type", "command")
            tools = entry.get("tools")
            if isinstance(tools, list) and tools and key in ("preToolUse", "postToolUse"):
                # Re-translate through the current map on the way in. Legacy
                # files were written when the map only renamed Bash→Shell, so
                # they can carry names Cursor has no tool for (`Edit`); left
                # alone those become matcher alternatives that can never match.
                converted["matcher"] = _tools_to_matcher(
                    _translate_tools([str(t) for t in tools], AIToolID.CURSOR)
                )
            if not any(
                isinstance(e, dict) and e.get("command") == converted["command"] for e in target
            ):
                target.append(converted)
        if target:
            migrated[key] = target

    existing["hooks"] = migrated
    return existing, True


def _cursor_upsert(
    hooks_section: dict[str, Any],
    event_name: str,
    hook: HookEntry,
    desired_tools: list[str],
    *,
    scoped: bool,
) -> bool:
    """Merge one hook into ``hooks_section[event_name]``; return True if changed.

    Dedup key is the command. An existing entry is only ever *widened* — its
    matcher grows to cover new tools, ``failClosed`` is added if missing, and a
    ``timeout`` is filled in only when absent so a hand-tuned value survives.
    """
    event_list: list[Any] = hooks_section.get(event_name, [])
    if not isinstance(event_list, list):
        event_list = []

    desired_matcher = _tools_to_matcher(desired_tools) if scoped else None
    if desired_matcher == ".*":
        # Cursor treats a missing matcher as "all tools"; don't write a
        # redundant catch-all regex.
        desired_matcher = None

    for entry in event_list:
        if not isinstance(entry, dict) or entry.get("command") != hook.command:
            continue
        changed = False
        if hook.fail_closed and entry.get("failClosed") is not True:
            entry["failClosed"] = True
            changed = True
        if hook.timeout is not None and "timeout" not in entry:
            entry["timeout"] = hook.timeout
            changed = True
        if not scoped:
            if "matcher" in entry:
                del entry["matcher"]
                changed = True
            return changed
        existing_matcher = entry.get("matcher")
        # A missing matcher already means "all tools" — never narrow it.
        if existing_matcher is None:
            return changed
        widened = _widen_matcher(
            existing_matcher if isinstance(existing_matcher, str) else None,
            desired_tools,
        )
        if widened != existing_matcher:
            entry["matcher"] = widened
            changed = True
        return changed

    new_entry: dict[str, Any] = {"type": "command", "command": hook.command}
    if desired_matcher is not None:
        new_entry["matcher"] = desired_matcher
    if hook.fail_closed:
        # Cursor defaults hooks to fail-open; a security guard must block the
        # action when the hook itself crashes or times out.
        new_entry["failClosed"] = True
    if hook.timeout is not None:
        new_entry["timeout"] = hook.timeout
    event_list.append(new_entry)
    hooks_section[event_name] = event_list
    return True


class CursorHooksWriter(AbstractSyncWriter):
    """Merges hooks into .cursor/hooks.json → ``hooks.<eventName>[]``.

    Format::

        {
          "version": 1,
          "hooks": {
            "preToolUse": [
              {"type": "command", "command": "...", "matcher": "Write|Shell",
               "failClosed": true, "timeout": 30}
            ],
            "beforeShellExecution": [
              {"type": "command", "command": "...", "failClosed": true}
            ]
          }
        }

    The ``{"version", "hooks"}`` wrapper is mandatory — Cursor's loader rejects a
    config without a top-level ``hooks`` object and drops the file entirely.
    crossby <=0.12 wrote the event arrays at the top level, so every Cursor hook
    it ever wrote was inert; :func:`_migrate_legacy_cursor_config` repairs such a
    file in place. Per-tool scope is a single ``matcher`` **regex string**;
    Cursor's schema has no ``tools`` array.

    Merge key: ``entry.command`` within the event's array. When the command
    matches, the ``matcher`` is widened if the desired coverage has grown
    (upgrade-safe). ``failClosed: true`` is emitted for hooks marked
    :attr:`HookEntry.fail_closed` — Cursor otherwise treats a hook that
    crashes/times out as *allow*, silently defeating a security guard.

    **Shell fan-out.** A ``pre_tool_use`` hook scoped to a shell tool also gets a
    second entry under ``beforeShellExecution``, written *unscoped* because that
    event matches against the command string rather than a tool name. Both
    events fire for a single shell call, so the hook command runs twice — which
    is deliberate and safe, because a guard is a pure allow/deny decision and
    therefore idempotent. `preToolUse` alone does already cover shell on current
    cursor-agent builds (verified); the extra registration is there because which
    events cursor-agent fires has varied by version, and a security guard that
    silently stops covering shell is the failure this issue exists to prevent.
    """

    tool_id = AIToolID.CURSOR
    concern = SyncConcern.HOOKS

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not data.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".cursor" / "hooks.json"
        file_data, error, was_new = read_json_file(path)
        if error is not None:
            msg = f"{path} {error} — skipping hooks sync. Fix the file manually or delete it."
            warnings.warn(msg, stacklevel=2)
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                file_path=path,
                message=msg,
            )

        kept, notes = _filter_supported_hooks(data.hooks, _CURSOR_SUPPORTED_EVENTS)
        existing = file_data or {}
        existing, changed = _migrate_legacy_cursor_config(existing)
        hooks_section: dict[str, Any] = existing.get("hooks", {})
        if not isinstance(hooks_section, dict):
            hooks_section = {}
        dropped_tool_scope_events: set[str] = set()

        for hook in kept:
            event_name = _translate_event(hook.event, self.tool_id)
            allow_tool_scope = hook.event in _CURSOR_TOOL_SCOPE_EVENTS
            raw_desired = hook.tools or []
            desired_tools = _translate_tools(raw_desired, self.tool_id) if allow_tool_scope else []
            if not allow_tool_scope and raw_desired:
                dropped_tool_scope_events.add(hook.event)

            if _cursor_upsert(
                hooks_section, event_name, hook, desired_tools, scoped=allow_tool_scope
            ):
                changed = True

            # Shell fan-out: mirror a shell-scoped pre-tool hook onto Cursor's
            # dedicated shell event, unscoped (it matches the command string,
            # not a tool name). See the class docstring for why this is a
            # deliberate double-registration.
            if (
                hook.event == "pre_tool_use"
                and any(tool in _CURSOR_SHELL_TOOLS for tool in desired_tools)
                and _cursor_upsert(hooks_section, _CURSOR_SHELL_EVENT, hook, [], scoped=False)
            ):
                changed = True

        for event in sorted(dropped_tool_scope_events):
            notes.append(
                ManualFixNote(
                    category=f"hooks.{event}.tools",
                    message=(
                        f"Cursor `{_translate_event(event, self.tool_id)}` hooks have no "
                        "per-tool scope; the source `tools` filter was dropped on write."
                    ),
                )
            )

        version_correct = existing.get("version") == 1

        if not changed and version_correct:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=path,
                message=_message_with_notes(None, notes),
            )

        action: _HookAction = "created" if was_new else "updated"
        if not dry_run:
            # `version` and `hooks` are both mandatory — Cursor's validator
            # rejects the file without them and loads no hooks at all.
            existing["version"] = 1
            existing["hooks"] = hooks_section
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=_message_with_notes(None, notes),
        )


# ---------------------------------------------------------------------------
# CopilotHooksWriter
# ---------------------------------------------------------------------------


_COPILOT_SUPPORTED_EVENTS: frozenset[str] = frozenset(
    {"pre_tool_use", "post_tool_use", "session_start", "stop"}
)


class CopilotHooksWriter(AbstractSyncWriter):
    """Merges hooks into .github/hooks/hooks.json → hooks.<eventName>[].

    Format::

        {
          "version": 1,
          "hooks": {
            "preToolUse": [
              {"type": "command", "bash": "...", "comment": "..."}
            ]
          }
        }

    Copilot has no tool filter field — if the canonical hook specifies ``tools``,
    a warning is emitted in the SyncResult message.
    Dedup key: ``entry.bash`` within the event's array.
    """

    tool_id = AIToolID.COPILOT
    concern = SyncConcern.HOOKS

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not data.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".github" / "hooks" / "hooks.json"
        file_data, error, was_new = read_json_file(path)
        if error is not None:
            msg = f"{path} {error} — skipping hooks sync. Fix the file manually or delete it."
            warnings.warn(msg, stacklevel=2)
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                file_path=path,
                message=msg,
            )

        kept, notes = _filter_supported_hooks(data.hooks, _COPILOT_SUPPORTED_EVENTS)
        existing = file_data or {}
        hooks_section: dict[str, Any] = existing.get("hooks", {})
        if not isinstance(hooks_section, dict):
            hooks_section = {}

        changed = False
        seen_tool_filter_drop = False

        for hook in kept:
            event_name = _translate_event(hook.event, self.tool_id)
            event_list: list[Any] = hooks_section.get(event_name, [])
            if not isinstance(event_list, list):
                event_list = []

            # Dedup: check if bash command already present
            command = hook.command
            already_exists = any(
                isinstance(entry, dict) and entry.get("bash") == command for entry in event_list
            )

            if not already_exists:
                if hook.tools and hook.tools != ["*"] and not seen_tool_filter_drop:
                    seen_tool_filter_drop = True
                    notes.append(
                        ManualFixNote(
                            category="hooks.tools",
                            message=(
                                "crossby writes Copilot hooks unscoped; the source "
                                "`tools` scope was dropped and the hook applies to all "
                                "tools. Copilot itself does support a `matcher` regex "
                                "on its tool events, so add one by hand if the hook "
                                "must be scoped."
                            ),
                        )
                    )
                new_entry: dict[str, Any] = {
                    "type": "command",
                    "bash": command,
                    "comment": hook.description or "",
                }
                if hook.timeout is not None:
                    # Copilot spells it `timeoutSec` (seconds, default 30).
                    # `timeout` is an accepted alias but `timeoutSec` wins, so
                    # emit the canonical name.
                    new_entry["timeoutSec"] = hook.timeout
                event_list.append(new_entry)
                hooks_section[event_name] = event_list
                changed = True

        version_correct = existing.get("version") == 1

        if not changed and version_correct:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=path,
                message=_message_with_notes(None, notes),
            )

        action: _HookAction = "created" if was_new else "updated"
        if not dry_run:
            existing["version"] = 1
            existing["hooks"] = hooks_section
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=_message_with_notes(None, notes),
        )


# ---------------------------------------------------------------------------
# AntigravityCLIHooksWriter
# ---------------------------------------------------------------------------


# agy exposes PreToolUse/PostToolUse/Pre/PostInvocation/Stop. Among crossby's
# canonical events only these three map cleanly; session_start / user_prompt_submit
# have no agy equivalent (agy's Pre/PostInvocation fire per model call, not once
# at session start) and are dropped with a manual-fix note.
_ANTIGRAVITY_CLI_SUPPORTED_EVENTS: frozenset[str] = frozenset(
    {"pre_tool_use", "post_tool_use", "stop"}
)
# agy honours a `matcher` (regex over tool names) only on the tool-execution
# events; Stop handlers sit directly under the event key with no matcher.
_ANTIGRAVITY_CLI_MATCHER_EVENTS: frozenset[str] = frozenset({"pre_tool_use", "post_tool_use"})


def _agy_slug(text: str) -> str:
    """Lowercase, hyphen-separated slug of ``text`` (alnum runs → single ``-``)."""
    out: list[str] = []
    prev_hyphen = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_hyphen = False
        elif not prev_hyphen:
            out.append("-")
            prev_hyphen = True
    return "".join(out).strip("-")


def _agy_command_present(container_map: dict[str, Any], agy_event: str, command: str) -> bool:
    """True if ``command`` is already registered under ``agy_event`` anywhere.

    agy's file maps arbitrary *container* names → ``{event: [...]}``. A
    tool-execution event's entries wrap handlers in ``{"matcher", "hooks": [...]}``;
    Stop's entries are handlers (``{"type", "command"}``) directly. This scans
    every container so a re-sync dedups against a hand-authored container too.
    """
    for container in container_map.values():
        if not isinstance(container, dict):
            continue
        event_list = container.get(agy_event)
        if not isinstance(event_list, list):
            continue
        for entry in event_list:
            if not isinstance(entry, dict):
                continue
            if entry.get("command") == command:
                return True  # Stop-shape handler
            inner = entry.get("hooks")
            if isinstance(inner, list) and any(
                isinstance(h, dict) and h.get("command") == command for h in inner
            ):
                return True  # matcher-wrapped handler
    return False


def _agy_find_matcher_entry(
    container_map: dict[str, Any], agy_event: str, command: str
) -> dict[str, Any] | None:
    """Return the matcher-wrapped entry registering ``command``, or ``None``.

    Used to *widen* an existing PreToolUse/PostToolUse entry's ``matcher`` on
    re-sync when the desired tool coverage has grown — mirroring the
    upgrade-safe merge the Claude/Codex writers perform — instead of silently
    dropping the newly-covered tools.
    """
    for container in container_map.values():
        if not isinstance(container, dict):
            continue
        event_list = container.get(agy_event)
        if not isinstance(event_list, list):
            continue
        for entry in event_list:
            if not isinstance(entry, dict):
                continue
            inner = entry.get("hooks")
            if isinstance(inner, list) and any(
                isinstance(h, dict) and h.get("command") == command for h in inner
            ):
                return entry
    return None


class AntigravityCLIHooksWriter(AbstractSyncWriter):
    """Merges hooks into .agents/hooks.json for Antigravity CLI (``agy``).

    agy's format maps arbitrary hook *container* names to per-event configs::

        {
          "crossby-pretooluse": {
            "PreToolUse": [
              {"matcher": "Edit|Write",
               "hooks": [{"type": "command", "command": "..."}]}
            ]
          },
          "crossby-stop": {
            "Stop": [{"type": "command", "command": "..."}]
          }
        }

    Tool-execution events (PreToolUse/PostToolUse) wrap their handlers in a
    ``{"matcher", "hooks": [...]}`` object; ``Stop`` lists handlers directly and
    ignores any matcher (dropped on write with a note). agy has no
    session_start / user_prompt_submit event, so those are dropped too. Dedup key
    is the handler ``command`` across every container.
    """

    tool_id = AIToolID.ANTIGRAVITY_CLI
    concern = SyncConcern.HOOKS

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not data.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".agents" / "hooks.json"
        file_data, error, was_new = read_json_file(path)
        if error is not None:
            msg = f"{path} {error} — skipping hooks sync. Fix the file manually or delete it."
            warnings.warn(msg, stacklevel=2)
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                file_path=path,
                message=msg,
            )

        kept, notes = _filter_supported_hooks(data.hooks, _ANTIGRAVITY_CLI_SUPPORTED_EVENTS)
        existing = file_data if isinstance(file_data, dict) else {}
        changed = False
        dropped_matcher_events: set[str] = set()

        for hook in kept:
            agy_event = _translate_event(hook.event, self.tool_id)
            command = hook.command
            allow_matcher = hook.event in _ANTIGRAVITY_CLI_MATCHER_EVENTS
            # Translate to agy's native tool-call names first — its matcher is a
            # regex over the live `toolCall.name`, so a matcher built from
            # crossby's canonical names (Write|Edit|Bash) matches nothing agy
            # ever emits and the guard silently never fires.
            desired_tools = _translate_tools(hook.tools or [], self.tool_id)
            if not allow_matcher and desired_tools:
                dropped_matcher_events.add(hook.event)

            if allow_matcher:
                # Upgrade-safe: if this command is already registered, widen its
                # matcher when the desired tool coverage has grown (never narrow),
                # mirroring the Claude/Codex writers. Prevents a re-sync from
                # silently dropping newly-covered tools from a guard.
                existing_entry = _agy_find_matcher_entry(existing, agy_event, command)
                if existing_entry is not None:
                    existing_matcher = existing_entry.get("matcher")
                    base = existing_matcher if isinstance(existing_matcher, str) else None
                    widened = _widen_matcher(base, desired_tools)
                    if widened != existing_matcher:
                        existing_entry["matcher"] = widened
                        changed = True
                    continue
            elif _agy_command_present(existing, agy_event, command):
                # Stop: no matcher to widen — a present command is a no-op.
                continue

            container_name = _agy_slug(hook.description) or f"crossby-{agy_event.lower()}"
            container = existing.get(container_name)
            if not isinstance(container, dict):
                container = {}
            event_list = container.get(agy_event)
            if not isinstance(event_list, list):
                event_list = []

            handler: dict[str, Any] = _command_handler(hook)
            if allow_matcher:
                event_list.append({"matcher": _tools_to_matcher(desired_tools), "hooks": [handler]})
            else:
                # Stop: handlers sit directly under the event key, no matcher.
                event_list.append(handler)

            container[agy_event] = event_list
            existing[container_name] = container
            changed = True

        for event in sorted(dropped_matcher_events):
            notes.append(
                ManualFixNote(
                    category=f"hooks.{event}.matcher",
                    message=(
                        f"Antigravity CLI ignores `matcher` on "
                        f"`{_translate_event(event, self.tool_id)}`; tool scope was "
                        "dropped on write."
                    ),
                )
            )

        if not changed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=path,
                message=_message_with_notes(None, notes),
            )

        action: _HookAction = "created" if was_new else "updated"
        if not dry_run:
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=_message_with_notes(None, notes),
        )


# ---------------------------------------------------------------------------
# CodexHooksWriter
# ---------------------------------------------------------------------------


_CODEX_SUPPORTED_EVENTS: frozenset[str] = frozenset(
    {"pre_tool_use", "post_tool_use", "session_start", "user_prompt_submit", "stop"}
)
# Codex honours `matcher` only on these events; for others (UserPromptSubmit,
# Stop) it is silently ignored, so we drop it on write and surface a note.
_CODEX_MATCHER_EVENTS: frozenset[str] = frozenset(
    {"pre_tool_use", "post_tool_use", "session_start"}
)
# Fallback note, surfaced ONLY when the flag can't be written automatically
# (e.g. a pre-existing `.codex/config.toml` is malformed). On the happy path the
# writer sets the flag itself, so no manual step is reported.
_CODEX_FEATURES_FLAG_NOTE = ManualFixNote(
    category="features.hooks",
    message=(
        "Could not update `.codex/config.toml` automatically — set "
        "`[features].hooks = true` (and `codex_hooks = true` for older Codex "
        "builds) there manually so Codex loads these hooks."
    ),
)

# Canonical key first, deprecated alias second. `hooks` is the current name and
# has been stable and ON by default since Codex 0.146.0 (`codex features list`
# reports `hooks stable true`); `codex_hooks` resolves to it as a deprecated
# alias. Unknown feature keys are inert, so writing both is safe on every build
# and is purely defensive for older ones.
_CODEX_FEATURE_KEYS: tuple[str, ...] = ("hooks", "codex_hooks")


def _ensure_codex_hooks_feature_flag(project_root: Path, *, dry_run: bool) -> ManualFixNote | None:
    """Enable the Codex hooks feature flags in ``.codex/config.toml`` (idempotent).

    Writes both ``[features].hooks`` (canonical) and ``[features].codex_hooks``
    (deprecated alias) so the config works on current and older Codex alike.
    Merges into any existing config, preserving other keys/tables.

    On current Codex this is belt-and-braces — the feature is stable and enabled
    by default, so hooks load whether or not the flag is present. It still gets
    written so a project pinned to an older Codex is not silently left with
    inert hooks.

    Returns ``None`` on success (or when both flags are already set / ``dry_run``);
    returns :data:`_CODEX_FEATURES_FLAG_NOTE` when the existing file is malformed
    TOML and can't be updated automatically, so the caller can surface a
    manual-fix note instead of silently leaving the hooks inert.
    """
    import tomllib

    import tomli_w

    path = project_root / ".codex" / "config.toml"
    existing: dict[str, Any] = {}
    original = ""
    if path.exists():
        try:
            original = path.read_text(encoding="utf-8")
            existing = tomllib.loads(original)
        except (tomllib.TOMLDecodeError, OSError, ValueError):
            return _CODEX_FEATURES_FLAG_NOTE

    features = existing.get("features")
    if not isinstance(features, dict):
        features = {}
    missing = [key for key in _CODEX_FEATURE_KEYS if features.get(key) is not True]
    if not missing:
        return None  # already enabled — nothing to do

    if dry_run:
        return None

    for key in missing:
        features[key] = True
    existing["features"] = features

    # Splice each key in textually so the user's comments and key ordering
    # survive; fall back to the (lossy) full dump only if that can't be done.
    # Reuses the text read above — a second read could see a different file.
    spliced: str | None = original
    for key in missing:
        if spliced is None:
            break
        spliced = set_scalar(spliced, ("features",), key, "true")

    new_text = splice_or_none(spliced, existing)
    if new_text is None:
        new_text = tomli_w.dumps(existing)

    try:
        atomic_write_text(path, new_text)
    except OSError:
        return _CODEX_FEATURES_FLAG_NOTE
    return None


class CodexHooksWriter(AbstractSyncWriter):
    """Merges hooks into .codex/hooks.json with the Claude-shape JSON layout.

    Codex supports only a subset of Claude's hook events (PreToolUse,
    PostToolUse, SessionStart, UserPromptSubmit, Stop) and only honours
    ``matcher`` on the first three. Unsupported events and dropped matchers
    are reported as manual-fix notes in the ``SyncResult.message`` so the
    sync report classifies the row as ``Check before using``.

    The writer also sets ``[features].hooks = true`` and its deprecated alias
    ``codex_hooks`` in ``.codex/config.toml`` (a manual-fix note is surfaced only
    if that file can't be written). On current Codex this is defensive only —
    the hooks feature is stable and enabled by default since 0.146.0 — but it
    keeps a project pinned to an older Codex from ending up with inert hooks.
    """

    tool_id = AIToolID.CODEX
    concern = SyncConcern.HOOKS

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not data.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".codex" / "hooks.json"
        file_data, error, was_new = read_json_file(path)
        if error is not None:
            msg = f"{path} {error} — skipping hooks sync. Fix the file manually or delete it."
            warnings.warn(msg, stacklevel=2)
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                file_path=path,
                message=msg,
            )

        kept, notes = _filter_supported_hooks(data.hooks, _CODEX_SUPPORTED_EVENTS)

        existing = file_data or {}
        hooks_section: dict[str, Any] = existing.get("hooks", {})
        if not isinstance(hooks_section, dict):
            hooks_section = {}

        changed = False
        dropped_matcher_events: set[str] = set()
        for hook in kept:
            event_name = _translate_event(hook.event, self.tool_id)
            event_list: list[Any] = hooks_section.get(event_name, [])
            if not isinstance(event_list, list):
                event_list = []

            command = hook.command
            desired_tools = hook.tools or []
            allow_matcher = hook.event in _CODEX_MATCHER_EVENTS
            if not allow_matcher and desired_tools:
                dropped_matcher_events.add(hook.event)

            already_exists = False
            for entry in event_list:
                if not isinstance(entry, dict):
                    continue
                inner_hooks = entry.get("hooks")
                if not isinstance(inner_hooks, list):
                    continue
                found_in_entry = any(
                    (isinstance(inner, dict) and inner.get("command") == command)
                    or (isinstance(inner, str) and inner == command)
                    for inner in inner_hooks
                )
                if found_in_entry:
                    already_exists = True
                    if allow_matcher:
                        existing_matcher = entry.get("matcher")
                        widened = _widen_matcher(
                            existing_matcher if isinstance(existing_matcher, str) else None,
                            desired_tools,
                        )
                        if widened != existing_matcher:
                            entry["matcher"] = widened
                            changed = True
                    elif "matcher" in entry:
                        # Strip a matcher Codex ignores so the file stays clean.
                        del entry["matcher"]
                        changed = True
                    break

            if not already_exists:
                new_entry: dict[str, Any] = {
                    "hooks": [_command_handler(hook)],
                }
                if allow_matcher:
                    new_entry["matcher"] = _tools_to_matcher(desired_tools)
                event_list.append(new_entry)
                hooks_section[event_name] = event_list
                changed = True

        for event in sorted(dropped_matcher_events):
            notes.append(
                ManualFixNote(
                    category=f"hooks.{event}.matcher",
                    message=(
                        f"Codex ignores `matcher` on `{_translate_event(event, self.tool_id)}`; "
                        "tool scope was dropped on write."
                    ),
                )
            )

        if not changed and not kept:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=path,
                message=_message_with_notes(None, notes),
            )

        # Enable the feature flag so Codex actually loads these hooks. On success
        # this is silent; if it can't be written a manual-fix note is surfaced.
        if kept:
            flag_note = _ensure_codex_hooks_feature_flag(project_root, dry_run=dry_run)
            if flag_note is not None:
                notes.append(flag_note)

        action: _HookAction = "created" if was_new else "updated"
        if not dry_run:
            existing["hooks"] = hooks_section
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message=_message_with_notes(None, notes),
        )
