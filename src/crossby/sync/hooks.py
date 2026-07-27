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
        "user_prompt_submit": "beforeSubmitPrompt",
        "stop": "stop",
    },
    AIToolID.COPILOT: {
        "pre_tool_use": "preToolUse",
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
    AIToolID.CURSOR: {"Bash": "Shell"},
    AIToolID.COPILOT: {"Edit": "edit", "Write": "write", "Bash": "shell"},
}


def _translate_tools(tools: list[str], tool_id: AIToolID) -> list[str]:
    """Translate canonical tool names to tool-specific names."""
    mapping = _TOOL_NAME_MAP.get(tool_id, {})
    return [mapping.get(t, t) for t in tools]


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
                    "hooks": [{"type": "command", "command": command}],
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


_CURSOR_SUPPORTED_EVENTS: frozenset[str] = frozenset({"pre_tool_use", "user_prompt_submit", "stop"})
# Cursor honours per-tool scoping only on its tool-execution events.
_CURSOR_TOOL_SCOPE_EVENTS: frozenset[str] = frozenset({"pre_tool_use"})


class CursorHooksWriter(AbstractSyncWriter):
    """Merges hooks into .cursor/hooks.json → <eventName>[].

    Format::

        {
          "preToolUse": [
            {"event": "preToolUse", "command": "...", "tools": ["Edit", "Shell"],
             "failClosed": true}
          ]
        }

    Merge key: ``entry.command`` within the event's array.
    When the command matches, the ``tools`` list is widened if the desired
    coverage has grown (upgrade-safe). ``failClosed: true`` is emitted for hooks
    marked :attr:`HookEntry.fail_closed` — Cursor otherwise treats a hook that
    crashes/times out as *allow*, silently defeating a security guard.
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
        changed = False
        dropped_tool_scope_events: set[str] = set()

        for hook in kept:
            event_name = _translate_event(hook.event, self.tool_id)
            event_list: list[Any] = existing.get(event_name, [])
            if not isinstance(event_list, list):
                event_list = []

            # Dedup by command; widen tools list if coverage has grown.
            # An existing entry whose ``tools`` key is missing or set to ``[]``
            # is treated as "all tools" — that's how Cursor reads it — so we
            # leave it alone instead of narrowing to the desired subset.
            command = hook.command
            allow_tool_scope = hook.event in _CURSOR_TOOL_SCOPE_EVENTS
            raw_desired = hook.tools or []
            desired_tools = _translate_tools(raw_desired, self.tool_id) if allow_tool_scope else []
            if not allow_tool_scope and raw_desired:
                dropped_tool_scope_events.add(hook.event)
            already_exists = False
            for entry in event_list:
                if not isinstance(entry, dict) or entry.get("command") != command:
                    continue
                already_exists = True
                # Upgrade-safe: add failClosed to a pre-existing entry that lacks
                # it (e.g. written by an older crossby) so security guards harden
                # on re-sync. Never silently downgrade an entry to fail-open.
                if hook.fail_closed and entry.get("failClosed") is not True:
                    entry["failClosed"] = True
                    changed = True
                if not allow_tool_scope:
                    if "tools" in entry:
                        del entry["tools"]
                        changed = True
                    break
                raw_tools = entry.get("tools")
                # Missing key or empty list ⇒ "all tools"; don't narrow it.
                if raw_tools is None or (isinstance(raw_tools, list) and not raw_tools):
                    break
                existing_tools: list[str] = list(raw_tools) if isinstance(raw_tools, list) else []
                missing = [t for t in desired_tools if t not in existing_tools]
                if missing:
                    entry["tools"] = existing_tools + missing
                    changed = True
                break

            if not already_exists:
                new_entry: dict[str, Any] = {
                    "event": event_name,
                    "command": command,
                }
                if allow_tool_scope:
                    new_entry["tools"] = desired_tools
                if hook.fail_closed:
                    # Cursor defaults hooks to fail-open; a security guard must
                    # block the action when the hook itself crashes/times out.
                    new_entry["failClosed"] = True
                event_list.append(new_entry)
                existing[event_name] = event_list
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
# CopilotHooksWriter
# ---------------------------------------------------------------------------


_COPILOT_SUPPORTED_EVENTS: frozenset[str] = frozenset({"pre_tool_use"})


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
                                "Copilot hooks have no per-tool filter; source `tools` "
                                "scope was dropped and the hook applies to all tools."
                            ),
                        )
                    )
                new_entry: dict[str, Any] = {
                    "type": "command",
                    "bash": command,
                    "comment": hook.description or "",
                }
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
            desired_tools = hook.tools or []
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

            handler: dict[str, Any] = {"type": "command", "command": command}
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
    category="features.codex_hooks",
    message=(
        "Could not update `.codex/config.toml` automatically — set "
        "`[features].codex_hooks = true` there manually so Codex loads these hooks."
    ),
)


def _ensure_codex_hooks_feature_flag(project_root: Path, *, dry_run: bool) -> ManualFixNote | None:
    """Enable ``[features].codex_hooks`` in ``.codex/config.toml`` (idempotent).

    Codex ignores ``.codex/hooks.json`` unless this feature flag is set, so the
    hooks we write are inert without it. Merges the flag into any existing config
    (preserving other keys/tables) and writes it back.

    Returns ``None`` on success (or when the flag is already set / ``dry_run``);
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
    if features.get("codex_hooks") is True:
        return None  # already enabled — nothing to do

    if dry_run:
        return None

    features["codex_hooks"] = True
    existing["features"] = features

    # Splice the one key in textually so the user's comments and key ordering
    # survive; fall back to the (lossy) full dump only if that can't be done.
    # Reuses the text read above — a second read could see a different file.
    new_text = splice_or_none(set_scalar(original, ("features",), "codex_hooks", "true"), existing)
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
    sync report classifies the row as ``Check before using``. Because Codex
    ignores ``hooks.json`` unless ``[features].codex_hooks = true`` is set, the
    writer also enables that flag in ``.codex/config.toml`` automatically (a
    manual-fix note is surfaced only if that file can't be written).
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
                    "hooks": [{"type": "command", "command": command}],
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
