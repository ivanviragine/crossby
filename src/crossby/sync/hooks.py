"""Hooks sync writers — one per AI tool.

Each writer merges .crossby.yml hooks into the tool's native config format
using a non-destructive merge strategy (dedup by (event, command)):
- New hooks are appended to the tool's hooks list.
- Hooks with the same (event, command) are skipped (idempotent).
- Hooks in the target but NOT in .crossby.yml are preserved.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncResult
from crossby.sync.json_utils import read_json_file, write_json_file


# ---------------------------------------------------------------------------
# Event name translation
# ---------------------------------------------------------------------------

_EVENT_NAMES: dict[AIToolID, dict[str, str]] = {
    AIToolID.CLAUDE: {"pre_tool_use": "PreToolUse"},
    AIToolID.CURSOR: {"pre_tool_use": "preToolUse"},
    AIToolID.COPILOT: {"pre_tool_use": "preToolUse"},
    AIToolID.GEMINI: {"pre_tool_use": "BeforeTool"},
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


def _tools_to_gemini(tools: list[str]) -> list[str]:
    """Convert tools list to Gemini format (empty/wildcard → ['.*'])."""
    if not tools or tools == ["*"]:
        return [".*"]
    return tools


# ---------------------------------------------------------------------------
# ClaudeHooksWriter
# ---------------------------------------------------------------------------


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

    Dedup key: command value within any entry's inner ``hooks[]``.
    """

    tool_id = AIToolID.CLAUDE
    concern = SyncConcern.HOOKS

    def sync(
        self,
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not config.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".claude" / "settings.json"
        data, error, was_new = read_json_file(path)
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

        existing = data or {}
        hooks_section: dict[str, Any] = existing.get("hooks", {})
        if not isinstance(hooks_section, dict):
            hooks_section = {}

        changed = False
        for hook in config.hooks:
            event_name = _translate_event(hook.event, self.tool_id)
            event_list: list[Any] = hooks_section.get(event_name, [])
            if not isinstance(event_list, list):
                event_list = []

            # Dedup: check if command already exists in any entry's inner hooks[]
            command = hook.command
            already_exists = False
            for entry in event_list:
                if not isinstance(entry, dict):
                    continue
                inner_hooks = entry.get("hooks")
                if not isinstance(inner_hooks, list):
                    continue
                for inner in inner_hooks:
                    if isinstance(inner, dict) and inner.get("command") == command:
                        already_exists = True
                        break
                if already_exists:
                    break

            if not already_exists:
                tools = hook.tools or []
                new_entry: dict[str, Any] = {
                    "matcher": _tools_to_matcher(tools),
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
            )

        action = "created" if was_new else "updated"
        if not dry_run:
            existing["hooks"] = hooks_section
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
        )


# ---------------------------------------------------------------------------
# CursorHooksWriter
# ---------------------------------------------------------------------------


class CursorHooksWriter(AbstractSyncWriter):
    """Merges hooks into .cursor/hooks.json → <eventName>[].

    Format::

        {
          "preToolUse": [
            {"event": "preToolUse", "command": "...", "tools": ["Edit", "Shell"]}
          ]
        }

    Dedup key: ``entry.command`` within the event's array.
    """

    tool_id = AIToolID.CURSOR
    concern = SyncConcern.HOOKS

    def sync(
        self,
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not config.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".cursor" / "hooks.json"
        data, error, was_new = read_json_file(path)
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

        existing = data or {}
        changed = False

        for hook in config.hooks:
            event_name = _translate_event(hook.event, self.tool_id)
            event_list: list[Any] = existing.get(event_name, [])
            if not isinstance(event_list, list):
                event_list = []

            # Dedup: check if command already present
            command = hook.command
            already_exists = any(
                isinstance(entry, dict) and entry.get("command") == command
                for entry in event_list
            )

            if not already_exists:
                tools = _translate_tools(hook.tools or [], self.tool_id)
                new_entry: dict[str, Any] = {
                    "event": event_name,
                    "command": command,
                    "tools": tools,
                }
                event_list.append(new_entry)
                existing[event_name] = event_list
                changed = True

        if not changed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=path,
            )

        action = "created" if was_new else "updated"
        if not dry_run:
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
        )


# ---------------------------------------------------------------------------
# CopilotHooksWriter
# ---------------------------------------------------------------------------


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
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not config.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".github" / "hooks" / "hooks.json"
        data, error, was_new = read_json_file(path)
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

        existing = data or {}
        hooks_section: dict[str, Any] = existing.get("hooks", {})
        if not isinstance(hooks_section, dict):
            hooks_section = {}

        changed = False
        warnings_msgs: list[str] = []

        for hook in config.hooks:
            event_name = _translate_event(hook.event, self.tool_id)
            event_list: list[Any] = hooks_section.get(event_name, [])
            if not isinstance(event_list, list):
                event_list = []

            # Dedup: check if bash command already present
            command = hook.command
            already_exists = any(
                isinstance(entry, dict) and entry.get("bash") == command
                for entry in event_list
            )

            if not already_exists:
                if hook.tools:
                    warnings_msgs.append(
                        "Copilot hooks do not support tool filtering — "
                        f"'{command}' will apply to all tools."
                    )
                new_entry: dict[str, Any] = {
                    "type": "command",
                    "bash": command,
                    "comment": hook.description or "",
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
                message="; ".join(warnings_msgs) or None,
            )

        action = "created" if was_new else "updated"
        if not dry_run:
            existing.setdefault("version", 1)
            existing["hooks"] = hooks_section
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
            message="; ".join(warnings_msgs) or None,
        )


# ---------------------------------------------------------------------------
# GeminiHooksWriter
# ---------------------------------------------------------------------------


class GeminiHooksWriter(AbstractSyncWriter):
    """Merges hooks into .gemini/settings.json → hooks[] (flat array).

    Format::

        {
          "hooks": [
            {"event": "BeforeTool", "command": "...", "tools": ["Edit", "Write"]}
          ]
        }

    Note: ``event`` is a field value in the entry dict, not an array key.
    Dedup key: ``(entry.event, entry.command)`` pair.
    """

    tool_id = AIToolID.GEMINI
    concern = SyncConcern.HOOKS

    def sync(
        self,
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if not config.hooks:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no hooks config",
            )

        path = project_root / ".gemini" / "settings.json"
        data, error, was_new = read_json_file(path)
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

        existing = data or {}
        hooks_list: list[Any] = existing.get("hooks", [])
        if not isinstance(hooks_list, list):
            hooks_list = []

        changed = False

        for hook in config.hooks:
            event_name = _translate_event(hook.event, self.tool_id)
            command = hook.command

            # Dedup: check if (event, command) pair already present
            already_exists = any(
                isinstance(entry, dict)
                and entry.get("event") == event_name
                and entry.get("command") == command
                for entry in hooks_list
            )

            if not already_exists:
                tools = hook.tools or []
                new_entry: dict[str, Any] = {
                    "event": event_name,
                    "command": command,
                    "tools": _tools_to_gemini(tools),
                }
                hooks_list.append(new_entry)
                changed = True

        if not changed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=path,
            )

        action = "created" if was_new else "updated"
        if not dry_run:
            existing["hooks"] = hooks_list
            write_json_file(path, existing)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=path,
        )
