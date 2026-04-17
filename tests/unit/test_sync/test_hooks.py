"""Tests for hooks sync writers (Claude, Cursor, Copilot, Gemini)."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry
from crossby.sync.base import SyncData
from crossby.sync.hooks import (
    ClaudeHooksWriter,
    CopilotHooksWriter,
    CursorHooksWriter,
    GeminiHooksWriter,
    _tools_to_matcher,
    _translate_event,
    _translate_tools,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

GUARD_HOOK = HookEntry(
    event="pre_tool_use",
    command="python3 ./scripts/guard.py",
    tools=["Edit", "Write"],
    description="Plan write guard",
)

BARE_HOOK = HookEntry(
    event="pre_tool_use",
    command="python3 ./scripts/lint.py",
    tools=[],
    description="",
)


def _cfg(*hooks: HookEntry) -> SyncData:
    return SyncData(hooks=list(hooks))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


class TestTranslateEvent:
    def test_pre_tool_use_to_claude(self) -> None:
        assert _translate_event("pre_tool_use", AIToolID.CLAUDE) == "PreToolUse"

    def test_pre_tool_use_to_cursor(self) -> None:
        assert _translate_event("pre_tool_use", AIToolID.CURSOR) == "preToolUse"

    def test_pre_tool_use_to_copilot(self) -> None:
        assert _translate_event("pre_tool_use", AIToolID.COPILOT) == "preToolUse"

    def test_pre_tool_use_to_gemini(self) -> None:
        assert _translate_event("pre_tool_use", AIToolID.GEMINI) == "BeforeTool"

    def test_unknown_event_passthrough(self) -> None:
        assert _translate_event("post_tool_use", AIToolID.CLAUDE) == "post_tool_use"


class TestTranslateTools:
    def test_cursor_bash_to_shell(self) -> None:
        assert _translate_tools(["Bash"], AIToolID.CURSOR) == ["Shell"]

    def test_cursor_other_unchanged(self) -> None:
        assert _translate_tools(["Edit", "Write"], AIToolID.CURSOR) == ["Edit", "Write"]

    def test_copilot_name_lowercasing(self) -> None:
        result = _translate_tools(["Edit", "Write", "Bash"], AIToolID.COPILOT)
        assert result == ["edit", "write", "shell"]

    def test_gemini_no_translation(self) -> None:
        assert _translate_tools(["Edit", "Bash", "Write"], AIToolID.GEMINI) == ["Edit", "Bash", "Write"]

    def test_claude_no_translation(self) -> None:
        assert _translate_tools(["Edit", "Bash"], AIToolID.CLAUDE) == ["Edit", "Bash"]


class TestToolsToMatcher:
    def test_two_tools(self) -> None:
        assert _tools_to_matcher(["Edit", "Write"]) == "Edit|Write"

    def test_single_tool(self) -> None:
        assert _tools_to_matcher(["Edit"]) == "Edit"

    def test_empty_tools(self) -> None:
        assert _tools_to_matcher([]) == ".*"

    def test_wildcard(self) -> None:
        assert _tools_to_matcher(["*"]) == ".*"


# ---------------------------------------------------------------------------
# ClaudeHooksWriter
# ---------------------------------------------------------------------------


class TestClaudeHooksWriter:
    writer = ClaudeHooksWriter()

    def test_no_hooks_config_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(SyncData(), tmp_path)
        assert result.action == "skipped"
        assert result.message == "no hooks config"

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".claude" / "settings.json"
        assert path.exists()
        data = _read_json(path)
        pre = data["hooks"]["PreToolUse"]
        assert len(pre) == 1
        assert pre[0]["matcher"] == "Edit|Write"
        assert pre[0]["hooks"] == [{"type": "command", "command": "python3 ./scripts/guard.py"}]

    def test_empty_tools_uses_wildcard_matcher(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(BARE_HOOK), tmp_path)
        data = _read_json(tmp_path / ".claude" / "settings.json")
        assert data["hooks"]["PreToolUse"][0]["matcher"] == ".*"

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"permissions": {"allow": ["Bash(git *)"]}}), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        assert "Bash(git *)" in data["permissions"]["allow"]
        assert len(data["hooks"]["PreToolUse"]) == 1

    def test_preserves_existing_hooks(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        existing_hook = {"matcher": ".*", "hooks": [{"type": "command", "command": "echo existing"}]}
        path.write_text(
            json.dumps({"hooks": {"PreToolUse": [existing_hook]}}),
            encoding="utf-8",
        )

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        assert len(data["hooks"]["PreToolUse"]) == 2
        commands = [
            h["command"]
            for entry in data["hooks"]["PreToolUse"]
            for h in entry.get("hooks", [])
        ]
        assert "echo existing" in commands
        assert "python3 ./scripts/guard.py" in commands

    def test_idempotent_skipped(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "skipped"

    def test_dry_run_no_write(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".claude" / "settings.json").exists()

    def test_dry_run_no_change(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path, dry_run=True)
        assert result.action == "skipped"

    def test_malformed_json_error(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text("{invalid json!!", encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "error"

    def test_multiple_hooks_added(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK, BARE_HOOK), tmp_path)
        data = _read_json(tmp_path / ".claude" / "settings.json")
        assert len(data["hooks"]["PreToolUse"]) == 2

    def test_dedup_by_command_not_matcher(self, tmp_path: Path) -> None:
        """Same command with different tools is still a duplicate."""
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        different_tools_hook = HookEntry(
            event="pre_tool_use",
            command="python3 ./scripts/guard.py",
            tools=["Bash"],
        )
        result = self.writer.sync(_cfg(different_tools_hook), tmp_path)
        assert result.action == "skipped"

    def test_updated_action_on_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "updated"

    def test_legacy_string_hook_dedup(self, tmp_path: Path) -> None:
        """Legacy string entries in inner hooks[] are recognized as duplicates."""
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        # Simulate a legacy config where the inner hooks entry is a plain string
        existing = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": ".*", "hooks": ["python3 ./scripts/guard.py"]}
                ]
            }
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        assert result.action == "skipped"
        data = _read_json(path)
        assert len(data["hooks"]["PreToolUse"]) == 1


# ---------------------------------------------------------------------------
# CursorHooksWriter
# ---------------------------------------------------------------------------


class TestCursorHooksWriter:
    writer = CursorHooksWriter()

    def test_no_hooks_config_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(SyncData(), tmp_path)
        assert result.action == "skipped"

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".cursor" / "hooks.json"
        assert path.exists()
        data = _read_json(path)
        pre = data["preToolUse"]
        assert len(pre) == 1
        assert pre[0]["event"] == "preToolUse"
        assert pre[0]["command"] == "python3 ./scripts/guard.py"
        assert pre[0]["tools"] == ["Edit", "Write"]

    def test_bash_translated_to_shell(self, tmp_path: Path) -> None:
        hook = HookEntry(event="pre_tool_use", command="echo hi", tools=["Bash"])
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["preToolUse"][0]["tools"] == ["Shell"]

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {"preToolUse": [{"event": "preToolUse", "command": "echo existing", "tools": []}]}
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        assert len(data["preToolUse"]) == 2

    def test_preserves_other_keys(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"other": "setting"}), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        data = _read_json(path)
        assert data["other"] == "setting"

    def test_idempotent_skipped(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "skipped"

    def test_dry_run_no_write(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".cursor" / "hooks.json").exists()

    def test_malformed_json_error(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        path.write_text("[not an object]", encoding="utf-8")
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "error"

    def test_updated_action_on_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        path.write_text(json.dumps({}), encoding="utf-8")
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "updated"


# ---------------------------------------------------------------------------
# CopilotHooksWriter
# ---------------------------------------------------------------------------


class TestCopilotHooksWriter:
    writer = CopilotHooksWriter()

    def test_no_hooks_config_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(SyncData(), tmp_path)
        assert result.action == "skipped"

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        assert path.exists()
        data = _read_json(path)
        assert data["version"] == 1
        pre = data["hooks"]["preToolUse"]
        assert len(pre) == 1
        assert pre[0]["type"] == "command"
        assert pre[0]["bash"] == "python3 ./scripts/guard.py"
        assert pre[0]["comment"] == "Plan write guard"

    def test_tools_warning_in_message(self, tmp_path: Path) -> None:
        """CopilotHooksWriter warns when canonical hook specifies tools."""
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.message is not None
        assert "Copilot hooks do not support tool filtering" in result.message

    def test_no_tools_no_warning(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(BARE_HOOK), tmp_path)
        assert result.message is None

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        existing = {"version": 1, "hooks": {"preToolUse": [{"type": "command", "bash": "echo old", "comment": ""}]}}
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg(BARE_HOOK), tmp_path)

        data = _read_json(path)
        assert len(data["hooks"]["preToolUse"]) == 2

    def test_version_1_added_on_write(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(BARE_HOOK), tmp_path)
        data = _read_json(tmp_path / ".github" / "hooks" / "hooks.json")
        assert data["version"] == 1

    def test_idempotent_skipped(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "skipped"

    def test_idempotent_no_warning_on_already_present_hook(self, tmp_path: Path) -> None:
        """On the idempotent path, no tools-warning is emitted."""
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "skipped"
        assert result.message is None

    def test_dry_run_no_write(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".github" / "hooks" / "hooks.json").exists()

    def test_malformed_json_error(self, tmp_path: Path) -> None:
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        path.write_text("{bad}", encoding="utf-8")
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "error"

    def test_description_used_as_comment(self, tmp_path: Path) -> None:
        hook = HookEntry(event="pre_tool_use", command="echo hi", description="My description")
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".github" / "hooks" / "hooks.json")
        assert data["hooks"]["preToolUse"][0]["comment"] == "My description"

    def test_no_description_empty_comment(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(BARE_HOOK), tmp_path)
        data = _read_json(tmp_path / ".github" / "hooks" / "hooks.json")
        assert data["hooks"]["preToolUse"][0]["comment"] == ""

    def test_updated_action_on_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "hooks": {}}), encoding="utf-8")
        result = self.writer.sync(_cfg(BARE_HOOK), tmp_path)
        assert result.action == "updated"

    def test_wildcard_tools_no_warning(self, tmp_path: Path) -> None:
        """tools=['*'] is semantically 'all tools' and should not trigger a warning."""
        hook = HookEntry(event="pre_tool_use", command="echo hi", tools=["*"])
        result = self.writer.sync(_cfg(hook), tmp_path)
        assert result.message is None

    def test_fixes_missing_version(self, tmp_path: Path) -> None:
        """Writes file even on idempotent hook path when version is missing."""
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        # Hooks already present, but version is absent
        existing = {"hooks": {"preToolUse": [{"type": "command", "bash": "python3 ./scripts/lint.py", "comment": ""}]}}
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(BARE_HOOK), tmp_path)

        assert result.action == "updated"
        data = _read_json(path)
        assert data["version"] == 1

    def test_fixes_wrong_version(self, tmp_path: Path) -> None:
        """Writes file even on idempotent hook path when version value is wrong."""
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        existing = {"version": 99, "hooks": {"preToolUse": [{"type": "command", "bash": "python3 ./scripts/lint.py", "comment": ""}]}}
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(BARE_HOOK), tmp_path)

        assert result.action == "updated"
        data = _read_json(path)
        assert data["version"] == 1


# ---------------------------------------------------------------------------
# GeminiHooksWriter
# ---------------------------------------------------------------------------


class TestGeminiHooksWriter:
    writer = GeminiHooksWriter()

    def test_no_hooks_config_skipped(self, tmp_path: Path) -> None:
        result = self.writer.sync(SyncData(), tmp_path)
        assert result.action == "skipped"

    def test_creates_new_file(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".gemini" / "settings.json"
        assert path.exists()
        data = _read_json(path)
        assert isinstance(data["hooks"], dict)
        before_tool = data["hooks"]["BeforeTool"]
        assert len(before_tool) == 1
        assert before_tool[0]["matcher"] == "Edit|Write"
        assert before_tool[0]["hooks"] == [
            {"type": "command", "command": "python3 ./scripts/guard.py"}
        ]

    def test_empty_tools_uses_wildcard_matcher(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(BARE_HOOK), tmp_path)
        data = _read_json(tmp_path / ".gemini" / "settings.json")
        assert data["hooks"]["BeforeTool"][0]["matcher"] == ".*"

    def test_hooks_nested_object_not_flat_array(self, tmp_path: Path) -> None:
        """Gemini stores hooks as a nested object keyed by event name."""
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        data = _read_json(tmp_path / ".gemini" / "settings.json")
        assert isinstance(data["hooks"], dict)
        assert "BeforeTool" in data["hooks"]

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        existing_entry = {
            "matcher": ".*",
            "hooks": [{"type": "command", "command": "echo old"}],
        }
        existing = {
            "mcpServers": {"ctx": {"command": "npx"}},
            "hooks": {"BeforeTool": [existing_entry]},
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        assert data["mcpServers"] == existing["mcpServers"]
        assert len(data["hooks"]["BeforeTool"]) == 2

    def test_migrates_old_flat_array_format(self, tmp_path: Path) -> None:
        """Old flat-array hooks are converted to nested format, preserving commands."""
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        old_format = {
            "hooks": [{"event": "BeforeTool", "command": "echo old", "tools": [".*"]}],
        }
        path.write_text(json.dumps(old_format), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        # After migration, hooks should be a dict
        assert isinstance(data["hooks"], dict)
        assert "BeforeTool" in data["hooks"]
        # Both the migrated old hook and the new hook should be present
        commands = [
            h["command"]
            for entry in data["hooks"]["BeforeTool"]
            for h in entry.get("hooks", [])
        ]
        assert "echo old" in commands
        assert "python3 ./scripts/guard.py" in commands

    def test_preserves_other_gemini_settings(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"theme": "dark", "hooks": {}}), encoding="utf-8")
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        data = _read_json(path)
        assert data["theme"] == "dark"

    def test_idempotent_skipped(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "skipped"

    def test_dedup_by_command(self, tmp_path: Path) -> None:
        """Same command with different tools is still a duplicate."""
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        different_tools_hook = HookEntry(
            event="pre_tool_use",
            command="python3 ./scripts/guard.py",
            tools=["Bash"],
        )
        result = self.writer.sync(_cfg(different_tools_hook), tmp_path)
        assert result.action == "skipped"

    def test_different_events_not_deduped(self, tmp_path: Path) -> None:
        """Same command for different events is NOT a duplicate."""
        hook2 = HookEntry(event="post_tool_use", command="python3 ./scripts/guard.py", tools=[])
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(hook2), tmp_path)
        assert result.action == "updated"

    def test_dry_run_no_write(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not (tmp_path / ".gemini" / "settings.json").exists()

    def test_dry_run_no_change(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path, dry_run=True)
        assert result.action == "skipped"

    def test_malformed_json_error(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        path.write_text("{bad}", encoding="utf-8")
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "error"

    def test_updated_action_on_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".gemini" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({}), encoding="utf-8")
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "updated"


# ---------------------------------------------------------------------------
# Config model validation
# ---------------------------------------------------------------------------


class TestHookEntryModel:
    def test_defaults(self) -> None:
        hook = HookEntry(event="pre_tool_use", command="echo hi")
        assert hook.tools == []
        assert hook.description == ""

    def test_full_fields(self) -> None:
        hook = HookEntry(
            event="pre_tool_use",
            command="python3 ./guard.py",
            tools=["Edit", "Write"],
            description="My guard",
        )
        assert hook.event == "pre_tool_use"
        assert hook.command == "python3 ./guard.py"
        assert hook.tools == ["Edit", "Write"]
        assert hook.description == "My guard"


class TestSyncDataHooksField:
    def test_hooks_defaults_to_empty_list(self) -> None:
        data = SyncData()
        assert data.hooks == []

    def test_hooks_from_hook_entries(self) -> None:
        data = SyncData(
            hooks=[
                HookEntry(event="pre_tool_use", command="echo hi", tools=["Edit"]),
            ]
        )
        assert len(data.hooks) == 1
        assert data.hooks[0].command == "echo hi"
