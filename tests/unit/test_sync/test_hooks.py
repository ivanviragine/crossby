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
    _widen_matcher,
)
from crossby.sync.readers import discover_hooks

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
        # `nonexistent_event` is not in any tool's mapping, so it falls through
        # unchanged. (Note: `post_tool_use` is now a canonical event with a
        # mapping for every supporting writer.)
        assert _translate_event("nonexistent_event", AIToolID.CLAUDE) == "nonexistent_event"


class TestTranslateTools:
    def test_cursor_bash_to_shell(self) -> None:
        assert _translate_tools(["Bash"], AIToolID.CURSOR) == ["Shell"]

    def test_cursor_other_unchanged(self) -> None:
        assert _translate_tools(["Edit", "Write"], AIToolID.CURSOR) == ["Edit", "Write"]

    def test_copilot_name_lowercasing(self) -> None:
        result = _translate_tools(["Edit", "Write", "Bash"], AIToolID.COPILOT)
        assert result == ["edit", "write", "shell"]

    def test_gemini_no_translation(self) -> None:
        result = _translate_tools(["Edit", "Bash", "Write"], AIToolID.GEMINI)
        assert result == ["Edit", "Bash", "Write"]

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


class TestWidenMatcher:
    def test_no_existing_returns_desired(self) -> None:
        assert _widen_matcher(None, ["Edit"]) == "Edit"
        assert _widen_matcher("", ["Edit"]) == "Edit"

    def test_catchall_existing_wins(self) -> None:
        # ".*" covers everything — narrowing it would drop coverage.
        assert _widen_matcher(".*", ["Edit", "Write"]) == ".*"

    def test_catchall_desired_wins(self) -> None:
        # An empty desired tool list expands to ".*".
        assert _widen_matcher("Edit", []) == ".*"

    def test_unions_disjoint_tokens(self) -> None:
        result = _widen_matcher("Edit|Write", ["Bash"])
        assert set(result.split("|")) == {"Edit", "Write", "Bash"}

    def test_idempotent_subset(self) -> None:
        # Desired is already covered → matcher unchanged.
        assert _widen_matcher("Edit|Write", ["Edit"]) == "Edit|Write"


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
        existing_hook = {
            "matcher": ".*",
            "hooks": [{"type": "command", "command": "echo existing"}],
        }
        path.write_text(
            json.dumps({"hooks": {"PreToolUse": [existing_hook]}}),
            encoding="utf-8",
        )

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        assert len(data["hooks"]["PreToolUse"]) == 2
        commands = [
            h["command"] for entry in data["hooks"]["PreToolUse"] for h in entry.get("hooks", [])
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

    def test_dedup_by_command_widens_matcher(self, tmp_path: Path) -> None:
        """Same command with different tools widens the matcher (no duplicate entry).

        Widen-not-replace protects existing coverage: re-running with a
        narrower hook spec must not silently shrink the matcher.
        """
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        different_tools_hook = HookEntry(
            event="pre_tool_use",
            command="python3 ./scripts/guard.py",
            tools=["Bash"],
        )
        result = self.writer.sync(_cfg(different_tools_hook), tmp_path)
        assert result.action == "updated"
        path = tmp_path / ".claude" / "settings.json"
        data = _read_json(path)
        pre_tool = data["hooks"]["PreToolUse"]
        assert len(pre_tool) == 1, "should not add a duplicate entry"
        # Matcher is the union of the two sync inputs, not the latest.
        tokens = pre_tool[0]["matcher"].split("|")
        assert set(tokens) == {"Edit", "Write", "Bash"}

    def test_updated_action_on_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "updated"

    def test_legacy_string_hook_dedup(self, tmp_path: Path) -> None:
        """Legacy string entries in inner hooks[] are recognized as duplicates.

        Existing ``.*`` matcher is broader than any concrete tool set, so the
        widen logic leaves it alone — narrowing it to ``Edit|Write`` would
        silently drop coverage.
        """
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        # Simulate a legacy config where the inner hooks entry is a plain string
        existing = {
            "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": ["python3 ./scripts/guard.py"]}]}
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        # No duplicate entry added; catch-all matcher preserved.
        assert result.action == "skipped"
        data = _read_json(path)
        assert len(data["hooks"]["PreToolUse"]) == 1
        assert data["hooks"]["PreToolUse"][0]["matcher"] == ".*"

    def test_widen_does_not_narrow_catchall(self, tmp_path: Path) -> None:
        """Existing ``.*`` matcher must not be replaced by a narrower one."""
        path = tmp_path / ".claude" / "settings.json"
        path.parent.mkdir()
        existing = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": ".*",
                        "hooks": [{"type": "command", "command": "python3 ./scripts/guard.py"}],
                    }
                ]
            }
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        # GUARD_HOOK has tools=["Edit", "Write"] → matcher would be "Edit|Write",
        # but ".*" is broader so the widen logic keeps the existing matcher.
        assert result.action == "skipped"
        data = _read_json(path)
        assert data["hooks"]["PreToolUse"][0]["matcher"] == ".*"


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

    def test_fail_closed_emitted(self, tmp_path: Path) -> None:
        """A fail-closed hook writes ``failClosed: true`` (Cursor is fail-open by default)."""
        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Edit"], fail_closed=True)
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["preToolUse"][0]["failClosed"] is True

    def test_fail_closed_absent_by_default(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert "failClosed" not in data["preToolUse"][0]

    def test_fail_closed_added_to_existing_entry(self, tmp_path: Path) -> None:
        """Re-syncing a fail-closed hook hardens a pre-existing fail-open entry."""
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {"preToolUse": [{"event": "preToolUse", "command": "guard", "tools": ["Edit"]}]}
        path.write_text(json.dumps(existing), encoding="utf-8")

        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Edit"], fail_closed=True)
        result = self.writer.sync(_cfg(hook), tmp_path)

        assert result.action == "updated"
        data = _read_json(path)
        assert len(data["preToolUse"]) == 1  # same command → merged, not duplicated
        assert data["preToolUse"][0]["failClosed"] is True

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {
            "preToolUse": [{"event": "preToolUse", "command": "echo existing", "tools": []}]
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        assert len(data["preToolUse"]) == 2

    def test_empty_existing_tools_not_narrowed(self, tmp_path: Path) -> None:
        """``tools: []`` in Cursor means "all tools" — must not be narrowed.

        Previously the writer appended desired tools onto the empty list,
        silently shrinking coverage from all-tools to just the desired set.
        """
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {
            "preToolUse": [
                {
                    "event": "preToolUse",
                    "command": "python3 ./scripts/guard.py",
                    "tools": [],
                }
            ]
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        # No-op: matched by command, existing tools=[] means all → leave alone.
        assert result.action == "skipped"
        data = _read_json(path)
        assert data["preToolUse"][0]["tools"] == []

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
        """CopilotHooksWriter warns when canonical hook specifies tools.

        The note is emitted as a manual-fix entry so the sync report flips
        the row to ``Check before using``; the literal substring
        ``manual_fix`` plus ``hooks.tools`` survives in the message.
        """
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.message is not None
        assert "manual_fix" in result.message
        assert "hooks.tools" in result.message

    def test_no_tools_no_warning(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(BARE_HOOK), tmp_path)
        assert result.message is None

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        existing = {
            "version": 1,
            "hooks": {
                "preToolUse": [{"type": "command", "bash": "echo old", "comment": ""}],
            },
        }
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
        existing = {
            "hooks": {
                "preToolUse": [
                    {"type": "command", "bash": "python3 ./scripts/lint.py", "comment": ""},
                ],
            },
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(BARE_HOOK), tmp_path)

        assert result.action == "updated"
        data = _read_json(path)
        assert data["version"] == 1

    def test_fixes_wrong_version(self, tmp_path: Path) -> None:
        """Writes file even on idempotent hook path when version value is wrong."""
        path = tmp_path / ".github" / "hooks" / "hooks.json"
        path.parent.mkdir(parents=True)
        existing = {
            "version": 99,
            "hooks": {
                "preToolUse": [
                    {"type": "command", "bash": "python3 ./scripts/lint.py", "comment": ""},
                ],
            },
        }
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
            h["command"] for entry in data["hooks"]["BeforeTool"] for h in entry.get("hooks", [])
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

    def test_dedup_by_command_widens_matcher(self, tmp_path: Path) -> None:
        """Same command with different tools widens the existing matcher.

        Before the widen fix, Gemini's writer left the matcher untouched on
        a command match, so coverage grew via duplicate entries or got stuck
        at the first sync's tools. Now the matcher unions both tool sets.
        """
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        different_tools_hook = HookEntry(
            event="pre_tool_use",
            command="python3 ./scripts/guard.py",
            tools=["Bash"],
        )
        result = self.writer.sync(_cfg(different_tools_hook), tmp_path)
        assert result.action == "updated"
        path = tmp_path / ".gemini" / "settings.json"
        data = _read_json(path)
        entries = data["hooks"]["BeforeTool"]
        assert len(entries) == 1
        tokens = entries[0]["matcher"].split("|")
        assert set(tokens) == {"Edit", "Write", "Bash"}

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


# ---------------------------------------------------------------------------
# discover_hooks — cross-tool union of tool scopes
# ---------------------------------------------------------------------------


def _write_claude_hook(root: Path, command: str, matcher: str) -> None:
    path = root / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": matcher,
                            "hooks": [{"type": "command", "command": command}],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_cursor_hook(root: Path, command: str, tools: list[str]) -> None:
    path = root / ".cursor" / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "preToolUse": [
                    {"command": command, "tools": tools},
                ]
            }
        ),
        encoding="utf-8",
    )


class TestDiscoverHooksUnion:
    def test_same_command_different_tool_scopes_unioned(self, tmp_path: Path) -> None:
        _write_claude_hook(tmp_path, "python3 guard.py", "Edit")
        _write_cursor_hook(tmp_path, "python3 guard.py", ["edit", "write"])

        hooks = discover_hooks(tmp_path)

        assert len(hooks) == 1
        assert hooks[0].event == "pre_tool_use"
        assert hooks[0].command == "python3 guard.py"
        assert set(hooks[0].tools) == {"Edit", "Write"}

    def test_empty_tools_means_all_and_wins(self, tmp_path: Path) -> None:
        # Claude matcher ".*" → empty canonical tools (means all).
        _write_claude_hook(tmp_path, "python3 guard.py", ".*")
        _write_cursor_hook(tmp_path, "python3 guard.py", ["edit"])

        hooks = discover_hooks(tmp_path)

        assert len(hooks) == 1
        assert hooks[0].tools == []

    def test_distinct_commands_not_merged(self, tmp_path: Path) -> None:
        _write_claude_hook(tmp_path, "python3 guard.py", "Edit")
        _write_cursor_hook(tmp_path, "python3 other.py", ["write"])

        hooks = discover_hooks(tmp_path)

        assert len(hooks) == 2


# ---------------------------------------------------------------------------
# CodexHooksWriter
# ---------------------------------------------------------------------------


def _stop_hook() -> HookEntry:
    return HookEntry(event="stop", command="python3 ./scripts/post.py", tools=[])


def _post_tool_use_hook() -> HookEntry:
    return HookEntry(event="post_tool_use", command="python3 ./scripts/audit.py", tools=["Edit"])


def _notification_hook() -> HookEntry:
    return HookEntry(event="notification", command="python3 ./scripts/notify.py", tools=[])


class TestCodexHooksWriter:
    """CodexHooksWriter — supports a subset of Claude's events."""

    def setup_method(self) -> None:
        from crossby.sync.hooks import CodexHooksWriter

        self.writer = CodexHooksWriter()

    def test_writes_supported_events_to_codex_hooks_json(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK, _post_tool_use_hook()), tmp_path)
        assert result.action == "created"
        path = tmp_path / ".codex" / "hooks.json"
        data = _read_json(path)
        assert "PreToolUse" in data["hooks"]
        assert "PostToolUse" in data["hooks"]

    def test_drops_notification_event(self, tmp_path: Path) -> None:
        """Codex has no notification event; it should be dropped with a note."""
        result = self.writer.sync(_cfg(_notification_hook()), tmp_path)
        # No supported hook → nothing kept, so no feature flag is written; the
        # dropped-event note is what surfaces.
        assert result.message is not None
        assert "manual_fix" in result.message
        assert "hooks.notification" in result.message

    def test_user_prompt_submit_drops_matcher(self, tmp_path: Path) -> None:
        ups_with_tools = HookEntry(
            event="user_prompt_submit",
            command="python3 ./scripts/ups.py",
            tools=["Edit"],
        )
        result = self.writer.sync(_cfg(ups_with_tools), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".codex" / "hooks.json")
        entries = data["hooks"]["UserPromptSubmit"]
        assert len(entries) == 1
        assert "matcher" not in entries[0]
        assert result.message is not None
        assert "user_prompt_submit.matcher" in result.message

    def test_stop_drops_matcher(self, tmp_path: Path) -> None:
        stop_with_tools = HookEntry(event="stop", command="python3 stop.py", tools=["Bash"])
        result = self.writer.sync(_cfg(stop_with_tools), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".codex" / "hooks.json")
        assert "matcher" not in data["hooks"]["Stop"][0]

    def test_enables_codex_hooks_feature_flag(self, tmp_path: Path) -> None:
        """A sync writes [features].codex_hooks = true so Codex loads the hooks."""
        import tomllib

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        # Flag written automatically → no manual-fix note on the happy path.
        assert result.message is None or "features.codex_hooks" not in result.message

        config = tmp_path / ".codex" / "config.toml"
        assert config.is_file()
        parsed = tomllib.loads(config.read_text(encoding="utf-8"))
        assert parsed["features"]["codex_hooks"] is True

    def test_feature_flag_preserves_existing_config(self, tmp_path: Path) -> None:
        """Enabling the flag merges into an existing config, keeping other keys."""
        import tomllib

        config = tmp_path / ".codex" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text('model = "gpt-5"\n\n[features]\nother_flag = true\n', encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        parsed = tomllib.loads(config.read_text(encoding="utf-8"))
        assert parsed["model"] == "gpt-5"
        assert parsed["features"]["other_flag"] is True
        assert parsed["features"]["codex_hooks"] is True

    def test_feature_flag_idempotent_when_already_set(self, tmp_path: Path) -> None:
        """A config that already enables the flag is left untouched."""
        config = tmp_path / ".codex" / "config.toml"
        config.parent.mkdir(parents=True)
        original = "[features]\ncodex_hooks = true\n"
        config.write_text(original, encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        assert config.read_text(encoding="utf-8") == original

    def test_malformed_config_surfaces_manual_fix_note(self, tmp_path: Path) -> None:
        """If .codex/config.toml is invalid TOML, surface a manual-fix note."""
        config = tmp_path / ".codex" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text("this is = = not valid toml", encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.message is not None
        assert "features.codex_hooks" in result.message
        # The malformed file is left as-is (not clobbered).
        assert config.read_text(encoding="utf-8") == "this is = = not valid toml"

    def test_merges_with_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".codex" / "hooks.json"
        path.parent.mkdir(parents=True)
        existing = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Read", "hooks": [{"type": "command", "command": "echo old"}]}
                ]
            }
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        # Old entry preserved; new entry appended.
        assert len(data["hooks"]["PreToolUse"]) == 2

    def test_supported_event_round_trip_writes_matcher(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".codex" / "hooks.json")
        entry = data["hooks"]["PreToolUse"][0]
        assert entry["matcher"] == "Edit|Write"


# ---------------------------------------------------------------------------
# Cross-writer parity — every existing writer drops unsupported events
# ---------------------------------------------------------------------------


import pytest  # noqa: E402


class TestCrossWriterUnsupportedEvents:
    """Every hooks writer flags unsupported events with a manual_fix note."""

    @pytest.mark.parametrize(
        ("writer_cls", "unsupported_event"),
        [
            # Cursor only supports pre_tool_use + stop.
            ("CursorHooksWriter", "post_tool_use"),
            ("CursorHooksWriter", "session_start"),
            ("CursorHooksWriter", "user_prompt_submit"),
            ("CursorHooksWriter", "notification"),
            # Copilot only supports pre_tool_use.
            ("CopilotHooksWriter", "post_tool_use"),
            ("CopilotHooksWriter", "stop"),
            ("CopilotHooksWriter", "user_prompt_submit"),
            # Gemini supports pre_tool_use + post_tool_use.
            ("GeminiHooksWriter", "stop"),
            ("GeminiHooksWriter", "notification"),
            ("GeminiHooksWriter", "user_prompt_submit"),
            # Codex supports everything except notification.
            ("CodexHooksWriter", "notification"),
        ],
    )
    def test_writer_drops_unsupported_event(
        self, tmp_path: Path, writer_cls: str, unsupported_event: str
    ) -> None:
        from crossby.sync import hooks as hooks_mod

        writer = getattr(hooks_mod, writer_cls)()
        hook = HookEntry(event=unsupported_event, command="echo x", tools=[])
        result = writer.sync(_cfg(hook), tmp_path)
        # The writer may write nothing (no supported hooks left) OR write
        # extras like the Codex features-flag note. Either way, the manual_fix
        # substring must surface so report.classify_status flips the row.
        assert result.message is not None, (
            f"{writer_cls} must emit a message when {unsupported_event} is dropped"
        )
        assert "manual_fix" in result.message
        assert f"hooks.{unsupported_event}" in result.message
