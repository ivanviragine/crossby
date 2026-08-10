"""Tests for hooks sync writers (Claude, Cursor, Copilot, Codex)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry
from crossby.sync.base import SyncData
from crossby.sync.hooks import (
    ClaudeHooksWriter,
    CopilotHooksWriter,
    CursorHooksWriter,
    _tools_to_matcher,
    _translate_event,
    _translate_tools,
    _widen_matcher,
)
from crossby.sync.readers import (
    _read_agy_hooks,
    _read_codex_hooks,
    _read_copilot_hooks,
    _read_cursor_hooks,
    discover_hooks,
    reader_available,
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

    def test_unknown_event_passthrough(self) -> None:
        # `nonexistent_event` is not in any tool's mapping, so it falls through
        # unchanged. (Note: `post_tool_use` is now a canonical event with a
        # mapping for every supporting writer.)
        assert _translate_event("nonexistent_event", AIToolID.CLAUDE) == "nonexistent_event"


class TestTranslateTools:
    def test_cursor_bash_to_shell(self) -> None:
        assert _translate_tools(["Bash"], AIToolID.CURSOR) == ["Shell"]

    def test_cursor_edit_collapses_into_write(self) -> None:
        """Cursor has no Edit tool — Edit/MultiEdit both mean Write, deduped."""
        assert _translate_tools(["Edit", "Write"], AIToolID.CURSOR) == ["Write"]
        assert _translate_tools(["Edit", "MultiEdit"], AIToolID.CURSOR) == ["Write"]

    def test_cursor_unknown_name_unchanged(self) -> None:
        assert _translate_tools(["Delete"], AIToolID.CURSOR) == ["Delete"]

    def test_antigravity_cli_native_names(self) -> None:
        """agy's matcher is a regex over its live toolCall.name values.

        Without translation a `Write|Edit|Bash` matcher matches none of them and
        the guard installs but never fires.
        """
        result = _translate_tools(
            ["Write", "Edit", "MultiEdit", "Bash", "Read", "Grep"], AIToolID.ANTIGRAVITY_CLI
        )
        assert result == [
            "write_to_file",
            "replace_file_content",
            "multi_replace_file_content",
            "run_command",
            "view_file",
            "grep_search",
        ]

    def test_copilot_has_no_tool_map_entry(self) -> None:
        """CopilotHooksWriter drops the ``tools`` scope and emits a manual-fix
        note instead of calling ``_translate_tools``, so ``_TOOL_NAME_MAP`` has
        no Copilot entry — the translator is an identity passthrough for it.
        """
        result = _translate_tools(["Edit", "Write", "Bash"], AIToolID.COPILOT)
        assert result == ["Edit", "Write", "Bash"]

    def test_claude_no_translation(self) -> None:
        assert _translate_tools(["Edit", "Bash"], AIToolID.CLAUDE) == ["Edit", "Bash"]


class TestHookEntryTimeout:
    def test_rejects_non_positive_timeout(self) -> None:
        """Cursor rejects a non-positive timeout and then loads NO hooks at all.

        Catching it here beats writing a config that silently disables the guard.
        """
        import pytest
        from pydantic import ValidationError

        for bad in (0, -1):
            with pytest.raises(ValidationError):
                HookEntry(event="pre_tool_use", command="guard", timeout=bad)

    def test_none_means_tool_default(self) -> None:
        assert HookEntry(event="pre_tool_use", command="guard").timeout is None


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

    def test_plain_alternation_existing_still_broadens_on_empty_desired(self) -> None:
        # A plain alternation is a tool list crossby understands, so empty desired
        # (all tools) still legitimately broadens it to ".*".
        assert _widen_matcher("Edit|Write", []) == ".*"

    def test_non_plain_existing_not_broadened_on_empty_desired(self) -> None:
        # A concrete regex crossby can't represent as a tool list reads back as
        # tools=[] (desired ".*"); broadening it to ".*" would silently widen a
        # deliberately-narrow guard, so it is kept unchanged instead.
        assert _widen_matcher("Write.*", []) == "Write.*"
        assert _widen_matcher("(Write|Shell)", []) == "(Write|Shell)"


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
        # Cursor's loader requires BOTH keys — without a top-level `hooks`
        # object it reports "missing 'hooks' property" and loads nothing.
        assert data["version"] == 1
        pre = data["hooks"]["preToolUse"]
        assert len(pre) == 1
        assert pre[0]["type"] == "command"
        assert pre[0]["command"] == "python3 ./scripts/guard.py"
        # Scope is a matcher regex; Cursor's schema has no `tools` array, and
        # Cursor has no Edit tool (Edit collapses into Write).
        assert pre[0]["matcher"] == "Write"
        assert "tools" not in pre[0]
        assert "event" not in pre[0]

    def test_bash_translated_to_shell(self, tmp_path: Path) -> None:
        hook = HookEntry(event="pre_tool_use", command="echo hi", tools=["Bash"])
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["hooks"]["preToolUse"][0]["matcher"] == "Shell"

    def test_edit_and_multiedit_collapse_to_write(self, tmp_path: Path) -> None:
        """Cursor has no Edit tool — Edit/MultiEdit both map to Write, deduped."""
        hook = HookEntry(
            event="pre_tool_use", command="guard", tools=["Edit", "MultiEdit", "Write"]
        )
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["hooks"]["preToolUse"][0]["matcher"] == "Write"

    def test_fail_closed_emitted(self, tmp_path: Path) -> None:
        """A fail-closed hook writes ``failClosed: true`` (Cursor is fail-open by default)."""
        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Edit"], fail_closed=True)
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["hooks"]["preToolUse"][0]["failClosed"] is True

    def test_fail_closed_absent_by_default(self, tmp_path: Path) -> None:
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert "failClosed" not in data["hooks"]["preToolUse"][0]

    def test_timeout_emitted_in_seconds(self, tmp_path: Path) -> None:
        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Write"], timeout=15)
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["hooks"]["preToolUse"][0]["timeout"] == 15

    def test_existing_timeout_not_overwritten(self, tmp_path: Path) -> None:
        """A hand-tuned timeout survives re-sync; crossby only fills in a missing one."""
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {
            "version": 1,
            "hooks": {
                "preToolUse": [
                    {"type": "command", "command": "guard", "matcher": "Write", "timeout": 5}
                ]
            },
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Write"], timeout=99)
        self.writer.sync(_cfg(hook), tmp_path)

        data = _read_json(path)
        assert data["hooks"]["preToolUse"][0]["timeout"] == 5

    def test_fail_closed_added_to_existing_entry(self, tmp_path: Path) -> None:
        """Re-syncing a fail-closed hook hardens a pre-existing fail-open entry."""
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {
            "version": 1,
            "hooks": {"preToolUse": [{"type": "command", "command": "guard", "matcher": "Write"}]},
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Edit"], fail_closed=True)
        result = self.writer.sync(_cfg(hook), tmp_path)

        assert result.action == "updated"
        data = _read_json(path)
        # same command → merged, not duplicated
        assert len(data["hooks"]["preToolUse"]) == 1
        assert data["hooks"]["preToolUse"][0]["failClosed"] is True

    def test_merges_into_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {
            "version": 1,
            "hooks": {"preToolUse": [{"type": "command", "command": "echo existing"}]},
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        data = _read_json(path)
        assert len(data["hooks"]["preToolUse"]) == 2

    def test_missing_matcher_not_narrowed(self, tmp_path: Path) -> None:
        """A missing ``matcher`` means "all tools" — must not be narrowed.

        Narrowing an unscoped guard to a subset would silently shrink its
        coverage on every re-sync.
        """
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        existing = {
            "version": 1,
            "hooks": {"preToolUse": [{"type": "command", "command": "python3 ./scripts/guard.py"}]},
        }
        path.write_text(json.dumps(existing), encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        # No-op: matched by command, no existing matcher means all → leave alone.
        assert result.action == "skipped"
        data = _read_json(path)
        assert "matcher" not in data["hooks"]["preToolUse"][0]

    def test_shell_hook_fans_out_to_before_shell_execution(self, tmp_path: Path) -> None:
        """A shell-scoped pre-tool hook also registers on Cursor's shell event.

        Cursor is the only tool with a dedicated ``beforeShellExecution``. The
        fan-out lets a caller register once against ``pre_tool_use`` + ``Bash``
        and get shell coverage on every tool.
        """
        hook = HookEntry(
            event="pre_tool_use", command="guard", tools=["Write", "Bash"], fail_closed=True
        )
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")["hooks"]

        assert data["preToolUse"][0]["matcher"] == "Write|Shell"
        shell = data["beforeShellExecution"]
        assert len(shell) == 1
        assert shell[0]["command"] == "guard"
        assert shell[0]["failClosed"] is True
        # beforeShellExecution matches the COMMAND STRING, not a tool name, so a
        # tool matcher there would match nothing.
        assert "matcher" not in shell[0]

    def test_no_fan_out_without_shell_tool(self, tmp_path: Path) -> None:
        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Write"])
        self.writer.sync(_cfg(hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")["hooks"]
        assert "beforeShellExecution" not in data

    def test_fan_out_round_trips_to_single_entry(self, tmp_path: Path) -> None:
        """The fanned-out pair reads back as ONE hook, not two.

        Without dedup, every read → write cycle would grow a spurious extra
        HookEntry.
        """
        from crossby.sync.readers import _read_cursor_hooks

        hook = HookEntry(event="pre_tool_use", command="guard", tools=["Write", "Bash"])
        self.writer.sync(_cfg(hook), tmp_path)

        entries = _read_cursor_hooks(tmp_path)
        assert len(entries) == 1
        assert entries[0].event == "pre_tool_use"
        assert entries[0].command == "guard"
        # The scoped half wins — the unscoped shell twin must not erase it.
        assert entries[0].tools == ["Write", "Bash"]

    def test_hand_authored_regex_matcher_is_not_read_as_tools(self, tmp_path: Path) -> None:
        """A real regex matcher must not be split into bogus tool names.

        ``matcher`` is a regex, so splitting on ``|`` unconditionally turns
        ``(Write|Shell)`` into ``(Write`` / ``Shell)``. Those fragments would
        become ``HookEntry.tools`` and get unioned back into the matcher on the
        next write, corrupting the user's file a little more each sync.
        """
        from crossby.sync.readers import _read_cursor_hooks

        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {
                        "preToolUse": [
                            {"command": "wildcard", "matcher": "Write.*"},
                            {"command": "grouped", "matcher": "(Write|Shell)"},
                            {"command": "plain", "matcher": "Write|Shell"},
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        by_command = {e.command: e.tools for e in _read_cursor_hooks(tmp_path)}
        assert by_command["wildcard"] == []
        assert by_command["grouped"] == []
        # A bare alternation still round-trips: Cursor's `Shell` maps back to
        # the canonical `Bash`.
        assert by_command["plain"] == ["Write", "Bash"]

    def test_migrates_legacy_flat_shape(self, tmp_path: Path) -> None:
        """Pre-0.13 crossby wrote a shape Cursor rejects outright; repair it.

        The old top-level layout had no ``hooks`` wrapper, so Cursor discarded
        the whole file and every hook crossby wrote was inert.
        """
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir()
        legacy = {
            "preToolUse": [
                {"event": "preToolUse", "command": "old-guard", "tools": ["Edit", "Write"]}
            ],
            "beforeSubmitPrompt": [{"event": "beforeSubmitPrompt", "command": "ctx"}],
            "someOtherSetting": {"keep": True},
        }
        path.write_text(json.dumps(legacy), encoding="utf-8")

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "updated"

        data = _read_json(path)
        assert data["version"] == 1
        assert "preToolUse" not in data, "legacy key must be lifted into hooks"
        assert "beforeSubmitPrompt" not in data
        # Unrelated top-level keys are left alone.
        assert data["someOtherSetting"] == {"keep": True}

        commands = [e["command"] for e in data["hooks"]["preToolUse"]]
        assert "old-guard" in commands, "existing hooks must survive migration"
        assert "python3 ./scripts/guard.py" in commands
        migrated = next(e for e in data["hooks"]["preToolUse"] if e["command"] == "old-guard")
        # Legacy `tools` are re-translated on the way in: pre-0.13 files stored
        # `Edit`, which Cursor has no tool for, so carrying it into the matcher
        # would leave a dead alternative that can never match.
        assert migrated["matcher"] == "Write"
        assert "tools" not in migrated
        assert "event" not in migrated
        assert data["hooks"]["beforeSubmitPrompt"][0]["command"] == "ctx"

    def test_migration_does_not_mutate_its_input(self) -> None:
        """The migrator returns a new config instead of editing the caller's.

        It used to `pop()` legacy keys off the dict it was handed, so an early
        return would have left a half-migrated config behind.
        """
        from crossby.sync.hooks import _migrate_legacy_cursor_config

        legacy: dict[str, object] = {
            "preToolUse": [{"event": "preToolUse", "command": "old-guard", "tools": ["Write"]}],
            "someOtherSetting": {"keep": True},
        }
        before = copy.deepcopy(legacy)

        migrated, changed = _migrate_legacy_cursor_config(legacy)

        assert changed is True
        assert legacy == before, "input dict must be untouched"
        assert "preToolUse" not in migrated
        assert migrated["hooks"]["preToolUse"][0]["matcher"] == "Write"

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

    def test_user_prompt_submit_maps_to_before_submit_prompt(self, tmp_path: Path) -> None:
        hook = HookEntry(event="user_prompt_submit", command="python3 ./scripts/context.py")
        result = self.writer.sync(_cfg(hook), tmp_path)
        assert result.action == "created"
        assert result.message is None
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["hooks"]["beforeSubmitPrompt"] == [
            {"type": "command", "command": "python3 ./scripts/context.py"}
        ]

    def test_session_start_supported(self, tmp_path: Path) -> None:
        """Cursor does fire sessionStart (verified against cursor-agent)."""
        hook = HookEntry(event="session_start", command="python3 ./scripts/context.py")
        result = self.writer.sync(_cfg(hook), tmp_path)
        assert result.action == "created"
        assert result.message is None
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["hooks"]["sessionStart"][0]["command"] == "python3 ./scripts/context.py"

    def test_user_prompt_submit_tools_filter_dropped_with_note(self, tmp_path: Path) -> None:
        hook = HookEntry(
            event="user_prompt_submit", command="python3 ./scripts/context.py", tools=["Edit"]
        )
        result = self.writer.sync(_cfg(hook), tmp_path)
        assert result.message is not None
        assert "manual_fix" in result.message
        assert "hooks.user_prompt_submit.tools" in result.message
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert "matcher" not in data["hooks"]["beforeSubmitPrompt"][0]


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

    def test_stop_maps_to_agent_stop(self, tmp_path: Path) -> None:
        """Copilot names the turn-complete event `agentStop`, not `stop`."""
        hook = HookEntry(event="stop", command="wade-hook stop", description="stop guard")
        result = self.writer.sync(_cfg(hook), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".github" / "hooks" / "hooks.json")
        assert data["hooks"]["agentStop"][0]["bash"] == "wade-hook stop"
        assert "stop" not in data["hooks"]

    def test_session_start_and_post_tool_use_supported(self, tmp_path: Path) -> None:
        hooks = [
            HookEntry(event="session_start", command="ctx"),
            HookEntry(event="post_tool_use", command="audit"),
        ]
        result = self.writer.sync(_cfg(*hooks), tmp_path)
        assert result.action == "created"
        data = _read_json(tmp_path / ".github" / "hooks" / "hooks.json")
        assert data["hooks"]["sessionStart"][0]["bash"] == "ctx"
        assert data["hooks"]["postToolUse"][0]["bash"] == "audit"

    def test_timeout_uses_timeout_sec_key(self, tmp_path: Path) -> None:
        """Copilot spells it `timeoutSec`; every other tool uses `timeout`."""
        hook = HookEntry(event="pre_tool_use", command="guard", timeout=45)
        self.writer.sync(_cfg(hook), tmp_path)
        entry = _read_json(tmp_path / ".github" / "hooks" / "hooks.json")["hooks"]["preToolUse"][0]
        assert entry["timeoutSec"] == 45
        assert "timeout" not in entry

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
# Reader data-type hardening — malformed JSON must skip, not crash
# ---------------------------------------------------------------------------


def _write_cursor_hooks_raw(root: Path, entries: list[dict]) -> None:
    path = root / ".cursor" / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "hooks": {"preToolUse": entries}}),
        encoding="utf-8",
    )


def _write_copilot_hooks_raw(root: Path, entries: list[dict]) -> None:
    path = root / ".github" / "hooks" / "hooks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "hooks": {"preToolUse": entries}}),
        encoding="utf-8",
    )


class TestReadCursorHooksHardening:
    """HookEntry is lax but not coercive, so the reader must isinstance-filter."""

    def test_list_command_is_skipped(self, tmp_path: Path) -> None:
        # A non-str command is unhashable for the (event, command) dedupe key
        # AND rejected by HookEntry — skip it before either fails.
        _write_cursor_hooks_raw(
            tmp_path,
            [
                {"command": ["a", "b"], "matcher": "Write"},
                {"command": "ok", "matcher": "Write"},
            ],
        )
        entries = _read_cursor_hooks(tmp_path)
        assert [e.command for e in entries] == ["ok"]

    def test_all_invalid_tools_scope_skips_entry(self, tmp_path: Path) -> None:
        # tools=[1, 2] was a scoping intent; emitting it as unscoped [] would
        # broaden the hook to ALL tools, so the whole entry is dropped instead.
        _write_cursor_hooks_raw(
            tmp_path,
            [
                {"command": "bad", "tools": [1, 2]},
                {"command": "good", "tools": ["write"]},
            ],
        )
        by_command = {e.command: e.tools for e in _read_cursor_hooks(tmp_path)}
        assert "bad" not in by_command
        assert by_command["good"] == ["Write"]

    def test_mixed_valid_invalid_tools_keeps_valid(self, tmp_path: Path) -> None:
        _write_cursor_hooks_raw(tmp_path, [{"command": "guard", "tools": [1, "write", 3]}])
        by_command = {e.command: e.tools for e in _read_cursor_hooks(tmp_path)}
        assert by_command["guard"] == ["Write"]

    def test_absent_scope_maps_to_all_tools(self, tmp_path: Path) -> None:
        # No matcher and no tools key → genuinely unscoped [] (all tools).
        _write_cursor_hooks_raw(tmp_path, [{"command": "guard"}])
        entries = _read_cursor_hooks(tmp_path)
        assert len(entries) == 1
        assert entries[0].tools == []

    def test_empty_tools_list_maps_to_all_tools(self, tmp_path: Path) -> None:
        # An explicit empty list carries no scoping intent → unscoped.
        _write_cursor_hooks_raw(tmp_path, [{"command": "guard", "tools": []}])
        entries = _read_cursor_hooks(tmp_path)
        assert entries[0].tools == []

    def test_non_list_tools_scope_skips_entry(self, tmp_path: Path) -> None:
        _write_cursor_hooks_raw(
            tmp_path,
            [
                {"command": "bad", "tools": "Write"},
                {"command": "good", "matcher": "Write"},
            ],
        )
        by_command = {e.command: e.tools for e in _read_cursor_hooks(tmp_path)}
        assert "bad" not in by_command
        assert by_command["good"] == ["Write"]

    def test_non_string_matcher_skips_entry(self, tmp_path: Path) -> None:
        # A wrongly-typed matcher (e.g. matcher: 123) was a scoping intent —
        # _matcher_tools silently treats any non-str input as "no plain
        # alternation" ([]), which would otherwise fall through to the
        # absent-scope case and broaden the hook to all tools. Skip instead.
        _write_cursor_hooks_raw(
            tmp_path,
            [
                {"command": "bad", "matcher": 123},
                {"command": "good", "matcher": "Write"},
            ],
        )
        by_command = {e.command: e.tools for e in _read_cursor_hooks(tmp_path)}
        assert "bad" not in by_command
        assert by_command["good"] == ["Write"]

    def test_distinct_scopes_for_same_command_are_unioned(self, tmp_path: Path) -> None:
        # Two hand-authored guards on the same command, scoped to different tools,
        # must not collapse to one — the local dedupe unions their scopes so
        # neither guard is silently dropped. (`Shell` maps back to `Bash`.)
        _write_cursor_hooks_raw(
            tmp_path,
            [
                {"command": "guard", "matcher": "Write"},
                {"command": "guard", "matcher": "Shell"},
            ],
        )
        entries = _read_cursor_hooks(tmp_path)
        assert len(entries) == 1
        assert entries[0].tools == ["Write", "Bash"]

    def test_genuinely_unscoped_duplicate_is_not_narrowed_by_scoped(
        self, tmp_path: Path
    ) -> None:
        # A hand-authored unscoped preToolUse entry ("all tools") sharing a
        # command with a scoped duplicate must dominate — only the
        # beforeShellExecution fan-out mirror is allowed to yield to a scoped
        # duplicate; a real unscoped declaration must never be silently
        # narrowed.
        _write_cursor_hooks_raw(
            tmp_path,
            [
                {"command": "guard"},
                {"command": "guard", "matcher": "Write"},
            ],
        )
        entries = _read_cursor_hooks(tmp_path)
        assert len(entries) == 1
        assert entries[0].tools == []

    def test_genuinely_unscoped_duplicate_dominates_regardless_of_order(
        self, tmp_path: Path
    ) -> None:
        # Same as above with the scoped entry listed first — the merge must
        # not depend on which half of the duplicate pair is read first.
        _write_cursor_hooks_raw(
            tmp_path,
            [
                {"command": "guard", "matcher": "Write"},
                {"command": "guard"},
            ],
        )
        entries = _read_cursor_hooks(tmp_path)
        assert len(entries) == 1
        assert entries[0].tools == []


class TestReadCopilotHooksHardening:
    def test_non_string_bash_is_skipped(self, tmp_path: Path) -> None:
        _write_copilot_hooks_raw(
            tmp_path,
            [
                {"bash": ["a", "b"], "comment": "bad"},
                {"bash": 5, "comment": "also bad"},
                {"bash": "echo ok", "comment": "good"},
            ],
        )
        entries = _read_copilot_hooks(tmp_path)
        assert [e.command for e in entries] == ["echo ok"]

    def test_non_string_comment_coerced_to_empty(self, tmp_path: Path) -> None:
        _write_copilot_hooks_raw(tmp_path, [{"bash": "echo hi", "comment": {"nested": True}}])
        entries = _read_copilot_hooks(tmp_path)
        assert len(entries) == 1
        assert entries[0].description == ""


# ---------------------------------------------------------------------------
# fail_closed round-trip — survives local dedupe AND cross-tool merge
# ---------------------------------------------------------------------------


class TestFailClosedRoundTrip:
    def _write_cursor_pair(self, root: Path, entries: list[dict]) -> None:
        path = root / ".cursor" / "hooks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "hooks": {"preToolUse": entries}}),
            encoding="utf-8",
        )

    def _write_cursor_fanout_pair(
        self, root: Path, scoped_entry: dict, shell_entry: dict
    ) -> None:
        path = root / ".cursor" / "hooks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {"preToolUse": [scoped_entry], "beforeShellExecution": [shell_entry]},
                }
            ),
            encoding="utf-8",
        )

    def test_read_single_fail_closed(self, tmp_path: Path) -> None:
        self._write_cursor_pair(
            tmp_path, [{"command": "guard", "matcher": "Write", "failClosed": True}]
        )
        entries = _read_cursor_hooks(tmp_path)
        assert entries[0].fail_closed is True

    def test_non_bool_fail_closed_defaults_false(self, tmp_path: Path) -> None:
        # A malformed (non-bool) failClosed must not reach HookEntry as-is.
        self._write_cursor_pair(
            tmp_path, [{"command": "guard", "matcher": "Write", "failClosed": "yes"}]
        )
        entries = _read_cursor_hooks(tmp_path)
        assert entries[0].fail_closed is False

    def test_local_dedupe_true_wins(self, tmp_path: Path) -> None:
        # The real fan-out shape: a scoped preToolUse half and its unscoped
        # beforeShellExecution mirror. The True must win, and the scoped half
        # must not be erased by the unscoped fan-out artifact.
        self._write_cursor_fanout_pair(
            tmp_path,
            {"command": "guard", "matcher": "Write", "failClosed": True},
            {"command": "guard", "failClosed": False},
        )
        entries = _read_cursor_hooks(tmp_path)
        assert len(entries) == 1
        assert entries[0].fail_closed is True
        assert entries[0].tools == ["Write"]

    def test_cross_tool_merge_true_wins(self, tmp_path: Path) -> None:
        # Cursor sets failClosed: true; Claude defines the same (event, command)
        # with no such flag. The True must survive the cross-tool merge.
        self._write_cursor_pair(
            tmp_path, [{"command": "python3 guard.py", "matcher": "Write", "failClosed": True}]
        )
        _write_claude_hook(tmp_path, "python3 guard.py", "Write")
        hooks = discover_hooks(tmp_path)
        assert len(hooks) == 1
        assert hooks[0].fail_closed is True


# ---------------------------------------------------------------------------
# reader_available — CLI-layer missing-reader diagnostics
# ---------------------------------------------------------------------------


class TestReaderAvailable:
    def test_permissions_covers_claude_and_cursor_only(self) -> None:
        from crossby.sync.base import SyncConcern

        assert reader_available(AIToolID.CLAUDE, SyncConcern.PERMISSIONS) is True
        assert reader_available(AIToolID.CURSOR, SyncConcern.PERMISSIONS) is True
        assert reader_available(AIToolID.CODEX, SyncConcern.PERMISSIONS) is False
        assert reader_available(AIToolID.COPILOT, SyncConcern.PERMISSIONS) is False
        assert reader_available(AIToolID.ANTIGRAVITY_CLI, SyncConcern.PERMISSIONS) is False

    def test_hooks_covers_all_five_tools(self) -> None:
        from crossby.sync.base import SyncConcern

        for tool in (
            AIToolID.CLAUDE,
            AIToolID.CURSOR,
            AIToolID.COPILOT,
            AIToolID.CODEX,
            AIToolID.ANTIGRAVITY_CLI,
        ):
            assert reader_available(tool, SyncConcern.HOOKS) is True

    def test_rules_agents_skills_mcp_cover_only_their_source_tools(self) -> None:
        from crossby.sync.base import SyncConcern

        # CODEX has a source mapping for all four (AGENTS.md, .codex/agents,
        # .agents/skills, .codex/config.toml) — reader_available is True.
        assert reader_available(AIToolID.CODEX, SyncConcern.RULES) is True
        assert reader_available(AIToolID.CODEX, SyncConcern.AGENTS) is True
        assert reader_available(AIToolID.CODEX, SyncConcern.SKILLS) is True
        assert reader_available(AIToolID.CODEX, SyncConcern.MCP) is True

        # VSCODE/OPENCODE/ANTIGRAVITY (distinct from ANTIGRAVITY_CLI) have no
        # source mapping in detect_rules/detect_agents/detect_skills or
        # discover_mcp_servers — no code path can ever populate them, so a
        # `--from` sync must be flagged unavailable rather than silently
        # producing nothing.
        for tool in (AIToolID.VSCODE, AIToolID.OPENCODE, AIToolID.ANTIGRAVITY):
            assert reader_available(tool, SyncConcern.RULES) is False
            assert reader_available(tool, SyncConcern.AGENTS) is False
            assert reader_available(tool, SyncConcern.SKILLS) is False
            assert reader_available(tool, SyncConcern.MCP) is False

    def test_plugins_has_no_per_tool_dimension_always_available(self) -> None:
        from crossby.sync.base import SyncConcern

        # discover_plugins() doesn't take a --from tool at all, so there is no
        # "unsupported source tool" case to gate.
        for tool in AIToolID:
            assert reader_available(tool, SyncConcern.PLUGINS) is True


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

    def test_enables_both_codex_hooks_feature_flags(self, tmp_path: Path) -> None:
        """Writes the canonical `hooks` key AND the deprecated `codex_hooks` alias.

        `hooks` is stable and on by default since Codex 0.146.0, so this is
        defensive; the alias keeps a project pinned to an older Codex working.
        Unknown feature keys are inert, so writing both is safe everywhere.
        """
        import tomllib

        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        # Flags written automatically → no manual-fix note on the happy path.
        assert result.message is None or "features.hooks" not in result.message

        config = tmp_path / ".codex" / "config.toml"
        assert config.is_file()
        parsed = tomllib.loads(config.read_text(encoding="utf-8"))
        assert parsed["features"]["hooks"] is True
        assert parsed["features"]["codex_hooks"] is True

    def test_adds_missing_canonical_key_to_legacy_config(self, tmp_path: Path) -> None:
        """A config written by an older crossby only has the alias — top it up.

        The splicer must preserve surrounding comments and key order while
        adding just the missing key.
        """
        import tomllib

        config = tmp_path / ".codex" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text(
            "# my codex config\nmodel = 'gpt-5'\n\n"
            "[features]\n# enable hooks\ncodex_hooks = true\n",
            encoding="utf-8",
        )

        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)

        text = config.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert parsed["features"]["hooks"] is True
        assert parsed["features"]["codex_hooks"] is True
        assert parsed["model"] == "gpt-5"
        # Comments survive the splice.
        assert "# my codex config" in text
        assert "# enable hooks" in text

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
        original = "[features]\nhooks = true\ncodex_hooks = true\n"
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
        assert "features.hooks" in result.message
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
            # Cursor supports everything except notification.
            ("CursorHooksWriter", "notification"),
            # Copilot has no prompt-submit hook wired up (its native event is
            # `userPromptSubmitted`, which crossby does not register yet).
            ("CopilotHooksWriter", "user_prompt_submit"),
            ("CopilotHooksWriter", "notification"),
            # Codex supports everything except notification.
            ("CodexHooksWriter", "notification"),
            # Antigravity CLI supports pre_tool_use + post_tool_use + stop.
            ("AntigravityCLIHooksWriter", "session_start"),
            ("AntigravityCLIHooksWriter", "user_prompt_submit"),
            ("AntigravityCLIHooksWriter", "notification"),
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


# ---------------------------------------------------------------------------
# AntigravityCLIHooksWriter
# ---------------------------------------------------------------------------


class TestAntigravityCLIHooksWriter:
    """AntigravityCLIHooksWriter — agy's container-wrapped `.agents/hooks.json`."""

    def setup_method(self) -> None:
        from crossby.sync.hooks import AntigravityCLIHooksWriter

        self.writer = AntigravityCLIHooksWriter()

    def _path(self, root: Path) -> Path:
        return root / ".agents" / "hooks.json"

    def test_pre_tool_use_is_matcher_wrapped(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "created"
        data = _read_json(self._path(tmp_path))
        # One container keyed by the description slug, holding a PreToolUse entry.
        container = data["plan-write-guard"]
        entry = container["PreToolUse"][0]
        # Translated to agy's native tool-call names — a matcher built from
        # crossby's canonical names would match nothing agy ever emits.
        assert entry["matcher"] == "replace_file_content|write_to_file"
        assert entry["hooks"] == [{"type": "command", "command": "python3 ./scripts/guard.py"}]

    def test_post_tool_use_is_matcher_wrapped(self, tmp_path: Path) -> None:
        hook = HookEntry(
            event="post_tool_use",
            command="python3 ./scripts/audit.py",
            tools=["Edit"],
            description="audit",
        )
        result = self.writer.sync(_cfg(hook), tmp_path)
        assert result.action == "created"
        entry = _read_json(self._path(tmp_path))["audit"]["PostToolUse"][0]
        assert entry["matcher"] == "replace_file_content"
        assert entry["hooks"] == [{"type": "command", "command": "python3 ./scripts/audit.py"}]

    def test_dry_run_reports_created_without_writing(self, tmp_path: Path) -> None:
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path, dry_run=True)
        assert result.action == "created"
        assert not self._path(tmp_path).exists()

    def test_stop_handlers_are_direct_no_matcher(self, tmp_path: Path) -> None:
        stop_hook = HookEntry(
            event="stop", command="wade hook stop", tools=[], description="session complete"
        )
        result = self.writer.sync(_cfg(stop_hook), tmp_path)
        assert result.action == "created"
        data = _read_json(self._path(tmp_path))
        stop_entries = data["session-complete"]["Stop"]
        # Stop handlers sit directly under the event key — no matcher wrapper.
        assert stop_entries == [{"type": "command", "command": "wade hook stop"}]

    def test_stop_drops_matcher_with_note(self, tmp_path: Path) -> None:
        stop_with_tools = HookEntry(event="stop", command="wade hook stop", tools=["Bash"])
        result = self.writer.sync(_cfg(stop_with_tools), tmp_path)
        assert result.action == "created"
        data = _read_json(self._path(tmp_path))
        # No "matcher" key anywhere in the Stop entry.
        for container in data.values():
            for entry in container.get("Stop", []):
                assert "matcher" not in entry
        assert result.message is not None
        assert "hooks.stop.matcher" in result.message

    def test_drops_session_start_event(self, tmp_path: Path) -> None:
        hook = HookEntry(event="session_start", command="echo hi", tools=[])
        result = self.writer.sync(_cfg(hook), tmp_path)
        assert result.message is not None
        assert "hooks.session_start" in result.message
        # Nothing supported was kept, so no file is written.
        assert not self._path(tmp_path).exists()

    def test_resync_is_idempotent(self, tmp_path: Path) -> None:
        cfg = _cfg(GUARD_HOOK)
        self.writer.sync(cfg, tmp_path)
        first = _read_json(self._path(tmp_path))
        result = self.writer.sync(cfg, tmp_path)
        assert result.action == "skipped"
        assert _read_json(self._path(tmp_path)) == first

    def test_resync_widens_matcher_when_coverage_grows(self, tmp_path: Path) -> None:
        # A guard whose tool scope grows across syncs must widen its matcher, not
        # silently drop the newly-covered tools (upgrade-safe merge).
        narrow = HookEntry(
            event="pre_tool_use", command="guard", tools=["Edit"], description="grow guard"
        )
        wide = HookEntry(
            event="pre_tool_use",
            command="guard",
            tools=["Edit", "Write"],
            description="grow guard",
        )
        self.writer.sync(_cfg(narrow), tmp_path)
        first = _read_json(self._path(tmp_path))["grow-guard"]["PreToolUse"][0]
        assert first["matcher"] == "replace_file_content"

        result = self.writer.sync(_cfg(wide), tmp_path)
        assert result.action == "updated"
        entry = _read_json(self._path(tmp_path))["grow-guard"]["PreToolUse"][0]
        assert set(entry["matcher"].split("|")) == {"replace_file_content", "write_to_file"}
        # Still one entry — widened in place, not duplicated.
        assert len(_read_json(self._path(tmp_path))["grow-guard"]["PreToolUse"]) == 1

    def test_resync_never_narrows_matcher(self, tmp_path: Path) -> None:
        # A later sync with a subset must NOT shrink existing coverage.
        wide = HookEntry(
            event="pre_tool_use",
            command="guard",
            tools=["Edit", "Write"],
            description="grow guard",
        )
        narrow = HookEntry(
            event="pre_tool_use", command="guard", tools=["Edit"], description="grow guard"
        )
        self.writer.sync(_cfg(wide), tmp_path)
        result = self.writer.sync(_cfg(narrow), tmp_path)
        assert result.action == "skipped"
        entry = _read_json(self._path(tmp_path))["grow-guard"]["PreToolUse"][0]
        assert set(entry["matcher"].split("|")) == {"replace_file_content", "write_to_file"}

    def test_preserves_hand_authored_container(self, tmp_path: Path) -> None:
        path = self._path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"my-own": {"PreToolUse": [{"matcher": ".*", "hooks": []}]}}),
            encoding="utf-8",
        )
        self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        data = _read_json(path)
        # The user's container survives; crossby's is added alongside it.
        assert "my-own" in data
        assert "plan-write-guard" in data

    def test_dedups_command_across_containers(self, tmp_path: Path) -> None:
        path = self._path(tmp_path)
        path.parent.mkdir(parents=True)
        # Same command already present under a differently-named container.
        path.write_text(
            json.dumps(
                {
                    "hand-rolled": {
                        "PreToolUse": [
                            {
                                "matcher": ".*",
                                "hooks": [{"type": "command", "command": GUARD_HOOK.command}],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        result = self.writer.sync(_cfg(GUARD_HOOK), tmp_path)
        assert result.action == "skipped"
        data = _read_json(path)
        assert "plan-write-guard" not in data


# ---------------------------------------------------------------------------
# Revocation — driven ONLY by SyncData.hooks_remove (never inferred)
# ---------------------------------------------------------------------------


def _remove(*pairs: tuple[str, str]) -> SyncData:
    return SyncData(hooks=[], hooks_remove=list(pairs))


class TestClaudeHooksRemoval:
    def _path(self, root: Path) -> Path:
        return root / ".claude" / "settings.json"

    def test_removes_named_hook_and_drops_empty_event(self, tmp_path: Path) -> None:
        writer = ClaudeHooksWriter()
        writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = writer.sync(_remove(("pre_tool_use", GUARD_HOOK.command)), tmp_path)
        assert result.action == "updated"
        assert result.revoked == 1
        assert result.added == 0
        # Removing the last hook for the event drops the event key, leaving a
        # valid (parseable, empty) structure — not malformed scaffolding.
        assert "PreToolUse" not in _read_json(self._path(tmp_path)).get("hooks", {})

    def test_removal_leaves_sibling_hooks(self, tmp_path: Path) -> None:
        writer = ClaudeHooksWriter()
        writer.sync(_cfg(GUARD_HOOK, BARE_HOOK), tmp_path)
        writer.sync(_remove(("pre_tool_use", GUARD_HOOK.command)), tmp_path)
        commands = {
            inner["command"]
            for entry in _read_json(self._path(tmp_path))["hooks"]["PreToolUse"]
            for inner in entry["hooks"]
        }
        assert GUARD_HOOK.command not in commands
        assert BARE_HOOK.command in commands

    def test_removing_an_absent_hook_is_skipped(self, tmp_path: Path) -> None:
        writer = ClaudeHooksWriter()
        writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = writer.sync(_remove(("pre_tool_use", "never-written")), tmp_path)
        assert result.action == "skipped"
        assert result.revoked == 0

    def test_empty_data_and_empty_removal_is_skipped(self, tmp_path: Path) -> None:
        result = ClaudeHooksWriter().sync(SyncData(), tmp_path)
        assert result.action == "skipped"
        assert result.message == "no hooks config"

    def test_add_and_remove_in_one_run(self, tmp_path: Path) -> None:
        writer = ClaudeHooksWriter()
        writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = writer.sync(
            SyncData(hooks=[BARE_HOOK], hooks_remove=[("pre_tool_use", GUARD_HOOK.command)]),
            tmp_path,
        )
        assert result.added == 1
        assert result.revoked == 1

    def test_dry_run_reports_revocation_but_writes_nothing(self, tmp_path: Path) -> None:
        writer = ClaudeHooksWriter()
        writer.sync(_cfg(GUARD_HOOK), tmp_path)
        before = self._path(tmp_path).read_text(encoding="utf-8")
        result = writer.sync(_remove(("pre_tool_use", GUARD_HOOK.command)), tmp_path, dry_run=True)
        # The intended revocation is reported...
        assert result.revoked == 1
        # ...but the file on disk is untouched.
        assert self._path(tmp_path).read_text(encoding="utf-8") == before


class TestClaudeMatcherNarrowing:
    """A crossby-owned matcher narrows on scope shrink; a human's never does."""

    def _entry(self, root: Path) -> dict:
        data = _read_json(root / ".claude" / "settings.json")
        return data["hooks"]["PreToolUse"][0]

    def test_owned_matcher_narrows(self, tmp_path: Path) -> None:
        writer = ClaudeHooksWriter()
        wide = HookEntry(event="pre_tool_use", command="guard", tools=["Edit", "Write"])
        writer.sync(_cfg(wide), tmp_path)
        assert set(self._entry(tmp_path)["matcher"].split("|")) == {"Edit", "Write"}

        narrow = HookEntry(event="pre_tool_use", command="guard", tools=["Edit"])
        result = writer.sync(
            SyncData(hooks=[narrow], hooks_owned=frozenset({("pre_tool_use", "guard")})),
            tmp_path,
        )
        assert result.action == "updated"
        assert self._entry(tmp_path)["matcher"] == "Edit"

    def test_unowned_matcher_is_never_narrowed(self, tmp_path: Path) -> None:
        writer = ClaudeHooksWriter()
        wide = HookEntry(event="pre_tool_use", command="guard", tools=["Edit", "Write"])
        writer.sync(_cfg(wide), tmp_path)
        narrow = HookEntry(event="pre_tool_use", command="guard", tools=["Edit"])
        # hooks_owned empty → crossby doesn't own it → widen-only (no change).
        result = writer.sync(SyncData(hooks=[narrow]), tmp_path)
        assert result.action == "skipped"
        assert set(self._entry(tmp_path)["matcher"].split("|")) == {"Edit", "Write"}


class TestOtherWritersRemoval:
    """Each native hooks writer removes the named (event, command) entry."""

    def test_cursor_removes_and_drops_shell_fanout(self, tmp_path: Path) -> None:
        writer = CursorHooksWriter()
        shell_hook = HookEntry(event="pre_tool_use", command="guard", tools=["Bash"])
        writer.sync(_cfg(shell_hook), tmp_path)
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        assert data["hooks"]["preToolUse"]
        assert data["hooks"]["beforeShellExecution"]  # fanned out

        result = writer.sync(_remove(("pre_tool_use", "guard")), tmp_path)
        assert result.revoked == 1
        data = _read_json(tmp_path / ".cursor" / "hooks.json")
        # Both the primary entry and its shell mirror are gone.
        assert "preToolUse" not in data["hooks"]
        assert "beforeShellExecution" not in data["hooks"]

    def test_copilot_removes_named_hook(self, tmp_path: Path) -> None:
        writer = CopilotHooksWriter()
        writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = writer.sync(_remove(("pre_tool_use", GUARD_HOOK.command)), tmp_path)
        assert result.revoked == 1
        data = _read_json(tmp_path / ".github" / "hooks" / "hooks.json")
        assert "preToolUse" not in data["hooks"]

    def test_codex_removes_named_hook(self, tmp_path: Path) -> None:
        from crossby.sync.hooks import CodexHooksWriter

        writer = CodexHooksWriter()
        writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = writer.sync(_remove(("pre_tool_use", GUARD_HOOK.command)), tmp_path)
        assert result.revoked == 1
        data = _read_json(tmp_path / ".codex" / "hooks.json")
        assert "PreToolUse" not in data.get("hooks", {})

    def test_antigravity_removes_named_hook(self, tmp_path: Path) -> None:
        from crossby.sync.hooks import AntigravityCLIHooksWriter

        writer = AntigravityCLIHooksWriter()
        writer.sync(_cfg(GUARD_HOOK), tmp_path)
        result = writer.sync(_remove(("pre_tool_use", GUARD_HOOK.command)), tmp_path)
        assert result.revoked == 1
        data = _read_json(tmp_path / ".agents" / "hooks.json")
        # The container's command is gone; an emptied container is pruned.
        commands = [
            handler.get("command")
            for container in data.values()
            if isinstance(container, dict)
            for entries in container.values()
            if isinstance(entries, list)
            for entry in entries
            for handler in [entry, *entry.get("hooks", [])]
            if isinstance(handler, dict)
        ]
        assert GUARD_HOOK.command not in commands


# ---------------------------------------------------------------------------
# Codex hooks reader — .codex/hooks.json is Claude-shaped
# ---------------------------------------------------------------------------


class TestCodexHooksReader:
    def _write(self, root: Path, hooks_section: dict) -> Path:
        path = root / ".codex" / "hooks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"hooks": hooks_section}), encoding="utf-8")
        return path

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _read_codex_hooks(tmp_path) == []

    def test_reads_written_codex_hooks(self, tmp_path: Path) -> None:
        """The writer's own output reads back as canonical entries."""
        from crossby.sync.hooks import CodexHooksWriter

        CodexHooksWriter().sync(_cfg(GUARD_HOOK, _stop_hook()), tmp_path)
        entries = {(e.event, e.command): e for e in _read_codex_hooks(tmp_path)}
        assert (GUARD_HOOK.event, GUARD_HOOK.command) in entries
        assert entries[(GUARD_HOOK.event, GUARD_HOOK.command)].tools == ["Edit", "Write"]
        stop = entries[("stop", "python3 ./scripts/post.py")]
        assert stop.tools == []

    def test_matcher_becomes_tools(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {
                "PreToolUse": [
                    {"matcher": "Edit|Bash", "hooks": [{"type": "command", "command": "g"}]}
                ]
            },
        )
        entry = _read_codex_hooks(tmp_path)[0]
        assert entry.tools == ["Edit", "Bash"]

    def test_catchall_and_missing_matcher_are_unscoped(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {
                "PreToolUse": [
                    {"matcher": ".*", "hooks": [{"type": "command", "command": "wild"}]},
                    {"hooks": [{"type": "command", "command": "nomatcher"}]},
                ]
            },
        )
        by_command = {e.command: e.tools for e in _read_codex_hooks(tmp_path)}
        assert by_command["wild"] == []
        assert by_command["nomatcher"] == []

    def test_malformed_handler_entries_are_skipped(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {
                "PreToolUse": [
                    {"matcher": "Edit", "hooks": ["not-a-dict", {"type": "command"}]},
                    {"matcher": "Edit", "hooks": [{"type": "command", "command": 123}]},
                    {"matcher": "Edit", "hooks": [{"type": "command", "command": ""}]},
                    {"matcher": "Edit", "hooks": [{"type": "command", "command": "ok"}]},
                ]
            },
        )
        entries = _read_codex_hooks(tmp_path)
        assert [e.command for e in entries] == ["ok"]

    def test_writer_reader_writer_round_trip_is_noop(self, tmp_path: Path) -> None:
        from crossby.sync.hooks import CodexHooksWriter

        writer = CodexHooksWriter()
        writer.sync(_cfg(GUARD_HOOK, _stop_hook()), tmp_path)
        path = tmp_path / ".codex" / "hooks.json"
        before = _read_json(path)

        read_back = _read_codex_hooks(tmp_path)
        writer.sync(SyncData(hooks=read_back), tmp_path)
        assert _read_json(path) == before

    def test_hand_authored_wildcard_matcher_round_trips_unchanged(self, tmp_path: Path) -> None:
        """A hand-authored `Write.*` re-syncs unchanged — never broadened to `.*`."""
        from crossby.sync.hooks import CodexHooksWriter

        path = self._write(
            tmp_path,
            {
                "PreToolUse": [
                    {"matcher": "Write.*", "hooks": [{"type": "command", "command": "g"}]}
                ]
            },
        )
        before = _read_json(path)

        read_back = _read_codex_hooks(tmp_path)
        CodexHooksWriter().sync(SyncData(hooks=read_back), tmp_path)

        after = _read_json(path)
        assert after == before
        assert after["hooks"]["PreToolUse"][0]["matcher"] == "Write.*"

    def test_exotic_matcher_syncs_cross_tool_as_catchall_not_inert(self, tmp_path: Path) -> None:
        """An exotic matcher reaches another tool as `.*` (fires), not a dead token.

        `.split("|")` would read `Write.*` as `tools=["Write.*"]`; synced on to
        Antigravity that token translates to nothing agy ever emits, so the guard
        installs but never fires. Reading it as unscoped re-scopes to `.*`, which
        still fires (fail-safe) — the point of the security guard.
        """
        from crossby.sync.hooks import AntigravityCLIHooksWriter

        self._write(
            tmp_path,
            {
                "PreToolUse": [
                    {"matcher": "Write.*", "hooks": [{"type": "command", "command": "guard"}]}
                ]
            },
        )
        read_back = _read_codex_hooks(tmp_path)
        assert read_back[0].tools == []  # unrepresentable regex → unscoped, not ["Write.*"]

        AntigravityCLIHooksWriter().sync(SyncData(hooks=read_back), tmp_path)
        agy = _read_json(tmp_path / ".agents" / "hooks.json")
        matchers = [e["matcher"] for c in agy.values() for e in c.get("PreToolUse", [])]
        assert matchers == [".*"]


# ---------------------------------------------------------------------------
# Antigravity CLI hooks reader — .agents/hooks.json is a container map
# ---------------------------------------------------------------------------


class TestAntigravityCLIHooksReader:
    def _write(self, root: Path, data: dict) -> Path:
        path = root / ".agents" / "hooks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _read_agy_hooks(tmp_path) == []

    def test_matcher_wrapped_and_bare_stop_both_parse(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {
                "guard": {
                    "PreToolUse": [
                        {"matcher": "write_to_file", "hooks": [{"type": "command", "command": "g"}]}
                    ]
                },
                "done": {"Stop": [{"type": "command", "command": "s"}]},
            },
        )
        by_command = {(e.event, e.command): e for e in _read_agy_hooks(tmp_path)}
        assert by_command[("pre_tool_use", "g")].tools == ["Write"]
        assert by_command[("stop", "s")].tools == []

    def test_native_matcher_recovers_canonical_tools(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {
                "guard": {
                    "PreToolUse": [
                        {
                            "matcher": "write_to_file|run_command",
                            "hooks": [{"type": "command", "command": "g"}],
                        }
                    ]
                }
            },
        )
        assert _read_agy_hooks(tmp_path)[0].tools == ["Write", "Bash"]

    def test_all_six_writer_emitted_natives_reverse_map(self, tmp_path: Path) -> None:
        """Every native the writer can emit reverses cleanly (writer→reader parity)."""
        from crossby.sync.hooks import AntigravityCLIHooksWriter

        canonical = ["Write", "Edit", "MultiEdit", "Bash", "Read", "Grep"]
        hook = HookEntry(
            event="pre_tool_use", command="g", tools=canonical, description="six natives"
        )
        AntigravityCLIHooksWriter().sync(_cfg(hook), tmp_path)
        assert _read_agy_hooks(tmp_path)[0].tools == canonical

    def test_catchall_and_non_plain_matcher_are_unscoped(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {
                "c": {
                    "PreToolUse": [
                        {"matcher": ".*", "hooks": [{"type": "command", "command": "wild"}]},
                        {
                            "matcher": "write_to_file.*",
                            "hooks": [{"type": "command", "command": "re"}],
                        },
                    ]
                }
            },
        )
        by_command = {e.command: e.tools for e in _read_agy_hooks(tmp_path)}
        assert by_command["wild"] == []
        assert by_command["re"] == []

    def test_pre_post_invocation_keys_are_skipped(self, tmp_path: Path) -> None:
        """agy's Pre/PostInvocation have no canonical crossby event — skip them."""
        self._write(
            tmp_path,
            {
                "c": {
                    "PreInvocation": [
                        {"matcher": ".*", "hooks": [{"type": "command", "command": "pre"}]}
                    ],
                    "PostInvocation": [{"type": "command", "command": "post"}],
                    "PreToolUse": [
                        {
                            "matcher": "write_to_file",
                            "hooks": [{"type": "command", "command": "keep"}],
                        }
                    ],
                }
            },
        )
        commands = {e.command for e in _read_agy_hooks(tmp_path)}
        assert commands == {"keep"}

    def test_malformed_entries_are_skipped(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {
                "bad": {
                    "PreToolUse": [
                        "not-a-dict",
                        {"matcher": "write_to_file", "hooks": "not-a-list"},
                        {"matcher": "write_to_file", "hooks": ["nope", {"type": "command"}]},
                        {"matcher": "write_to_file", "hooks": [{"type": "command", "command": 5}]},
                    ],
                    "Stop": [{"type": "command", "command": ""}, {"type": "command"}],
                },
                "not-a-container": [1, 2, 3],
                "ok": {"Stop": [{"type": "command", "command": "good"}]},
            },
        )
        assert [e.command for e in _read_agy_hooks(tmp_path)] == ["good"]

    def test_writer_reader_writer_round_trip_is_noop(self, tmp_path: Path) -> None:
        from crossby.sync.hooks import AntigravityCLIHooksWriter

        writer = AntigravityCLIHooksWriter()
        writer.sync(
            _cfg(GUARD_HOOK, HookEntry(event="stop", command="wade stop", description="done")),
            tmp_path,
        )
        path = tmp_path / ".agents" / "hooks.json"
        before = _read_json(path)

        read_back = _read_agy_hooks(tmp_path)
        result = writer.sync(SyncData(hooks=read_back), tmp_path)
        assert result.action == "skipped"
        assert _read_json(path) == before


# ---------------------------------------------------------------------------
# Round-trip guard — a hand-authored non-plain matcher is never broadened to .*
# ---------------------------------------------------------------------------


class TestHooksRoundTripGuard:
    def test_agy_hand_authored_wildcard_matcher_not_broadened(self, tmp_path: Path) -> None:
        from crossby.sync.hooks import AntigravityCLIHooksWriter

        path = tmp_path / ".agents" / "hooks.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "mine": {
                        "PreToolUse": [
                            {
                                "matcher": "Write.*",
                                "hooks": [{"type": "command", "command": "guard"}],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        before = _read_json(path)

        read_back = _read_agy_hooks(tmp_path)
        assert read_back[0].tools == []  # non-plain matcher → unscoped
        result = AntigravityCLIHooksWriter().sync(SyncData(hooks=read_back), tmp_path)

        assert result.action == "skipped"
        after = _read_json(path)
        assert after == before
        assert after["mine"]["PreToolUse"][0]["matcher"] == "Write.*"

    def test_cursor_hand_authored_wildcard_matcher_not_broadened(self, tmp_path: Path) -> None:
        path = tmp_path / ".cursor" / "hooks.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "hooks": {
                        "preToolUse": [
                            {"type": "command", "command": "guard", "matcher": "Write.*"}
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        before = _read_json(path)

        read_back = _read_cursor_hooks(tmp_path)
        assert read_back[0].tools == []  # non-plain matcher → unscoped
        result = CursorHooksWriter().sync(SyncData(hooks=read_back), tmp_path)

        assert result.action == "skipped"
        after = _read_json(path)
        assert after == before
        assert after["hooks"]["preToolUse"][0]["matcher"] == "Write.*"


# ---------------------------------------------------------------------------
# discover_hooks(from_tool=...) — Codex / agy are no longer read-blind
# ---------------------------------------------------------------------------


class TestDiscoverHooksFromTool:
    def test_codex_from_tool_returns_real_entries(self, tmp_path: Path) -> None:
        from crossby.sync.hooks import CodexHooksWriter

        CodexHooksWriter().sync(_cfg(GUARD_HOOK), tmp_path)
        hooks = discover_hooks(tmp_path, AIToolID.CODEX)
        assert [(h.event, h.command) for h in hooks] == [("pre_tool_use", GUARD_HOOK.command)]

    def test_antigravity_cli_from_tool_returns_real_entries(self, tmp_path: Path) -> None:
        from crossby.sync.hooks import AntigravityCLIHooksWriter

        AntigravityCLIHooksWriter().sync(_cfg(GUARD_HOOK), tmp_path)
        hooks = discover_hooks(tmp_path, AIToolID.ANTIGRAVITY_CLI)
        assert [(h.event, h.command) for h in hooks] == [("pre_tool_use", GUARD_HOOK.command)]

    def test_codex_from_tool_empty_when_no_config(self, tmp_path: Path) -> None:
        assert discover_hooks(tmp_path, AIToolID.CODEX) == []
