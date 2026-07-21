"""Tests for the runtime hook I/O contract (crossby.hooks.runtime)."""

from __future__ import annotations

import json

import pytest

from crossby.hooks.runtime import (
    HookDecision,
    HookEvent,
    detect_tool_id,
    emit_decision,
    emit_stop_decision,
    parse_event,
)
from crossby.models.ai import AIToolID, HookOutputDialect


class TestParseEventDialects:
    """parse_event normalizes every tool's stdin field naming."""

    @pytest.mark.parametrize(
        "payload",
        [
            # Claude
            {"tool_name": "Write", "tool_input": {"file_path": "/repo/a.py"}},
            {"tool_name": "Write", "tool_input": {"path": "/repo/a.py"}},
            # Cursor
            {"tool_name": "Write", "tool_input": {"filePath": "/repo/a.py"}},
            {"toolName": "Write", "toolInput": {"file_path": "/repo/a.py"}},
        ],
    )
    def test_tool_input_dialects(self, payload: dict) -> None:
        ev = parse_event(json.dumps(payload))
        assert ev.file_path == "/repo/a.py"
        assert ev.tool_name == "write"

    @pytest.mark.parametrize("key", ["file", "path", "filePath", "file_path"])
    def test_copilot_toolargs_json_string(self, key: str) -> None:
        payload = {"toolName": "edit", "toolArgs": json.dumps({key: "/repo/b.py"})}
        ev = parse_event(json.dumps(payload))
        assert ev.file_path == "/repo/b.py"
        assert ev.tool_name == "edit"

    @pytest.mark.parametrize(
        "payload",
        [
            # NotebookEdit puts its target in notebook_path, not file_path.
            {"tool_name": "NotebookEdit", "tool_input": {"notebook_path": "/repo/a.py"}},
            {"toolName": "NotebookEdit", "toolInput": {"notebookPath": "/repo/a.py"}},
            {"toolName": "notebookedit", "toolArgs": '{"notebook_path": "/repo/a.py"}'},
        ],
    )
    def test_notebook_path_extracted(self, payload: dict) -> None:
        # Without this the worktree guard can't see a NotebookEdit target and
        # would fail open, letting an out-of-worktree notebook write through.
        ev = parse_event(json.dumps(payload))
        assert ev.file_path == "/repo/a.py"
        assert ev.is_write is True

    def test_command_extraction(self) -> None:
        ev = parse_event(json.dumps({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}))
        assert ev.command == "rm -rf /"
        assert ev.file_path is None

    def test_cursor_top_level_command(self) -> None:
        # Cursor beforeShellExecution puts command at the top level, no wrapper.
        ev = parse_event(json.dumps({"command": "git push --force"}))
        assert ev.command == "git push --force"

    def test_cursor_top_level_file_path(self) -> None:
        # Cursor's file-scoped event hooks put the path at the top level.
        ev = parse_event(json.dumps({"file_path": "/repo/wt/a.py"}))
        assert ev.file_path == "/repo/wt/a.py"

    def test_wrapped_value_wins_over_top_level(self) -> None:
        # A nested tool_input command still takes precedence over a top-level one.
        payload = {"command": "top", "tool_input": {"command": "wrapped"}}
        assert parse_event(json.dumps(payload)).command == "wrapped"

    def test_cwd_and_raw_preserved(self) -> None:
        payload = {"tool_name": "Write", "tool_input": {"file_path": "/x"}, "cwd": "/repo"}
        ev = parse_event(json.dumps(payload))
        assert ev.cwd == "/repo"
        assert ev.raw == payload


class TestParseEventEvents:
    """Event-name resolution from payload or override."""

    @pytest.mark.parametrize(
        "raw_name,canonical",
        [
            ("PreToolUse", "pre_tool_use"),
            ("preToolUse", "pre_tool_use"),
            ("SessionStart", "session_start"),
            ("Stop", "stop"),
        ],
    )
    def test_event_from_payload(self, raw_name: str, canonical: str) -> None:
        ev = parse_event(json.dumps({"hook_event_name": raw_name}))
        assert ev.event == canonical

    def test_event_override_used_when_absent(self) -> None:
        ev = parse_event(json.dumps({"tool_name": "Write"}), event="stop")
        assert ev.event == "stop"

    @pytest.mark.parametrize("key", ["stop_hook_active", "stopHookActive"])
    def test_stop_hook_active_parsed(self, key: str) -> None:
        assert parse_event(json.dumps({key: True})).stop_hook_active is True

    def test_stop_hook_active_defaults_false(self) -> None:
        assert parse_event(json.dumps({"event": "stop"})).stop_hook_active is False


class TestParseEventMalformed:
    """Malformed input never raises; yields an all-None event."""

    @pytest.mark.parametrize("raw", ["", "   ", "not json", "[]", "null", "123"])
    def test_never_raises(self, raw: str) -> None:
        ev = parse_event(raw)
        assert ev.file_path is None
        assert ev.tool_name is None
        assert ev.raw == {}

    def test_malformed_toolargs_ignored(self) -> None:
        ev = parse_event(json.dumps({"tool_name": "write", "toolArgs": "{not json"}))
        assert ev.file_path is None


class TestIsWrite:
    """HookEvent.is_write distinguishes write intents (unknown → treated as write)."""

    @pytest.mark.parametrize("name", ["write", "edit", "multiedit", "notebookedit", "delete"])
    def test_write_tools(self, name: str) -> None:
        assert HookEvent(tool_name=name).is_write is True

    @pytest.mark.parametrize("name", ["read", "view", "grep", "bash"])
    def test_non_write_tools(self, name: str) -> None:
        assert HookEvent(tool_name=name).is_write is False

    def test_unknown_tool_name_treated_as_write(self) -> None:
        assert HookEvent(tool_name=None).is_write is True


class TestEmitDecisionAllow:
    @pytest.mark.parametrize("dialect", list(HookOutputDialect))
    def test_allow_is_silent_exit_zero(self, dialect: HookOutputDialect) -> None:
        em = emit_decision(HookDecision.allow(), dialect)
        assert em.exit_code == 0
        assert em.stdout == ""
        assert em.stderr == ""


class TestEmitDecisionDeny:
    def test_hook_specific_output(self) -> None:
        em = emit_decision(
            HookDecision.deny("nope"),
            HookOutputDialect.HOOK_SPECIFIC_OUTPUT,
            event="pre_tool_use",
        )
        assert em.exit_code == 2
        assert em.stderr == "nope"
        payload = json.loads(em.stdout)
        assert list(payload.keys()) == ["hookSpecificOutput"]
        hso = payload["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert hso["permissionDecisionReason"] == "nope"
        assert hso["hookEventName"] == "PreToolUse"

    def test_permission_dialect(self) -> None:
        em = emit_decision(HookDecision.deny("blocked"), HookOutputDialect.PERMISSION)
        assert em.exit_code == 2
        payload = json.loads(em.stdout)
        assert payload["permission"] == "deny"
        assert payload["agent_message"] == "blocked"

    def test_exit_code_dialect_has_no_stdout(self) -> None:
        em = emit_decision(HookDecision.deny("blocked"), HookOutputDialect.EXIT_CODE)
        assert em.exit_code == 2
        assert em.stdout == ""
        assert em.stderr == "blocked"

    @pytest.mark.parametrize("dialect", list(HookOutputDialect))
    def test_deny_always_exits_two(self, dialect: HookOutputDialect) -> None:
        assert emit_decision(HookDecision.deny("x"), dialect).exit_code == 2


class TestEmitDecisionContext:
    def test_context_injected_for_hook_specific_output(self) -> None:
        em = emit_decision(
            HookDecision.context("hello"),
            HookOutputDialect.HOOK_SPECIFIC_OUTPUT,
            event="session_start",
        )
        assert em.exit_code == 0
        payload = json.loads(em.stdout)
        assert payload["hookSpecificOutput"]["additionalContext"] == "hello"
        assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    @pytest.mark.parametrize("dialect", [HookOutputDialect.PERMISSION, HookOutputDialect.EXIT_CODE])
    def test_context_no_op_for_other_dialects(self, dialect: HookOutputDialect) -> None:
        em = emit_decision(HookDecision.context("hello"), dialect)
        assert em.exit_code == 0
        assert em.stdout == ""


class TestEmitStopDecision:
    @pytest.mark.parametrize(
        "dialect",
        [HookOutputDialect.HOOK_SPECIFIC_OUTPUT, HookOutputDialect.PERMISSION],
    )
    def test_no_block_emits_continue_on_stdout_dialects(self, dialect: HookOutputDialect) -> None:
        # Codex rejects an empty-stdout Stop hook, so a no-op must still emit
        # valid JSON; {"continue": true} is the universal harmless no-op.
        em = emit_stop_decision(False, "unused", dialect)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {"continue": True}

    def test_no_block_is_silent_for_exit_code_dialect(self) -> None:
        # EXIT_CODE tools ignore stdout entirely, so a no-op stays truly silent.
        em = emit_stop_decision(False, "unused", HookOutputDialect.EXIT_CODE)
        assert em.exit_code == 0
        assert em.stdout == ""

    def test_block_hook_specific_output(self) -> None:
        em = emit_stop_decision(True, "finish first", HookOutputDialect.HOOK_SPECIFIC_OUTPUT)
        assert em.exit_code == 0  # Stop hooks block via the JSON decision, not exit code
        payload = json.loads(em.stdout)
        assert payload == {"decision": "block", "reason": "finish first"}

    def test_block_cursor_followup(self) -> None:
        em = emit_stop_decision(True, "finish first", HookOutputDialect.PERMISSION)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {"followup_message": "finish first"}

    def test_block_exit_code_dialect_is_noop(self) -> None:
        em = emit_stop_decision(True, "finish first", HookOutputDialect.EXIT_CODE)
        assert em.exit_code == 0
        assert em.stdout == ""


class TestDetectToolId:
    """Best-effort tool detection from a hook payload's shape."""

    def test_cursor_by_conversation_id(self) -> None:
        assert detect_tool_id({"conversation_id": "abc"}) is AIToolID.CURSOR

    def test_cursor_by_workspace_roots(self) -> None:
        assert detect_tool_id({"workspace_roots": ["/repo"]}) is AIToolID.CURSOR

    def test_codex_by_model(self) -> None:
        # Codex includes `model` in hook stdin; session_id present too but Cursor absent.
        assert detect_tool_id({"model": "gpt-5", "session_id": "s"}) is AIToolID.CODEX

    def test_claude_by_session_id(self) -> None:
        assert detect_tool_id({"session_id": "s"}) is AIToolID.CLAUDE

    def test_cursor_wins_over_codex_and_claude(self) -> None:
        payload = {"conversation_id": "c", "model": "x", "session_id": "s"}
        assert detect_tool_id(payload) is AIToolID.CURSOR

    def test_unknown_returns_none(self) -> None:
        # No distinguishing field → None, so the caller applies its own default.
        assert detect_tool_id({"tool_name": "Write"}) is None
        assert detect_tool_id({"workspace_roots": []}) is None
