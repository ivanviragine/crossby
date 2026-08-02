"""Tests for the runtime hook I/O contract (crossby.hooks.runtime)."""

from __future__ import annotations

import json

import pytest

from crossby.hooks.runtime import (
    READ_TOOL_NAMES,
    SHELL_TOOL_NAMES,
    HookDecision,
    HookEvent,
    detect_tool_id,
    emit_decision,
    emit_stop_decision,
    parse_event,
)
from crossby.models.ai import AIToolID, HookOutputDialect, HookStopDialect


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

    @pytest.mark.parametrize("key", ["file_path", "filePath", "path"])
    def test_antigravity_cli_toolcall_wrapper(self, key: str) -> None:
        # agy nests the call as {"toolCall": {"name", "args": {...}}}.
        payload = {"toolCall": {"name": "write_file", "args": {key: "/repo/a.py"}}}
        ev = parse_event(json.dumps(payload))
        assert ev.tool_name == "write_file"
        assert ev.file_path == "/repo/a.py"

    def test_antigravity_cli_toolcall_command(self) -> None:
        payload = {"toolCall": {"name": "run_command", "args": {"command": "git push"}}}
        ev = parse_event(json.dumps(payload))
        assert ev.tool_name == "run_command"
        assert ev.command == "git push"

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
            # agy (Gemini-family successor) emits Claude-style Pre/PostToolUse.
            ("PostToolUse", "post_tool_use"),
            ("SessionStart", "session_start"),
            ("Stop", "stop"),
        ],
    )
    def test_event_from_payload(self, raw_name: str, canonical: str) -> None:
        ev = parse_event(json.dumps({"hook_event_name": raw_name}))
        assert ev.event == canonical

    @pytest.mark.parametrize("gemini_name", ["BeforeTool", "AfterTool"])
    def test_removed_gemini_event_names_pass_through(self, gemini_name: str) -> None:
        # Gemini CLI was removed (#69); its BeforeTool/AfterTool names have no
        # emitter, so they are no longer normalized and pass through unchanged.
        ev = parse_event(json.dumps({"hook_event_name": gemini_name}))
        assert ev.event == gemini_name

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
    """HookEvent.is_write is a fail-CLOSED denylist, not an allowlist.

    The old allowlist returned False for anything it didn't recognise, so a
    guard waved through Codex's ``apply_patch`` and every agy write tool.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "write",
            "edit",
            "multiedit",
            "notebookedit",
            "delete",
            "create",
            # The real fall-throughs the allowlist missed.
            "apply_patch",  # Codex
            "write_stdin",  # Codex
            "str_replace_editor",  # Copilot
            "write_to_file",  # agy
            "replace_file_content",  # agy
            "multi_replace_file_content",  # agy
        ],
    )
    def test_write_tools(self, name: str) -> None:
        assert HookEvent(tool_name=name).is_write is True

    @pytest.mark.parametrize(
        "name",
        [
            "some_brand_new_tool",
            "edit_notebook_v2",
            "",
        ],
    )
    def test_unknown_tool_name_fails_closed(self, name: str) -> None:
        """A name crossby has never seen must be treated as a write."""
        assert HookEvent(tool_name=name).is_write is True

    @pytest.mark.parametrize(
        "name",
        [
            # Claude
            "read",
            "grep",
            "glob",
            "webfetch",
            "websearch",
            "todowrite",
            "task",
            # Cursor
            "tabread",
            # Copilot
            "view",
            "rg",
            "web_fetch",
            "ask_user",
            # agy
            "view_file",
            "list_dir",
            "grep_search",
            "read_url_content",
            "search_web",
            "list_permissions",
        ],
    )
    def test_documented_read_tools(self, name: str) -> None:
        assert HookEvent(tool_name=name).is_write is False

    @pytest.mark.parametrize(
        "name",
        ["bash", "shell", "run_command", "powershell", "exec_command"],
    )
    def test_shell_is_carved_out(self, name: str) -> None:
        """Shell calls are NOT path-addressed writes — deliberate carve-out.

        wade's worktree_containment denies a write whose file_path is absent, so
        marking a shell call as a write would deny every shell command. Codex is
        the extreme case: it has no read-only tools at all and routes every read
        through `rg` in the shell, so a session would be blocked outright.
        Shell is guarded through HookEvent.command instead.
        """
        event = HookEvent(tool_name=name, command="rm -rf /")
        assert event.is_write is False
        # The command channel is still populated, so a policy can inspect it.
        assert event.command == "rm -rf /"

    def test_unknown_tool_name_treated_as_write(self) -> None:
        assert HookEvent(tool_name=None).is_write is True

    def test_read_and_shell_sets_are_disjoint(self) -> None:
        assert not (READ_TOOL_NAMES & SHELL_TOOL_NAMES)

    def test_name_sets_are_lowercase(self) -> None:
        """_extract_tool_name lowercases, so the sets must be lowercase to match."""
        for name in READ_TOOL_NAMES | SHELL_TOOL_NAMES:
            assert name == name.lower()

    def test_matching_is_case_insensitive_via_parse(self) -> None:
        assert parse_event(json.dumps({"tool_name": "Bash"})).is_write is False
        assert parse_event(json.dumps({"tool_name": "Read"})).is_write is False
        assert parse_event(json.dumps({"tool_name": "Write"})).is_write is True


class TestEmitDecisionAllow:
    @pytest.mark.parametrize(
        "dialect",
        [d for d in HookOutputDialect if d is not HookOutputDialect.DECISION],
    )
    def test_allow_is_silent_exit_zero(self, dialect: HookOutputDialect) -> None:
        em = emit_decision(HookDecision.allow(), dialect)
        assert em.exit_code == 0
        assert em.stdout == ""
        assert em.stderr == ""

    def test_allow_emits_empty_object_for_decision_dialect(self) -> None:
        # agy rejects a truly empty stdout; {} is its "no opinion, proceed" no-op.
        em = emit_decision(HookDecision.allow(), HookOutputDialect.DECISION)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {}
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

    def test_decision_dialect(self) -> None:
        # agy blocks a tool call via a top-level {"decision": "deny"}; exit 2
        # keeps the guard fail-closed even if agy ignores the stdout decision.
        em = emit_decision(HookDecision.deny("nope"), HookOutputDialect.DECISION)
        assert em.exit_code == 2
        assert em.stderr == "nope"
        assert json.loads(em.stdout) == {"decision": "deny", "reason": "nope"}

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

    def test_context_no_op_for_other_dialects(self) -> None:
        em = emit_decision(HookDecision.context("hello"), HookOutputDialect.EXIT_CODE)
        assert em.exit_code == 0
        assert em.stdout == ""

    def test_context_no_op_for_decision_dialect(self) -> None:
        # agy has no verified context-injection field, so context degrades to a
        # bare {} proceed rather than blocking or injecting.
        em = emit_decision(HookDecision.context("hello"), HookOutputDialect.DECISION)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {}

    def test_context_injected_for_permission_dialect_on_session_start(self) -> None:
        # Cursor injects context via a top-level `additional_context` field, not
        # hookSpecificOutput — and only on the events whose output schema
        # actually reads it.
        em = emit_decision(
            HookDecision.context("hello"), HookOutputDialect.PERMISSION, event="session_start"
        )
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {"additional_context": "hello"}

    def test_context_gated_off_for_cursor_prompt_submit(self) -> None:
        """Cursor's beforeSubmitPrompt ignores `additional_context`.

        Its documented output is `continue` + `user_message` only, so emitting
        the key there injects nothing — don't pretend it landed.
        """
        em = emit_decision(
            HookDecision.context("hello"),
            HookOutputDialect.PERMISSION,
            event="user_prompt_submit",
        )
        assert em.exit_code == 0
        assert em.stdout == ""

    def test_context_flat_for_copilot(self) -> None:
        """Copilot reads a flat top-level additionalContext, never nested."""
        em = emit_decision(HookDecision.context("hello"), HookOutputDialect.PERMISSION_DECISION)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {"additionalContext": "hello"}


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

    def test_no_block_emits_empty_object_for_decision_dialect(self) -> None:
        # agy's Stop no-op is a bare {} — a top-level {"continue": true} is not a
        # key it recognizes, and {"decision": "continue"} would *block* the stop.
        em = emit_stop_decision(False, "unused", HookOutputDialect.DECISION)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {}

    def test_block_decision_dialect_continues(self) -> None:
        # agy blocks a Stop by telling the agent to continue.
        em = emit_stop_decision(True, "finish first", HookOutputDialect.DECISION)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {"decision": "continue", "reason": "finish first"}

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


class TestEmitStopDecisionStopDialects:
    """The Stop channel is declared independently of the tool-call channel."""

    def test_block_decision(self) -> None:
        em = emit_stop_decision(True, "finish first", HookStopDialect.BLOCK_DECISION)
        assert em.exit_code == 0
        assert json.loads(em.stdout) == {"decision": "block", "reason": "finish first"}

    def test_followup_message(self) -> None:
        em = emit_stop_decision(True, "finish first", HookStopDialect.FOLLOWUP_MESSAGE)
        assert json.loads(em.stdout) == {"followup_message": "finish first"}

    def test_continue_decision(self) -> None:
        # agy blocks a stop by telling the agent to *continue* — inverted
        # polarity vs the `continue` boolean every other dialect uses as a no-op.
        em = emit_stop_decision(True, "finish first", HookStopDialect.CONTINUE_DECISION)
        assert json.loads(em.stdout) == {"decision": "continue", "reason": "finish first"}

    def test_none_dialect_is_noop(self) -> None:
        em = emit_stop_decision(True, "finish first", HookStopDialect.NONE)
        assert em.exit_code == 0
        assert em.stdout == ""

    @pytest.mark.parametrize("dialect", list(HookStopDialect))
    def test_stop_never_exits_nonzero(self, dialect: HookStopDialect) -> None:
        """The Stop channel is fail-OPEN — a guard must never trap the agent."""
        assert emit_stop_decision(True, "r", dialect).exit_code == 0
        assert emit_stop_decision(False, "", dialect).exit_code == 0

    def test_copilot_stop_now_blocks_instead_of_no_op(self) -> None:
        """Regression: Copilot's stop was a silent no-op under EXIT_CODE.

        A Copilot Stop hook would install and then never block anything.
        """
        from crossby.ai_tools.copilot import CopilotAdapter

        caps = CopilotAdapter().capabilities()
        assert caps.supports_stop_hook is True
        em = emit_stop_decision(True, "keep going", caps.hook_stop_dialect)
        assert json.loads(em.stdout) == {"decision": "block", "reason": "keep going"}

    @pytest.mark.parametrize(
        ("legacy", "expected_stop"),
        [
            (HookOutputDialect.HOOK_SPECIFIC_OUTPUT, HookStopDialect.BLOCK_DECISION),
            (HookOutputDialect.PERMISSION, HookStopDialect.FOLLOWUP_MESSAGE),
            (HookOutputDialect.PERMISSION_DECISION, HookStopDialect.BLOCK_DECISION),
            (HookOutputDialect.DECISION, HookStopDialect.CONTINUE_DECISION),
            (HookOutputDialect.EXIT_CODE, HookStopDialect.NONE),
        ],
    )
    def test_legacy_output_dialect_maps_to_stop_dialect(
        self, legacy: HookOutputDialect, expected_stop: HookStopDialect
    ) -> None:
        """wade still passes a HookOutputDialect positionally; keep it working.

        Behaviour must be byte-identical to the new enum's, so a wade pin bump
        is a no-op rather than a silent change of stop semantics.
        """
        for should_block in (True, False):
            assert emit_stop_decision(should_block, "r", legacy) == emit_stop_decision(
                should_block, "r", expected_stop
            )


class TestEmitDecisionCopilot:
    """Copilot's preToolUse reads a flat permission-decision object."""

    def test_deny_is_flat_with_required_reason(self) -> None:
        em = emit_decision(
            HookDecision.deny("outside worktree"), HookOutputDialect.PERMISSION_DECISION
        )
        payload = json.loads(em.stdout)
        assert payload == {
            "permissionDecision": "deny",
            "permissionDecisionReason": "outside worktree",
        }
        # Never nested — hookSpecificOutput appears nowhere in GitHub's docs.
        assert "hookSpecificOutput" not in payload
        # Exit 2 is an additional deny channel that overrides an allow in stdout,
        # so the guard stays fail-closed even if stdout is dropped.
        assert em.exit_code == 2
        assert em.stderr == "outside worktree"

    def test_deny_emits_exactly_one_json_object(self) -> None:
        """Copilot does a single JSON.parse — two objects would be invalid JSON.

        Concatenated objects parse as garbage and are silently ignored, which
        would turn a deny into an allow.
        """
        em = emit_decision(HookDecision.deny("nope"), HookOutputDialect.PERMISSION_DECISION)
        assert json.loads(em.stdout)  # parses as a single value
        assert em.stdout.count("{") == 1

    def test_claude_deny_stays_nested_only(self) -> None:
        """Claude validates strictly and fails open on unexpected root keys."""
        em = emit_decision(HookDecision.deny("nope"), HookOutputDialect.HOOK_SPECIFIC_OUTPUT)
        payload = json.loads(em.stdout)
        assert set(payload) == {"hookSpecificOutput"}


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

    @pytest.mark.parametrize(
        "payload",
        [
            {"workspacePaths": ["/repo"]},
            {"conversationId": "c1"},
            {"artifactDirectoryPath": "/repo/.artifacts"},
            {"toolCall": {"name": "write_file", "args": {}}},
        ],
    )
    def test_antigravity_cli_by_camelcase_shape(self, payload: dict) -> None:
        assert detect_tool_id(payload) is AIToolID.ANTIGRAVITY_CLI

    def test_antigravity_cli_wins_over_codex_model_field(self) -> None:
        # agy hook stdin can also carry `model`; its camelCase shape must win so
        # it isn't misread as Codex.
        payload = {"workspacePaths": ["/repo"], "model": "gemini-3-flash"}
        assert detect_tool_id(payload) is AIToolID.ANTIGRAVITY_CLI

    def test_unknown_returns_none(self) -> None:
        # No distinguishing field → None, so the caller applies its own default.
        assert detect_tool_id({"tool_name": "Write"}) is None
        assert detect_tool_id({"workspace_roots": []}) is None
