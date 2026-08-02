"""Adapter capability flags that the runtime hook contract and consumers rely on."""

from __future__ import annotations

from crossby.ai_tools import AbstractAITool
from crossby.models.ai import AIToolID, HookOutputDialect


def _caps(tool: AIToolID):
    return AbstractAITool.get(tool).capabilities()


class TestHookOutputDialect:
    def test_claude_and_codex_use_hook_specific_output(self) -> None:
        assert _caps(AIToolID.CLAUDE).hook_output_dialect is HookOutputDialect.HOOK_SPECIFIC_OUTPUT
        assert _caps(AIToolID.CODEX).hook_output_dialect is HookOutputDialect.HOOK_SPECIFIC_OUTPUT

    def test_cursor_uses_permission(self) -> None:
        assert _caps(AIToolID.CURSOR).hook_output_dialect is HookOutputDialect.PERMISSION

    def test_copilot_uses_permission_decision(self) -> None:
        # Copilot's preToolUse has a documented structured stdout schema, FLAT at
        # the top level. Modelling it as EXIT_CODE (crossby <=0.12) threw away
        # both the deny reason and the `ask` decision.
        assert _caps(AIToolID.COPILOT).hook_output_dialect is HookOutputDialect.PERMISSION_DECISION

    def test_antigravity_cli_uses_decision(self) -> None:
        # agy reads a hook decision as a top-level {"decision": …} object.
        caps = _caps(AIToolID.ANTIGRAVITY_CLI)
        assert caps.hook_output_dialect is HookOutputDialect.DECISION


class TestStopHookSupport:
    def test_supported(self) -> None:
        for tool in (AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.CURSOR):
            assert _caps(tool).supports_stop_hook is True

    def test_copilot_supports_stop(self) -> None:
        # Copilot fires `agentStop`, and emit_stop_decision produces a real
        # blocking payload for it rather than the old EXIT_CODE no-op.
        assert _caps(AIToolID.COPILOT).supports_stop_hook is True

    def test_antigravity_cli_supports_stop_but_not_session_start(self) -> None:
        # agy fires a Stop hook (the reliable enforcement surface) but has no
        # session_start / user_prompt_submit event.
        assert _caps(AIToolID.ANTIGRAVITY_CLI).supports_stop_hook is True
        assert _caps(AIToolID.ANTIGRAVITY_CLI).supports_session_start_hook is False


class TestUserPromptSubmitHookSupport:
    def test_supported(self) -> None:
        for tool in (AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.CURSOR):
            assert _caps(tool).supports_user_prompt_submit_hook is True

    def test_unsupported(self) -> None:
        for tool in (AIToolID.COPILOT, AIToolID.ANTIGRAVITY_CLI):
            assert _caps(tool).supports_user_prompt_submit_hook is False


class TestSandboxAndFailOpen:
    def test_codex_sandboxes_writes(self) -> None:
        assert _caps(AIToolID.CODEX).sandboxes_writes is True

    def test_antigravity_cli_fails_closed_but_sandbox_unclaimed(self) -> None:
        # agy denies a tool call when its PreToolUse hook errors (fail-closed),
        # but its write sandbox is an opt-in flag, not a verified default — so we
        # do NOT claim sandboxes_writes, keeping wade's own containment guard.
        caps = _caps(AIToolID.ANTIGRAVITY_CLI)
        assert caps.hook_fail_open_default is False
        assert caps.sandboxes_writes is False

    def test_claude_does_not_hard_sandbox(self) -> None:
        # Claude adds trusted dirs but prompts rather than hard-blocking.
        assert _caps(AIToolID.CLAUDE).supports_trusted_dirs is True
        assert _caps(AIToolID.CLAUDE).sandboxes_writes is False

    def test_cursor_fails_open_by_default(self) -> None:
        assert _caps(AIToolID.CURSOR).hook_fail_open_default is True

    def test_others_fail_closed_by_default(self) -> None:
        for tool in (AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.COPILOT):
            assert _caps(tool).hook_fail_open_default is False


class TestUsageReporting:
    def test_claude_and_codex_report_usage(self) -> None:
        assert _caps(AIToolID.CLAUDE).supports_usage_reporting is True
        assert _caps(AIToolID.CODEX).supports_usage_reporting is True

    def test_cursor_does_not_report_usage(self) -> None:
        assert _caps(AIToolID.CURSOR).supports_usage_reporting is False
