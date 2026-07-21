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

    def test_copilot_uses_exit_code(self) -> None:
        assert _caps(AIToolID.COPILOT).hook_output_dialect is HookOutputDialect.EXIT_CODE

    def test_antigravity_cli_uses_hook_specific_output_default(self) -> None:
        # AntigravityCLIAdapter.capabilities() doesn't override this field,
        # so it falls back to the base-class default.
        caps = _caps(AIToolID.ANTIGRAVITY_CLI)
        assert caps.hook_output_dialect is HookOutputDialect.HOOK_SPECIFIC_OUTPUT


class TestStopHookSupport:
    def test_supported(self) -> None:
        for tool in (AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.CURSOR):
            assert _caps(tool).supports_stop_hook is True

    def test_unsupported(self) -> None:
        assert _caps(AIToolID.COPILOT).supports_stop_hook is False

    def test_antigravity_cli_has_no_hook_system(self) -> None:
        # agy has no hook system at all, by design — unlike Codex.
        assert _caps(AIToolID.ANTIGRAVITY_CLI).supports_stop_hook is False
        assert _caps(AIToolID.ANTIGRAVITY_CLI).supports_session_start_hook is False


class TestSandboxAndFailOpen:
    def test_codex_sandboxes_writes(self) -> None:
        assert _caps(AIToolID.CODEX).sandboxes_writes is True

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
