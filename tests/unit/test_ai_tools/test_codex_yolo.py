"""Codex yolo mode must skip approvals without disabling the OS sandbox."""

from __future__ import annotations

from crossby.ai_tools.codex import CodexAdapter


class TestCodexYoloIsSandboxPreserving:
    def test_yolo_uses_approval_bypass_not_sandbox_bypass(self) -> None:
        args = CodexAdapter().yolo_args()
        assert args == ["-a", "never"]

    def test_yolo_never_disables_sandbox(self) -> None:
        # Regression guard: --yolo / --dangerously-bypass-approvals-and-sandbox
        # would remove the OS sandbox boundary. Yolo means skip prompts only.
        args = CodexAdapter().yolo_args()
        assert "--yolo" not in args
        assert "--dangerously-bypass-approvals-and-sandbox" not in args
