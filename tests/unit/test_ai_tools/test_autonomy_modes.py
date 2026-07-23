"""Accept-edits and auto (classifier) autonomy tiers.

Covers per-adapter ``accept_edits_args()`` / ``auto_args()``, capability flags,
``build_launch_command()`` composition, the precedence chain
(``yolo > auto > accept_edits > plan``), and the downgrade/fallback warning
paths.
"""

from __future__ import annotations

import pytest

from crossby.ai_tools.base import AbstractAITool


def _permission_mode(cmd: list[str]) -> str | None:
    """Return the value following ``--permission-mode`` in *cmd*, if present."""
    if "--permission-mode" in cmd:
        return cmd[cmd.index("--permission-mode") + 1]
    return None


class TestAcceptEditsArgs:
    """Per-adapter accept-edits flags verified against the July 2026 matrix."""

    def test_claude(self) -> None:
        assert AbstractAITool.get("claude").accept_edits_args() == [
            "--permission-mode",
            "acceptEdits",
        ]

    def test_codex_uses_workspace_write_untrusted(self) -> None:
        # The removed --approval-mode auto-edit must NOT be used.
        args = AbstractAITool.get("codex").accept_edits_args()
        assert args == ["-s", "workspace-write", "-a", "untrusted"]
        assert "--approval-mode" not in args

    def test_cursor_native_default_no_flag(self) -> None:
        assert AbstractAITool.get("cursor").accept_edits_args() == []

    def test_copilot(self) -> None:
        assert AbstractAITool.get("copilot").accept_edits_args() == ["--allow-tool", "write"]

    def test_antigravity_cli(self) -> None:
        assert AbstractAITool.get("antigravity-cli").accept_edits_args() == [
            "--mode",
            "accept-edits",
        ]

    def test_unsupported_tools_return_empty(self) -> None:
        for tool in ("opencode", "vscode", "antigravity"):
            assert AbstractAITool.get(tool).accept_edits_args() == []


class TestAutoArgs:
    """Only Claude exposes a real launch-time classifier auto mode."""

    def test_claude(self) -> None:
        assert AbstractAITool.get("claude").auto_args() == ["--permission-mode", "auto"]

    def test_other_tools_return_empty(self) -> None:
        for tool in ("codex", "cursor", "copilot", "antigravity-cli", "opencode"):
            assert AbstractAITool.get(tool).auto_args() == []


class TestCapabilityFlags:
    def test_accept_edits_supported_tools(self) -> None:
        for tool in ("claude", "codex", "cursor", "copilot", "antigravity-cli"):
            assert AbstractAITool.get(tool).capabilities().supports_accept_edits is True

    def test_accept_edits_unsupported_tools(self) -> None:
        for tool in ("opencode", "vscode", "antigravity"):
            assert AbstractAITool.get(tool).capabilities().supports_accept_edits is False

    def test_auto_is_claude_only(self) -> None:
        assert AbstractAITool.get("claude").capabilities().supports_auto is True
        for tool in ("codex", "cursor", "copilot", "antigravity-cli", "opencode", "vscode"):
            assert AbstractAITool.get(tool).capabilities().supports_auto is False


class TestAcceptEditsComposition:
    """``build_launch_command(accept_edits=True)`` per supported adapter."""

    def test_claude(self) -> None:
        cmd = AbstractAITool.get("claude").build_launch_command(accept_edits=True)
        assert _permission_mode(cmd) == "acceptEdits"

    def test_codex(self) -> None:
        cmd = AbstractAITool.get("codex").build_launch_command(accept_edits=True)
        assert cmd[-4:] == ["-s", "workspace-write", "-a", "untrusted"]

    def test_cursor_emits_no_extra_flag(self) -> None:
        adapter = AbstractAITool.get("cursor")
        base = adapter.build_launch_command(model="sonnet-4.6")
        with_accept = adapter.build_launch_command(model="sonnet-4.6", accept_edits=True)
        assert with_accept == base

    def test_copilot(self) -> None:
        cmd = AbstractAITool.get("copilot").build_launch_command(accept_edits=True)
        assert cmd[-2:] == ["--allow-tool", "write"]

    def test_antigravity_cli(self) -> None:
        cmd = AbstractAITool.get("antigravity-cli").build_launch_command(accept_edits=True)
        assert cmd[-2:] == ["--mode", "accept-edits"]


class TestAutoComposition:
    def test_claude_uses_classifier(self) -> None:
        cmd = AbstractAITool.get("claude").build_launch_command(auto=True)
        assert _permission_mode(cmd) == "auto"


class TestAutoDowngrade:
    """``auto`` is Claude-only and downgrades to accept-edits, then default."""

    def test_codex_downgrades_to_accept_edits(self) -> None:
        adapter = AbstractAITool.get("codex")
        with pytest.warns(UserWarning, match="downgrading to accept-edits"):
            cmd = adapter.build_launch_command(auto=True)
        assert cmd[-4:] == ["-s", "workspace-write", "-a", "untrusted"]

    def test_cursor_downgrades_to_accept_edits_no_flag(self) -> None:
        adapter = AbstractAITool.get("cursor")
        base = adapter.build_launch_command(model="sonnet-4.6")
        with pytest.warns(UserWarning, match="downgrading to accept-edits"):
            cmd = adapter.build_launch_command(model="sonnet-4.6", auto=True)
        assert cmd == base

    def test_opencode_downgrades_to_default_prompting(self) -> None:
        adapter = AbstractAITool.get("opencode")
        base = adapter.build_launch_command()
        with pytest.warns(UserWarning, match="using default prompting"):
            cmd = adapter.build_launch_command(auto=True)
        assert cmd == base
        # Never escalates to yolo.
        assert "--yolo" not in cmd


class TestAcceptEditsFallback:
    """Accept-edits degrades to default prompting where unsupported."""

    def test_opencode_default_prompting_with_warning(self) -> None:
        adapter = AbstractAITool.get("opencode")
        base = adapter.build_launch_command()
        with pytest.warns(UserWarning, match="using default prompting"):
            cmd = adapter.build_launch_command(accept_edits=True)
        assert cmd == base

    def test_vscode_default_prompting_with_warning(self) -> None:
        adapter = AbstractAITool.get("vscode")
        with pytest.warns(UserWarning, match="does not support accept-edits"):
            adapter.build_launch_command(accept_edits=True)

    def test_explicit_plan_is_honored_as_lower_fallback(self) -> None:
        # accept_edits unsupported + explicit plan requested → warn about plan.
        adapter = AbstractAITool.get("opencode")
        with pytest.warns(UserWarning, match="falling back to plan mode"):
            adapter.build_launch_command(accept_edits=True, plan_mode=True)


class TestPrecedence:
    """yolo > auto > accept_edits > plan (most permissive wins)."""

    def test_yolo_supersedes_all(self) -> None:
        cmd = AbstractAITool.get("claude").build_launch_command(
            yolo=True, auto=True, accept_edits=True, plan_mode=True
        )
        assert "--dangerously-skip-permissions" in cmd
        assert _permission_mode(cmd) is None  # no acceptEdits/auto/plan value

    def test_auto_supersedes_accept_edits_and_plan(self) -> None:
        cmd = AbstractAITool.get("claude").build_launch_command(
            auto=True, accept_edits=True, plan_mode=True
        )
        assert _permission_mode(cmd) == "auto"

    def test_accept_edits_supersedes_plan(self) -> None:
        cmd = AbstractAITool.get("claude").build_launch_command(accept_edits=True, plan_mode=True)
        assert _permission_mode(cmd) == "acceptEdits"

    def test_plan_alone(self) -> None:
        cmd = AbstractAITool.get("claude").build_launch_command(plan_mode=True)
        assert _permission_mode(cmd) == "plan"

    def test_no_autonomy_flags_is_bare(self) -> None:
        cmd = AbstractAITool.get("claude").build_launch_command()
        assert cmd == ["claude"]


class TestCapabilityInvariants:
    """The downgrade cascade relies on higher tiers implying lower ones.

    ``_autonomy_launch_args`` collapses the requested flags to the single highest
    tier and walks down. If a tool supported ``auto`` but not ``yolo``, a
    ``--yolo --accept-edits`` request could land on ``auto`` (never requested).
    Guard the invariant so a future adapter can't break it silently.
    """

    def test_auto_implies_yolo_and_accept_edits(self) -> None:
        for tool in AbstractAITool.available_tools():
            caps = AbstractAITool.get(tool).capabilities()
            if caps.supports_auto:
                assert caps.supports_yolo, f"{tool}: supports_auto without supports_yolo"
                assert caps.supports_accept_edits, (
                    f"{tool}: supports_auto without supports_accept_edits"
                )


class TestYoloFallbackUnchangedTools:
    """Yolo-unsupported tools still degrade the same way (regression guard)."""

    def test_opencode_yolo_default_prompting(self) -> None:
        adapter = AbstractAITool.get("opencode")
        base = adapter.build_launch_command()
        with pytest.warns(UserWarning, match="using default prompting"):
            cmd = adapter.build_launch_command(yolo=True)
        assert cmd == base

    def test_opencode_yolo_with_plan_falls_back_to_plan(self) -> None:
        adapter = AbstractAITool.get("opencode")
        with pytest.warns(UserWarning, match="falling back to plan mode"):
            adapter.build_launch_command(yolo=True, plan_mode=True)
