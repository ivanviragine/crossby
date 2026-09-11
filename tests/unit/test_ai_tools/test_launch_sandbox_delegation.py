"""The public launch contract forwards sandbox selection to command building."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from unittest.mock import patch

from crossby.ai_tools.antigravity import AntigravityAdapter
from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter
from crossby.ai_tools.copilot import CopilotAdapter
from crossby.ai_tools.vscode import VSCodeAdapter


def test_launch_forwards_unsandboxed_request(tmp_path: Path) -> None:
    adapter = CodexAdapter()
    with (
        patch.object(adapter, "build_launch_command", return_value=["codex"]) as build,
        patch("crossby.utils.process.run_with_transcript", return_value=0),
    ):
        assert adapter.launch(tmp_path, sandbox=False) == 0

    assert build.call_args.kwargs["sandbox"] is False


def test_launch_omits_sandbox_for_legacy_command_builder(tmp_path: Path) -> None:
    """Unsupported adapters may retain the public pre-toggle hook signature."""
    adapter = ClaudeAdapter()

    def legacy_build_launch_command(
        *,
        model: object,
        initial_message: object,
        plan_mode: object,
        trusted_dirs: object,
        effort: object,
        allowed_commands: object,
        yolo: object,
        accept_edits: object,
        auto: object,
        scene: object,
        working_dir: object,
        network_access: object,
    ) -> list[str]:
        return ["claude"]

    with patch("crossby.utils.process.run_with_transcript", return_value=0):
        adapter.build_launch_command = legacy_build_launch_command  # type: ignore[method-assign]
        assert adapter.launch(tmp_path) == 0


def test_launch_renders_native_profile_approvals_once_after_command_build(tmp_path: Path) -> None:
    """The native channel composes around legacy-safe command construction."""
    adapter = CodexAdapter()
    with (
        patch.object(adapter, "build_launch_command", return_value=["codex"]) as build,
        patch.object(adapter, "allow_tools_args", return_value=["--native", "entry"]) as native,
        patch("crossby.utils.process.run_with_transcript", return_value=0) as run,
    ):
        assert adapter.launch(tmp_path, allow_tools=["entry"]) == 0

    assert build.call_count == 1
    native.assert_called_once_with(["entry"], None)
    assert run.call_args.args[0] == ["codex", "--native", "entry"]


def test_launch_preserves_legacy_positional_bindings_before_allow_tools(tmp_path: Path) -> None:
    """The profile-only argument must not shift the public launch contract."""
    positional_args = (
        tmp_path,
        "model",
        "prompt",
        True,
        tmp_path / "transcript",
        ["trusted-dir"],
        None,
        ["command:*"],
        True,
        True,
        True,
        True,
        None,
        True,
        tmp_path / "plans",
    )

    for adapter in (CopilotAdapter(), VSCodeAdapter(), AntigravityAdapter()):
        bound = signature(adapter.launch).bind(*positional_args)

        assert bound.arguments["yolo"] is True
        assert bound.arguments["plan_mode"] is True
        assert bound.arguments["accept_edits"] is True
        assert bound.arguments["auto"] is True
        assert bound.arguments["network_access"] is True
        assert bound.arguments["plan_output_dir"] == tmp_path / "plans"
        assert "allow_tools" not in bound.arguments
