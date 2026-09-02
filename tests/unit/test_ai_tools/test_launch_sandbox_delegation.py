"""The public launch contract forwards sandbox selection to command building."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from crossby.ai_tools.claude import ClaudeAdapter
from crossby.ai_tools.codex import CodexAdapter


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
