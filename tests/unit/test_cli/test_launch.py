"""Targeted unit tests for the launch command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import TokenUsage
from crossby.models.config import CrossbyConfig

runner = CliRunner()


class TestLaunchCommand:
    def test_creates_transcript_parent_dir(self, tmp_path) -> None:
        transcript = tmp_path / "nested" / "deep" / "session.txt"
        adapter = Mock()
        adapter.capabilities.return_value = SimpleNamespace(
            display_name="Claude Code",
            supports_initial_message=True,
        )
        adapter.launch.return_value = 0
        adapter.parse_transcript.return_value = TokenUsage()

        with (
            patch("crossby.config.loader.load_config", return_value=CrossbyConfig()),
            patch("crossby.services.ai_resolution.resolve_ai_tool", return_value="claude"),
            patch("crossby.services.ai_resolution.resolve_model", return_value=None),
            patch("crossby.services.ai_resolution.resolve_effort", return_value=None),
            patch("crossby.services.ai_resolution.resolve_yolo", return_value=False),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
        ):
            result = runner.invoke(
                app,
                [
                    "launch",
                    str(tmp_path),
                    "--tool",
                    "claude",
                    "--transcript",
                    str(transcript),
                ],
            )

        assert result.exit_code == 0
        assert transcript.parent.exists()
        assert adapter.launch.call_args.kwargs["transcript_path"] == transcript

    def test_resolution_errors_are_shown_cleanly(self, tmp_path) -> None:
        with (
            patch("crossby.config.loader.load_config", return_value=CrossbyConfig()),
            patch("crossby.services.ai_resolution.resolve_ai_tool", return_value="claude"),
            patch(
                "crossby.services.ai_resolution.resolve_model",
                side_effect=ValueError("bad model for claude"),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "launch",
                    str(tmp_path),
                    "--tool",
                    "claude",
                    "--model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 1
        assert "bad model for claude" in result.output
