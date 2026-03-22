"""Tests for launch CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


class TestTranscriptParentDir:
    """Fix 3: transcript parent directory is created before launch."""

    def test_transcript_parent_dir_created_before_launch(self, tmp_path: Path) -> None:
        """The real launch() must create the transcript parent dir before adapter.launch()."""
        transcript = tmp_path / "deep" / "nested" / "transcript.txt"
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: claude\n")

        call_order: list[str] = []

        mock_adapter = MagicMock()

        def fake_launch(**kwargs: object) -> int:
            if transcript.parent.is_dir():
                call_order.append("launch_after_mkdir")
            else:
                call_order.append("launch_before_mkdir")
            return 0

        mock_adapter.launch.side_effect = fake_launch
        mock_adapter.capabilities.return_value = MagicMock(
            display_name="Claude Code",
            supports_initial_message=True,
        )
        mock_adapter.parse_transcript.return_value = MagicMock(total_tokens=None, session_id=None)

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
        ):
            result = runner.invoke(
                app,
                ["launch", str(tmp_path), "--tool", "claude", "--transcript", str(transcript)],
            )

        assert result.exit_code == 0, result.output
        assert call_order == ["launch_after_mkdir"]
        assert transcript.parent.is_dir()

    def test_launch_works_without_transcript(self, tmp_path: Path) -> None:
        """Launch without --transcript should not fail."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: claude\n")

        mock_adapter = MagicMock()
        mock_adapter.launch.return_value = 0
        mock_adapter.capabilities.return_value = MagicMock(
            display_name="Claude Code",
            supports_initial_message=True,
        )

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
        ):
            result = runner.invoke(app, ["launch", str(tmp_path), "--tool", "claude"])

        assert result.exit_code == 0, result.output
        mock_adapter.launch.assert_called_once()
