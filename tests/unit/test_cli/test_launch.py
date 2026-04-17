"""Tests for launch CLI command."""

from __future__ import annotations

import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


def _mock_adapter() -> MagicMock:
    """Create a standard mock adapter for launch tests."""
    mock = MagicMock()
    mock.launch.return_value = 0
    mock.capabilities.return_value = MagicMock(
        display_name="Claude Code",
        supports_initial_message=True,
    )
    mock.parse_transcript.return_value = MagicMock(total_tokens=None, session_id=None)
    return mock


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


class TestProfileFlag:
    """Test --profile flag loads settings from .crossby.yml profiles."""

    def test_profile_applies_tool_override(self, tmp_path: Path) -> None:
        config = {
            "version": 1,
            "profiles": {
                "ccyolo": {"tool": "claude", "yolo": True, "effort": "max"},
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(config))

        adapter = _mock_adapter()
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, True),
            ),
        ):
            result = runner.invoke(
                app,
                ["launch", str(tmp_path), "--profile", "ccyolo"],
            )

        assert result.exit_code == 0, result.output
        adapter.launch.assert_called_once()

    def test_unknown_profile_exits_1(self, tmp_path: Path) -> None:
        (tmp_path / ".crossby.yml").write_text("version: 1\n")
        result = runner.invoke(
            app,
            ["launch", str(tmp_path), "--profile", "nonexistent"],
        )
        assert result.exit_code == 1
        assert "Unknown profile" in result.output

    def test_cli_flags_override_profile(self, tmp_path: Path) -> None:
        config = {
            "version": 1,
            "profiles": {
                "fast": {"tool": "cursor", "model": "haiku"},
            },
        }
        (tmp_path / ".crossby.yml").write_text(yaml.dump(config))

        adapter = _mock_adapter()
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", "opus", None, False),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "launch", str(tmp_path),
                    "--profile", "fast",
                    "--tool", "claude",
                    "--model", "opus",
                ],
            )

        assert result.exit_code == 0, result.output

    def test_launch_without_config_file(self, tmp_path: Path) -> None:
        """Launch works without any .crossby.yml — auto-detects everything."""
        adapter = _mock_adapter()
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                return_value=["claude"],
            ),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
        ):
            result = runner.invoke(
                app,
                ["launch", str(tmp_path), "--tool", "claude"],
            )

        assert result.exit_code == 0, result.output
        adapter.launch.assert_called_once()
