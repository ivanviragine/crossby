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


class TestResumeFlag:
    """--resume routes to build_resume_command and run_with_transcript."""

    def _make_adapter(self, supports_resume: bool = True) -> MagicMock:
        adapter = MagicMock()
        adapter.capabilities.return_value = MagicMock(
            display_name="Claude Code",
            supports_resume=supports_resume,
            supports_initial_message=True,
        )
        adapter.build_resume_command.return_value = ["claude", "--resume", "abc-123"]
        adapter.parse_transcript.return_value = MagicMock(total_tokens=None, session_id=None)
        return adapter

    def test_resume_calls_build_resume_command(self, tmp_path: Path) -> None:
        """--resume reaches adapter.build_resume_command with the correct session ID."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: claude\n")
        mock_adapter = self._make_adapter()

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
            patch("crossby.utils.process.run_with_transcript", return_value=0) as mock_run,
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "claude", "--resume", "abc-123"]
            )

        assert result.exit_code == 0, result.output
        mock_adapter.build_resume_command.assert_called_once_with("abc-123")
        mock_run.assert_called_once()
        launched_cmd = mock_run.call_args[0][0]
        assert launched_cmd == ["claude", "--resume", "abc-123"]
        # Normal adapter.launch() must NOT be called
        mock_adapter.launch.assert_not_called()

    def test_resume_unsupported_tool_exits_1(self, tmp_path: Path) -> None:
        """--resume with a tool that lacks supports_resume exits 1."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: cursor\n")
        mock_adapter = self._make_adapter(supports_resume=False)

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("cursor", None, None, False),
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "cursor", "--resume", "any"]
            )

        assert result.exit_code == 1
        assert "does not support session resume" in result.output

    def test_resume_build_command_returns_none_exits_1(self, tmp_path: Path) -> None:
        """Adapter claims supports_resume but build_resume_command returns None → exits 1."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: claude\n")
        mock_adapter = self._make_adapter(supports_resume=True)
        mock_adapter.build_resume_command.return_value = None

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
        ):
            result = runner.invoke(
                app, ["launch", str(tmp_path), "--tool", "claude", "--resume", "abc-123"]
            )

        assert result.exit_code == 1
        assert "build_resume_command returned None" in result.output

    def test_resume_with_transcript(self, tmp_path: Path) -> None:
        """--resume + --transcript creates transcript dir and parses the transcript."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: claude\n")
        transcript = tmp_path / "deep" / "session.txt"
        mock_adapter = self._make_adapter(supports_resume=True)
        mock_adapter.parse_transcript.return_value = MagicMock(
            total_tokens=1000, session_id="abc-123"
        )
        # Make the transcript file exist so parse_transcript is triggered
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text("")

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
            patch("crossby.utils.process.run_with_transcript", return_value=0),
        ):
            result = runner.invoke(
                app,
                [
                    "launch",
                    str(tmp_path),
                    "--tool",
                    "claude",
                    "--resume",
                    "abc-123",
                    "--transcript",
                    str(transcript),
                ],
            )

        assert result.exit_code == 0, result.output
        assert transcript.parent.is_dir()
        mock_adapter.parse_transcript.assert_called_once_with(transcript)


class TestTrustedDirFlag:
    """--trusted-dir passes directories through to adapter.launch."""

    def _make_adapter(self) -> MagicMock:
        adapter = MagicMock()
        adapter.launch.return_value = 0
        adapter.capabilities.return_value = MagicMock(
            display_name="Claude Code",
            supports_resume=False,
            supports_initial_message=True,
        )
        adapter.parse_transcript.return_value = MagicMock(total_tokens=None, session_id=None)
        return adapter

    def test_single_trusted_dir_passed_to_launch(self, tmp_path: Path) -> None:
        """A single --trusted-dir value reaches adapter.launch as trusted_dirs."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: claude\n")
        mock_adapter = self._make_adapter()

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
        ):
            result = runner.invoke(
                app,
                ["launch", str(tmp_path), "--tool", "claude", "--trusted-dir", "/tmp/plan"],
            )

        assert result.exit_code == 0, result.output
        mock_adapter.launch.assert_called_once()
        _, kwargs = mock_adapter.launch.call_args
        assert kwargs.get("trusted_dirs") == ["/tmp/plan"]

    def test_multiple_trusted_dirs_passed_to_launch(self, tmp_path: Path) -> None:
        """Multiple --trusted-dir values are collected and passed as a list."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: claude\n")
        mock_adapter = self._make_adapter()

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "launch",
                    str(tmp_path),
                    "--tool",
                    "claude",
                    "--trusted-dir",
                    "/a",
                    "--trusted-dir",
                    "/b",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_adapter.launch.assert_called_once()
        _, kwargs = mock_adapter.launch.call_args
        assert set(kwargs.get("trusted_dirs", [])) == {"/a", "/b"}
