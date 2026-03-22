"""Tests for subprocess helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crossby.utils.process import CommandError, run, run_silent, run_with_transcript


class TestRun:
    def test_success(self) -> None:
        result = run([sys.executable, "-c", "print('hello')"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_failure_raises(self) -> None:
        with pytest.raises(CommandError) as exc_info:
            run([sys.executable, "-c", "import sys; sys.exit(7)"])
        assert exc_info.value.returncode == 7

    def test_failure_no_check(self) -> None:
        result = run([sys.executable, "-c", "import sys; sys.exit(9)"], check=False)
        assert result.returncode == 9

    def test_command_not_found(self) -> None:
        with pytest.raises(CommandError) as exc_info:
            run(["nonexistent_command_xyz"])
        assert exc_info.value.returncode == 127


class TestRunSilent:
    def test_success(self) -> None:
        assert run_silent([sys.executable, "-c", "import sys; sys.exit(0)"]) is True

    def test_failure(self) -> None:
        assert run_silent([sys.executable, "-c", "import sys; sys.exit(5)"]) is False


class TestRunWithTranscript:
    def test_no_transcript_path_runs_command_directly(self) -> None:
        with patch("crossby.utils.process.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = run_with_transcript(["echo", "hi"], transcript_path=None)
        assert result == 0
        mock_run.assert_called_once_with(["echo", "hi"], cwd=None)

    def test_script_not_found_falls_back(self, tmp_path: Path) -> None:
        transcript = tmp_path / ".transcript"
        with (
            patch("crossby.utils.process.shutil.which", return_value=None),
            patch("crossby.utils.process.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = run_with_transcript(["echo", "hi"], transcript_path=transcript)
        assert result == 0
        mock_run.assert_called_once_with(["echo", "hi"], cwd=None)

    def test_gnu_script_linux_syntax(self, tmp_path: Path) -> None:
        transcript = tmp_path / ".transcript"
        with (
            patch("crossby.utils.process.shutil.which", return_value="/usr/bin/script"),
            patch("crossby.utils.process.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=0),
            ]
            result = run_with_transcript(
                ["claude", "--permission-mode", "plan"],
                transcript_path=transcript,
            )
        assert result == 0
        actual_cmd = mock_run.call_args_list[-1].args[0]
        assert actual_cmd[0:4] == ["script", "-q", "-e", "-c"]
        assert "claude" in actual_cmd[4]
        assert "--permission-mode" in actual_cmd[4]
        assert actual_cmd[5] == str(transcript)

    def test_bsd_script_macos_syntax(self, tmp_path: Path) -> None:
        transcript = tmp_path / ".transcript"
        with (
            patch("crossby.utils.process.shutil.which", return_value="/usr/bin/script"),
            patch("crossby.utils.process.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                MagicMock(returncode=1),
                MagicMock(returncode=0),
            ]
            result = run_with_transcript(
                ["claude", "--permission-mode", "plan"],
                transcript_path=transcript,
            )
        assert result == 0
        assert mock_run.call_args_list[-1].args[0] == [
            "script",
            "-q",
            str(transcript),
            "claude",
            "--permission-mode",
            "plan",
        ]

    def test_cwd_is_passed_through(self, tmp_path: Path) -> None:
        cwd = tmp_path / "work"
        cwd.mkdir()
        with (
            patch("crossby.utils.process.shutil.which", return_value=None),
            patch("crossby.utils.process.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            run_with_transcript(["true"], transcript_path=None, cwd=cwd)
        mock_run.assert_called_once_with(["true"], cwd=cwd)

    def test_returns_script_exit_code(self, tmp_path: Path) -> None:
        transcript = tmp_path / ".transcript"
        with (
            patch("crossby.utils.process.shutil.which", return_value="/usr/bin/script"),
            patch("crossby.utils.process.subprocess.run") as mock_run,
        ):
            mock_run.side_effect = [
                MagicMock(returncode=1),
                MagicMock(returncode=42),
            ]
            result = run_with_transcript(["somecommand"], transcript_path=transcript)
        assert result == 42
