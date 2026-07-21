"""Tests for launch CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml
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


class TestTranscriptRelativePath:
    """Relative --transcript path is resolved against work_dir before use."""

    def test_relative_transcript_resolved_against_work_dir(self, tmp_path: Path) -> None:
        """crossby launch /proj --transcript deep/session.txt creates /proj/deep/ and passes
        the absolute path to adapter.launch."""
        work_dir = tmp_path / "proj"
        work_dir.mkdir()
        (work_dir / ".crossby.yml").write_text("version: 1\nai:\n  default_tool: claude\n")

        mock_adapter = MagicMock()
        mock_adapter.launch.return_value = 0
        mock_adapter.capabilities.return_value = MagicMock(
            display_name="Claude Code",
            supports_initial_message=True,
            supports_trusted_dirs=False,
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
                ["launch", str(work_dir), "--tool", "claude", "--transcript", "deep/session.txt"],
            )

        assert result.exit_code == 0, result.output
        expected_transcript = work_dir / "deep" / "session.txt"
        assert expected_transcript.parent.is_dir()
        _, kwargs = mock_adapter.launch.call_args
        assert kwargs.get("transcript_path") == expected_transcript


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

    def _make_adapter(self, supports_trusted_dirs: bool = True) -> MagicMock:
        adapter = MagicMock()
        adapter.launch.return_value = 0
        adapter.capabilities.return_value = MagicMock(
            display_name="Claude Code",
            supports_resume=False,
            supports_initial_message=True,
            supports_trusted_dirs=supports_trusted_dirs,
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

    def test_trusted_dir_unsupported_tool_exits_1(self, tmp_path: Path) -> None:
        """--trusted-dir with a tool that lacks supports_trusted_dirs exits 1."""
        config_file = tmp_path / ".crossby.yml"
        config_file.write_text("version: 1\nai:\n  default_tool: cursor\n")
        mock_adapter = self._make_adapter(supports_trusted_dirs=False)

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=mock_adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("cursor", None, None, False),
            ),
        ):
            result = runner.invoke(
                app,
                ["launch", str(tmp_path), "--tool", "cursor", "--trusted-dir", "/tmp/plan"],
            )

        assert result.exit_code == 1
        assert "does not support --trusted-dir" in result.output


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
                    "launch",
                    str(tmp_path),
                    "--profile",
                    "fast",
                    "--tool",
                    "claude",
                    "--model",
                    "opus",
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


class TestPlanFlag:
    """`--plan` forwards plan_mode=True to the adapter when supported."""

    def _launch_with_plan(
        self,
        tmp_path: Path,
        tool: str,
        *,
        supports_plan_mode: bool,
    ) -> tuple[MagicMock, Any]:
        (tmp_path / ".crossby.yml").write_text(f"version: 1\nai:\n  default_tool: {tool}\n")
        adapter = MagicMock()
        adapter.launch.return_value = 0
        adapter.capabilities.return_value = MagicMock(
            display_name=tool.title(),
            supports_initial_message=True,
            supports_trusted_dirs=False,
            supports_plan_mode=supports_plan_mode,
        )
        adapter.parse_transcript.return_value = MagicMock(total_tokens=None, session_id=None)

        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=(tool, None, None, False),
            ),
        ):
            result = runner.invoke(app, ["launch", str(tmp_path), "--tool", tool, "--plan"])
        return adapter, result

    def test_plan_flag_forwards_to_copilot(self, tmp_path: Path) -> None:
        adapter, result = self._launch_with_plan(tmp_path, "copilot", supports_plan_mode=True)
        assert result.exit_code == 0, result.output
        _, kwargs = adapter.launch.call_args
        assert kwargs.get("plan_mode") is True

    def test_plan_flag_forwards_to_claude(self, tmp_path: Path) -> None:
        adapter, result = self._launch_with_plan(tmp_path, "claude", supports_plan_mode=True)
        assert result.exit_code == 0, result.output
        _, kwargs = adapter.launch.call_args
        assert kwargs.get("plan_mode") is True

    def test_plan_flag_forwards_to_antigravity_cli(self, tmp_path: Path) -> None:
        adapter, result = self._launch_with_plan(
            tmp_path, "antigravity-cli", supports_plan_mode=True
        )
        assert result.exit_code == 0, result.output
        _, kwargs = adapter.launch.call_args
        assert kwargs.get("plan_mode") is True

    def test_plan_flag_rejected_when_unsupported(self, tmp_path: Path) -> None:
        adapter, result = self._launch_with_plan(tmp_path, "codex", supports_plan_mode=False)
        assert result.exit_code == 1, result.output
        adapter.launch.assert_not_called()

    def test_no_plan_flag_forwards_false(self, tmp_path: Path) -> None:
        (tmp_path / ".crossby.yml").write_text("version: 1\nai:\n  default_tool: claude\n")
        adapter = _mock_adapter()
        adapter.capabilities.return_value = MagicMock(
            display_name="Claude",
            supports_initial_message=True,
            supports_trusted_dirs=False,
            supports_plan_mode=True,
        )
        with (
            patch("crossby.ai_tools.base.AbstractAITool.get", return_value=adapter),
            patch(
                "crossby.services.ai_resolution.confirm_ai_selection",
                return_value=("claude", None, None, False),
            ),
        ):
            result = runner.invoke(app, ["launch", str(tmp_path), "--tool", "claude"])

        assert result.exit_code == 0, result.output
        _, kwargs = adapter.launch.call_args
        assert kwargs.get("plan_mode") is False
