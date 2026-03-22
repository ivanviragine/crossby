"""Basic CLI smoke tests for version/help and a few key error paths."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

import crossby
from crossby.cli.main import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "crossby" in result.output
        assert crossby.__version__ in result.output


class TestHelp:
    def test_root_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "crossby" in result.output
        assert "init" in result.output
        assert "launch" in result.output
        assert "stats" in result.output
        assert "convert" in result.output
        assert "config" in result.output


class TestCommandBehaviorWithoutContext:
    def test_launch_without_tool_or_detection_exits(self) -> None:
        with patch("crossby.ai_tools.base.AbstractAITool.detect_installed", return_value=[]):
            result = runner.invoke(app, ["launch"])
        assert result.exit_code == 1
        assert "No AI tool specified or detected" in result.output

    def test_config_show_without_config_exits(self) -> None:
        with runner.isolated_filesystem():
            result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 1
        assert "No .crossby.yml found" in result.output
