"""Focused tests for `crossby init`."""

from __future__ import annotations

from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID

runner = CliRunner()


def test_init_uses_prompt_selection_for_default_tool(tmp_path) -> None:
    with (
        patch(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            return_value=[AIToolID.CLAUDE, AIToolID.CODEX],
        ),
        patch("crossby.ui.prompts.is_tty", return_value=True),
        patch("crossby.ui.prompts.select", return_value=1),
    ):
        result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / ".crossby.yml").read_text(encoding="utf-8"))
    assert config["ai"]["default_tool"] == "codex"
    assert "claude" in config["models"]
    assert "codex" in config["models"]
