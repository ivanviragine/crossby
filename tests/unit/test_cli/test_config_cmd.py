"""Focused tests for `crossby config show`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.config import (
    AIConfig,
    CommandConfig,
    ComplexityModelMapping,
    CrossbyConfig,
    PermissionsConfig,
)

runner = CliRunner()


def test_config_show_displays_models_and_permissions() -> None:
    config = CrossbyConfig(
        ai=AIConfig(default_tool="claude", default_model="claude-sonnet-4.6", effort="medium"),
        models={"claude": ComplexityModelMapping(medium="claude-sonnet-4.6")},
        permissions=PermissionsConfig(allowed_commands=["git:*"]),
    )
    config_path = Path("project") / ".crossby.yml"

    with (
        patch("crossby.config.loader.find_config_file", return_value=config_path),
        patch("crossby.config.loader.load_config", return_value=config),
    ):
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "CROSSBY Configuration" in result.output
    assert str(config_path) in result.output
    assert "Default tool" in result.output
    assert "Models: claude" in result.output
    assert "git:*" in result.output


def test_config_show_displays_command_overrides() -> None:
    config = CrossbyConfig(
        ai=AIConfig(
            default_tool="claude",
            yolo=False,
            commands={
                "plan": CommandConfig(
                    tool="claude",
                    model="claude-opus-4.6",
                    effort="high",
                    yolo=True,
                )
            },
        ),
    )

    with (
        patch("crossby.config.loader.find_config_file", return_value=Path(".crossby.yml")),
        patch("crossby.config.loader.load_config", return_value=config),
    ):
        result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "YOLO mode" in result.output
    assert "off" in result.output
    assert "Command Overrides" in result.output
    assert "tool=claude" in result.output
    assert "model=claude-opus-4.6" in result.output
    assert "effort=high" in result.output
    assert "yolo=on" in result.output
