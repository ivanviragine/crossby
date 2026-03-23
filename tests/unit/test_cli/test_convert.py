"""Focused tests for `crossby convert`."""

from __future__ import annotations

from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


def test_convert_shell_wrapper_to_canonical() -> None:
    result = runner.invoke(
        app,
        ["convert", "shell(crossby:*)", "--from", "copilot", "--to", "canonical"],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "crossby:*"


def test_convert_rejects_unknown_target_tool() -> None:
    result = runner.invoke(app, ["convert", "crossby:*", "--from", "canonical", "--to", "unknown"])

    assert result.exit_code == 1
    assert "Unknown target tool" in result.output


def test_convert_canonical_to_cursor_wrapper() -> None:
    result = runner.invoke(
        app,
        ["convert", "./scripts/check.sh:*", "--from", "canonical", "--to", "cursor"],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "Shell(./scripts/check.sh:*)"


def test_convert_rejects_unknown_source_tool() -> None:
    result = runner.invoke(app, ["convert", "crossby:*", "--from", "unknown", "--to", "claude"])

    assert result.exit_code == 1
    assert "Unknown source tool" in result.output
