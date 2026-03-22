"""Focused tests for `crossby stats`."""

from __future__ import annotations

from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


def test_stats_rejects_unknown_tool(tmp_path) -> None:
    transcript = tmp_path / "session.txt"
    transcript.write_text("Total tokens: 123\n", encoding="utf-8")

    result = runner.invoke(app, ["stats", str(transcript), "--tool", "unknown"])

    assert result.exit_code == 1
    assert "Unknown AI tool: unknown" in result.output


def test_stats_reports_when_no_usage_data_found(tmp_path) -> None:
    transcript = tmp_path / "session.txt"
    transcript.write_text("No stats here\n", encoding="utf-8")

    result = runner.invoke(app, ["stats", str(transcript)])

    assert result.exit_code == 0
    assert "No token usage data found in transcript." in result.output
