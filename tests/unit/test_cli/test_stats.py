"""Focused tests for `crossby stats`."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "transcripts"


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


def test_stats_displays_usage_summary_and_session_id(tmp_path) -> None:
    session_id = "11111111-1111-1111-1111-111111111111"
    transcript = tmp_path / "session.txt"
    transcript.write_text(
        "\n".join(
            [
                "Input tokens: 100",
                "Output tokens: 25",
                "Cached tokens: 5",
                "Total tokens: 130",
                f"claude --resume {session_id}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["stats", str(transcript)])

    assert result.exit_code == 0
    assert "Session Statistics" in result.output
    assert "Input tokens" in result.output
    assert "Cached tokens" in result.output
    assert session_id in result.output


def test_stats_displays_copilot_model_breakdown() -> None:
    transcript = FIXTURES / "copilot_session.txt"

    result = runner.invoke(app, ["stats", str(transcript), "--tool", "copilot"])

    assert result.exit_code == 0
    assert "Per-Model Breakdown" in result.output
    assert "gpt-4.1" in result.output
    assert "Session ID" in result.output
