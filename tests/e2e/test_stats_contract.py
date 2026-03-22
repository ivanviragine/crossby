"""Deterministic contract tests for `crossby stats`."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.e2e._support import run_crossby

pytestmark = [pytest.mark.contract, pytest.mark.e2e_deterministic]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts"


def test_stats_parses_generic_transcript(e2e_context) -> None:
    transcript = FIXTURES / "generic_session.txt"

    result = run_crossby(["stats", str(transcript)], cwd=e2e_context.project, env=e2e_context.env)

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Session Statistics" in combined
    assert "Total tokens" in combined
    assert "5,678" in combined


def test_stats_uses_explicit_tool_parser(e2e_context) -> None:
    transcript = FIXTURES / "copilot_session.txt"

    result = run_crossby(
        ["stats", str(transcript), "--tool", "copilot"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "gpt-4.1" in combined
    assert "Session ID" in combined


def test_stats_missing_file_exits_cleanly(e2e_context) -> None:
    missing = e2e_context.project / "missing.txt"

    result = run_crossby(["stats", str(missing)], cwd=e2e_context.project, env=e2e_context.env)

    assert result.returncode == 1
    assert f"File not found: {missing}" in result.stderr.replace("\n", "")
