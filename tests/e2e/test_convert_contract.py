"""Deterministic contract tests for `crossby convert`."""

from __future__ import annotations

import pytest
from tests.e2e._support import run_crossby

pytestmark = [pytest.mark.contract, pytest.mark.e2e_deterministic]


def test_convert_canonical_to_copilot(e2e_context) -> None:
    result = run_crossby(
        ["convert", "crossby:*", "--from", "canonical", "--to", "copilot"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "shell(crossby:*)"


def test_convert_claude_to_cursor_round_trip_shape(e2e_context) -> None:
    result = run_crossby(
        ["convert", "Bash(myapp:*)", "--from", "claude", "--to", "cursor"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Shell(myapp:*)"


def test_convert_rejects_unknown_source_tool(e2e_context) -> None:
    result = run_crossby(
        ["convert", "myapp:*", "--from", "unknown", "--to", "claude"],
        cwd=e2e_context.project,
        env=e2e_context.env,
    )

    assert result.returncode == 1
    assert "Unknown source tool" in result.stderr
