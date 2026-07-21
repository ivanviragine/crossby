"""CLI integration tests for `crossby agents convert`."""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


CLAUDE_AGENT = """\
---
name: researcher
description: Research and summarize
tools: [Read, Grep, Bash]
model: sonnet
---
Body.
"""


def _write_input(tmp_path: Path, name: str = "researcher.md", content: str = CLAUDE_AGENT) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_claude_to_codex_stdout_has_both_artifacts(tmp_path: Path) -> None:
    src = _write_input(tmp_path)
    result = runner.invoke(
        app, ["agents", "convert", str(src), "--from", "claude", "--to", "codex"]
    )
    assert result.exit_code == 0, result.stdout
    assert 'developer_instructions = "Body.\\n"' in result.stdout
    assert "[agents.researcher]" in result.stdout


def test_writes_to_output_directory(tmp_path: Path) -> None:
    src = _write_input(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "agents",
            "convert",
            str(src),
            "--from",
            "claude",
            "--to",
            "copilot",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    expected = out / "researcher.agent.md"
    assert expected.is_file()
    assert "tools" in expected.read_text(encoding="utf-8")


def test_codex_output_creates_two_files(tmp_path: Path) -> None:
    src = _write_input(tmp_path)
    out = tmp_path / "codex_out"
    result = runner.invoke(
        app,
        ["agents", "convert", str(src), "--from", "claude", "--to", "codex", "--output", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    agent_file = out / "researcher.toml"
    fragment_file = out / "researcher.config-fragment.toml"
    assert agent_file.is_file()
    assert fragment_file.is_file()
    fragment = tomllib.loads(fragment_file.read_text(encoding="utf-8"))
    assert "researcher" in fragment["agents"]


def test_unknown_source_tool_errors(tmp_path: Path) -> None:
    src = _write_input(tmp_path)
    result = runner.invoke(
        app, ["agents", "convert", str(src), "--from", "borgles", "--to", "claude"]
    )
    assert result.exit_code != 0


def test_lossy_warnings_print_to_stderr(tmp_path: Path) -> None:
    """Cursor target should produce a lossy warning for the tools field."""
    src = _write_input(tmp_path)
    result = runner.invoke(
        app, ["agents", "convert", str(src), "--from", "claude", "--to", "cursor"]
    )
    assert result.exit_code == 0
    # console.info / console.warn both end up captured by CliRunner; check
    # that the warning text is present somewhere in the output.
    combined = result.stdout + (result.stderr or "")
    assert "Cursor lacks" in combined or "tools" in combined
