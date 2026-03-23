"""Tests for crossby sync CLI command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


def _setup_claude_source(root: Path) -> None:
    """Create a minimal Claude source config in root."""
    (root / "CLAUDE.md").write_text("# Instructions")
    skills = root / ".claude" / "skills"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# Skill")
    settings = {"permissions": {"allow": ["Bash(myapp:*)"]}}
    (root / ".claude" / "settings.json").write_text(
        json.dumps(settings), encoding="utf-8"
    )


class TestSyncDirect:
    def test_sync_from_claude_to_cursor(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = runner.invoke(app, ["sync", "--from", "claude", "--to", "cursor"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

    def test_sync_dry_run(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = runner.invoke(
            app, ["sync", "--from", "claude", "--to", "cursor", "--dry-run"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "would" in result.output.lower() or "plan" in result.output.lower()

    def test_sync_instructions_only(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = runner.invoke(
            app, ["sync", "--from", "claude", "--to", "cursor", "--instructions"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_missing_from_in_direct_mode(self) -> None:
        result = runner.invoke(app, ["sync", "--to", "cursor"])
        assert result.exit_code == 1

    def test_missing_to_and_all(self) -> None:
        result = runner.invoke(app, ["sync", "--from", "claude"])
        assert result.exit_code == 1

    def test_unknown_tool(self) -> None:
        result = runner.invoke(app, ["sync", "--from", "unknown_tool", "--to", "cursor"])
        assert result.exit_code == 1

    def test_unsupported_target_warns(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = runner.invoke(
            app, ["sync", "--from", "claude", "--to", "vscode"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    def test_multiple_to_targets(self, tmp_path: Path) -> None:
        _setup_claude_source(tmp_path)
        result = runner.invoke(
            app, ["sync", "--from", "claude", "--to", "cursor", "--to", "gemini"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0


class TestSyncHelp:
    def test_help_output(self) -> None:
        result = runner.invoke(app, ["sync", "--help"])
        assert result.exit_code == 0
        assert "--from" in result.output
        assert "--to" in result.output
        assert "--all" in result.output
        assert "--dry-run" in result.output
