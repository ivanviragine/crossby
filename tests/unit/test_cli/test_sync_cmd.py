"""Integration tests for the `crossby sync` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app


runner = CliRunner()


@pytest.fixture()
def project_with_config(tmp_path: Path) -> Path:
    """A project directory with a .crossby.yml and allowed_commands."""
    config = tmp_path / ".crossby.yml"
    config.write_text(
        "version: 1\n"
        "ai:\n"
        "  default_tool: claude\n"
        "permissions:\n"
        "  allowed_commands:\n"
        "    - 'myapp:*'\n"
        "sync:\n"
        "  auto: true\n"
        "  tools: []\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _patch_cursor_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect Cursor global config path so tests don't touch the real one."""
    fake = tmp_path / ".cursor" / "cli-config.json"
    monkeypatch.setattr("crossby.sync.permissions._GLOBAL_CURSOR_CONFIG_PATH", fake)


class TestSyncCommandPermissions:
    def test_sync_permissions_creates_claude_settings(
        self, project_with_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """crossby sync permissions creates .claude/settings.json for Claude."""
        monkeypatch.chdir(project_with_config)

        # Patch detect_installed to return only Claude
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude"],
            )
            result = runner.invoke(
                app,
                ["sync", "permissions", "--path", str(project_with_config)],
            )

        assert result.exit_code == 0, result.output
        settings = project_with_config / ".claude" / "settings.json"
        assert settings.exists()
        data = json.loads(settings.read_text())
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_sync_all_creates_files_for_all_installed(
        self, project_with_config: Path
    ) -> None:
        """crossby sync (no concern) runs all writers for installed tools."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                ["sync", "--path", str(project_with_config)],
            )

        assert result.exit_code == 0, result.output

        claude_settings = project_with_config / ".claude" / "settings.json"
        cursor_config = project_with_config / ".cursor" / "cli.json"
        assert claude_settings.exists()
        assert cursor_config.exists()

    def test_sync_unknown_concern_exits_1(self, project_with_config: Path) -> None:
        result = runner.invoke(
            app,
            ["sync", "nonexistent", "--path", str(project_with_config)],
        )
        assert result.exit_code == 1
        assert "Unknown concern" in result.output

    def test_sync_unknown_tool_exits_1(self, project_with_config: Path) -> None:
        result = runner.invoke(
            app,
            ["sync", "--tool", "nonexistent", "--path", str(project_with_config)],
        )
        assert result.exit_code == 1
        assert "Unknown tool" in result.output

    def test_sync_dry_run_does_not_write(self, project_with_config: Path) -> None:
        """--dry-run reports changes without writing files."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude"],
            )
            result = runner.invoke(
                app,
                ["sync", "--dry-run", "--path", str(project_with_config)],
            )

        assert result.exit_code == 0, result.output
        assert not (project_with_config / ".claude" / "settings.json").exists()

    def test_sync_tool_filter_claude(self, project_with_config: Path) -> None:
        """--tool claude only runs Claude writer."""
        result = runner.invoke(
            app,
            ["sync", "--tool", "claude", "--path", str(project_with_config)],
        )
        assert result.exit_code == 0, result.output
        settings = project_with_config / ".claude" / "settings.json"
        assert settings.exists()
        data = json.loads(settings.read_text())
        assert "Bash(myapp:*)" in data["permissions"]["allow"]

    def test_sync_idempotent(self, project_with_config: Path) -> None:
        """Running sync twice leaves files in the expected state."""
        for _ in range(2):
            runner.invoke(
                app,
                ["sync", "--tool", "claude", "--path", str(project_with_config)],
            )
        settings = project_with_config / ".claude" / "settings.json"
        data = json.loads(settings.read_text())
        assert data["permissions"]["allow"].count("Bash(myapp:*)") == 1
