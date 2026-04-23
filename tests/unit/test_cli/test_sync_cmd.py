"""Integration tests for the `crossby sync` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


@pytest.fixture()
def project_with_claude_perms(tmp_path: Path) -> Path:
    """A project directory with Claude allowlist (source for permissions sync)."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps({"permissions": {"allow": ["Bash(myapp:*)"]}}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _patch_cursor_global(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect Cursor global config path so tests don't touch the real one."""
    fake = tmp_path / ".cursor" / "cli-config.json"
    monkeypatch.setattr("crossby.sync.permissions._GLOBAL_CURSOR_CONFIG_PATH", fake)


class TestSyncCommandPermissions:
    def test_sync_permissions_from_claude(
        self, project_with_claude_perms: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """crossby sync permissions --from claude creates Cursor config."""
        monkeypatch.chdir(project_with_claude_perms)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync", "permissions",
                    "--from", "claude",
                    "--to", "cursor",
                    "--path", str(project_with_claude_perms),
                ],
            )

        assert result.exit_code == 0, result.output
        cursor_config = project_with_claude_perms / ".cursor" / "cli.json"
        assert cursor_config.exists()
        data = json.loads(cursor_config.read_text())
        assert "Shell(myapp:*)" in data["permissions"]["allow"]

    def test_sync_all_from_claude(
        self, project_with_claude_perms: Path
    ) -> None:
        """crossby sync --from claude runs all concerns for installed tools."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                ["sync", "--from", "claude", "--path", str(project_with_claude_perms)],
            )

        assert result.exit_code == 0, result.output

    def test_sync_unknown_concern_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["sync", "nonexistent", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "Unknown concern" in result.output

    def test_sync_unknown_from_tool_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["sync", "--from", "nonexistent", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "Unknown tool" in result.output

    def test_sync_unknown_to_tool_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["sync", "--from", "claude", "--to", "nonexistent", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "Unknown tool" in result.output

    def test_sync_dry_run_does_not_write(
        self, project_with_claude_perms: Path
    ) -> None:
        """--dry-run reports changes without writing files."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync", "permissions",
                    "--from", "claude",
                    "--to", "cursor",
                    "--dry-run",
                    "--path", str(project_with_claude_perms),
                ],
            )

        assert result.exit_code == 0, result.output
        assert not (project_with_claude_perms / ".cursor" / "cli.json").exists()

    def test_sync_idempotent(
        self, project_with_claude_perms: Path
    ) -> None:
        """Running sync twice leaves files in the expected state."""
        for _ in range(2):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    "crossby.ai_tools.base.AbstractAITool.detect_installed",
                    lambda: ["claude", "cursor"],
                )
                runner.invoke(
                    app,
                    [
                        "sync", "permissions",
                        "--from", "claude",
                        "--to", "cursor",
                        "--path", str(project_with_claude_perms),
                    ],
                )
        cursor_config = project_with_claude_perms / ".cursor" / "cli.json"
        data = json.loads(cursor_config.read_text())
        assert data["permissions"]["allow"].count("Shell(myapp:*)") == 1


@pytest.fixture()
def project_with_claude_skills(tmp_path: Path) -> Path:
    """A project directory with Claude skills as source."""
    skills_dir = tmp_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    skill = skills_dir / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# my-skill\n", encoding="utf-8")
    return tmp_path


class TestSyncCommandSkills:
    def test_sync_skills_from_claude_to_cursor(
        self, project_with_claude_skills: Path
    ) -> None:
        """crossby sync skills --from claude --to cursor creates symlink."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync", "skills",
                    "--from", "claude",
                    "--to", "cursor",
                    "--path", str(project_with_claude_skills),
                ],
            )

        assert result.exit_code == 0, result.output
        cursor_skills = project_with_claude_skills / ".cursor" / "skills"
        assert cursor_skills.is_symlink()

    def test_sync_skills_concern_filter_non_interactive(
        self, project_with_claude_skills: Path
    ) -> None:
        """--concern skills filters to only skills writers."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync", "skills",
                    "--from", "claude",
                    "--path", str(project_with_claude_skills),
                ],
            )

        assert result.exit_code == 0, result.output
        assert (project_with_claude_skills / ".cursor" / "skills").is_symlink()

    def test_sync_skills_dry_run(self, project_with_claude_skills: Path) -> None:
        """--dry-run does not create skill symlinks."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync", "skills",
                    "--from", "claude",
                    "--to", "cursor",
                    "--dry-run",
                    "--path", str(project_with_claude_skills),
                ],
            )

        assert result.exit_code == 0, result.output
        assert not (project_with_claude_skills / ".cursor" / "skills").exists()

    def test_sync_skills_idempotent(self, project_with_claude_skills: Path) -> None:
        """Running sync skills twice is a no-op on second run."""
        for _ in range(2):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    "crossby.ai_tools.base.AbstractAITool.detect_installed",
                    lambda: ["claude", "cursor"],
                )
                result = runner.invoke(
                    app,
                    [
                        "sync", "skills",
                        "--from", "claude",
                        "--to", "cursor",
                        "--path", str(project_with_claude_skills),
                    ],
                )
            assert result.exit_code == 0, result.output

        target = project_with_claude_skills / ".cursor" / "skills"
        assert target.is_symlink()

    def test_wizard_shows_skills_in_scan_output(
        self, project_with_claude_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wizard mode shows Skills line in scan output."""
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )
        monkeypatch.setattr("crossby.ui.prompts.confirm", lambda *a, **kw: False)
        result = runner.invoke(
            app,
            ["sync", "--path", str(project_with_claude_skills)],
        )
        assert result.exit_code == 0, result.output
        assert "Skills:" in result.output
