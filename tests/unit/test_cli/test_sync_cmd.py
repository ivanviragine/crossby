"""Integration tests for the `crossby sync` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncData

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
                    "sync",
                    "permissions",
                    "--from",
                    "claude",
                    "--to",
                    "cursor",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )

        assert result.exit_code == 0, result.output
        cursor_config = project_with_claude_perms / ".cursor" / "cli.json"
        assert cursor_config.exists()
        data = json.loads(cursor_config.read_text())
        assert "Shell(myapp:*)" in data["permissions"]["allow"]

    def test_sync_all_from_claude(self, project_with_claude_perms: Path) -> None:
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

    def test_sync_dry_run_does_not_write(self, project_with_claude_perms: Path) -> None:
        """--dry-run reports changes without writing files."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "permissions",
                    "--from",
                    "claude",
                    "--to",
                    "cursor",
                    "--dry-run",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )

        assert result.exit_code == 0, result.output
        assert not (project_with_claude_perms / ".cursor" / "cli.json").exists()

    def test_sync_idempotent(self, project_with_claude_perms: Path) -> None:
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
                        "sync",
                        "permissions",
                        "--from",
                        "claude",
                        "--to",
                        "cursor",
                        "--path",
                        str(project_with_claude_perms),
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
    def test_sync_skills_from_claude_to_cursor(self, project_with_claude_skills: Path) -> None:
        """crossby sync skills --from claude --to cursor creates symlink."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "skills",
                    "--from",
                    "claude",
                    "--to",
                    "cursor",
                    "--path",
                    str(project_with_claude_skills),
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
                    "sync",
                    "skills",
                    "--from",
                    "claude",
                    "--path",
                    str(project_with_claude_skills),
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
                    "sync",
                    "skills",
                    "--from",
                    "claude",
                    "--to",
                    "cursor",
                    "--dry-run",
                    "--path",
                    str(project_with_claude_skills),
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
                        "sync",
                        "skills",
                        "--from",
                        "claude",
                        "--to",
                        "cursor",
                        "--path",
                        str(project_with_claude_skills),
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


class TestSyncDefaultsBypassWizard:
    """Sync respects ``sync_defaults`` from ``.crossby.yml`` without the wizard.

    Regression for #44 — previously, plain ``crossby sync`` with a config
    default fell through to the per-concern wizard and synced **all** tools
    instead of using the configured source/target/concern.
    """

    def test_config_defaults_drive_non_interactive_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_yaml = (
            "version: 1\nsync_defaults:\n"
            "  from: cursor\n  to: antigravity-cli\n  concern: permissions\n"
        )
        (tmp_path / ".crossby.yml").write_text(config_yaml, encoding="utf-8")

        captured: dict[str, Any] = {}

        def fake_build_sync_data(project_root: Path, from_tool: AIToolID | None = None) -> SyncData:
            captured["build_from"] = from_tool
            captured["build_root"] = project_root
            return SyncData()

        def fake_run_sync(
            data: SyncData,
            project_root: Path,
            **kwargs: Any,
        ) -> list[Any]:
            captured["run_kwargs"] = kwargs
            return []

        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(
                lambda _cls: [AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.ANTIGRAVITY_CLI]
            ),
        )
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr("crossby.sync.readers.build_sync_data", fake_build_sync_data)
        monkeypatch.setattr("crossby.sync.run_sync", fake_run_sync)

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert captured["build_from"] is AIToolID.CURSOR
        assert captured["run_kwargs"]["concern"] is SyncConcern.PERMISSIONS
        assert captured["run_kwargs"]["installed_tools"] == [AIToolID.ANTIGRAVITY_CLI]
        assert captured["run_kwargs"]["tool_id"] is AIToolID.ANTIGRAVITY_CLI


class TestValidateTarget:
    """``crossby sync --validate-target`` runs every validator without writing."""

    def test_clean_project_exit_zero(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Project\nBe helpful.\n", encoding="utf-8")
        result = runner.invoke(
            app,
            ["sync", "--validate-target", "--path", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        # The OK rows should appear in the output table.
        assert "ok" in result.output

    def test_invalid_codex_toml_exit_one(self, tmp_path: Path) -> None:
        codex = tmp_path / ".codex" / "config.toml"
        codex.parent.mkdir()
        codex.write_text("[[ broken", encoding="utf-8")
        result = runner.invoke(
            app,
            ["sync", "--validate-target", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1, result.output

    def test_missing_skill_field_exit_one(self, tmp_path: Path) -> None:
        skill_md = tmp_path / ".claude" / "skills" / "broken" / "SKILL.md"
        skill_md.parent.mkdir(parents=True)
        # Frontmatter without name/description.
        skill_md.write_text("---\nfoo: bar\n---\nBody.", encoding="utf-8")
        result = runner.invoke(
            app,
            ["sync", "--validate-target", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1, result.output

    def test_empty_project_no_findings(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["sync", "--validate-target", "--path", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "Nothing to validate" in result.output


class TestPlanAndDoctor:
    """``crossby sync --plan`` and ``--doctor`` are pre-write inspection modes."""

    def test_mutually_exclusive_with_validate_target(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "sync",
                "--plan",
                "--validate-target",
                "--path",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_plan_with_no_source_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No tools detected → auto-detect can't resolve a source.
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: [],
        )
        result = runner.invoke(
            app,
            ["sync", "--plan", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "needs a source tool" in result.output

    def test_plan_writes_nothing(
        self,
        project_with_claude_perms: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(project_with_claude_perms)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "--plan",
                    "--from",
                    "claude",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Migration plan" in result.output
        # No cursor allowlist file should have been written.
        assert not (project_with_claude_perms / ".cursor" / "cli.json").exists()

    def test_doctor_renders_readiness(
        self,
        project_with_claude_perms: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(project_with_claude_perms)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "--doctor",
                    "--from",
                    "claude",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )
        # readiness=high means exit 0; medium/low may or may not — just verify the section.
        assert "Crossby doctor" in result.output
        assert "readiness:" in result.output


class TestStrategyAndReportFormatValidation:
    """``--strategy`` and ``--report-format`` reject unknown values."""

    def test_invalid_strategy_exits_one(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["sync", "--strategy", "wat", "--from", "claude", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "--strategy" in result.output

    def test_invalid_report_format_exits_one(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "sync",
                "--report-format",
                "yaml",
                "--from",
                "claude",
                "--path",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 1
        assert "--report-format" in result.output

    def test_strategy_translate_threads_to_skills_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--strategy translate`` flips skills_strategy on the run."""
        # Set up a Claude skill so the skills writer has work.
        skill = tmp_path / ".claude" / "skills" / "my-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: x\nallowed-tools:\n  - Read\n---\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "codex"],
        )
        result = runner.invoke(
            app,
            [
                "sync",
                "skills",
                "--from",
                "claude",
                "--strategy",
                "translate",
                "--path",
                str(tmp_path),
            ],
        )
        # Translate ran end-to-end and produced the codex skill copy with a
        # manual-fix block (Codex doesn't honour Claude's allowed-tools).
        codex_skill = tmp_path / ".agents" / "skills" / "my-skill" / "SKILL.md"
        assert result.exit_code == 0, result.output
        assert codex_skill.is_file()
        assert "<!-- crossby:manual-fix:start -->" in codex_skill.read_text(encoding="utf-8")


class TestPersistReportGate:
    """The persistent report file is only written for real (non-dry-run) syncs."""

    def test_dry_run_does_not_persist_report(
        self,
        project_with_claude_perms: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(project_with_claude_perms)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            runner.invoke(
                app,
                [
                    "sync",
                    "--from",
                    "claude",
                    "--dry-run",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )
        # No .crossby/sync-report.md should exist after a dry-run.
        assert not (project_with_claude_perms / ".crossby" / "sync-report.md").exists()

    def test_real_run_writes_report(
        self,
        project_with_claude_perms: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(project_with_claude_perms)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            runner.invoke(
                app,
                [
                    "sync",
                    "--from",
                    "claude",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )
        assert (project_with_claude_perms / ".crossby" / "sync-report.md").is_file()

    def test_no_persist_report_skips_file(
        self,
        project_with_claude_perms: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(project_with_claude_perms)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            runner.invoke(
                app,
                [
                    "sync",
                    "--from",
                    "claude",
                    "--no-persist-report",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )
        assert not (project_with_claude_perms / ".crossby" / "sync-report.md").exists()


class TestWizardScanShowsPlugins:
    """The wizard scan output should list a Plugins row when plugins exist."""

    def test_plugin_dir_appears_in_scan_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Plugin source + a Claude rules file so the wizard doesn't skip
        # outright with "no tool configs found to sync".
        (tmp_path / "CLAUDE.md").write_text("# x\n", encoding="utf-8")
        (tmp_path / ".claude" / "plugins" / "team-macros").mkdir(parents=True)

        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )
        # Send empty stdin so the wizard exits at the first prompt instead of
        # blocking. We only care about the scan output that prints first.
        result = runner.invoke(
            app,
            ["sync", "--path", str(tmp_path)],
            input="\n",
        )
        # Don't assert exit_code (wizard may abort on EOF); the scan section
        # must mention Plugins regardless of how the run ends.
        assert "Plugins:" in result.output


class TestPlanDoctorNoTargets:
    """When only the source tool is installed, --plan/--doctor warns clearly
    instead of just saying "no sync rows produced"."""

    def test_plan_warns_when_no_target_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "CLAUDE.md").write_text("# x\n", encoding="utf-8")
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude"],
        )
        result = runner.invoke(
            app,
            ["sync", "--plan", "--from", "claude", "--path", str(tmp_path)],
        )
        # Run still succeeds (plugins still discovered, etc.) but the warning
        # fires so the user knows why their plan is empty.
        assert "No target tools detected" in result.output

    def test_doctor_warns_when_no_target_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude"],
        )
        result = runner.invoke(
            app,
            ["sync", "--doctor", "--from", "claude", "--path", str(tmp_path)],
        )
        assert "No target tools detected" in result.output


class TestSyncMalformedConfig:
    """Malformed ``.crossby.yml`` should surface a clean error, not a traceback."""

    def test_malformed_config_exits_with_clean_error(self, tmp_path: Path) -> None:
        (tmp_path / ".crossby.yml").write_text(
            "version: 1\nai:\n  - this is malformed\n", encoding="utf-8"
        )
        result = runner.invoke(
            app,
            ["sync", "--plan", "--from", "claude", "--path", str(tmp_path)],
        )
        assert result.exit_code == 1
        # No Python traceback / no ConfigError class name leaking through.
        assert "Traceback" not in result.output
        assert "ConfigError" not in result.output
        # The error message itself does land.
        assert "ai" in result.output and "mapping" in result.output
