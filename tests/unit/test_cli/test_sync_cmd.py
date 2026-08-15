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
from crossby.sync.ownership import LEDGER_PATH, OwnershipLedger, load_ledger, save_ledger
from crossby.sync.readers import ProjectScan
from crossby.sync.report import REPORT_PATH

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

        def fake_build_sync_data(
            project_root: Path,
            from_tool: AIToolID | None = None,
            *,
            include_user_scope: bool = False,
        ) -> SyncData:
            captured["build_from"] = from_tool
            captured["build_root"] = project_root
            captured["include_user_scope"] = include_user_scope
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
            classmethod(lambda _cls: [AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.ANTIGRAVITY_CLI]),
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


class TestMissingReaderMessage:
    """A ``--from`` tool with no reader for the requested concern must warn
    rather than silently sync nothing (permissions from codex/copilot/agy)."""

    @pytest.mark.parametrize("source", ["codex", "copilot", "antigravity-cli"])
    def test_direct_sync_permissions_no_reader_warns(self, tmp_path: Path, source: str) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor", "codex", "copilot", "antigravity-cli"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "permissions",
                    "--from",
                    source,
                    "--to",
                    "cursor",
                    "--path",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "no reader for" in result.output
        assert "permissions" in result.output

    def test_plan_permissions_no_reader_warns(self, tmp_path: Path) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["codex", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "permissions",
                    "--plan",
                    "--from",
                    "codex",
                    "--path",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "no reader for" in result.output

    def test_hooks_from_codex_does_not_warn(self, tmp_path: Path) -> None:
        # Hooks cover all five tools, so codex must NOT trigger the message.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["codex", "cursor"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "hooks",
                    "--from",
                    "codex",
                    "--to",
                    "cursor",
                    "--path",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "no reader for" not in result.output

    def test_no_reader_does_not_revoke_existing_permissions(
        self, project_with_claude_perms: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A no-reader source must SKIP the concern, not revoke it.

        Regression for the review P1: an empty source is read by the ownership
        diff as "source removed everything", so a plain warn-and-continue would
        delete permissions crossby previously synced to the target. Skipping the
        concern (skip_concerns) leaves the ledger and the target untouched.
        """
        monkeypatch.chdir(project_with_claude_perms)
        cursor_config = project_with_claude_perms / ".cursor" / "cli.json"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor", "codex"],
            )
            # 1) Seed: sync claude permissions → cursor, recording ownership.
            first = runner.invoke(
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
            assert first.exit_code == 0, first.output
            allow = json.loads(cursor_config.read_text())["permissions"]["allow"]
            assert "Shell(myapp:*)" in allow

            # 2) Now sync permissions from codex (no reader): must warn and skip,
            #    NOT revoke the previously-synced permission.
            second = runner.invoke(
                app,
                [
                    "sync",
                    "permissions",
                    "--from",
                    "codex",
                    "--to",
                    "cursor",
                    "--path",
                    str(project_with_claude_perms),
                ],
            )
            assert second.exit_code == 0, second.output
            assert "no reader for" in second.output

        # The previously-synced permission survives — nothing was revoked.
        allow_after = json.loads(cursor_config.read_text())["permissions"]["allow"]
        assert "Shell(myapp:*)" in allow_after


class TestDoctorSurfacesMalformedHooks:
    """``--doctor`` reports invalid JSON in the hooks files crossby writes,
    including the newly-registered ``.codex`` / ``.agents`` locations."""

    @pytest.mark.parametrize("rel", [".codex/hooks.json", ".agents/hooks.json"])
    def test_doctor_reports_malformed_hooks(self, tmp_path: Path, rel: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True)
        path.write_text("{ broken", encoding="utf-8")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["claude", "cursor"],
            )
            result = runner.invoke(
                app,
                ["sync", "--doctor", "--from", "claude", "--path", str(tmp_path)],
            )
        assert result.exit_code == 1, result.output
        assert "hooks.json" in result.output
        assert "invalid JSON" in result.output


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


# ---------------------------------------------------------------------------
# Interactive-wizard revocation (Issue #111, follow-up to #100)
# ---------------------------------------------------------------------------


_GUARD_HOOK = ("pre_tool_use", "guard")
_OWNED_HOOK = ("pre_tool_use", "owned")


def _seed_hook_ledger(project_root: Path, tool: AIToolID, pairs: set[tuple[str, str]]) -> None:
    """Seed ``.crossby/owned.json`` so crossby owns *pairs* of hooks for *tool*."""
    ledger = OwnershipLedger()
    ledger.record_hooks(tool, pairs)
    save_ledger(project_root, ledger)


def _write_claude_settings(project_root: Path, settings: dict[str, Any]) -> Path:
    path = project_root / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path


def _claude_hook_commands(project_root: Path) -> set[str]:
    settings = json.loads((project_root / ".claude" / "settings.json").read_text())
    return {
        inner["command"]
        for entry in settings.get("hooks", {}).get("PreToolUse", [])
        for inner in entry["hooks"]
    }


def _empty_project_scan(*_args: Any, **_kwargs: Any) -> ProjectScan:
    """A scan that discovers nothing — simulates a concern that has emptied
    across the *whole environment* while the ledger + a target file still carry
    crossby's entry."""
    return ProjectScan(installed_tools=[])


class TestWizardRevocation:
    """The wizard dispatches revocable concerns on *data OR ownership*.

    When a concern has emptied across the environment (nothing discovered) but
    the ledger still owns entries, the wizard runs ``run_sync`` so crossby can
    revoke what it wrote earlier — the follow-up behaviour from Issue #111.
    """

    def test_wizard_migrates_codex_alias_without_discovered_or_owned_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tomllib

        config = tmp_path / ".codex" / "config.toml"
        config.parent.mkdir(parents=True)
        config.write_text(
            "# preserve config\n[features]\ncodex_hooks = false\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["codex"],
        )
        monkeypatch.setattr("crossby.sync.readers.scan_project", _empty_project_scan)

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        text = config.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert parsed["features"]["hooks"] is False
        assert "codex_hooks" not in parsed["features"]
        assert "# preserve config" in text
        assert not (tmp_path / ".codex" / "hooks.json").exists()

    def test_wizard_revokes_environment_wide_emptied_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Target file physically holds a crossby-owned hook; the ledger owns it.
        _write_claude_settings(
            tmp_path,
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit", "hooks": [{"type": "command", "command": "owned"}]}
                    ]
                }
            },
        )
        _seed_hook_ledger(tmp_path, AIToolID.CLAUDE, {("pre_tool_use", "owned")})

        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )
        # Discovery finds nothing across every tool → environment-wide absence.
        monkeypatch.setattr("crossby.sync.readers.scan_project", _empty_project_scan)

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        # The crossby-owned hook is gone from the target …
        assert "PreToolUse" not in json.loads(
            (tmp_path / ".claude" / "settings.json").read_text()
        ).get("hooks", {})
        # … and the ledger no longer lists it.
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset()

    def test_wizard_revokes_environment_wide_emptied_permission(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_claude_settings(tmp_path, {"permissions": {"allow": ["Bash(myapp:*)"]}})
        ledger = OwnershipLedger()
        ledger.record_permissions(AIToolID.CLAUDE, {"myapp:*"})
        save_ledger(tmp_path, ledger)

        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )
        monkeypatch.setattr("crossby.sync.readers.scan_project", _empty_project_scan)

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        allow = json.loads((tmp_path / ".claude" / "settings.json").read_text())["permissions"][
            "allow"
        ]
        assert "Bash(myapp:*)" not in allow
        assert load_ledger(tmp_path).permissions(AIToolID.CLAUDE) == frozenset()

    def test_declining_a_still_present_port_revokes_owned_copies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Discovery IS non-empty (the hook is still present), but the user answers
        # "no" to the port prompt while the ledger owns the entry. With the literal
        # `data OR ownership` gate, data.hooks is empty so the ownership arm fires
        # and revokes crossby's owned copy. Pins this as a conscious, stable choice.
        _write_claude_settings(
            tmp_path,
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit", "hooks": [{"type": "command", "command": "guard"}]}
                    ]
                }
            },
        )
        _seed_hook_ledger(tmp_path, AIToolID.CLAUDE, {("pre_tool_use", "guard")})

        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        # Real scan discovers the hook, so the port prompt is reached — then declined.
        monkeypatch.setattr("crossby.ui.prompts.confirm", lambda *a, **k: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "guard" not in _claude_hook_commands(tmp_path)
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset()

    def test_hook_still_present_on_another_tool_is_not_revoked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Merged discovery: an owned entry still present on *any* installed tool is
        # rediscovered as current and therefore NOT revoked (environment-wide
        # absence, not entry-for-entry --from parity).
        _write_claude_settings(
            tmp_path,
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit", "hooks": [{"type": "command", "command": "guard"}]}
                    ]
                }
            },
        )
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )

        # Establish: a real wizard run ports Claude's hook to Cursor and records
        # crossby ownership of it for Cursor.
        first = runner.invoke(app, ["sync", "--path", str(tmp_path)])
        assert first.exit_code == 0, first.output
        assert load_ledger(tmp_path).hooks(AIToolID.CURSOR) == frozenset({_GUARD_HOOK})

        # Re-run: the hook is still present on Claude, so it is rediscovered as
        # current and Cursor's owned copy is preserved, not revoked.
        from crossby.sync.readers import discover_hooks

        second = runner.invoke(app, ["sync", "--path", str(tmp_path)])
        assert second.exit_code == 0, second.output
        cursor_hooks = {
            (h.event, h.command) for h in discover_hooks(tmp_path, from_tool=AIToolID.CURSOR)
        }
        assert ("pre_tool_use", "guard") in cursor_hooks
        assert load_ledger(tmp_path).hooks(AIToolID.CURSOR) == frozenset({_GUARD_HOOK})

    def test_explicit_concern_does_not_trigger_other_owned_concerns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `crossby sync permissions` must not fire the hooks dispatch even though
        # the ledger owns hooks — the concern filter gates the ownership arm too.
        _write_claude_settings(
            tmp_path,
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit", "hooks": [{"type": "command", "command": "guard"}]}
                    ]
                }
            },
        )
        _seed_hook_ledger(tmp_path, AIToolID.CLAUDE, {("pre_tool_use", "guard")})

        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )

        result = runner.invoke(app, ["sync", "permissions", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "Nothing to sync." in result.output
        # The owned hook is untouched on disk and in the ledger.
        assert "guard" in _claude_hook_commands(tmp_path)
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset({_GUARD_HOOK})

    def test_dry_run_ownership_revocation_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_claude_settings(
            tmp_path,
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit", "hooks": [{"type": "command", "command": "owned"}]}
                    ]
                }
            },
        )
        _seed_hook_ledger(tmp_path, AIToolID.CLAUDE, {("pre_tool_use", "owned")})

        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )
        monkeypatch.setattr("crossby.sync.readers.scan_project", _empty_project_scan)

        result = runner.invoke(app, ["sync", "--dry-run", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        # No target change …
        assert "owned" in _claude_hook_commands(tmp_path)
        # … no ledger change …
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset({_OWNED_HOOK})
        # … and no report persisted.
        assert not (tmp_path / REPORT_PATH).exists()

    def test_hand_authored_entry_survives_alongside_revoked_owned_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A target file holding both a crossby-owned entry and a hand-authored one
        # keeps the hand-authored entry after a wizard revocation run.
        _write_claude_settings(
            tmp_path,
            {
                "hooks": {
                    "PreToolUse": [
                        {"matcher": "Edit", "hooks": [{"type": "command", "command": "owned"}]},
                        {"matcher": "Write", "hooks": [{"type": "command", "command": "human"}]},
                    ]
                }
            },
        )
        # The ledger owns ONLY crossby's entry, never the human one.
        _seed_hook_ledger(tmp_path, AIToolID.CLAUDE, {("pre_tool_use", "owned")})

        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )
        monkeypatch.setattr("crossby.sync.readers.scan_project", _empty_project_scan)

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        commands = _claude_hook_commands(tmp_path)
        assert "human" in commands
        assert "owned" not in commands

    def test_owned_but_emptied_mcp_is_a_no_op_that_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # MCP revokes only via `disabled ∩ owned`, so an owned-but-emptied MCP
        # dispatch removes nothing, leaves the ledger unchanged, and only emits
        # skipped writer rows — pinned so the tradeoff is a conscious choice.
        mcp = tmp_path / ".mcp.json"
        mcp.write_text(json.dumps({"mcpServers": {"srv": {"command": "npx"}}}), encoding="utf-8")
        ledger = OwnershipLedger()
        ledger.record_mcp(AIToolID.CLAUDE, {"srv"})
        save_ledger(tmp_path, ledger)

        captured: list[Any] = []

        def _capture(results: list[Any], **_kwargs: Any) -> None:
            captured.extend(results)

        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )
        monkeypatch.setattr("crossby.sync.readers.scan_project", _empty_project_scan)
        monkeypatch.setattr("crossby.cli.sync._display_results", _capture)

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        # (a) No MCP removals.
        mcp_rows = [r for r in captured if r.concern == SyncConcern.MCP]
        assert mcp_rows, "expected the MCP dispatch to run and report rows"
        assert all(r.revoked == 0 for r in mcp_rows)
        assert any(r.action == "skipped" for r in mcp_rows)
        # (b) Target config and ledger left unchanged.
        assert "srv" in json.loads(mcp.read_text())["mcpServers"]
        assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset({"srv"})

    def test_no_data_and_no_ownership_is_idempotent_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No new data AND no ledger ownership → no run_sync for revocable concerns
        # and no ledger / report churn.
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            lambda: ["claude", "cursor"],
        )

        result = runner.invoke(app, ["sync", "--path", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "No tool configs found to sync." in result.output
        assert not (tmp_path / LEDGER_PATH).exists()
        assert not (tmp_path / REPORT_PATH).exists()


class TestSyncCommandHooks:
    """`crossby sync hooks --from codex|antigravity-cli` is no longer read-blind."""

    def _claude_hook_commands(self, project: Path) -> list[str]:
        data = json.loads((project / ".claude" / "settings.json").read_text())
        return [
            handler["command"]
            for entries in data.get("hooks", {}).values()
            for entry in entries
            for handler in entry.get("hooks", [])
        ]

    def test_sync_hooks_from_codex_reaches_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real .codex/hooks.json is read and written to the target tool."""
        codex = tmp_path / ".codex" / "hooks.json"
        codex.parent.mkdir()
        codex.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Edit", "hooks": [{"type": "command", "command": "guard"}]}
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["codex", "claude"],
            )
            result = runner.invoke(
                app,
                ["sync", "hooks", "--from", "codex", "--to", "claude", "--path", str(tmp_path)],
            )

        assert result.exit_code == 0, result.output
        assert "no hooks config" not in result.output
        assert "guard" in self._claude_hook_commands(tmp_path)

    def test_sync_hooks_from_antigravity_cli_reaches_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real .agents/hooks.json is read and written to the target tool."""
        agy = tmp_path / ".agents" / "hooks.json"
        agy.parent.mkdir()
        agy.write_text(
            json.dumps(
                {
                    "guard": {
                        "PreToolUse": [
                            {
                                "matcher": "write_to_file",
                                "hooks": [{"type": "command", "command": "guard"}],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "crossby.ai_tools.base.AbstractAITool.detect_installed",
                lambda: ["antigravity-cli", "claude"],
            )
            result = runner.invoke(
                app,
                [
                    "sync",
                    "hooks",
                    "--from",
                    "antigravity-cli",
                    "--to",
                    "claude",
                    "--path",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "no hooks config" not in result.output
        # agy's native tool name reverse-maps to the canonical `Write` on write.
        assert "guard" in self._claude_hook_commands(tmp_path)
        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
        matchers = [e["matcher"] for e in settings["hooks"]["PreToolUse"]]
        assert matchers == ["Write"]
