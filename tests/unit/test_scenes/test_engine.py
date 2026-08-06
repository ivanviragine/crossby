"""End-to-end scene apply / clear tests against a real temp project."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossby.models.config import SceneConfig, SceneSelector
from crossby.scenes import apply_scene, clear_scene
from crossby.sync.ownership import SceneDeclareKey, load_ledger
from tests.unit.test_scenes.conftest import populate_project, read_json, resolve

# A scene that keeps 2 of 3 skills, 1 of 2 agents, 1 of 2 MCP servers.
SCENE = SceneConfig(
    skills=SceneSelector(include=["review-*", "knowledge"]),
    agents=SceneSelector(include=["code-reviewer"]),
    mcp=SceneSelector(include=["github"]),
)


def _settings(root: Path) -> dict:
    return read_json(root / ".claude" / "settings.json")


class TestApplyDeclare:
    def test_claude_skilloverrides_disables_deselected(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert _settings(tmp_path)["skillOverrides"] == {"deploy-prod": "off"}

    def test_claude_deny_blocks_deselected_agent(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert _settings(tmp_path)["permissions"]["deny"] == ["Agent(deployer)"]

    def test_claude_disables_deselected_mcp(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert _settings(tmp_path)["disabledMcpjsonServers"] == ["linear"]

    def test_real_skills_dir_left_intact(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        # All three still present in the real source; only skillOverrides filters.
        names = {p.name for p in (tmp_path / ".claude" / "skills").iterdir() if p.is_dir()}
        assert names == {"review-skill", "knowledge", "deploy-prod"}


class TestApplyProject:
    def test_shared_agents_skills_repointed_to_selection(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        shared = tmp_path / ".agents" / "skills"
        assert shared.is_symlink()
        resolved = {p.name for p in shared.iterdir() if p.name != ".crossby-managed"}
        assert resolved == {"review-skill", "knowledge"}

    def test_shared_path_reported_once_for_both_tools(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        rows = [
            r
            for r in results
            if r.concern.value == "skills" and r.message and "shared by" in r.message
        ]
        assert len(rows) == 1
        assert "antigravity-cli" in rows[0].message and "codex" in rows[0].message

    def test_projection_gitignored_dir_created(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert (tmp_path / ".crossby" / "scene" / "active" / "skills").is_dir()


class TestProvenance:
    def test_ledger_records_declare_keys(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        from crossby.models.ai import AIToolID

        ledger = load_ledger(tmp_path)
        assert ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES) == frozenset(
            {"deploy-prod"}
        )
        assert ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS) == frozenset(
            {"Agent(deployer)"}
        )
        assert ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DISABLED_MCP) == frozenset(
            {"linear"}
        )

    def test_user_authored_entries_survive_clear(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        # A user's own skillOverride + deny + disabled server crossby never wrote.
        (tmp_path / ".claude").mkdir(exist_ok=True)
        (tmp_path / ".claude" / "settings.json").write_text(
            '{"skillOverrides": {"user-skill": "off"}, '
            '"permissions": {"deny": ["Agent(user-agent)"]}, '
            '"disabledMcpjsonServers": ["user-server"]}',
            encoding="utf-8",
        )
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        clear_scene(tmp_path)
        s = _settings(tmp_path)
        assert s["skillOverrides"] == {"user-skill": "off"}
        assert s["permissions"]["deny"] == ["Agent(user-agent)"]
        assert s["disabledMcpjsonServers"] == ["user-server"]


class TestIdempotency:
    def test_reapply_reports_skipped_and_changes_nothing(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        settings_before = (tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8")
        link_before = (tmp_path / ".agents" / "skills").readlink()

        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)

        settings_after = (tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8")
        assert settings_after == settings_before
        assert (tmp_path / ".agents" / "skills").readlink() == link_before
        declare_rows = [r for r in results if r.tool_id and r.message == "already applied"]
        assert declare_rows  # the DECLARE surfaces report skipped on a re-run


class TestClearRoundTrip:
    def test_clear_reverts_declare_and_restores_sources(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        clear_scene(tmp_path)

        s = _settings(tmp_path)
        assert "skillOverrides" not in s
        assert s.get("permissions", {}).get("deny", []) == []
        assert "disabledMcpjsonServers" not in s
        # Projection removed; shared dir points back at the unfiltered source.
        assert not (tmp_path / ".crossby" / "scene").exists()
        shared = tmp_path / ".agents" / "skills"
        resolved = {p.name for p in shared.iterdir() if p.name != ".crossby-managed"}
        assert resolved == {"review-skill", "knowledge", "deploy-prod"}

    def test_switch_scene_leaves_no_trace_of_previous(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        # Scene B: keep a different single skill and the other agent.
        scene_b = SceneConfig(
            skills=SceneSelector(include=["deploy-*"]),
            agents=SceneSelector(include=["deployer"]),
            mcp=SceneSelector(include=["linear"]),
        )
        apply_scene(resolve(tmp_path, scene_b), tmp_path)
        s = _settings(tmp_path)
        assert s["skillOverrides"] == {"knowledge": "off", "review-skill": "off"}
        assert s["permissions"]["deny"] == ["Agent(code-reviewer)"]
        assert s["disabledMcpjsonServers"] == ["github"]
        shared = tmp_path / ".agents" / "skills"
        assert {p.name for p in shared.iterdir() if p.name != ".crossby-managed"} == {"deploy-prod"}


class TestDryRun:
    def test_dry_run_leaves_filesystem_byte_identical(self, tmp_path: Path) -> None:
        populate_project(tmp_path)

        def snapshot() -> dict[str, str]:
            out: dict[str, str] = {}
            for p in sorted(tmp_path.rglob("*")):
                if p.is_symlink():
                    out[str(p.relative_to(tmp_path))] = "->" + str(p.readlink())
                elif p.is_file():
                    out[str(p.relative_to(tmp_path))] = p.read_text(encoding="utf-8")
            return out

        before = snapshot()
        results = apply_scene(resolve(tmp_path, SCENE), tmp_path, dry_run=True)
        assert snapshot() == before
        assert results  # a full result set is still produced


class TestUnsupportedAndVersionGate:
    def test_old_claude_skips_skilloverrides_additions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "crossby.scenes.versioning.detect_tool_version", lambda _tool: (2, 1, 100)
        )
        populate_project(tmp_path)
        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert "skillOverrides" not in _settings(tmp_path)
        gated = [r for r in results if r.message and "claude >=" in r.message]
        assert gated

    def test_cursor_mcp_reported_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.models.ai import AIToolID

        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: [AIToolID.CLAUDE, AIToolID.CURSOR]),
        )
        populate_project(tmp_path)
        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        rows = [r for r in results if r.message and "no per-server disable key" in r.message]
        assert rows and any(r.tool_id == AIToolID.CURSOR for r in rows)


class TestCodexTrust:
    def test_untrusted_project_reports_caveat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("crossby.scenes.trust.codex_trusts_project", lambda *a, **k: False)
        populate_project(tmp_path)
        (tmp_path / ".codex").mkdir()
        (tmp_path / ".codex" / "config.toml").write_text(
            '[mcp_servers.linear]\ncommand = "lin"\n', encoding="utf-8"
        )
        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        from crossby.models.ai import AIToolID

        codex_rows = [r for r in results if r.tool_id == AIToolID.CODEX and r.message]
        assert any("does not trust" in r.message for r in codex_rows)


class TestGitignore:
    def test_scene_dir_gitignored_in_managed_block(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".crossby/scene/" in gitignore
        assert "crossby scene projection" in gitignore


class TestPartialFailureSafe:
    def test_malformed_settings_isolates_and_rerun_converges(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        # Claude settings.json malformed → its DECLARE rows error, others proceed.
        (tmp_path / ".claude").mkdir(exist_ok=True)
        (tmp_path / ".claude" / "settings.json").write_text("{ broken", encoding="utf-8")

        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert any(r.action == "error" for r in results)
        # The shared skills projection still happened (a different tool succeeded).
        assert (tmp_path / ".agents" / "skills").is_symlink()

        # Fix the file and re-run from the half-applied state → converges.
        (tmp_path / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
        results2 = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert not any(r.action == "error" for r in results2)
        assert _settings(tmp_path)["skillOverrides"] == {"deploy-prod": "off"}


class TestNonManagedDirRefusal:
    def _project_with_real_agents_skills(self, tmp_path: Path) -> None:
        from tests.unit.test_scenes.conftest import make_skill

        for name in ("review-skill", "knowledge", "deploy-prod"):
            make_skill(tmp_path, ".claude/skills", name)
        # A real, non-crossby-managed .agents/skills (a stray file makes it unmanaged).
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        (tmp_path / ".agents" / "skills" / "user-notes.txt").write_text("mine", encoding="utf-8")

    def test_refused_without_force(self, tmp_path: Path) -> None:
        self._project_with_real_agents_skills(tmp_path)
        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        skills_errs = [r for r in results if r.concern.value == "skills" and r.action == "error"]
        assert skills_errs
        # The user's real directory is untouched.
        assert (tmp_path / ".agents" / "skills" / "user-notes.txt").is_file()

    def test_force_backs_up_and_replaces(self, tmp_path: Path) -> None:
        self._project_with_real_agents_skills(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path, force=True)
        assert (tmp_path / ".agents" / "skills").is_symlink()
        backups = list(tmp_path.glob(".agents/skills.bak*"))
        assert backups and (backups[0] / "user-notes.txt").is_file()

    def test_clear_refuses_real_non_crossby_target(self, tmp_path: Path) -> None:
        # clear must not back up and replace a real, non-crossby directory it never
        # managed — the same refusal the apply path enforces.
        self._project_with_real_agents_skills(tmp_path)
        results = clear_scene(tmp_path)
        assert not (tmp_path / ".agents" / "skills").is_symlink()
        assert (tmp_path / ".agents" / "skills" / "user-notes.txt").is_file()
        assert not list(tmp_path.glob(".agents/skills.bak*"))
        skills_errs = [r for r in results if r.concern.value == "skills" and r.action == "error"]
        assert skills_errs


class TestHooksPermissionsFilter:
    def test_selecting_all_permissions_does_not_touch_them(self, tmp_path: Path) -> None:
        from tests.unit.test_scenes.conftest import write_json

        populate_project(tmp_path)
        write_json(
            tmp_path / ".claude" / "settings.json",
            {"permissions": {"allow": ["Bash(git diff:*)", "Bash(npm test)"]}},
        )
        # Scene with no permissions selector → selects all → no permissions rows.
        results = apply_scene(resolve(tmp_path, SCENE), tmp_path)
        perm_rows = [r for r in results if r.concern.value == "permissions"]
        assert perm_rows == []


class TestToolScope:
    """The ``tools=`` argument that scopes apply / clear to a subset of tools."""

    def _install_four(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from crossby.models.ai import AIToolID

        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(
                lambda _cls: [
                    AIToolID.CLAUDE,
                    AIToolID.CODEX,
                    AIToolID.ANTIGRAVITY_CLI,
                    AIToolID.CURSOR,
                ]
            ),
        )

    def test_apply_scoped_touches_only_the_named_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.models.ai import AIToolID

        self._install_four(monkeypatch)
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path, tools=[AIToolID.CURSOR])

        # Cursor got a filtered skills projection; Claude was never DECLARE-filtered.
        assert (tmp_path / ".cursor" / "skills").is_symlink()
        assert not (tmp_path / ".claude" / "settings.json").exists()

    def test_scoped_clear_keeps_shared_projection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.models.ai import AIToolID

        self._install_four(monkeypatch)
        populate_project(tmp_path)
        # Apply to every tool: codex + antigravity share the .agents/skills tree.
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert (tmp_path / ".crossby" / "scene").exists()

        # Clearing only cursor must not yank the projection from the sharers.
        clear_scene(tmp_path, tools=[AIToolID.CURSOR])
        assert (tmp_path / ".crossby" / "scene").exists()
        # The other tools still resolve through the shared projection tree.
        assert (tmp_path / ".agents" / "skills").is_symlink()

    def test_unscoped_clear_removes_projection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_four(monkeypatch)
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        clear_scene(tmp_path)
        assert not (tmp_path / ".crossby" / "scene").exists()

    def test_empty_scope_clear_reverts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # tools=[] must mean "revert nothing" — it must NOT collapse to "all".
        self._install_four(monkeypatch)
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        clear_scene(tmp_path, tools=[])
        assert _settings(tmp_path)["skillOverrides"] == {"deploy-prod": "off"}
        assert (tmp_path / ".crossby" / "scene").exists()

    def test_clear_restores_uninstalled_recorded_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.models.ai import AIToolID

        self._install_four(monkeypatch)
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert (tmp_path / ".cursor" / "skills").is_symlink()

        # Cursor is uninstalled, but a clear of the recorded scope must still
        # re-point its on-disk symlink before deleting the projection.
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI]),
        )
        clear_scene(
            tmp_path,
            tools=[AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI, AIToolID.CURSOR],
        )
        assert not (tmp_path / ".crossby" / "scene").exists()
        # Cursor's dir resolves to the real source, not a dangling deleted path.
        resolved = {p.name for p in (tmp_path / ".cursor" / "skills").iterdir()}
        assert {"review-skill", "knowledge", "deploy-prod"} <= resolved

    def test_clear_keeps_projection_when_restore_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.models.ai import AIToolID
        from crossby.sync.base import SyncConcern, SyncResult

        self._install_four(monkeypatch)
        populate_project(tmp_path)
        apply_scene(resolve(tmp_path, SCENE), tmp_path)
        assert (tmp_path / ".crossby" / "scene").exists()

        def _err(*_a: object, **_kw: object) -> SyncResult:
            return SyncResult(
                tool_id=AIToolID.CURSOR, concern=SyncConcern.SKILLS, action="error", message="boom"
            )

        monkeypatch.setattr("crossby.scenes.projection.restore_source", _err)
        clear_scene(tmp_path)
        # A failed re-point must not leave a tool dangling — keep the projection.
        assert (tmp_path / ".crossby" / "scene").exists()
