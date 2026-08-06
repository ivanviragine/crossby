"""Integration tests for the ``crossby scene`` CLI command group.

These drive the real activation engine end to end: a project is built on disk
(via the scene-engine test helpers), the installed-tool set / version / trust are
pinned for determinism, and each subcommand is invoked through ``CliRunner``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.scenes.state import SCENE_STATE_PATH
from tests.unit.test_scenes.conftest import populate_project, read_json

runner = CliRunner()

# Every installed tool that participates in scenes, including Cursor so the
# per-tool (--tool cursor) scoping paths are exercised.
INSTALLED = [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI, AIToolID.CURSOR]

# Scene A keeps the review skills / agent / MCP; scene B keeps the deploy set.
SCENES_YAML = """\
version: 1
scenes:
  pr-review:
    description: Review a pull request
    skills:
      include: ["review-*", "knowledge"]
    agents:
      include: ["code-reviewer"]
    mcp:
      include: ["github"]
  deploy:
    description: Ship to production
    skills:
      include: ["deploy-*"]
    agents:
      include: ["deployer"]
    mcp:
      include: ["linear"]
"""


@pytest.fixture(autouse=True)
def _pinned_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin tool detection, version and trust so activation is deterministic.

    The scene-engine conftest autouses these, but that conftest does not apply to
    ``tests/unit/test_cli`` — so they are replicated here. ``is_tty`` is forced
    off so every subcommand takes its non-interactive defaults.
    """
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        classmethod(lambda _cls: list(INSTALLED)),
    )
    monkeypatch.setattr("crossby.scenes.versioning.detect_tool_version", lambda _tool: (2, 1, 218))
    monkeypatch.setattr("crossby.scenes.trust.codex_trusts_project", lambda *a, **k: True)
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)


def _write_config(root: Path, body: str = SCENES_YAML) -> None:
    (root / ".crossby.yml").write_text(body, encoding="utf-8")


def _settings(root: Path) -> dict[str, object]:
    return read_json(root / ".claude" / "settings.json")


def _project(tmp_path: Path, body: str = SCENES_YAML) -> Path:
    populate_project(tmp_path)
    _write_config(tmp_path, body)
    return tmp_path


def _invoke(args: list[str], root: Path) -> Result:
    return runner.invoke(app, [*args, "--path", str(root)])


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_lists_scenes_with_counts(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "list"], root)
        assert result.exit_code == 0, result.output
        assert "pr-review" in result.output
        assert "deploy" in result.output
        assert "Review a pull request" in result.output

    def test_no_scenes_prints_hint(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        _write_config(tmp_path, "version: 1\n")
        result = _invoke(["scene", "list"], tmp_path)
        assert result.exit_code == 0, result.output
        assert "No scenes defined" in result.output


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


class TestShow:
    def test_shows_resolved_items_and_mechanisms(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "show", "pr-review"], root)
        assert result.exit_code == 0, result.output
        # Selected items and a mechanism column appear.
        assert "review-skill" in result.output
        assert "code-reviewer" in result.output
        assert "declare" in result.output
        assert "project" in result.output

    def test_unmatched_selector_warns(self, tmp_path: Path) -> None:
        body = 'version: 1\nscenes:\n  odd:\n    skills:\n      include: ["does-not-exist-*"]\n'
        root = _project(tmp_path, body)
        result = _invoke(["scene", "show", "odd"], root)
        assert result.exit_code == 0, result.output
        assert "matched no detected item" in result.output

    def test_unknown_scene_lists_available(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "show", "nope"], root)
        assert result.exit_code == 1
        assert "Unknown scene" in result.output
        assert "pr-review" in result.output


# ---------------------------------------------------------------------------
# use
# ---------------------------------------------------------------------------


class TestUse:
    def test_applies_and_records_state(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "use", "pr-review"], root)
        assert result.exit_code == 0, result.output

        # DECLARE filtered Claude to the selected review set.
        assert _settings(root)["skillOverrides"] == {"deploy-prod": "off"}
        assert _settings(root)["disabledMcpjsonServers"] == ["linear"]

        # State recorded, with per-tool mechanisms.
        state = read_json(root / SCENE_STATE_PATH)
        assert state["scene"] == "pr-review"
        assert state["status"] == "applied"
        assert state["tools"]["claude"]["mechanisms"]["skills"] == "declare"
        assert state["tools"]["cursor"]["mechanisms"]["skills"] == "project"

    def test_state_file_is_gitignored(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _invoke(["scene", "use", "pr-review"], root)
        gitignore = (root / ".gitignore").read_text(encoding="utf-8")
        assert SCENE_STATE_PATH.as_posix() in gitignore

    def test_plan_writes_nothing(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "use", "pr-review", "--plan"], root)
        assert result.exit_code == 0, result.output
        assert not (root / ".claude" / "settings.json").exists()
        assert not (root / SCENE_STATE_PATH).exists()

    def test_unknown_scene_exits_1(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "use", "ghost"], root)
        assert result.exit_code == 1
        assert "Unknown scene" in result.output
        assert "pr-review" in result.output


# ---------------------------------------------------------------------------
# Scene switching — the central correctness requirement
# ---------------------------------------------------------------------------


class TestSwitching:
    def test_a_then_b_then_clear_restores_original(self, tmp_path: Path) -> None:
        root = _project(tmp_path)

        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        assert _invoke(["scene", "use", "deploy"], root).exit_code == 0

        # After switching to B, only B's disable set is present.
        assert _settings(root)["skillOverrides"] == {"knowledge": "off", "review-skill": "off"}
        assert _settings(root)["disabledMcpjsonServers"] == ["github"]

        assert _invoke(["scene", "clear"], root).exit_code == 0

        # Clear reverts to the true pre-A baseline, not B's (or A's) output.
        settings = _settings(root)
        assert "skillOverrides" not in settings
        assert settings.get("permissions", {}).get("deny", []) == []
        assert "disabledMcpjsonServers" not in settings
        assert not (root / SCENE_STATE_PATH).exists()
        assert not (root / ".crossby" / "scene").exists()

    def test_reapply_active_scene_repairs_state(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        # Re-applying the active scene is a revert-then-reapply, not a no-op.
        result = _invoke(["scene", "use", "pr-review"], root)
        assert result.exit_code == 0, result.output
        assert _settings(root)["skillOverrides"] == {"deploy-prod": "off"}

    def test_scoped_reapply_preserves_other_tools(self, tmp_path: Path) -> None:
        # Applying globally then re-applying the SAME scene scoped to one tool
        # must repair only that tool and keep the rest recorded/active.
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        assert _invoke(["scene", "use", "pr-review", "--tool", "cursor"], root).exit_code == 0

        state = read_json(root / SCENE_STATE_PATH)
        assert state["scene"] == "pr-review"
        assert set(state["tools"]) == {str(t) for t in INSTALLED}
        # Claude's DECLARE was never reverted.
        assert _settings(root)["skillOverrides"] == {"deploy-prod": "off"}

    def test_unscoped_switch_reverts_recorded_but_uninstalled_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A tool the active scene recorded but that has since been uninstalled must
        # still be reverted by an unscoped switch — its owned keys would otherwise
        # stay applied with no state record (the switch replaces, not merges).
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        assert _settings(root)["skillOverrides"] == {"deploy-prod": "off"}

        # Uninstall Claude, then switch scenes without --tool.
        remaining = [t for t in INSTALLED if t != AIToolID.CLAUDE]
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: list(remaining)),
        )
        assert _invoke(["scene", "use", "deploy"], root).exit_code == 0

        # Claude's stale pr-review DECLARE was reverted, not left dangling, and the
        # new state records only the still-installed tools.
        assert "skillOverrides" not in _settings(root)
        state = read_json(root / SCENE_STATE_PATH)
        assert "claude" not in state["tools"]
        assert set(state["tools"]) == {str(t) for t in remaining}


class TestPerToolScope:
    def test_use_tool_cursor_touches_only_cursor(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "use", "pr-review", "--tool", "cursor"], root)
        assert result.exit_code == 0, result.output

        # Cursor got a filtered skills tree; Claude was never DECLARE-filtered.
        assert (root / ".cursor" / "skills").is_symlink()
        assert not (root / ".claude" / "settings.json").exists()

        state = read_json(root / SCENE_STATE_PATH)
        assert list(state["tools"]) == ["cursor"]

    def test_clear_after_partial_apply_reverts_only_cursor(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review", "--tool", "cursor"], root).exit_code == 0

        # A full clear reverts exactly the recorded tool (cursor) and leaves the
        # projection cleaned; nothing was ever written for the other tools.
        assert _invoke(["scene", "clear"], root).exit_code == 0
        assert not (root / ".claude" / "settings.json").exists()
        assert not (root / SCENE_STATE_PATH).exists()
        # Cursor's dir is restored to the unfiltered source.
        cursor_skills = root / ".cursor" / "skills"
        names = {p.name for p in cursor_skills.iterdir() if p.name != ".crossby-managed"}
        assert names == {"review-skill", "knowledge", "deploy-prod"}

    def test_switch_after_partial_apply_reverts_only_cursor(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review", "--tool", "cursor"], root).exit_code == 0
        # Switching to a full scene reverts the recorded cursor state, then
        # applies the new scene across all tools.
        assert _invoke(["scene", "use", "deploy"], root).exit_code == 0
        assert _settings(root)["skillOverrides"] == {"knowledge": "off", "review-skill": "off"}
        state = read_json(root / SCENE_STATE_PATH)
        assert state["scene"] == "deploy"
        assert set(state["tools"]) == {str(t) for t in INSTALLED}

    def test_scoped_clear_prunes_tool_from_state(self, tmp_path: Path) -> None:
        # A scoped clear drops exactly the cleared tool (and its drift hashes),
        # leaving the other tools recorded and scene-managed.
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        assert _invoke(["scene", "clear", "--tool", "cursor"], root).exit_code == 0

        state = read_json(root / SCENE_STATE_PATH)
        assert "cursor" not in state["tools"]
        assert "claude" in state["tools"]
        # Claude's DECLARE is still in force — only cursor was reverted.
        assert _settings(root)["skillOverrides"] == {"deploy-prod": "off"}

    def test_clear_tool_not_in_active_scene_is_noop(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review", "--tool", "cursor"], root).exit_code == 0
        # codex was never part of the cursor-only scene.
        result = _invoke(["scene", "clear", "--tool", "codex"], root)
        assert result.exit_code == 0, result.output
        assert "not part of the active scene" in result.output
        # The cursor scene is left intact.
        assert (root / SCENE_STATE_PATH).exists()
        assert read_json(root / SCENE_STATE_PATH)["scene"] == "pr-review"

    def test_tool_scope_expands_to_shared_skills_dir(self, tmp_path: Path) -> None:
        # codex and antigravity-cli share .agents/skills, so --tool codex must
        # also cover antigravity-cli (and record it) to stay honest.
        root = _project(tmp_path)
        result = _invoke(["scene", "use", "pr-review", "--tool", "codex"], root)
        assert result.exit_code == 0, result.output
        assert "shared skills directory" in result.output
        state = read_json(root / SCENE_STATE_PATH)
        assert set(state["tools"]) == {"codex", "antigravity-cli"}

    def test_scoped_switch_to_different_scene_is_rejected(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        # A different scene, scoped to one tool, would strand the other tools.
        result = _invoke(["scene", "use", "deploy", "--tool", "cursor"], root)
        assert result.exit_code == 1
        assert "strand" in result.output.lower()
        # The active scene is unchanged.
        assert read_json(root / SCENE_STATE_PATH)["scene"] == "pr-review"

    def test_scoped_reapply_same_scene_allowed(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review", "--tool", "cursor"], root).exit_code == 0
        # Re-applying the SAME scene scoped to the same tool is fine (repair).
        result = _invoke(["scene", "use", "pr-review", "--tool", "cursor"], root)
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class TestDrift:
    def test_status_flags_content_edit(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0

        # Hand-edit a scene-managed file so its content changes.
        settings_path = root / ".claude" / "settings.json"
        data = _settings(root)
        data["skillOverrides"] = {"deploy-prod": "off", "knowledge": "off"}
        settings_path.write_text(json.dumps(data), encoding="utf-8")

        result = _invoke(["scene", "status"], root)
        assert result.exit_code == 0, result.output
        assert "Drift detected" in result.output

    def test_status_ignores_neutral_reformat(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0

        # Re-serialise with different indentation / key order — same content.
        settings_path = root / ".claude" / "settings.json"
        data = _settings(root)
        reordered = dict(reversed(list(data.items())))
        settings_path.write_text(json.dumps(reordered, indent=4), encoding="utf-8")

        result = _invoke(["scene", "status"], root)
        assert result.exit_code == 0, result.output
        assert "No drift" in result.output

    def test_use_refuses_switch_on_drift(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        _drift_settings(root)

        result = _invoke(["scene", "use", "deploy"], root)
        assert result.exit_code == 1
        assert "drift" in result.output.lower()
        # The active scene is unchanged (switch refused).
        assert read_json(root / SCENE_STATE_PATH)["scene"] == "pr-review"

    def test_force_overwrites_drift(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        _drift_settings(root)

        result = _invoke(["scene", "use", "pr-review", "--force"], root)
        assert result.exit_code == 0, result.output
        # Re-applied cleanly, repairing the drift.
        assert _settings(root)["skillOverrides"] == {"deploy-prod": "off"}

    def test_plan_previews_even_when_drifted(self, tmp_path: Path) -> None:
        # --plan writes nothing, so it must preview rather than refuse on drift.
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        _drift_settings(root)

        use_plan = _invoke(["scene", "use", "deploy", "--plan"], root)
        assert use_plan.exit_code == 0, use_plan.output
        clear_plan = _invoke(["scene", "clear", "--plan"], root)
        assert clear_plan.exit_code == 0, clear_plan.output
        # Neither preview mutated the active scene.
        assert read_json(root / SCENE_STATE_PATH)["scene"] == "pr-review"

    def test_clear_refuses_on_drift_without_force(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        _drift_settings(root)

        result = _invoke(["scene", "clear"], root)
        assert result.exit_code == 1
        assert "drift" in result.output.lower()
        assert (root / SCENE_STATE_PATH).exists()

        # --force lets the clear through.
        assert _invoke(["scene", "clear", "--force"], root).exit_code == 0
        assert not (root / SCENE_STATE_PATH).exists()

    def test_tool_scope_ignores_other_tool_drift(self, tmp_path: Path) -> None:
        # Drift on a Claude-managed file must not block or be reported by a
        # cursor-scoped clear / status.
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        _drift_settings(root)  # edits .claude/settings.json (Claude's file)

        status = _invoke(["scene", "status", "--tool", "cursor"], root)
        assert status.exit_code == 0, status.output
        assert "No drift" in status.output

        # cursor's own files are untouched, so its scoped clear is not refused.
        clear = _invoke(["scene", "clear", "--tool", "cursor"], root)
        assert clear.exit_code == 0, clear.output


def _drift_settings(root: Path) -> None:
    """Hand-edit a scene-managed file so drift detection fires.

    Deleting the crossby-written ``skillOverrides`` key is a change the engine can
    cleanly repair on ``--force`` (the revert is a no-op, the reapply re-adds it),
    unlike overwriting the value — which the ledger treats as a user edit to keep.
    """
    settings_path = root / ".claude" / "settings.json"
    data = read_json(settings_path)
    data.pop("skillOverrides", None)
    settings_path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_no_active_scene(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "status"], root)
        assert result.exit_code == 0, result.output
        assert "No active scene" in result.output

    def test_reports_active_scene_and_mechanism(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        result = _invoke(["scene", "status"], root)
        assert result.exit_code == 0, result.output
        assert "pr-review" in result.output
        assert "declare" in result.output

    def test_deleted_scene_definition(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        # Remove the scene from the config after applying it.
        _write_config(root, "version: 1\n")

        status = _invoke(["scene", "status"], root)
        assert status.exit_code == 0, status.output
        assert "no longer defined" in status.output

        # Clear still succeeds without re-resolving the scene definition.
        clear = _invoke(["scene", "clear"], root)
        assert clear.exit_code == 0, clear.output
        assert "skillOverrides" not in _settings(root)
        assert not (root / SCENE_STATE_PATH).exists()


# ---------------------------------------------------------------------------
# clear edge cases + schema versioning
# ---------------------------------------------------------------------------


class TestClearAndSchema:
    def test_clear_with_no_active_scene_is_noop(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "clear"], root)
        assert result.exit_code == 0, result.output
        assert "nothing to clear" in result.output.lower()

    def test_unrecognised_schema_version_is_no_active_scene(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        state_path = root / SCENE_STATE_PATH
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"version": 999, "scene": "pr-review"}), encoding="utf-8")

        status = _invoke(["scene", "status"], root)
        assert status.exit_code == 0, status.output
        assert "unrecognised schema version" in status.output
        assert "No active scene" in status.output

        # Clear treats it as no active scene (no revert from uninterpretable state).
        clear = _invoke(["scene", "clear"], root)
        assert clear.exit_code == 0, clear.output
        assert "nothing to clear" in clear.output.lower()

    def test_empty_tools_state_is_no_active_scene_and_retained(self, tmp_path: Path) -> None:
        # A state file with no readable tool records is uninterpretable: report
        # no active scene, and do NOT delete the file (so a bug can't orphan a
        # ledger-owned setting).
        root = _project(tmp_path)
        state_path = root / SCENE_STATE_PATH
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "scene": "pr-review",
                    "applied_at": "2026-01-01T00:00:00Z",
                    "tools": {},
                }
            ),
            encoding="utf-8",
        )
        result = _invoke(["scene", "clear"], root)
        assert result.exit_code == 0, result.output
        assert "no readable tool records" in result.output
        assert state_path.exists()

    def test_state_with_only_unknown_tools_is_retained(self, tmp_path: Path) -> None:
        # A state recording only tool ids this build doesn't know is
        # uninterpretable: clear leaves it in place rather than deleting it (which
        # could orphan a ledger-owned setting).
        root = _project(tmp_path)
        state_path = root / SCENE_STATE_PATH
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "scene": "pr-review",
                    "applied_at": "2026-01-01T00:00:00Z",
                    "tools": {"notatool": {"mechanisms": {}, "status": "applied", "hashes": {}}},
                }
            ),
            encoding="utf-8",
        )
        result = _invoke(["scene", "clear"], root)
        assert result.exit_code == 0, result.output
        assert "unknown tools" in result.output.lower()
        assert state_path.exists()


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------


class TestPartialFailure:
    def test_partial_apply_exits_nonzero_and_records_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path)

        # Make Cursor's skills re-point fail; run_sync converts the writer
        # exception into an ``error`` row, so the apply is partial.
        from crossby.sync.skills import CursorSkillsWriter

        original = CursorSkillsWriter.sync

        def _boom(self, data, project_root, *, dry_run=False, force=False):  # type: ignore[no-untyped-def]
            if self.concern.value == "skills":
                raise RuntimeError("cursor skills writer failed")
            return original(self, data, project_root, dry_run=dry_run, force=force)

        monkeypatch.setattr(CursorSkillsWriter, "sync", _boom)

        result = _invoke(["scene", "use", "pr-review"], root)
        assert result.exit_code == 1
        assert "partial" in result.output.lower()

        state = read_json(root / SCENE_STATE_PATH)
        assert state["status"] == "partial"
        assert state["tools"]["cursor"]["status"] == "failed"
        # A tool that succeeded is still marked applied.
        assert state["tools"]["claude"]["status"] == "applied"

    def test_clear_after_partial_reverts_succeeded_tools(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path)
        from crossby.sync.skills import CursorSkillsWriter

        original = CursorSkillsWriter.sync

        def _boom(self, data, project_root, *, dry_run=False, force=False):  # type: ignore[no-untyped-def]
            if self.concern.value == "skills":
                raise RuntimeError("cursor skills writer failed")
            return original(self, data, project_root, dry_run=dry_run, force=force)

        monkeypatch.setattr(CursorSkillsWriter, "sync", _boom)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 1

        # Clearing after a partial apply cleanly reverts the tools that succeeded.
        monkeypatch.setattr(CursorSkillsWriter, "sync", original)
        assert _invoke(["scene", "clear"], root).exit_code == 0
        settings = _settings(root)
        assert "skillOverrides" not in settings
        assert not (root / SCENE_STATE_PATH).exists()

    def test_apply_exception_records_recovery_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If apply raises after the ledger already holds provenance, a recovery
        # state is written so `clear` still knows a scene is active.
        root = _project(tmp_path)

        def _boom(*_a: object, **_kw: object) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr("crossby.scenes.engine.apply_scene", _boom)
        result = _invoke(["scene", "use", "pr-review"], root)
        assert result.exit_code == 1
        assert "apply failed" in result.output.lower()

        state = read_json(root / SCENE_STATE_PATH)
        assert state["scene"] == "pr-review"
        assert state["status"] == "partial"

    def test_failed_clear_preserves_state_for_retry(self, tmp_path: Path) -> None:
        # If the revert errors (e.g. a managed file is now malformed), the state
        # file must survive so the clear can be retried. --force bypasses the
        # drift refusal so the clear actually runs and hits the writer error.
        root = _project(tmp_path)
        assert _invoke(["scene", "use", "pr-review"], root).exit_code == 0
        (root / ".claude" / "settings.json").write_text("{ broken", encoding="utf-8")

        result = _invoke(["scene", "clear", "--force"], root)
        assert result.exit_code == 1
        assert "left intact" in result.output.lower()
        assert (root / SCENE_STATE_PATH).exists()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_unknown_tool_exits_1(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _invoke(["scene", "use", "pr-review", "--tool", "notatool"], root)
        assert result.exit_code == 1
        assert "Unknown tool" in result.output

    def test_no_installed_tools_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: []),
        )
        result = _invoke(["scene", "use", "pr-review"], root)
        assert result.exit_code == 1
        assert "No AI tools found" in result.output

    def test_missing_config_lists_no_scenes(self, tmp_path: Path) -> None:
        # A missing .crossby.yml loads a default (empty) config, so `list`
        # reports no scenes rather than erroring.
        populate_project(tmp_path)  # no .crossby.yml written
        result = _invoke(["scene", "list"], tmp_path)
        assert result.exit_code == 0, result.output
        assert "No scenes defined" in result.output
