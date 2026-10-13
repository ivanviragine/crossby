"""Source-scoped command migration through real CLI readers and filesystem writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncData, SyncResult
from crossby.sync.readers import build_sync_data
from crossby.sync.skills import CodexSkillsWriter
from crossby.sync.skills_provenance import PROVENANCE_NAME

runner = CliRunner()


def _skill(root: Path, tool: str, name: str) -> Path:
    directory = root / (".agents" if tool == "codex" else f".{tool}") / "skills" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Native {tool}.\n---\nNative {tool} body.\n",
        encoding="utf-8",
    )
    return directory


def _command(root: Path, tool: str, name: str = "deploy") -> Path:
    command = root / f".{tool}" / "commands" / f"{name}.md"
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text(f"DEPLOY FROM {tool.upper()} ONLY\n", encoding="utf-8")
    return command


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.CODEX],
    )
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
    for tool in ("claude", "cursor"):
        _skill(tmp_path, tool, f"native-{tool}")
        _command(tmp_path, tool)
    return tmp_path


def _sync(root: Path, source: str, *flags: str, target: str = "codex") -> str:
    result = runner.invoke(
        app,
        [
            "sync",
            "skills",
            "--from",
            source,
            "--to",
            target,
            "--strategy",
            "translate",
            "--no-persist-report",
            "--path",
            str(root),
            *flags,
        ],
    )
    assert result.exit_code == 0, result.output
    return result.output


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("source,other", [("claude", "cursor"), ("cursor", "claude")])
def test_explicit_source_plan_dry_run_and_writes_agree(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    other: str,
) -> None:
    rows: list[SyncResult] = []
    monkeypatch.setattr(
        "crossby.cli.sync._display_results", lambda results, **kw: rows.extend(results)
    )
    before = _snapshot(project)
    plan = _sync(project, source, "--plan")
    assert "1 manual-fix" in plan
    assert _snapshot(project) == before
    _sync(project, source, "--dry-run")
    dry = next(row for row in rows if row.concern == SyncConcern.SKILLS and row.tool_id)
    assert dry.action == "created"
    assert "1 manual-fix" in (dry.message or "")
    assert _snapshot(project) == before
    rows.clear()
    _sync(project, source)
    real = next(row for row in rows if row.concern == SyncConcern.SKILLS and row.tool_id)
    assert real.action == dry.action
    target = project / ".agents/skills"
    text = (target / f"{source}-command-deploy/SKILL.md").read_text()
    assert f"DEPLOY FROM {source.upper()} ONLY" in text
    assert f"DEPLOY FROM {other.upper()} ONLY" not in text
    assert not (target / f"{other}-command-deploy").exists()
    assert not (target / f"native-{other}").exists()
    ownership = json.loads((target / PROVENANCE_NAME).read_text())["skills"]
    assert ownership[f"{source}-command-deploy"]["source"] == f"command:{source}"
    assert ownership[f"native-{source}"]["source"] == f"skills:.{source}/skills"
    before_repeat = _snapshot(project)
    rows.clear()
    _sync(project, source)
    assert (
        next(row for row in rows if row.concern == SyncConcern.SKILLS and row.tool_id).action
        == "skipped"
    )
    assert _snapshot(project) == before_repeat


@pytest.mark.parametrize("first,second", [("claude", "cursor"), ("cursor", "claude")])
def test_switch_scope_preserves_other_source_and_human_files(
    project: Path,
    first: str,
    second: str,
) -> None:
    _sync(project, first)
    target = project / ".agents/skills"
    first_output = target / f"{first}-command-deploy"
    original = _snapshot(first_output)
    _sync(project, second)
    human = target / "human/SKILL.md"
    human.parent.mkdir()
    human.write_text("Human-authored skill\n")
    _sync(project, second)
    assert _snapshot(first_output) == original
    (project / f".{second}/commands/deploy.md").unlink()
    before = _snapshot(project)
    _sync(project, second, "--dry-run")
    assert _snapshot(project) == before
    _sync(project, second)
    assert not (target / f"{second}-command-deploy").exists()
    assert _snapshot(first_output) == original
    assert human.read_text() == "Human-authored skill\n"
    _sync(project, first)
    assert _snapshot(first_output) == original
    assert human.read_text() == "Human-authored skill\n"


def test_stale_human_edits_and_added_support_files_survive(project: Path) -> None:
    _sync(project, "claude")
    target = project / ".agents/skills"
    human_edit = target / "claude-command-deploy/SKILL.md"
    human_edit.write_text("Human replaced the generated instructions\n")
    human_support = target / "native-claude/notes.txt"
    human_support.write_text("Human notes\n")
    (project / ".claude/commands/deploy.md").unlink()
    (project / ".claude/skills/native-claude/SKILL.md").unlink()
    _skill(project, "claude", "keep-source-detectable")
    _sync(project, "claude")
    assert human_edit.read_text() == "Human replaced the generated instructions\n"
    assert human_support.read_text() == "Human notes\n"


def test_no_command_primitive_does_not_scan_other_sources(project: Path) -> None:
    _skill(project, "codex", "native-codex")
    _sync(project, "codex", target="copilot")
    target = project / ".github/skills"
    assert not (target / "claude-command-deploy").exists()
    assert not (target / "cursor-command-deploy").exists()
    assert (target / "native-codex/SKILL.md").is_file()


def test_configured_source_is_effective_scope(project: Path) -> None:
    (project / ".crossby.yml").write_text("sync_defaults:\n  from: cursor\n  to: codex\n")
    result = runner.invoke(
        app, ["sync", "skills", "--strategy", "translate", "--path", str(project)]
    )
    assert result.exit_code == 0, result.output
    assert (project / ".agents/skills/cursor-command-deploy/SKILL.md").is_file()
    assert not (project / ".agents/skills/claude-command-deploy").exists()


@pytest.mark.parametrize("source_tools", [None, (), (AIToolID.CODEX,)])
def test_unscoped_api_and_empty_scope_are_distinct(
    project: Path,
    source_tools: tuple[AIToolID, ...] | None,
) -> None:
    data = SyncData(
        skills_source=".claude/skills",
        skills_strategy="translate",
        skills_source_tools=source_tools,
    )
    result = CodexSkillsWriter().sync(data, project)
    assert result.action == "created"
    for tool in ("claude", "cursor"):
        assert (project / f".agents/skills/{tool}-command-deploy/SKILL.md").exists() == (
            source_tools is None
        )
    assert build_sync_data(project, from_tool=AIToolID.CLAUDE).skills_source_tools == (
        AIToolID.CLAUDE,
    )
    assert build_sync_data(project).skills_source_tools is None


@pytest.mark.parametrize("collision", ["native-command", "other-source", "human", "nested-command"])
@pytest.mark.parametrize("preview", [[], ["--dry-run"], ["--plan"]])
def test_collisions_fail_before_changing_target(
    project: Path,
    collision: str,
    preview: list[str],
) -> None:
    if collision == "native-command":
        _skill(project, "claude", "claude-command-deploy")
    elif collision == "nested-command":
        _command(project, "claude", "release/cut")
        _command(project, "claude", "release-cut")
    else:
        _sync(project, "cursor")
        if collision == "other-source":
            _skill(project, "claude", "native-cursor")
        else:
            _skill(project, "codex", "claude-command-deploy")
    before = _snapshot(project)
    result = runner.invoke(
        app,
        [
            "sync",
            "skills",
            "--from",
            "claude",
            "--to",
            "codex",
            "--strategy",
            "translate",
            "--path",
            str(project),
            "--no-persist-report",
            *preview,
        ],
    )
    assert "collision" in result.output, result.output
    if not preview or preview == ["--dry-run"]:
        assert result.exit_code == 1
    assert _snapshot(project) == before


def test_wizard_selected_source_scopes_secondary_discovery(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CLAUDE, AIToolID.CODEX],
    )
    result = runner.invoke(
        app, ["sync", "skills", "--strategy", "translate", "--path", str(project)]
    )
    assert result.exit_code == 0, result.output
    assert (project / ".agents/skills/claude-command-deploy/SKILL.md").is_file()
    assert not (project / ".agents/skills/cursor-command-deploy").exists()


@pytest.mark.parametrize("broken_provenance", ["invalid-json", "symlink"])
def test_invalid_provenance_cannot_authorize_cleanup(project: Path, broken_provenance: str) -> None:
    _sync(project, "claude")
    target = project / ".agents/skills"
    provenance = target / PROVENANCE_NAME
    if broken_provenance == "invalid-json":
        provenance.write_text("{broken")
    else:
        provenance.unlink()
        external = project / "external.json"
        external.write_text("{}")
        provenance.symlink_to(external)
    before = _snapshot(project)
    result = runner.invoke(
        app,
        [
            "sync",
            "skills",
            "--from",
            "claude",
            "--to",
            "codex",
            "--strategy",
            "translate",
            "--path",
            str(project),
            "--no-persist-report",
        ],
    )
    assert result.exit_code == 1
    assert _snapshot(project) == before


@pytest.mark.parametrize("collision", ["native-command", "nested-command"])
@pytest.mark.parametrize("target_kind", ["directory", "symlink"])
@pytest.mark.parametrize("preview", [[], ["--dry-run"], ["--plan"]])
def test_forced_source_collision_preserves_target_without_backup(
    project: Path,
    collision: str,
    target_kind: str,
    preview: list[str],
) -> None:
    _sync(project, "cursor")
    target = project / ".agents/skills"
    if target_kind == "symlink":
        target.rename(project / "existing-skills")
        target.symlink_to(project / "existing-skills", target_is_directory=True)
    if collision == "native-command":
        _skill(project, "claude", "claude-command-deploy")
    else:
        _command(project, "claude", "release/cut")
        _command(project, "claude", "release-cut")
    before = _snapshot(project)
    original_link = target.readlink() if target.is_symlink() else None
    result = runner.invoke(
        app,
        [
            "sync",
            "skills",
            "--from",
            "claude",
            "--to",
            "codex",
            "--strategy",
            "translate",
            "--path",
            str(project),
            "--force",
            "--no-persist-report",
            *preview,
        ],
    )
    assert result.exit_code == 1, result.output
    assert "collision" in result.output
    assert _snapshot(project) == before
    assert target.is_symlink() == (original_link is not None)
    if original_link is not None:
        assert target.readlink() == original_link
    assert not list(target.parent.glob("skills.bak*"))


@pytest.mark.parametrize("target_kind", ["human", "edited-generated", "symlink"])
def test_forced_preview_and_execution_agree_on_replacement(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    target = project / ".agents/skills"
    if target_kind == "human":
        _skill(project, "codex", "native-claude")
    else:
        _sync(project, "claude")
        if target_kind == "edited-generated":
            (target / "native-claude/SKILL.md").write_text("Human edit\n")
        else:
            target.rename(project / "existing-skills")
            target.symlink_to(project / "existing-skills", target_is_directory=True)
    original = _snapshot(target)
    rows: list[SyncResult] = []
    monkeypatch.setattr(
        "crossby.cli.sync._display_results", lambda results, **kw: rows.extend(results)
    )
    before = _snapshot(project)
    plan = _sync(project, "claude", "--force", "--plan")
    assert "1 manual-fix" in plan
    assert _snapshot(project) == before
    _sync(project, "claude", "--force", "--dry-run")
    dry = next(row for row in rows if row.concern == SyncConcern.SKILLS and row.tool_id)
    assert dry.action == "created"
    assert "1 manual-fix" in (dry.message or "")
    assert _snapshot(project) == before
    rows.clear()
    _sync(project, "claude", "--force")
    real = next(row for row in rows if row.concern == SyncConcern.SKILLS and row.tool_id)
    assert real.action == dry.action
    assert not target.is_symlink()
    assert "Native claude body" in (target / "native-claude/SKILL.md").read_text()
    assert not (target / "cursor-command-deploy").exists()
    if target_kind == "symlink":
        assert _snapshot(project / "existing-skills") == original
    else:
        backups = list(target.parent.glob("skills.bak*"))
        assert len(backups) == 1
        assert _snapshot(backups[0]) == original
