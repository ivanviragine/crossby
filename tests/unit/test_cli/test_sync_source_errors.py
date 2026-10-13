"""Source failures must not authorize removal through the real sync CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern
from crossby.sync.ownership import LEDGER_PATH, load_ledger
from crossby.sync.readers import build_sync_data, discover_hooks

runner = CliRunner()
SOURCE = Path(".claude/settings.json")
TARGETS = {"hooks": Path(".cursor/hooks.json"), "permissions": Path(".cursor/cli.json")}
CONFIG = {
    "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"command": "./guard.sh"}]}]},
    "permissions": {"allow": ["Bash(myapp:*)"]},
}


@pytest.fixture(autouse=True)
def isolated_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CLAUDE, AIToolID.CURSOR],
    )
    monkeypatch.setattr(
        "crossby.sync.permissions._GLOBAL_CURSOR_CONFIG_PATH", tmp_path / "global-cursor.json"
    )
    monkeypatch.setattr("crossby.ui.prompts.confirm", lambda *a, **kw: True)
    monkeypatch.chdir(tmp_path)


def write_source(root: Path, value: object = CONFIG) -> Path:
    source = root / SOURCE
    source.parent.mkdir(exist_ok=True)
    source.write_text(json.dumps(value), encoding="utf-8")
    return source


def sync(root: Path, concern: str | None, *options: str) -> Result:
    return runner.invoke(
        app, ["sync", *([concern] if concern else []), "--path", str(root), *options]
    )


@pytest.mark.parametrize("concern", ["hooks", "permissions"])
@pytest.mark.parametrize(
    "failure", ["json", "io", "utf8", "root", "section", "directory", "symlink"]
)
@pytest.mark.parametrize("mode", ["write", "dry-run", "plan"])
def test_invalid_source_preserves_target_and_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concern: str, failure: str, mode: str
) -> None:
    source = write_source(tmp_path)
    target = tmp_path / TARGETS[concern]
    options = ("--from", "claude", "--to", "cursor")
    first = sync(tmp_path, concern, *options)
    assert first.exit_code == 0, first.output
    before_target = target.read_bytes()
    before_ledger = (tmp_path / LEDGER_PATH).read_bytes()

    if failure == "json":
        source.write_text("{ BROKEN JSON", encoding="utf-8")
    elif failure == "utf8":
        source.write_bytes(b"\xff")
    elif failure == "root":
        write_source(tmp_path, [])
    elif failure == "section":
        write_source(tmp_path, {concern: []})
    elif failure == "directory":
        source.unlink()
        source.mkdir()
    elif failure == "symlink":
        source.unlink()
        source.symlink_to("missing.json")
    else:
        real_read = Path.read_text

        def denied(path: Path, *args, **kwargs):
            if path == source:
                raise PermissionError("simulated source denial")
            return real_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)

    extra = () if mode == "write" else (f"--{mode}",)
    result = sync(tmp_path, concern, *options, *extra)
    assert result.exit_code == 1, result.output
    assert "source" in result.output.lower(), result.output
    assert target.read_bytes() == before_target
    assert (tmp_path / LEDGER_PATH).read_bytes() == before_ledger


@pytest.mark.parametrize("concern", ["hooks", "permissions"])
@pytest.mark.parametrize("empty", ["explicit", "absent"])
def test_valid_removal_revokes_owned_entries_and_preserves_human_entries(
    tmp_path: Path, concern: str, empty: str
) -> None:
    source = write_source(tmp_path)
    target = tmp_path / TARGETS[concern]
    target.parent.mkdir(exist_ok=True)
    human = (
        {"version": 1, "hooks": {"preToolUse": [{"command": "./human.sh"}]}}
        if concern == "hooks"
        else {"permissions": {"allow": ["Shell(human:*)"]}}
    )
    target.write_text(json.dumps(human), encoding="utf-8")
    options = ("--from", "claude", "--to", "cursor")
    assert sync(tmp_path, concern, *options).exit_code == 0
    if empty == "explicit":
        write_source(tmp_path, {"hooks": {}, "permissions": {"allow": []}})
    else:
        source.unlink()
    result = sync(tmp_path, concern, *options)
    assert result.exit_code == 0, result.output
    text = target.read_text()
    assert "human" in text
    assert "./guard.sh" not in text
    assert "myapp" not in text
    ledger = load_ledger(tmp_path)
    assert (
        not ledger.hooks(AIToolID.CURSOR)
        if concern == "hooks"
        else not ledger.permissions(AIToolID.CURSOR)
    )


@pytest.mark.parametrize("concern", ["hooks", "permissions"])
def test_multisource_wizard_refuses_incomplete_discovery(tmp_path: Path, concern: str) -> None:
    source = write_source(tmp_path)
    assert sync(tmp_path, concern, "--from", "claude", "--to", "cursor").exit_code == 0
    target = tmp_path / TARGETS[concern]
    before_target = target.read_bytes()
    before_ledger = (tmp_path / LEDGER_PATH).read_bytes()
    source.write_text("{ BROKEN JSON", encoding="utf-8")
    # Cursor remains a valid second source; partial discovery cannot authorize
    # even additive writes or ownership changes for this concern.
    data = build_sync_data(tmp_path)
    assert data.source_errors[SyncConcern(concern)]
    assert data.hooks if concern == "hooks" else data.allowed_commands
    result = sync(tmp_path, concern)
    assert result.exit_code == 1, result.output
    assert target.read_bytes() == before_target
    assert (tmp_path / LEDGER_PATH).read_bytes() == before_ledger
    assert source.read_text() == "{ BROKEN JSON"


def test_wizard_invalid_source_without_ownership_is_an_error(tmp_path: Path) -> None:
    write_source(tmp_path).write_text("{ BROKEN JSON", encoding="utf-8")
    result = sync(tmp_path, "hooks")
    assert result.exit_code == 1, result.output
    assert not (tmp_path / TARGETS["hooks"]).exists()
    assert not (tmp_path / LEDGER_PATH).exists()


def test_unaffected_concerns_still_run_and_keep_hook_ownership(tmp_path: Path) -> None:
    write_source(tmp_path)
    assert sync(tmp_path, "hooks", "--from", "claude", "--to", "cursor").exit_code == 0
    hook_target = tmp_path / TARGETS["hooks"]
    before_target = hook_target.read_bytes()
    owned = load_ledger(tmp_path).hooks(AIToolID.CURSOR)
    write_source(tmp_path, {"hooks": [], "permissions": {"allow": ["Bash(other:*)"]}})
    (tmp_path / "CLAUDE.md").write_text("Project rules\n", encoding="utf-8")
    result = sync(tmp_path, None, "--from", "claude", "--to", "cursor")
    assert result.exit_code == 1, result.output
    assert (tmp_path / ".cursorrules").read_text() == "Project rules\n"
    assert "Shell(other:*)" in (tmp_path / TARGETS["permissions"]).read_text()
    assert hook_target.read_bytes() == before_target
    assert load_ledger(tmp_path).hooks(AIToolID.CURSOR) == owned


def test_invalid_other_concern_does_not_fail_selected_concern(tmp_path: Path) -> None:
    write_source(tmp_path, {"hooks": [], "permissions": {"allow": ["Bash(other:*)"]}})
    result = sync(tmp_path, "permissions", "--from", "claude", "--to", "cursor")
    assert result.exit_code == 0, result.output
    assert "Shell(other:*)" in (tmp_path / TARGETS["permissions"]).read_text()


@pytest.mark.parametrize(
    "tool,path",
    [
        ("claude", ".claude/settings.json"),
        ("cursor", ".cursor/hooks.json"),
        ("codex", ".codex/hooks.json"),
        ("copilot", ".github/hooks/hooks.json"),
        ("antigravity-cli", ".agents/hooks.json"),
    ],
)
def test_every_hook_reader_reports_invalid_source(tmp_path: Path, tool: str, path: str) -> None:
    source = tmp_path / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{ BROKEN JSON", encoding="utf-8")
    errors: list[str] = []
    assert discover_hooks(tmp_path, AIToolID(tool), errors=errors) == []
    assert len(errors) == 1
    assert path in errors[0]


@pytest.mark.parametrize("concern", ["hooks", "permissions"])
def test_scene_filter_preserves_invalid_source_concern(tmp_path: Path, concern: str) -> None:
    from crossby.models.config import SceneConfig, SceneSelector
    from crossby.scenes.engine import apply_scene
    from crossby.services.scene_resolution import resolve_scene
    from crossby.sync.readers import scan_project

    source = write_source(tmp_path)
    assert sync(tmp_path, concern, "--from", "claude", "--to", "cursor").exit_code == 0
    target = tmp_path / TARGETS[concern]
    before_target = target.read_bytes()
    ledger = load_ledger(tmp_path)
    source.write_text("{ BROKEN JSON", encoding="utf-8")
    scene = SceneConfig(**{concern: SceneSelector(include=[])})
    scan = scan_project(tmp_path, [AIToolID.CLAUDE, AIToolID.CURSOR])
    results = apply_scene(resolve_scene(scene, scan, tmp_path), tmp_path)
    assert any(r.concern == SyncConcern(concern) and r.action == "error" for r in results)
    assert target.read_bytes() == before_target
    after = load_ledger(tmp_path)
    assert after.hooks(AIToolID.CURSOR) == ledger.hooks(AIToolID.CURSOR)
    assert after.permissions(AIToolID.CURSOR) == ledger.permissions(AIToolID.CURSOR)
