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
    write_source(
        tmp_path,
        {"hooks": {"PreToolUse": {}}, "permissions": {"allow": ["Bash(other:*)"]}},
    )
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


HOOK_SOURCES = {
    "claude": Path(".claude/settings.json"),
    "codex": Path(".codex/hooks.json"),
    "cursor": Path(".cursor/hooks.json"),
    "copilot": Path(".github/hooks/hooks.json"),
    "antigravity-cli": Path(".agents/hooks.json"),
}


def native_hook_entry(tool: str) -> dict:
    if tool == "copilot":
        return {"type": "command", "bash": "./guard.sh"}
    handler = {"type": "command", "command": "./guard.sh"}
    if tool == "cursor":
        return {**handler, "matcher": "Shell", "failClosed": True}
    matcher = "run_command" if tool == "antigravity-cli" else "Bash"
    return {"matcher": matcher, "hooks": [handler]}


def native_hook_config(tool: str, entries: object) -> dict:
    event = "preToolUse" if tool in {"cursor", "copilot"} else "PreToolUse"
    events = {event: entries}
    return {"guard": events} if tool == "antigravity-cli" else {"hooks": events}


def hook_source_and_target(root: Path, tool: str) -> tuple[Path, Path, tuple[str, ...]]:
    source = root / HOOK_SOURCES[tool]
    source.parent.mkdir(parents=True, exist_ok=True)
    target_tool = "claude" if tool == "cursor" else "cursor"
    target = root / (SOURCE if target_tool == "claude" else TARGETS["hooks"])
    return source, target, ("--from", tool, "--to", target_tool)


NESTED_FAILURES = (
    [
        (tool, flaw)
        for tool in HOOK_SOURCES
        for flaw in (
            "event",
            "entry",
            "command",
            "empty-command",
            "missing-command",
            "handler-type",
        )
    ]
    + [
        (tool, flaw)
        for tool in ("claude", "codex", "antigravity-cli")
        for flaw in ("inner-hooks", "missing-hooks", "inner-entry", "matcher", "null-matcher")
    ]
    + [
        ("cursor", flaw)
        for flaw in ("matcher", "null-matcher", "tools", "mixed-tools", "empty-tool", "failClosed")
    ]
    + [("antigravity-cli", "container")]
)


@pytest.mark.parametrize("mode", ["write", "plan"])
@pytest.mark.parametrize("tool,flaw", NESTED_FAILURES)
def test_malformed_supported_hook_data_cannot_revoke(
    tmp_path: Path, tool: str, flaw: str, mode: str
) -> None:
    source, target, options = hook_source_and_target(tmp_path, tool)
    source.write_text(json.dumps(native_hook_config(tool, [native_hook_entry(tool)])))
    first = sync(tmp_path, "hooks", *options)
    assert first.exit_code == 0, first.output
    before_target = target.read_bytes()
    before_ledger = (tmp_path / LEDGER_PATH).read_bytes()

    entry = native_hook_entry(tool)
    handler = entry["hooks"][0] if "hooks" in entry else entry
    command_key = "bash" if tool == "copilot" else "command"
    entries: object = [entry]
    if flaw == "event":
        entries = entry  # Supervisor's P1: a valid JSON object in place of an event list.
    elif flaw == "entry":
        entries = ["not an object"]
    elif flaw == "command":
        handler[command_key] = ["./guard.sh"]
    elif flaw == "empty-command":
        handler[command_key] = ""
    elif flaw == "missing-command":
        handler.pop(command_key)
    elif flaw == "handler-type":
        handler["type"] = None
    elif flaw == "inner-hooks":
        entry["hooks"] = {"command": "./guard.sh"}
    elif flaw == "missing-hooks":
        entry.pop("hooks")
    elif flaw == "inner-entry":
        entry["hooks"] = ["./guard.sh"]
    elif flaw in {"matcher", "null-matcher"}:
        entry["matcher"] = 42 if flaw == "matcher" else None
    elif flaw in {"tools", "mixed-tools", "empty-tool"}:
        entry.pop("matcher")
        entry["tools"] = {"tools": None, "mixed-tools": ["Shell", 42], "empty-tool": [""]}[flaw]
    elif flaw == "failClosed":
        entry["failClosed"] = "true"
    config = native_hook_config(tool, entries)
    if flaw == "container":
        config = {"guard": [config["guard"]]}
    source.write_text(json.dumps(config))

    result = sync(tmp_path, "hooks", *options, *(("--plan",) if mode == "plan" else ()))
    assert result.exit_code == 1, result.output
    assert "Invalid source input" in result.output, result.output
    assert str(HOOK_SOURCES[tool]) in result.output, result.output
    assert target.read_bytes() == before_target
    assert (tmp_path / LEDGER_PATH).read_bytes() == before_ledger
    assert "./guard.sh" in target.read_text()


@pytest.mark.parametrize("tool", HOOK_SOURCES)
def test_native_extensions_do_not_block_valid_hook_sync_or_explicit_deletion(
    tmp_path: Path, tool: str
) -> None:
    source, target, options = hook_source_and_target(tmp_path, tool)
    entry = native_hook_entry(tool)
    # These fields do not determine whether the reader retains a command/scope.
    entry.update({"description": 42, "timeout": {"future": "metadata"}})
    unsupported = [
        {"type": kind, "prompt": "native-only", "command": None, "bash": None, "matcher": None}
        for kind in ("prompt", "agent", "http", "future-native-handler")
    ]
    if "hooks" in entry:
        entry["hooks"].extend(unsupported)
        entries = [entry]
    else:
        entries = [entry, *unsupported]
    config = native_hook_config(tool, entries)
    events = config["guard"] if tool == "antigravity-cli" else config["hooks"]
    events["FutureNativeEvent"] = {"unknown": "layout"}
    source.write_text(json.dumps(config))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    assert "./guard.sh" in target.read_text()

    # Valid empty known events retain deletion authority even alongside extensions.
    config = native_hook_config(tool, [])
    events = config["guard"] if tool == "antigravity-cli" else config["hooks"]
    events["FutureNativeEvent"] = {"unknown": "layout"}
    source.write_text(json.dumps(config))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    assert "./guard.sh" not in target.read_text()
    assert not load_ledger(tmp_path).hooks(AIToolID(options[-1]))


@pytest.mark.parametrize("tool", ["claude", "codex", "antigravity-cli"])
def test_explicit_empty_nested_handler_list_still_revokes(tmp_path: Path, tool: str) -> None:
    source, target, options = hook_source_and_target(tmp_path, tool)
    source.write_text(json.dumps(native_hook_config(tool, [native_hook_entry(tool)])))
    assert sync(tmp_path, "hooks", *options).exit_code == 0
    entry = native_hook_entry(tool)
    entry["hooks"] = []
    source.write_text(json.dumps(native_hook_config(tool, [entry])))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    assert "./guard.sh" not in target.read_text()
    assert not load_ledger(tmp_path).hooks(AIToolID(options[-1]))


def test_copilot_powershell_only_is_valid_unsupported_input(tmp_path: Path) -> None:
    source, target, options = hook_source_and_target(tmp_path, "copilot")
    source.write_text(
        json.dumps(native_hook_config("copilot", [{"type": "command", "powershell": "echo ok"}]))
    )
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    assert not target.exists()


def test_cursor_plain_matcher_ignores_unused_legacy_tools(tmp_path: Path) -> None:
    source, target, options = hook_source_and_target(tmp_path, "cursor")
    entry = native_hook_entry("cursor")
    entry["tools"] = None  # Native matcher already fully defines the recovered scope.
    source.write_text(json.dumps(native_hook_config("cursor", [entry])))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    assert "./guard.sh" in target.read_text()


def test_nested_malformation_blocks_partial_multisource_wizard(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    assert sync(tmp_path, "hooks", "--from", "claude", "--to", "cursor").exit_code == 0
    target = tmp_path / TARGETS["hooks"]
    before_target = target.read_bytes()
    before_ledger = (tmp_path / LEDGER_PATH).read_bytes()
    # Exact supervisor reproduction, alongside the valid Cursor discovery.
    malformed = {"hooks": {"PreToolUse": {"hooks": [{"command": "./guard.sh"}]}}}
    source.write_text(json.dumps(malformed))
    result = sync(tmp_path, "hooks")
    assert result.exit_code == 1, result.output
    assert target.read_bytes() == before_target
    assert (tmp_path / LEDGER_PATH).read_bytes() == before_ledger
    assert json.loads(source.read_text()) == malformed


@pytest.mark.parametrize("tool,event", [("codex", "Notification"), ("copilot", "UserPromptSubmit")])
def test_known_imported_event_is_validated_even_if_source_writer_cannot_emit_it(
    tmp_path: Path, tool: str, event: str
) -> None:
    source = tmp_path / HOOK_SOURCES[tool]
    source.parent.mkdir(parents=True)
    entry = native_hook_entry(tool)
    source.write_text(json.dumps({"hooks": {event: [entry]}}))
    options = ("--from", tool, "--to", "claude")
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    target = tmp_path / SOURCE
    before_target = target.read_bytes()
    before_ledger = (tmp_path / LEDGER_PATH).read_bytes()
    source.write_text(json.dumps({"hooks": {event: entry}}))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 1, result.output
    assert target.read_bytes() == before_target
    assert (tmp_path / LEDGER_PATH).read_bytes() == before_ledger


@pytest.mark.parametrize("tool", ["claude", "codex", "cursor", "antigravity-cli"])
def test_unrepresentable_native_matcher_still_syncs(tmp_path: Path, tool: str) -> None:
    source, target, options = hook_source_and_target(tmp_path, tool)
    entry = native_hook_entry(tool)
    entry["matcher"] = "(Write|Shell)"
    source.write_text(json.dumps(native_hook_config(tool, [entry])))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    assert "./guard.sh" in target.read_text()


def test_agy_stop_ignores_scope_metadata_but_refuses_bad_command(tmp_path: Path) -> None:
    source, target, options = hook_source_and_target(tmp_path, "antigravity-cli")
    entry = {"command": "./guard.sh", "matcher": None, "hooks": {"ignored": "metadata"}}
    source.write_text(json.dumps({"guard": {"Stop": [entry]}}))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 0, result.output
    before_target = target.read_bytes()
    before_ledger = (tmp_path / LEDGER_PATH).read_bytes()
    entry["command"] = None
    source.write_text(json.dumps({"guard": {"Stop": [entry]}}))
    result = sync(tmp_path, "hooks", *options)
    assert result.exit_code == 1, result.output
    assert target.read_bytes() == before_target
    assert (tmp_path / LEDGER_PATH).read_bytes() == before_ledger


@pytest.mark.parametrize("mode", ["write", "dry-run", "plan", "doctor"])
def test_source_error_is_reported_when_target_has_no_concern_writer(
    tmp_path: Path, mode: str
) -> None:
    source = write_source(tmp_path)
    assert sync(tmp_path, "permissions", "--from", "claude", "--to", "cursor").exit_code == 0
    target = tmp_path / TARGETS["permissions"]
    before_target = target.read_bytes()
    before_ledger = (tmp_path / LEDGER_PATH).read_bytes()
    source.write_text(json.dumps({"permissions": []}))
    extra = () if mode == "write" else (f"--{mode}",)
    result = sync(tmp_path, "permissions", "--from", "claude", "--to", "codex", *extra)
    assert result.exit_code == 1, result.output
    assert "Invalid source input" in result.output
    assert "permissions must be an object" in result.output
    if mode == "doctor":
        assert "readiness: low" in result.output.lower()
    assert target.read_bytes() == before_target
    assert (tmp_path / LEDGER_PATH).read_bytes() == before_ledger
    assert not (tmp_path / ".codex/config.toml").exists()


def test_source_error_is_reported_when_no_other_target_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_source(tmp_path)
    source.write_text(json.dumps({"hooks": {"PreToolUse": {"hooks": [{"command": "./guard.sh"}]}}}))
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed", lambda: [AIToolID.CLAUDE]
    )
    result = sync(tmp_path, "hooks", "--from", "claude", "--plan")
    assert result.exit_code == 1, result.output
    assert "Invalid source input" in result.output
    assert not (tmp_path / LEDGER_PATH).exists()
