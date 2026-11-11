"""A read-only session never starts in a folder whose own hooks the tool would run.

On Cursor, Copilot, OpenCode and Antigravity CLI the read-only profile keeps
hooks enabled, and a hook is a command that can change files (verified with
sentinel hooks; see ``docs/unattended-headless-verification.md#project-hooks``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import HeadlessProjectHooksError
from crossby.ai_tools.hook_inspection import project_hook_sources
from crossby.api import AskRequest, CliBackend, ProjectHooksError, ProjectWorkspace, ask
from crossby.models.ai import AIToolID, HeadlessTerminalStatus
from crossby.services import converse
from crossby.services.converse import DEPTH_ENV
from crossby.utils.versioning import BinaryVersion, parse_semver
from tests.unit.test_ai_tools.test_restrictions import SUCCESS, _recording_cli, _request

HOOK = {"version": 1, "hooks": {"preToolUse": [{"type": "command", "command": "touch x"}]}}
AGY_HOOK = {"crossby": {"PreToolUse": [{"matcher": ".*", "hooks": [{"command": "touch x"}]}]}}


@pytest.fixture(autouse=True)
def _top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


def _write(root: Path, relative: str, content: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))


@pytest.mark.parametrize(
    ("tool", "relative", "content"),
    [
        (AIToolID.CURSOR, ".cursor/hooks.json", HOOK),
        (AIToolID.COPILOT, ".github/hooks/hooks.json", HOOK),
        (AIToolID.COPILOT, ".github/hooks/team.json", HOOK),
        (AIToolID.ANTIGRAVITY_CLI, ".agents/hooks.json", AGY_HOOK),
        (AIToolID.OPENCODE, "opencode.json", {"plugin": ["opencode-notify"]}),
        (AIToolID.OPENCODE, ".opencode/plugin/notify.js", "export default {}"),
        (AIToolID.OPENCODE, ".opencode/plugins/notify.ts", "export default {}"),
    ],
)
def test_each_tools_project_hooks_are_found(
    tool: AIToolID, relative: str, content: object, tmp_path: Path
) -> None:
    _write(tmp_path, relative, content)
    sources = project_hook_sources(tool, tmp_path)
    assert not sources.clear
    assert sources.found and not sources.unreadable


@pytest.mark.parametrize(
    ("tool", "relative", "content"),
    [
        (AIToolID.CURSOR, ".cursor/hooks.json", {"version": 1, "hooks": {}}),
        (AIToolID.COPILOT, ".github/hooks/hooks.json", {"version": 1, "hooks": {"x": []}}),
        (AIToolID.OPENCODE, "opencode.json", {"plugin": []}),
        # Another tool's hooks are not this tool's.
        (AIToolID.CURSOR, ".github/hooks/hooks.json", HOOK),
    ],
)
def test_empty_or_foreign_sources_are_clear(
    tool: AIToolID, relative: str, content: object, tmp_path: Path
) -> None:
    _write(tmp_path, relative, content)
    assert project_hook_sources(tool, tmp_path).clear


@pytest.mark.parametrize(
    ("tool", "relative"),
    [
        (AIToolID.CURSOR, ".cursor/hooks.json"),
        (AIToolID.ANTIGRAVITY_CLI, ".agents/hooks.json"),
        (AIToolID.OPENCODE, "opencode.jsonc"),
    ],
)
def test_an_unreadable_source_is_never_treated_as_clear(
    tool: AIToolID, relative: str, tmp_path: Path
) -> None:
    _write(tmp_path, relative, '{"hooks": // comment\n{}}')
    sources = project_hook_sources(tool, tmp_path)
    assert sources.unreadable == (relative,)
    assert not sources.clear


def _at_floor(monkeypatch: pytest.MonkeyPatch, tool: AIToolID, argv: list[str]) -> AbstractAITool:
    adapter = AbstractAITool.get(tool)
    floor = type(adapter).HEADLESS_READ_ONLY_VERIFIED_VERSION or str(
        adapter.capabilities().headless.verified_version
    )
    normalized = parse_semver(floor)
    assert normalized is not None
    monkeypatch.setattr(
        type(adapter), "_headless_command", lambda self, request, **_k: list(argv), raising=False
    )
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion(normalized, floor),
        raising=True,
    )
    if tool is AIToolID.CLAUDE:
        from crossby.ai_tools import claude as claude_module
        from tests.unit.test_ai_tools.test_headless_adapters import CLAUDE_RESULT

        monkeypatch.setattr(claude_module, "_new_session_id", lambda: CLAUDE_RESULT["session_id"])
    return adapter


def test_a_read_only_session_is_refused_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, ".cursor/hooks.json", HOOK)
    record = tmp_path / "ran.json"
    adapter = _at_floor(
        monkeypatch, AIToolID.CURSOR, _recording_cli(record, SUCCESS[AIToolID.CURSOR], ())
    )
    with pytest.raises(HeadlessProjectHooksError) as raised:
        adapter.run_headless_session(_request(tmp_path, read_only=True))
    assert raised.value.sources == (".cursor/hooks.json",)
    assert "would run this folder's own hooks (.cursor/hooks.json)" in str(raised.value)
    assert not record.exists()
    # The same refusal surfaces at preflight.
    with pytest.raises(HeadlessProjectHooksError):
        adapter.preflight_headless_session(_request(tmp_path, read_only=True))
    # Accepting the hooks explicitly runs the session.
    accepted = adapter.run_headless_session(
        _request(tmp_path, read_only=True, allow_project_hooks=True)
    )
    assert accepted.status is HeadlessTerminalStatus.SUCCEEDED
    assert record.exists()


def test_sessions_that_may_write_and_tools_that_disable_hooks_are_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, ".cursor/hooks.json", HOOK)
    _write(tmp_path, ".claude/settings.json", {"hooks": {"PreToolUse": [{"hooks": []}]}})
    cursor = _at_floor(
        monkeypatch,
        AIToolID.CURSOR,
        _recording_cli(tmp_path / "c.json", SUCCESS[AIToolID.CURSOR], ()),
    )
    # The asker is the user's own agent: no read-only profile, no check.
    assert (
        cursor.run_headless_session(_request(tmp_path)).status is HeadlessTerminalStatus.SUCCEEDED
    )
    claude = _at_floor(
        monkeypatch,
        AIToolID.CLAUDE,
        _recording_cli(tmp_path / "a.json", SUCCESS[AIToolID.CLAUDE], ()),
    )
    # Claude Code's read-only profile disables every hook.
    result = claude.run_headless_session(_request(tmp_path, read_only=True))
    assert result.status is HeadlessTerminalStatus.SUCCEEDED


def test_ask_names_the_hooks_and_honors_the_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, ".agents/hooks.json", AGY_HOOK)
    _at_floor(
        monkeypatch,
        AIToolID.ANTIGRAVITY_CLI,
        _recording_cli(tmp_path / "ran.json", SUCCESS[AIToolID.ANTIGRAVITY_CLI], ()),
    )
    request = AskRequest(
        backend=CliBackend(tool="antigravity-cli"),
        prompt="q",
        workspace=ProjectWorkspace(path=tmp_path),
    )
    with pytest.raises(ProjectHooksError) as raised:
        ask(request)
    assert raised.value.sources == (".agents/hooks.json",)
    accepted = ask(request.model_copy(update={"allow_project_hooks": True}))
    assert accepted.succeeded


def test_converse_passes_the_override_to_every_read_only_session(tmp_path: Path) -> None:
    from tests.unit.test_services.test_converse import FakeRunner

    runner = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    converse.run_conversation(
        "goal",
        converse.make_participants(AIToolID.CLAUDE, AIToolID.CURSOR),
        working_dir=tmp_path,
        runner=runner,
        allow_project_hooks=True,
    )
    assert all(request.allow_project_hooks for _tool, request in runner.calls)
    assert all(request.allow_project_hooks for request in runner.preflight_requests)
    default = FakeRunner(["q\nNEXT: ASK", "a", "done\nNEXT: DONE"])
    converse.run_conversation(
        "goal",
        converse.make_participants(AIToolID.CLAUDE, AIToolID.CURSOR),
        working_dir=tmp_path,
        runner=default,
    )
    assert not any(request.allow_project_hooks for _tool, request in default.calls)


def test_the_cli_flag_reaches_the_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from crossby.cli.main import app
    from tests.unit.test_cli.test_converse_cmd import _use_ask_runner
    from tests.unit.test_services.test_converse import FakeRunner

    fake = _use_ask_runner(monkeypatch, FakeRunner(["ok"]))
    result = CliRunner().invoke(
        app, ["ask", "cursor", "q", "--allow-project-hooks", "--path", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert fake.calls[0][1].allow_project_hooks
