"""Read-only sessions and unattended resume, for every managed-headless adapter.

A responder must never change the project, and each side of a consultation
keeps its own native session. Every adapter therefore declares a read-only
profile and an unattended resume path, each verified live on a named CLI build
(see ``docs/unattended-headless-verification.md``). These tests pin the exact
native arguments of those profiles and the checks around them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import HeadlessRequestError, HeadlessUnsupportedError
from crossby.models.ai import (
    AIToolID,
    HeadlessInteractionMode,
    HeadlessTerminalStatus,
    PlanCommandPolicy,
)
from crossby.utils.versioning import BinaryVersion, parse_semver
from tests.unit.test_ai_tools.test_headless_adapters import (
    AGY_RESULT,
    CODEX_EVENTS,
    COPILOT_SESSION,
    CURSOR_RESULT,
    OPENCODE_EVENTS,
    _adapter,
    _agy_result,
    _command,
    _copilot_events,
    _fake_cli,
    _request,
)

MANAGED = (
    AIToolID.CLAUDE,
    AIToolID.CODEX,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.OPENCODE,
    AIToolID.ANTIGRAVITY_CLI,
)
UUID = "5f0c1a7e-2b7d-4c55-9a3e-0f6f7d2c9b11"


def _jsonl(*frames: dict[str, Any]) -> str:
    return "\n".join(json.dumps(frame) for frame in frames) + "\n"


def _unattended_floor(adapter: AbstractAITool) -> str:
    floor = adapter.capabilities().headless.verified_version
    assert floor is not None
    return floor


def _resume_floor(adapter: AbstractAITool) -> str:
    return type(adapter).HEADLESS_RESUME_VERIFIED_VERSION or _unattended_floor(adapter)


def _read_only_floor(adapter: AbstractAITool) -> str:
    return type(adapter).HEADLESS_READ_ONLY_VERIFIED_VERSION or _unattended_floor(adapter)


def _at_verified_build(
    monkeypatch: pytest.MonkeyPatch, tool: AIToolID, argv: list[str]
) -> AbstractAITool:
    """An adapter running ``argv`` whose CLI reports its resume-verified build."""
    adapter = _adapter(monkeypatch, tool, argv)
    version = _resume_floor(adapter)
    normalized = parse_semver(version)
    assert normalized is not None
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion(normalized, version),
        raising=True,
    )
    return adapter


# ---------------------------------------------------------------------------
# Capabilities and version gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", MANAGED)
def test_every_managed_tool_resumes_and_runs_read_only_on_its_verified_build(
    tool: AIToolID,
) -> None:
    adapter = AbstractAITool.get(tool)
    assert adapter.headless_resume_supported(_resume_floor(adapter))
    assert adapter.headless_read_only_supported(_read_only_floor(adapter))
    assert not adapter.headless_resume_supported("0.0.1")
    assert not adapter.headless_read_only_supported("0.0.1")


@pytest.mark.parametrize("tool", (AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.OPENCODE))
def test_resume_and_read_only_were_verified_after_the_unattended_floor(tool: AIToolID) -> None:
    adapter = AbstractAITool.get(tool)
    floor = _unattended_floor(adapter)
    assert not adapter.headless_resume_supported(floor)
    assert not adapter.headless_read_only_supported(floor)


@pytest.mark.parametrize(
    ("tool", "build"),
    [
        (AIToolID.CURSOR, "2026.11.05-86047df"),
        (AIToolID.COPILOT, "1.0.105"),
        (AIToolID.ANTIGRAVITY_CLI, "1.5.4"),
    ],
)
def test_stdin_prompt_tools_verified_everything_on_one_unattended_floor(
    tool: AIToolID, build: str
) -> None:
    """The piped prompt, resume and read-only were verified together on one build."""
    adapter = AbstractAITool.get(tool)
    assert _unattended_floor(adapter) == build
    assert type(adapter).HEADLESS_RESUME_VERIFIED_VERSION is None
    assert type(adapter).HEADLESS_READ_ONLY_VERIFIED_VERSION is None


def test_an_older_build_refuses_resume_and_read_only(tmp_path: Path) -> None:
    adapter = AbstractAITool.get(AIToolID.CODEX)
    old = BinaryVersion((0, 160, 0), "0.160.0")
    adapter._validate_headless_mode_version(_request(tmp_path), old)  # type: ignore[attr-defined]
    with pytest.raises(HeadlessUnsupportedError, match=r"verified on 0\.166\.0"):
        adapter._validate_headless_mode_version(  # type: ignore[attr-defined]
            _request(tmp_path, resume_id=UUID), old
        )
    with pytest.raises(HeadlessUnsupportedError, match=r"Read-only Codex CLI sessions"):
        adapter._validate_headless_mode_version(  # type: ignore[attr-defined]
            _request(tmp_path, read_only=True), old
        )


def test_a_mode_without_a_read_only_profile_is_refused(tmp_path: Path) -> None:
    adapter = AbstractAITool.get(AIToolID.CODEX)
    with pytest.raises(HeadlessUnsupportedError, match="no verified way to run a read-only"):
        adapter._validate_headless_request(  # type: ignore[attr-defined]
            _request(tmp_path, read_only=True, interaction_mode=HeadlessInteractionMode.BROKERED),
            interaction_handler=lambda _interaction: None,  # type: ignore[arg-type,return-value]
            require_existing_working_dir=True,
        )


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------


def test_claude_read_only_is_a_closed_tool_set_that_keeps_the_folder_setup(
    tmp_path: Path,
) -> None:
    command = _command(AIToolID.CLAUDE, _request(tmp_path, read_only=True))
    assert command[command.index("--tools") + 1] == "Read,Grep,Glob,Skill"
    assert command[command.index("--settings") + 1] == '{"disableAllHooks":true}'
    # Both would also drop the folder's CLAUDE.md and skills.
    assert "--setting-sources" not in command
    assert "--restricted" not in command


def test_claude_read_only_rejects_a_command_policy(tmp_path: Path) -> None:
    request = _request(
        tmp_path, read_only=True, command_policy=PlanCommandPolicy(allowed_commands=("git log",))
    )
    with pytest.raises(HeadlessRequestError, match="has no shell"):
        AbstractAITool.get(AIToolID.CLAUDE)._validate_headless_requirements(request)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------


def test_codex_read_only_pins_the_sandbox_and_approvals(tmp_path: Path) -> None:
    command = _command(
        AIToolID.CODEX,
        _request(tmp_path, read_only=True, trusted_dirs=(tmp_path,)),
        schema_path=None,
    )
    assert command[:8] == [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--sandbox",
        "read-only",
    ]
    assert 'sandbox_mode="read-only"' in command
    # Ambient approval settings (an auto reviewer) must not approve escalations.
    assert 'approval_policy="never"' in command
    assert command[command.index("--disable") + 1] == "hooks"
    # Read-only takes no writable roots, and no workspace network toggle.
    assert "--add-dir" not in command
    assert not any("network_access" in arg for arg in command)
    assert command[-1] == "-"


def test_codex_resume_continues_the_thread_with_config_overrides(tmp_path: Path) -> None:
    command = _command(
        AIToolID.CODEX, _request(tmp_path, resume_id=UUID, read_only=True), schema_path=None
    )
    assert command[:5] == ["codex", "exec", "resume", "--json", "--skip-git-repo-check"]
    # ``codex exec resume`` takes no --sandbox, --color or --add-dir.
    assert "--sandbox" not in command and "--color" not in command
    assert 'sandbox_mode="read-only"' in command
    assert 'approval_policy="never"' in command
    assert command[-2:] == [UUID, "-"]


def test_only_a_read_only_codex_session_may_run_outside_a_git_repository(
    tmp_path: Path,
) -> None:
    # The guard protects files no version control can restore; a session that
    # can write keeps it, fresh or resumed.
    for request in (_request(tmp_path), _request(tmp_path, resume_id=UUID)):
        assert "--skip-git-repo-check" not in _command(AIToolID.CODEX, request, schema_path=None)


def test_codex_resume_carries_writable_roots_as_config(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    command = _command(
        AIToolID.CODEX, _request(tmp_path, resume_id=UUID, trusted_dirs=(extra,)), schema_path=None
    )
    assert 'sandbox_mode="workspace-write"' in command
    assert f"sandbox_workspace_write.writable_roots={json.dumps([str(extra)])}" in command
    assert "--add-dir" not in command


def test_codex_resume_must_report_the_resumed_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _at_verified_build(
        monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(*CODEX_EVENTS))
    )
    result = adapter.run_headless_session(_request(tmp_path, resume_id=UUID))
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert "different thread" in " ".join(result.warnings)
    resumed = adapter.run_headless_session(
        _request(tmp_path, resume_id=CODEX_EVENTS[0]["thread_id"])
    )
    assert resumed.status is HeadlessTerminalStatus.SUCCEEDED
    assert resumed.thread_id == CODEX_EVENTS[0]["thread_id"]


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


def test_cursor_read_only_is_a_closed_tool_set_and_resume_names_the_chat(
    tmp_path: Path,
) -> None:
    command = _command(AIToolID.CURSOR, _request(tmp_path, read_only=True, resume_id=UUID))
    allowed = command[command.index("--allowed-tools") + 1].split(",")
    assert allowed == [
        "read_tool_call",
        "grep_tool_call",
        "glob_tool_call",
        "ls_tool_call",
        "sem_search_tool_call",
    ]
    # Neither mode stops print mode's edit tool, so neither stands in for it.
    assert "--mode" not in command
    assert command[command.index("--resume") + 1] == UUID
    assert _request(tmp_path).prompt not in command  # piped on stdin


def test_cursor_resume_must_report_the_resumed_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _at_verified_build(
        monkeypatch, AIToolID.CURSOR, _fake_cli(stdout=json.dumps(CURSOR_RESULT))
    )
    other = adapter.run_headless_session(_request(tmp_path, resume_id=UUID))
    assert other.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert "different chat" in " ".join(other.warnings)
    same = adapter.run_headless_session(_request(tmp_path, resume_id=CURSOR_RESULT["session_id"]))
    assert same.status is HeadlessTerminalStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# GitHub Copilot
# ---------------------------------------------------------------------------


def test_copilot_pins_new_sessions_and_resumes_them_by_id(tmp_path: Path) -> None:
    fresh = _command(AIToolID.COPILOT, _request(tmp_path), session_id=UUID)
    assert fresh[fresh.index("--session-id") + 1] == UUID
    other = "0c1e8b2a-6f4d-4a3b-9c2d-1e5f7a9b3c4d"
    resumed = _command(AIToolID.COPILOT, _request(tmp_path, resume_id=other), session_id=UUID)
    assert resumed[resumed.index("--session-id") + 1] == other
    assert resumed.count("--session-id") == 1


def test_copilot_read_only_denies_writes_and_the_shell(tmp_path: Path) -> None:
    command = _command(AIToolID.COPILOT, _request(tmp_path, read_only=True), session_id=UUID)
    denied = [command[i + 1] for i, arg in enumerate(command) if arg == "--deny-tool"]
    assert denied == ["write", "shell"]
    assert "--allow-all-tools" not in command


def test_copilot_rejects_bad_resume_ids_and_read_only_command_policies(tmp_path: Path) -> None:
    adapter = AbstractAITool.get(AIToolID.COPILOT)

    def validate(**updates: Any) -> None:
        adapter._validate_headless_request(  # type: ignore[attr-defined]
            _request(tmp_path, **updates),
            interaction_handler=None,
            require_existing_working_dir=True,
        )

    with pytest.raises(HeadlessRequestError, match="UUIDs"):
        validate(resume_id="not-a-uuid")
    with pytest.raises(HeadlessRequestError, match="denies the shell"):
        validate(read_only=True, command_policy=PlanCommandPolicy(allowed_commands=("ls",)))


def test_copilot_reports_the_session_it_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _at_verified_build(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=_copilot_events()))
    fresh = adapter.run_headless_session(_request(tmp_path))
    assert fresh.status is HeadlessTerminalStatus.SUCCEEDED
    assert fresh.session_id == COPILOT_SESSION
    adapter = _at_verified_build(
        monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=_copilot_events(session=UUID))
    )
    resumed = adapter.run_headless_session(_request(tmp_path, resume_id=UUID))
    assert resumed.status is HeadlessTerminalStatus.SUCCEEDED
    assert resumed.session_id == UUID
    # A resumed run must report the session it was given.
    adapter = _at_verified_build(monkeypatch, AIToolID.COPILOT, _fake_cli(stdout=_copilot_events()))
    other = adapter.run_headless_session(_request(tmp_path, resume_id=UUID))
    assert other.status is HeadlessTerminalStatus.INVALID_OUTPUT


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------


def test_opencode_read_only_runs_a_session_only_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    adapter = AbstractAITool.get(AIToolID.OPENCODE)
    request = _request(tmp_path, read_only=True, resume_id="ses_abc")
    command = _command(AIToolID.OPENCODE, request)
    assert command[command.index("--agent") + 1] == "crossby-read-only"
    assert command[command.index("--session") + 1] == "ses_abc"
    env = adapter._headless_environment(request)  # type: ignore[attr-defined]
    agent = json.loads(env["OPENCODE_CONFIG_CONTENT"])["agent"]["crossby-read-only"]
    assert agent["permission"] == {
        "edit": "deny",
        "bash": "deny",
        "task": "deny",
        "question": "deny",
    }
    # A session that may write gets no inline config at all.
    assert "OPENCODE_CONFIG_CONTENT" not in adapter._headless_environment(_request(tmp_path))  # type: ignore[attr-defined]


def test_opencode_read_only_keeps_existing_inline_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(AIToolID.OPENCODE)
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT", json.dumps({"model": "x/y", "agent": {"mine": {"mode": "all"}}})
    )
    env = adapter._headless_environment(_request(tmp_path, read_only=True))  # type: ignore[attr-defined]
    merged = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert merged["model"] == "x/y"
    assert set(merged["agent"]) == {"mine", "crossby-read-only"}
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", "not json")
    with pytest.raises(HeadlessRequestError, match="not a JSON object"):
        adapter._validate_headless_requirements(_request(tmp_path, read_only=True))  # type: ignore[attr-defined]


def test_opencode_ignores_plain_text_a_plugin_prints_to_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Recorded on 1.19.5: a plugin's update checker printed this before the
    # first event, which used to fail every run as malformed output.
    chatter = "[auto-update-checker] Package not found, nothing to invalidate: some-plugin\n"
    adapter = _adapter(
        monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=chatter + _jsonl(*OPENCODE_EVENTS))
    )
    result = adapter.run_headless_session(_request(tmp_path))
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"
    assert any("1 non-JSON line" in warning for warning in result.warnings)


def test_opencode_still_rejects_a_broken_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = '{"type": "text", "sessionID": \n' + _jsonl(*OPENCODE_EVENTS)
    adapter = _adapter(monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=broken))
    result = adapter.run_headless_session(_request(tmp_path))
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert "malformed raw JSON events" in " ".join(result.warnings)


def test_opencode_resume_must_report_the_resumed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _at_verified_build(
        monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=_jsonl(*OPENCODE_EVENTS))
    )
    other = adapter.run_headless_session(_request(tmp_path, resume_id="ses_other"))
    assert other.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert "different session" in " ".join(other.warnings)
    same = adapter.run_headless_session(
        _request(tmp_path, resume_id=OPENCODE_EVENTS[0]["sessionID"])
    )
    assert same.status is HeadlessTerminalStatus.SUCCEEDED


def test_children_see_their_own_folder_as_pwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OpenCode takes its working directory from PWD, not the process cwd
    # (verified on 1.19.5), so a stale inherited PWD would put a responder in
    # the caller's folder instead of its own.
    seat = tmp_path / "seat"
    seat.mkdir()
    monkeypatch.setenv("PWD", "/somewhere/else")
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    program = (
        "import json,os,sys\n"
        "sys.stdin.read()\n"
        "sid='ses_pwd'\n"
        "text=json.dumps({'pwd':os.environ.get('PWD'),"
        "'config':os.environ.get('OPENCODE_CONFIG_CONTENT')})\n"
        "print(json.dumps({'type':'text','sessionID':sid,"
        "'part':{'type':'text','text':text,'sessionID':sid}}))\n"
        "print(json.dumps({'type':'step_finish','sessionID':sid,"
        "'part':{'type':'step-finish','reason':'stop','sessionID':sid}}))\n"
    )
    adapter = _at_verified_build(monkeypatch, AIToolID.OPENCODE, [sys.executable, "-c", program])
    result = adapter.run_headless_session(_request(seat, read_only=True))
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    seen = json.loads(result.final_text or "{}")
    assert seen["pwd"] == os.path.abspath(seat)
    assert "crossby-read-only" in json.loads(seen["config"])["agent"]


# ---------------------------------------------------------------------------
# Antigravity CLI
# ---------------------------------------------------------------------------


def test_antigravity_read_only_replaces_the_allow_list_in_plan_mode(tmp_path: Path) -> None:
    command = _command(
        AIToolID.ANTIGRAVITY_CLI,
        _request(tmp_path, read_only=True),
        print_timeout_seconds=30,
        conversation_id=UUID,
    )
    assert command[:3] == ["agy", "--conversation", UUID]
    assert command[command.index("--mode") + 1] == "plan"
    assert command[command.index("--settings") + 1] == '{"permissions":{"allow":[]}}'


def test_antigravity_resume_continues_the_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = AGY_RESULT["conversation_id"]
    payload = {**AGY_RESULT, "response": "OK\n", "denied_actions": [{"action": "write_file"}]}
    payload.pop("json_schema")
    payload.pop("structured_output")
    adapter = _at_verified_build(
        monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(payload))
    )
    other = adapter.run_headless_session(_request(tmp_path, resume_id=UUID))
    assert other.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert "different conversation" in " ".join(other.warnings)
    same = adapter.run_headless_session(_request(tmp_path, resume_id=conversation))
    assert same.status is HeadlessTerminalStatus.SUCCEEDED
    # Denied actions are reported by count only, never by their native names.
    assert same.denials == (
        "Antigravity CLI denied 1 action(s) that needed a permission headless mode cannot grant",
    )
    assert "write_file" not in " ".join(same.denials)
