"""The public ``crossby.api.ask`` contract.

The product's use case is exercised through ``crossby.api`` imports only. The
native CLIs are real child processes replaying recorded payloads (see
``test_headless_adapters``), so prompt delivery, environments, cancellation
and parallel runs are real; a scripted runner covers outcomes a fake CLI
cannot produce on demand.
"""

from __future__ import annotations

import _thread
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import HeadlessTransportError
from crossby.api import (
    ENVELOPE_VERSION,
    SCHEMA_VERSION,
    AskRequest,
    AskRequestError,
    AskResult,
    AskStatus,
    AskUnsupportedError,
    CliBackend,
    EffortLevel,
    JsonOutput,
    NestedAskError,
    ProjectWorkspace,
    RequirementNotMetError,
    RestrictionChannel,
    RestrictionState,
    TextOutput,
    ask,
    preflight,
    render_envelope,
)
from crossby.models.ai import (
    AIToolID,
    HeadlessEvent,
    HeadlessEventKind,
    HeadlessSessionPreflight,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    Restriction,
    TokenUsage,
)
from crossby.services import ask as ask_service
from crossby.services.converse import DEPTH_ENV
from crossby.utils.versioning import BinaryVersion, parse_semver
from tests.unit.test_ai_tools.test_headless_adapters import CODEX_EVENTS, _jsonl

CODEX_FLOOR = "0.166.0"


@pytest.fixture(autouse=True)
def _top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


def _codex_cli(
    monkeypatch: pytest.MonkeyPatch,
    record_dir: Path,
    *,
    sleep_seconds: float = 0.0,
    version: str = CODEX_FLOOR,
) -> None:
    """Codex at its read-only floor, replaced by a child that records what it got.

    Each run writes ``<cwd name>.json`` into ``record_dir`` with its stdin,
    working directory and depth variable, so parallel runs stay separable.
    """
    program = "\n".join(
        (
            "import json,os,sys,time",
            "data=sys.stdin.read()",
            f"record={str(record_dir)!r}",
            "name=os.path.basename(os.getcwd())",
            "open(os.path.join(record, name + '.json'),'w').write(json.dumps({"
            f"'stdin': data, 'cwd': os.getcwd(), 'depth': os.environ.get({DEPTH_ENV!r}),"
            "'pid': os.getpid()}))",
            f"time.sleep({sleep_seconds!r})",
            f"sys.stdout.write({_jsonl(CODEX_EVENTS)!r})",
            "sys.stdout.flush()",
        )
    )
    argv = [sys.executable, "-c", program]
    adapter = AbstractAITool.get(AIToolID.CODEX)
    normalized = parse_semver(version)
    assert normalized is not None
    monkeypatch.setattr(
        type(adapter), "_headless_command", lambda self, request, **_k: list(argv), raising=False
    )
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion(normalized, version),
        raising=True,
    )


def _folder(tmp_path: Path, name: str = "persona") -> Path:
    folder = tmp_path / name
    folder.mkdir()
    return folder


def _record(record_dir: Path, folder: Path) -> dict[str, Any]:
    return dict(json.loads((record_dir / f"{folder.name}.json").read_text()))


def _ask_codex(folder: Path, **updates: Any) -> AskRequest:
    values: dict[str, Any] = {
        "backend": CliBackend(tool="codex"),
        "prompt": "What would make you reject this pricing page?",
        "workspace": ProjectWorkspace(path=folder),
    }
    values.update(updates)
    return AskRequest(**values)


# ---------------------------------------------------------------------------
# The product's use case, through public imports only
# ---------------------------------------------------------------------------


def test_an_ask_answers_with_its_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = _folder(tmp_path, "records")
    folder = _folder(tmp_path)
    _codex_cli(monkeypatch, records)
    result = ask(_ask_codex(folder))

    assert result.succeeded
    assert result.status is AskStatus.SUCCEEDED
    assert result.output == TextOutput(text="OK")
    assert result.text == "OK"
    assert result.schema_version == SCHEMA_VERSION
    # Sent exactly as written, in the persona's folder, one level deeper.
    seen = _record(records, folder)
    assert seen["stdin"] == "What would make you reject this pricing page?"
    assert seen["cwd"] == os.path.realpath(folder)
    assert seen["depth"] == "1"
    assert result.delivery == "prompt"
    assert result.envelope_version is None
    # Which build ran, and what it reported.
    provenance = result.provenance
    assert provenance.tool is AIToolID.CODEX
    assert provenance.installed_version == CODEX_FLOOR
    assert provenance.read_only_verified_version == CODEX_FLOOR
    assert provenance.verified_version == "0.154.0"
    assert provenance.thread_id == CODEX_EVENTS[0]["thread_id"]
    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.output_tokens) == (9810, 1)
    assert result.usage.complete is None
    # It answered without using a tool, which is not the same as not saying.
    assert result.usage.tool_steps == 0
    # Every channel, resolved for the read-only profile on that build.
    assert [item.channel for item in result.restrictions] == list(RestrictionChannel)
    assert result.restriction(RestrictionChannel.HOOKS).state is RestrictionState.ENFORCED
    assert result.restriction(RestrictionChannel.WEB).state is RestrictionState.NOT_ENFORCED
    assert result.restriction(RestrictionChannel.MCP_TOOLS).state is RestrictionState.UNKNOWN
    assert DEPTH_ENV not in os.environ


def test_instructions_travel_in_the_documented_envelope_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    folder = _folder(tmp_path)
    _codex_cli(monkeypatch, records)
    persona = "You are a CTO at a 40-person fintech.\nYou distrust usage-based pricing."
    result = ask(_ask_codex(folder, workspace=ProjectWorkspace(path=folder, instructions=persona)))

    stdin = _record(records, folder)["stdin"]
    assert stdin == (
        "<instructions>\n"
        f"{persona}\n"
        "</instructions>\n\n"
        "What would make you reject this pricing page?"
    )
    assert stdin == render_envelope("What would make you reject this pricing page?", persona)
    assert stdin.count(persona) == 1
    assert result.delivery == "prompt_envelope"
    assert result.envelope_version == ENVELOPE_VERSION == 1


def test_instructions_are_text_never_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    folder = _folder(tmp_path)
    (folder / "cto.md").write_text("FILE CONTENT")
    _codex_cli(monkeypatch, records)
    ask(_ask_codex(folder, workspace=ProjectWorkspace(path=folder, instructions="cto.md")))
    stdin = _record(records, folder)["stdin"]
    assert "<instructions>\ncto.md\n</instructions>" in stdin
    assert "FILE CONTENT" not in stdin


def test_an_unmet_requirement_is_refused_before_the_agent_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    folder = _folder(tmp_path)
    _codex_cli(monkeypatch, records)
    request = _ask_codex(
        folder,
        require_blocked=frozenset({RestrictionChannel.HOOKS, RestrictionChannel.WEB}),
    )
    with pytest.raises(RequirementNotMetError) as raised:
        ask(request)
    assert [item.channel for item in raised.value.unmet] == [RestrictionChannel.WEB]
    assert raised.value.unmet[0].state is RestrictionState.NOT_ENFORCED
    assert not any(records.iterdir())
    # A met requirement runs.
    met = _ask_codex(folder, require_blocked=frozenset({RestrictionChannel.HOOKS}))
    assert ask(met).succeeded


def test_preflight_reports_without_starting_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    folder = _folder(tmp_path)
    _codex_cli(monkeypatch, records)
    report = preflight(
        _ask_codex(
            folder,
            require_blocked=frozenset({RestrictionChannel.HOOKS, RestrictionChannel.WEB}),
        )
    )
    assert report.provenance.installed_version == CODEX_FLOOR
    assert [item.channel for item in report.unmet] == [RestrictionChannel.WEB]
    assert report.restriction(RestrictionChannel.FILE_EDITS).enforced
    assert "authentication" in report.deferred
    assert not any(records.iterdir())


def test_refusals_before_start_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = _folder(tmp_path)
    with pytest.raises(AskUnsupportedError, match="does not support managed headless"):
        ask(
            AskRequest(
                backend=CliBackend(tool="vscode"),
                prompt="q",
                workspace=ProjectWorkspace(path=folder),
            )
        )
    _codex_cli(monkeypatch, _folder(tmp_path, "records"))
    with pytest.raises(AskRequestError, match="existing working directory"):
        ask(_ask_codex(tmp_path / "missing"))


def test_a_call_from_inside_a_crossby_agent_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    _codex_cli(monkeypatch, records)
    monkeypatch.setenv(DEPTH_ENV, "1")
    with pytest.raises(NestedAskError, match="Nested consultations are disabled"):
        ask(_ask_codex(_folder(tmp_path)))
    assert not any(records.iterdir())


# ---------------------------------------------------------------------------
# Request objects
# ---------------------------------------------------------------------------


def test_requests_are_immutable_and_validated(tmp_path: Path) -> None:
    request = _ask_codex(tmp_path)
    with pytest.raises(ValidationError):
        request.prompt = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        request.backend.tool = AIToolID.CLAUDE  # type: ignore[misc]
    for timeout in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValidationError, match="timeout_seconds"):
            _ask_codex(tmp_path, timeout_seconds=timeout)
    with pytest.raises(ValidationError, match="the question is empty"):
        _ask_codex(tmp_path, prompt="  \n")
    with pytest.raises(ValidationError):
        ProjectWorkspace(path=tmp_path, instructions=" ")
    with pytest.raises(ValidationError):
        CliBackend(tool="codex", model="")
    with pytest.raises(ValidationError):
        AskRequest(
            backend=CliBackend(tool="codex"),
            prompt="q",
            workspace=ProjectWorkspace(path=tmp_path),
            extra=1,  # type: ignore[call-arg]
        )


def test_results_round_trip_through_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _codex_cli(monkeypatch, _folder(tmp_path, "records"))
    result = ask(_ask_codex(_folder(tmp_path)))
    payload = result.model_dump_json()
    assert json.loads(payload)["schema_version"] == 1
    assert AskResult.model_validate_json(payload) == result


# ---------------------------------------------------------------------------
# Outcomes once the agent has started
# ---------------------------------------------------------------------------


class ScriptedRunner:
    """A session runner that returns scripted outcomes and records requests."""

    def __init__(self, outcome: HeadlessSessionResult | BaseException) -> None:
        self.outcome = outcome
        self.requests: list[HeadlessSessionRequest] = []

    def preflight(
        self, tool: AIToolID, request: HeadlessSessionRequest, timeout_seconds: float
    ) -> HeadlessSessionPreflight:  # pragma: no cover - not used here
        raise AssertionError("not used")

    def run(
        self, tool: AIToolID, request: HeadlessSessionRequest, cancel: threading.Event | None
    ) -> HeadlessSessionResult:
        self.requests.append(request)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def restrictions(
        self, tool: AIToolID, request: HeadlessSessionRequest, version: str | None
    ) -> tuple[Restriction, ...]:
        return AbstractAITool.get(tool).headless_restrictions(request, version)


def _session(status: HeadlessTerminalStatus, **fields: Any) -> HeadlessSessionResult:
    values: dict[str, Any] = {
        "tool": AIToolID.CLAUDE,
        "version": "2.1.317",
        "status": status,
        "events": (
            HeadlessEvent(sequence=1, kind=HeadlessEventKind.TERMINAL, terminal_status=status),
        ),
        "duration_seconds": 1.0,
    }
    values.update(fields)
    return HeadlessSessionResult(**values)


def _claude_ask(tmp_path: Path, **updates: Any) -> AskRequest:
    values: dict[str, Any] = {
        "backend": CliBackend(tool="claude", model="opus", effort=EffortLevel.HIGH),
        "prompt": "q",
        "workspace": ProjectWorkspace(path=tmp_path),
    }
    values.update(updates)
    return AskRequest(**values)


def test_the_session_is_fresh_unattended_and_read_only(tmp_path: Path) -> None:
    runner = ScriptedRunner(_session(HeadlessTerminalStatus.SUCCEEDED, final_text="A"))
    ask_service.run_ask(_claude_ask(tmp_path, timeout_seconds=42.0), runner=runner)
    (request,) = runner.requests
    assert request.read_only
    assert request.resume_id is None
    assert request.interaction_mode.value == "unattended"
    assert (request.model, request.effort) == ("opus", EffortLevel.HIGH)
    assert request.timeout_seconds == 42.0
    assert request.environment == {DEPTH_ENV: "1"}


def _granted(tmp_path: Path, tool: str) -> tuple[Path, ...]:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    runner = ScriptedRunner(_session(HeadlessTerminalStatus.SUCCEEDED, final_text="A"))
    request = AskRequest(
        backend=CliBackend(tool=tool),
        prompt="q",
        workspace=ProjectWorkspace(path=_folder(tmp_path, f"seat-{tool}")),
        readable_dirs=(project,),
    )
    ask_service.run_ask(request, runner=runner)
    (session,) = runner.requests
    assert session.read_only
    return session.trusted_dirs


def test_readable_dirs_are_granted_only_where_the_tool_needs_it(tmp_path: Path) -> None:
    project = tmp_path / "project"
    # Claude Code and Copilot deny reads outside the folder they run in.
    assert _granted(tmp_path, "claude") == (project,)
    assert _granted(tmp_path, "copilot") == (project,)
    # These read any path under their read-only profile, so nothing is passed.
    for tool in ("codex", "cursor", "antigravity-cli"):
        assert _granted(tmp_path, tool) == ()
    # OpenCode has neither a grant nor verified outside reads.
    with pytest.raises(AskUnsupportedError, match="readable_dirs cannot be honored"):
        _granted(tmp_path, "opencode")


def test_readable_dirs_are_checked_before_the_agent_starts(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="absolute paths"):
        _claude_ask(tmp_path, readable_dirs=(Path("relative"),))
    with pytest.raises(ValidationError, match="asks to block"):
        _claude_ask(
            tmp_path,
            readable_dirs=(tmp_path,),
            require_blocked=frozenset({RestrictionChannel.READS_OUTSIDE_WORKSPACE}),
        )
    runner = ScriptedRunner(_session(HeadlessTerminalStatus.SUCCEEDED, final_text="A"))
    with pytest.raises(AskRequestError, match="not an existing folder"):
        ask_service.run_ask(
            _claude_ask(tmp_path, readable_dirs=(tmp_path / "missing",)), runner=runner
        )
    assert runner.requests == []


def test_a_readable_dir_reaches_claudes_command_line_without_a_write_tool(tmp_path: Path) -> None:
    project = _folder(tmp_path, "project")
    adapter = AbstractAITool.get(AIToolID.CLAUDE)
    session = ask_service.session_request(
        _claude_ask(tmp_path, readable_dirs=(project,)), timeout_seconds=30.0, environment={}
    )
    command = adapter._headless_command(session)  # type: ignore[attr-defined]
    assert command[command.index("--add-dir") + 1] == str(project)
    assert command[command.index("--tools") + 1] == "Read,Grep,Glob,Skill"


def _states(
    tool: str, tmp_path: Path, **updates: Any
) -> tuple[HeadlessSessionRequest, dict[str, str]]:
    request = AskRequest(
        backend=CliBackend(tool=tool),
        prompt="q",
        workspace=ProjectWorkspace(path=tmp_path),
        **updates,
    )
    session = ask_service.session_request(request, timeout_seconds=30.0, environment={})
    adapter = AbstractAITool.get(AIToolID(tool))
    _, floor, _ = ask_service._verified_versions(AIToolID(tool))
    found = adapter.headless_restrictions(session, floor)
    return session, {item.channel.value: item.state.value for item in found}


def test_search_commands_are_opt_in_and_reported_as_not_write_proof(tmp_path: Path) -> None:
    adapter = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI)

    session, states = _states("antigravity-cli", tmp_path)
    command = adapter._headless_command(session, print_timeout_seconds=30)  # type: ignore[attr-defined]
    assert command[command.index("--settings") + 1] == '{"permissions":{"allow":[]}}'
    assert (states["shell_commands"], states["shell_writes"]) == ("enforced", "enforced")

    session, states = _states("antigravity-cli", tmp_path, allow_search_commands=True)
    command = adapter._headless_command(session, print_timeout_seconds=30)  # type: ignore[attr-defined]
    allowed = json.loads(command[command.index("--settings") + 1])["permissions"]["allow"]
    assert "command(grep)" in allowed and "command(git grep)" in allowed
    assert not any("rm" in rule or "sed" in rule or "tee" in rule for rule in allowed)
    assert "plan" in command  # still asked not to edit
    # An allow rule matches a command's start, so a write can ride along: say so. A
    # caller that requires shell writes blocked is then refused like any unmet channel.
    assert (states["shell_commands"], states["shell_writes"]) == ("not_enforced", "not_enforced")
    assert states["file_edits"] == "enforced"

    # Tools with search tools of their own ignore the option.
    for tool in ("claude", "codex"):
        _, plain = _states(tool, tmp_path)
        _, asked = _states(tool, tmp_path, allow_search_commands=True)
        assert plain == asked


def test_json_null_is_present_output(tmp_path: Path) -> None:
    runner = ScriptedRunner(
        _session(HeadlessTerminalStatus.SUCCEEDED, final_json=None, final_json_present=True)
    )
    result = ask_service.run_ask(
        _claude_ask(tmp_path, response_schema={"type": "null"}), runner=runner
    )
    assert result.succeeded
    assert result.output == JsonOutput(value=None)
    assert result.text is None


def test_a_timed_out_ask_keeps_its_usage_and_native_ids(tmp_path: Path) -> None:
    runner = ScriptedRunner(
        _session(
            HeadlessTerminalStatus.TIMED_OUT,
            session_id="1492f340-106a-4548-b7ba-6c3aba132018",
            usage=TokenUsage(input_tokens=5, output_tokens=0),
            warnings=("The session reached its deadline.",),
        )
    )
    result = ask_service.run_ask(_claude_ask(tmp_path), runner=runner)
    assert result.status is AskStatus.TIMED_OUT
    assert not result.succeeded
    assert result.error == "claude session ended with status timed_out"
    assert result.usage is not None and result.usage.input_tokens == 5
    assert result.provenance.session_id == "1492f340-106a-4548-b7ba-6c3aba132018"
    assert result.warnings == ("The session reached its deadline.",)


def test_a_failed_ask_names_the_native_status_and_denials(tmp_path: Path) -> None:
    runner = ScriptedRunner(
        _session(
            HeadlessTerminalStatus.FAILED,
            native_status="api_error",
            denials=("Claude Code reported 1 permission denials",),
        )
    )
    result = ask_service.run_ask(_claude_ask(tmp_path), runner=runner)
    assert result.status is AskStatus.FAILED
    assert result.error == (
        "claude session ended with status failed (native: api_error) "
        "(Claude Code reported 1 permission denials)"
    )


def test_a_transport_failure_keeps_partial_evidence(tmp_path: Path) -> None:
    capability = AbstractAITool.get(AIToolID.CLAUDE).capabilities().headless
    partial = HeadlessSessionResult(
        tool=AIToolID.CLAUDE,
        version="2.1.317",
        status=HeadlessTerminalStatus.FAILED,
        session_id="1492f340-106a-4548-b7ba-6c3aba132018",
        duration_seconds=0.5,
        is_partial=True,
    )
    error = HeadlessTransportError(
        "The managed headless transport could not start the native CLI.",
        tool_id=AIToolID.CLAUDE,
        capability=capability,
        partial_result=partial,
    )
    result = ask_service.run_ask(_claude_ask(tmp_path), runner=ScriptedRunner(error))
    assert result.status is AskStatus.FAILED
    assert result.error == "claude: The managed headless transport could not start the native CLI."
    assert result.provenance.session_id == "1492f340-106a-4548-b7ba-6c3aba132018"


def test_cancelling_before_the_start_runs_nothing(tmp_path: Path) -> None:
    runner = ScriptedRunner(_session(HeadlessTerminalStatus.SUCCEEDED, final_text="A"))
    cancel = threading.Event()
    cancel.set()
    result = ask_service.run_ask(_claude_ask(tmp_path), runner=runner, cancel=cancel)
    assert result.status is AskStatus.CANCELLED
    assert runner.requests == []


def test_cancelling_a_running_ask_stops_its_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    _codex_cli(monkeypatch, records, sleep_seconds=30.0)
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    result = ask(_ask_codex(_folder(tmp_path)), cancel=cancel)
    assert result.status is AskStatus.CANCELLED
    assert time.monotonic() - started < 10


def test_ctrl_c_during_an_ask_stops_its_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    folder = _folder(tmp_path)
    _codex_cli(monkeypatch, records, sleep_seconds=30.0)
    threading.Timer(0.8, _thread.interrupt_main).start()
    with pytest.raises(KeyboardInterrupt):
        ask(_ask_codex(folder))
    pid = _record(records, folder)["pid"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the native CLI outlived the interrupted ask")


# ---------------------------------------------------------------------------
# Parallel asks
# ---------------------------------------------------------------------------


def test_parallel_asks_are_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = _folder(tmp_path, "records")
    _codex_cli(monkeypatch, records, sleep_seconds=0.3)
    folders = [_folder(tmp_path, f"persona-{index}") for index in range(6)]
    results: dict[int, AskResult] = {}

    def run(index: int) -> None:
        results[index] = ask(_ask_codex(folders[index], prompt=f"question {index}"))

    threads = [threading.Thread(target=run, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert sorted(results) == list(range(6))
    for index, folder in enumerate(folders):
        assert results[index].succeeded
        seen = _record(records, folder)
        assert seen["stdin"] == f"question {index}"
        assert seen["cwd"] == os.path.realpath(folder)
        assert seen["depth"] == "1"
    assert DEPTH_ENV not in os.environ


def test_cancelling_one_parallel_ask_leaves_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _folder(tmp_path, "records")
    _codex_cli(monkeypatch, records, sleep_seconds=1.5)
    cancels = [threading.Event() for _ in range(3)]
    results: dict[int, AskResult] = {}

    def run(index: int) -> None:
        folder = tmp_path / f"p{index}"
        folder.mkdir()
        results[index] = ask(_ask_codex(folder), cancel=cancels[index])

    threads = [threading.Thread(target=run, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    time.sleep(0.4)
    cancels[1].set()
    for thread in threads:
        thread.join(30)
    assert results[1].status is AskStatus.CANCELLED
    assert results[0].succeeded and results[2].succeeded


def test_a_recognized_failure_reason_and_reported_cost_reach_the_result(tmp_path: Path) -> None:
    from crossby.api import FailureReason

    runner = ScriptedRunner(
        _session(
            HeadlessTerminalStatus.FAILED,
            native_status="api_error",
            failure_reason=FailureReason.NOT_SIGNED_IN,
            usage=TokenUsage(input_tokens=0, output_tokens=0, reported_cost_usd=0.0),
        )
    )
    result = ask_service.run_ask(_claude_ask(tmp_path), runner=runner)
    assert result.failure_reason is FailureReason.NOT_SIGNED_IN
    assert result.error == (
        "claude session ended with status failed: not signed in (native: api_error)"
    )
    assert result.usage is not None and result.usage.reported_cost_usd == 0.0
    assert AskResult.model_validate_json(result.model_dump_json()) == result
