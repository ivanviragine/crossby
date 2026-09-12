"""Native OpenCode question API and exact-session export contracts."""

from __future__ import annotations

import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from crossby.ai_tools import (
    AbstractAITool,
    PlanArtifactMalformedError,
    PlanArtifactMissingError,
    PlanBindingMismatchError,
    PlanInteractionOutcome,
    PlanInteractionRequiredError,
    PlanInteractionResponse,
    PlanSessionRequest,
    PlanSessionUnsupportedError,
    PlanTransportError,
)
from crossby.ai_tools.opencode_server import OpenCodeServer
from crossby.ai_tools.plan_process import CapturedProcess
from crossby.models.ai import AIToolID, EffortLevel, PlanInteractionKind
from crossby.utils.versioning import BinaryVersion

FIXTURES = Path(__file__).parents[2] / "fixtures" / "plan_sessions"


class FakeServer:
    def __init__(self, directory: Path) -> None:
        self.export = json.loads((FIXTURES / "opencode_export_success.json").read_text())
        self.export["info"]["directory"] = str(directory)
        for message in self.export["messages"]:
            message["info"]["path"] = {"cwd": str(directory)}
        self.messages = deepcopy(self.export["messages"])
        self.questions: list[list[dict[str, Any]]] = []
        self.permissions: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str, Any]] = []
        self.exports: list[list[str]] = []
        self.export_timeouts: list[float] = []
        self.closed = False
        self.started = False
        self.deadline = 0.0
        self.tool_name = "question"
        self.tool_session = "ses_exact_123"

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def remaining(self) -> float:
        return 1.0

    def request(self, method: str, path: str, payload: Any = None) -> Any:
        self.calls.append((method, path, payload))
        if path == "/session":
            assert method == "POST"
            return deepcopy(self.export["info"]) | {"id": "ses_exact_123"}
        if path.endswith("/prompt_async"):
            assert method == "POST"
            return None
        if path == "/question":
            return self.questions[0] if self.questions else []
        if path.startswith("/question/"):
            self.questions.pop(0)
            return True
        if path == "/permission":
            return list(self.permissions)
        if path.startswith("/permission/"):
            self.permissions.clear()
            return True
        if path.endswith("/message/msg_question_123"):
            return {
                "info": {"id": "msg_question_123", "sessionID": self.tool_session},
                "parts": [{"type": "tool", "callID": "call_question_123", "tool": self.tool_name}],
            }
        if path.endswith("/message"):
            return self.messages[:1] if self.questions else self.messages
        raise AssertionError(f"Unexpected native API request: {method} {path}")

    def export_session(self, command: list[str], **kwargs: Any) -> CapturedProcess:
        self.exports.append(command)
        self.export_timeouts.append(kwargs["timeout"])
        return CapturedProcess(0, json.dumps(self.export), "")


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    peer = FakeServer(tmp_path)

    def create(_request: PlanSessionRequest, deadline: float) -> FakeServer:
        peer.deadline = deadline
        return peer

    monkeypatch.setattr("crossby.ai_tools.opencode_server.OpenCodeServer", create)
    monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", peer.export_session)
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda *_args, **_kwargs: BinaryVersion((1, 18, 29), "1.18.29"),
    )
    monkeypatch.setattr("crossby.ai_tools.opencode_server.time.sleep", lambda _seconds: None)
    return peer


def _request(tmp_path: Path, **changes: Any) -> PlanSessionRequest:
    return PlanSessionRequest(
        **{"working_dir": tmp_path, "prompt": "Create a native plan", **changes}
    )


def _run(tmp_path: Path, handler: Any = None, **changes: Any) -> Any:
    return AbstractAITool.get(AIToolID.OPENCODE).run_plan_session(
        _request(tmp_path, **changes), handler
    )


def _question(**changes: Any) -> dict[str, Any]:
    request = json.loads((FIXTURES / "opencode_question.json").read_text())
    request["questions"][0].update(changes)
    return request


def test_native_mode_model_effort_and_exact_export(server: FakeServer, tmp_path: Path) -> None:
    result = _run(tmp_path, model="anthropic/claude-sonnet-4", effort=EffortLevel.HIGH)

    assert result.session_id == "ses_exact_123"
    assert result.artifact_id == "msg-plan-1"
    assert result.version == "1.18.29"
    assert result.plan == "# Native plan\n\n1. Inspect.\n2. Implement."
    assert (
        "POST",
        "/session/ses_exact_123/prompt_async",
        {
            "agent": "plan",
            "model": {"providerID": "anthropic", "modelID": "claude-sonnet-4"},
            "variant": "high",
            "parts": [{"type": "text", "text": "Create a native plan"}],
        },
    ) in server.calls
    assert server.exports == [["opencode", "export", "ses_exact_123"]]
    assert server.closed


@pytest.mark.parametrize("effort", [EffortLevel.XHIGH, EffortLevel.MAX])
def test_unsupported_effort_is_rejected_before_start(
    effort: EffortLevel, server: FakeServer, tmp_path: Path
) -> None:
    with pytest.raises(PlanSessionUnsupportedError):
        _run(tmp_path, effort=effort)
    assert not server.started


@pytest.mark.parametrize("model", ["", "default-alias", "/model", "provider/"])
def test_malformed_model_is_rejected_before_start(
    model: str, server: FakeServer, tmp_path: Path
) -> None:
    with pytest.raises(PlanSessionUnsupportedError, match="provider/model"):
        _run(tmp_path, model=model)
    assert not server.started


def test_native_multiple_questions_and_multiple_rounds(server: FakeServer, tmp_path: Path) -> None:
    first = _question()
    first["questions"].append({"question": "Any constraints?", "options": []})
    second = _question(multiple=False)
    second["id"] = "que_second_123"
    server.questions = [[first], [second]]
    seen: list[Any] = []

    def answer(interaction: Any) -> PlanInteractionResponse:
        seen.append(interaction)
        if not interaction.options:
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED, answer="--keep-compatibility"
            )
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_ids=("Linux", "macOS") if interaction.allow_multiple else ("Linux",),
        )

    _run(tmp_path, answer)
    assert [interaction.question_id for interaction in seen] == [
        "que_exact_123:0",
        "que_exact_123:1",
        "que_second_123",
    ]
    assert seen[0].allow_multiple is True
    assert seen[0].artifact_id == "call_question_123"
    assert (
        "POST",
        "/question/que_exact_123/reply",
        {"answers": [["Linux", "macOS"], ["--keep-compatibility"]]},
    ) in server.calls
    assert ("POST", "/question/que_second_123/reply", {"answers": [["Linux"]]}) in server.calls
    assert len([call for call in server.calls if call[1].endswith("prompt_async")]) == 1
    assert server.closed


@pytest.mark.parametrize("multiple", [False, None, "false", 1])
def test_native_multiple_selection_is_validated(
    multiple: Any, server: FakeServer, tmp_path: Path
) -> None:
    server.questions = [[_question(multiple=multiple)]]
    error = PlanInteractionRequiredError if multiple is False else PlanArtifactMalformedError
    with pytest.raises(error):
        _run(
            tmp_path,
            lambda _: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED, option_ids=("Linux", "macOS")
            ),
        )
    assert not server.exports
    assert server.closed


@pytest.mark.parametrize(
    "response",
    [
        PlanInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="stale"),
        PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED, answer="stale answer"),
        PlanInteractionResponse(outcome=PlanInteractionOutcome.SKIPPED),
        PlanInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    ],
)
def test_unanswered_or_invalid_native_question_stops(
    response: PlanInteractionResponse, server: FakeServer, tmp_path: Path
) -> None:
    server.questions = [[_question()]]
    with pytest.raises(PlanInteractionRequiredError):
        _run(tmp_path, lambda _: response)
    assert not any(method == "POST" and path.endswith("/reply") for method, path, _ in server.calls)
    assert server.closed


def test_missing_handler_surfaces_native_question(server: FakeServer, tmp_path: Path) -> None:
    server.questions = [[_question()]]
    with pytest.raises(PlanInteractionRequiredError) as raised:
        _run(tmp_path)
    assert raised.value.interaction.question_id == "que_exact_123"
    assert raised.value.interaction.allow_multiple
    assert server.closed


def test_malformed_native_options_stop(server: FakeServer, tmp_path: Path) -> None:
    server.questions = [[_question(options=[{"id": "missing-label"}])]]
    with pytest.raises(PlanTransportError):
        _run(tmp_path)
    assert server.closed


def test_question_tool_session_must_match(server: FakeServer, tmp_path: Path) -> None:
    server.questions = [[_question()]]
    server.tool_session = "ses_decoy"
    with pytest.raises(PlanBindingMismatchError):
        _run(tmp_path)
    assert server.closed


@pytest.mark.parametrize(
    "outcome", [PlanInteractionOutcome.APPROVED, PlanInteractionOutcome.DENIED]
)
def test_plan_exit_is_a_separate_nonexecuting_decision(
    outcome: PlanInteractionOutcome, server: FakeServer, tmp_path: Path
) -> None:
    server.tool_name = "plan_exit"
    server.questions = [[_question(options=[{"label": "Yes"}, {"label": "No"}], multiple=False)]]

    def answer(interaction: Any) -> PlanInteractionResponse:
        assert interaction.kind is PlanInteractionKind.PLAN_APPROVAL
        assert [option.option_id for option in interaction.options] == ["No"]
        return PlanInteractionResponse(outcome=outcome)

    if outcome is PlanInteractionOutcome.APPROVED:
        with pytest.raises(PlanInteractionRequiredError):
            _run(tmp_path, answer)
        assert not server.exports
    else:
        _run(tmp_path, answer)
        assert ("POST", "/question/que_exact_123/reject", None) in server.calls
    assert not any(path.endswith("/reply") for _, path, _ in server.calls)
    assert server.closed


@pytest.mark.parametrize(
    ("outcome", "reply"),
    [(PlanInteractionOutcome.APPROVED, "once"), (PlanInteractionOutcome.DENIED, "reject")],
)
def test_native_permission_decision_is_preserved(
    outcome: PlanInteractionOutcome, reply: str, server: FakeServer, tmp_path: Path
) -> None:
    server.permissions = [
        {
            "id": "per_exact_123",
            "sessionID": "ses_exact_123",
            "permission": "bash",
            "patterns": ["git status"],
        }
    ]
    _run(tmp_path, lambda _: PlanInteractionResponse(outcome=outcome))
    assert ("POST", "/permission/per_exact_123/reply", {"reply": reply}) in server.calls


@pytest.mark.parametrize("failure", ["session", "directory", "message", "missing-mode", "blank"])
def test_export_cannot_substitute_another_artifact(
    failure: str, server: FakeServer, tmp_path: Path
) -> None:
    expected: type[Exception] = PlanBindingMismatchError
    if failure == "session":
        server.export["info"]["id"] = "ses_decoy"
    elif failure == "directory":
        server.export["info"]["directory"] = str(tmp_path.parent / "decoy")
    elif failure == "message":
        server.export["messages"][-1]["info"]["id"] = "msg_old_plan"
    elif failure == "missing-mode":
        server.export["messages"][-1]["info"].pop("agent")
        server.export["messages"][-1]["info"].pop("mode")
        expected = PlanArtifactMissingError
    else:
        server.export["messages"][-1]["parts"][0]["text"] = " "
        expected = PlanArtifactMissingError
    with pytest.raises(expected):
        _run(tmp_path)
    assert server.closed


def test_export_ignores_decoy_metadata_and_earlier_plan_text(
    server: FakeServer, tmp_path: Path
) -> None:
    server.export["sessionID"] = "ses_decoy"
    server.export["messages"][0]["parts"][0]["metadata"] = {"sessionID": "ses_decoy"}
    preamble = deepcopy(server.export["messages"][-1])
    preamble["info"]["id"] = "msg_old_plan"
    preamble["parts"][0]["text"] = "I will inspect the project."
    server.export["messages"].insert(1, preamble)
    result = _run(tmp_path)
    assert result.artifact_id == "msg-plan-1"
    assert result.plan.startswith("# Native plan")


@pytest.mark.parametrize("failure", ["error", "length", "build"])
def test_failed_or_incomplete_native_turn_is_never_exported(
    failure: str, server: FakeServer, tmp_path: Path
) -> None:
    if failure == "error":
        server.messages[-1]["info"]["error"] = {"message": "private-provider-error"}
    elif failure == "length":
        server.messages[-1]["info"]["finish"] = "length"
    else:
        server.messages[-1]["info"]["agent"] = "build"
    with pytest.raises(PlanTransportError) as raised:
        _run(tmp_path)
    assert "private-provider-error" not in str(raised.value)
    assert not server.exports
    assert server.closed


def test_native_timeout_redacts_prompt_and_closes(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise TimeoutError("private prompt or answer")

    monkeypatch.setattr(server, "request", timeout)
    with pytest.raises(PlanTransportError) as raised:
        _run(tmp_path)
    assert "private prompt or answer" not in str(raised.value)
    assert server.closed


def test_native_startup_failure_closes_server(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "start", lambda: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(PlanTransportError):
        _run(tmp_path)
    assert server.closed


def test_export_timeout_does_not_expose_command(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(command: list[str], **kwargs: Any) -> CapturedProcess:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", timeout)
    with pytest.raises(PlanTransportError, match="export timed out") as raised:
        _run(tmp_path)
    assert "Command '" not in str(raised.value)
    assert server.closed


def test_native_session_and_export_share_deadline(
    server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    original_close = server.close

    def close() -> None:
        original_close()
        clock[0] = 104.0

    monkeypatch.setattr("crossby.ai_tools.base.monotonic", lambda: clock[0])
    monkeypatch.setattr("crossby.ai_tools.opencode.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(server, "close", close)
    _run(tmp_path, timeout_seconds=10)
    assert server.deadline == 110.0
    assert server.export_timeouts == [6.0]


def test_server_uses_fresh_loopback_auth_and_keeps_questions_enabled(tmp_path: Path) -> None:
    with patch("crossby.ai_tools.opencode_server.JsonRpcProcess") as process:
        native = OpenCodeServer(_request(tmp_path), time.monotonic() + 10)
    command = process.call_args.args[0]
    environment = process.call_args.kwargs["env"]
    assert command == ["opencode", "serve", "--hostname", "127.0.0.1", "--port", "0"]
    assert environment["OPENCODE_ENABLE_QUESTION_TOOL"] == "true"
    assert len(environment["OPENCODE_SERVER_PASSWORD"]) > 32
    assert environment["OPENCODE_SERVER_PASSWORD"] not in repr(command)
    assert native.authorization.startswith("Basic ")
