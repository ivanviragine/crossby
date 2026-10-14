"""Real SDK dispatch with an owned native child; no model service is contacted."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from crossby.ai_tools.copilot import CopilotAdapter
from crossby.ai_tools.headless import HeadlessTransportError, HeadlessUnsupportedError
from crossby.models.ai import (
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessLaunchContext,
    HeadlessSessionRequest,
    HeadlessTerminalStatus,
    PlanInteractionOutcome,
    SessionInteractionResponse,
)
from crossby.utils.versioning import BinaryVersion


@pytest.fixture
def peer(monkeypatch, tmp_path):
    from crossby.ai_tools import copilot_sdk

    # Base installs skip the optional dispatcher; CI's all-extras job requires it.
    if importlib.util.find_spec("copilot") is None:
        if os.environ.get("CI"):
            pytest.fail("CI must install copilot-sdk for real dispatch regressions")
        pytest.skip("requires the optional copilot-sdk extra")
    copilot_sdk.load_sdk()
    script = Path(__file__).parents[2] / "fixtures" / "copilot_sdk_peer.py"
    log = tmp_path / "peer.jsonl"
    monkeypatch.setenv("CROSSBY_COPILOT_LOG", str(log))
    monkeypatch.setattr(copilot_sdk, "command", lambda: [sys.executable, str(script)])
    monkeypatch.setattr(
        CopilotAdapter,
        "_detect_headless_version",
        lambda *a, **k: BinaryVersion(text="1.0.97", normalized=(1, 0, 97)),
    )

    def run(scenario="success", handler=None, observer=None, cancel_event=None, **kwargs):
        monkeypatch.setenv("CROSSBY_COPILOT_SCENARIO", scenario)
        request = HeadlessSessionRequest(
            prompt="private task",
            working_dir=tmp_path,
            interaction_mode=HeadlessInteractionMode.BROKERED,
            timeout_seconds=kwargs.pop("timeout_seconds", 3),
            **kwargs,
        )
        result = CopilotAdapter().run_headless_session(
            request,
            interaction_handler=handler
            or (lambda _: SessionInteractionResponse(outcome=PlanInteractionOutcome.DENIED)),
            event_handler=observer,
            cancel_event=cancel_event,
        )
        return result, [json.loads(line) for line in log.read_text().splitlines()]

    return run


def test_real_sdk_success_resume_scoped_policy(peer):
    result, frames = peer(
        resume_id="exact-native-session",
        model="claude-sonnet-4-6",
        launch_context=HeadlessLaunchContext(
            allow_tools=("shell(git status)", "excluded(tool)"),
            allow_urls=("https://example.com",),
            excluded_mcp_servers=("excluded",),
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.session_id == "exact-native-session"
    assert result.final_text == "result ✓"
    resume = next(f for f in frames if f.get("method") == "session.resume")
    assert resume["params"]["continuePendingWork"] is False
    assert resume["params"]["model"] == "claude-sonnet-4.6"
    assert resume["params"]["disabledMcpServers"] == ["excluded"]
    configure = next(f for f in frames if f.get("method") == "session.permissions.configure")
    assert "rules" not in configure["params"] and "urls" not in configure["params"]
    rules = next(f for f in frames if f.get("method") == "session.permissions.modifyRules")
    assert rules["params"]["scope"] == "session"
    assert rules["params"]["add"] == [
        {"kind": "shell", "argument": "git status"},
        {"kind": "url", "argument": "https://example.com"},
    ]


def test_exact_question_and_choice(peer):
    seen = []

    def answer(interaction):
        seen.append(interaction)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="B")

    result, frames = peer("question", answer)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert seen[0].request_id == "101" and not seen[0].allow_other
    assert [o.option_id for o in seen[0].options] == ["A", "B"]
    assert next(f for f in frames if f.get("id") == 101)["result"] == {
        "answer": "B",
        "wasFreeform": False,
    }


def test_permission_authority_native_input_and_approve_once(peer):
    seen = []

    def answer(interaction):
        seen.append(interaction)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    result, frames = peer("permission", answer)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    operation = seen[0].operation
    assert operation.execution_dir == Path("/tmp/native-cwd")
    assert operation.shell_expression == "git status"
    assert operation.native_input["managedApprovalRequired"] is True
    response = next(
        f for f in frames if f.get("method") == "session.permissions.handlePendingPermissionRequest"
    )
    assert response["params"]["requestId"] == "permission-id"
    assert response["params"]["result"] == {"kind": "approve-once"}


def test_form_suggestions_preserve_freeform(peer):
    result, frames = peer(
        "form",
        lambda _: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, content={"choice": "outside-suggested-values"}
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    response = next(f for f in frames if f.get("method") == "session.ui.handlePendingElicitation")
    assert response["params"]["requestId"] == "form-id"
    assert response["params"]["result"]["content"] == {"choice": "outside-suggested-values"}


@pytest.mark.parametrize("scenario", ["malformed", "malformed-after-idle", "eof", "url-form"])
def test_sdk_parsing_and_unsupported_native_requests_fail_closed(peer, scenario):
    with pytest.raises(HeadlessTransportError) as error:
        peer(scenario)
    assert "private task" not in str(error.value)
    assert error.value.partial.status is not HeadlessTerminalStatus.SUCCEEDED


@pytest.mark.parametrize("scenario", ["native-error", "aborted"])
def test_native_failure_is_not_idle_success(peer, scenario):
    result, _ = peer(scenario)
    assert result.status is (
        HeadlessTerminalStatus.CANCELLED if scenario == "aborted" else HeadlessTerminalStatus.FAILED
    )
    assert result.native_status == ("aborted" if scenario == "aborted" else "failure")


@pytest.mark.parametrize("scenario", ["withdraw-question", "withdraw-permission"])
@pytest.mark.parametrize("blocked", ["handler", "observer"])
def test_native_withdrawal_wakes_handler_and_observer(peer, scenario, blocked):
    entered = threading.Event()
    release = threading.Event()

    def callback(_):
        entered.set()
        release.wait(2)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    def observer(event):
        if event.kind in {HeadlessEventKind.CLARIFICATION, HeadlessEventKind.PERMISSION}:
            callback(event)

    try:
        result, frames = peer(
            scenario,
            callback if blocked == "handler" else None,
            observer if blocked == "observer" else None,
            interaction_timeout_seconds=0.3,
        )
        assert result.status is HeadlessTerminalStatus.SUCCEEDED
        assert not any(
            f.get("id") == 101
            or f.get("method") == "session.permissions.handlePendingPermissionRequest"
            for f in frames
        )
    finally:
        release.set()


def test_moving_idle_deadline_and_fixed_overall(peer):
    result, _ = peer("idle-progress", idle_timeout_seconds=0.25)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    result, frames = peer("silent", idle_timeout_seconds=0.25)
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert any(f.get("method") == "session.abort" for f in frames)


def test_optional_sdk_missing_and_context_rejection(monkeypatch, tmp_path):
    from crossby.ai_tools import copilot_sdk

    monkeypatch.setattr(copilot_sdk.metadata, "version", lambda _: "0.0.0")
    assert CopilotAdapter().capabilities().headless.brokered is None
    with pytest.raises(HeadlessUnsupportedError, match="crossby\\[copilot-sdk\\]"):
        CopilotAdapter().preflight_headless_session(
            HeadlessSessionRequest(
                prompt="x", working_dir=tmp_path, interaction_mode=HeadlessInteractionMode.BROKERED
            ),
            lambda _: None,
        )


def test_queued_reply_is_withdrawn_before_physical_commit(peer, monkeypatch):
    from crossby.ai_tools.copilot_sdk import Bridge

    release = threading.Event()
    original_commit, original_withdraw = Bridge.commit, Bridge.withdraw

    @contextmanager
    def queued(self, key):
        assert release.wait(1)
        with original_commit(self, key):
            yield

    def withdraw(self, key):
        original_withdraw(self, key)
        release.set()

    monkeypatch.setattr(Bridge, "commit", queued)
    monkeypatch.setattr(Bridge, "withdraw", withdraw)
    result, frames = peer(
        "withdraw-question",
        lambda _: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_id="A"
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert not any(f.get("id") == 101 for f in frames)


def test_content_length_backpressure_attempts_abort_before_kill(tmp_path):
    from crossby.ai_tools.plan_process import ContentLengthJsonRpcProcess

    proc = ContentLengthJsonRpcProcess(
        [sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp_path, timeout=1
    )
    observed = []
    before = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            proc.send(
                {"method": "test", "params": {"blob": "x" * 500000}},
                timeout=0.1,
                before_stop=lambda: observed.append(proc.returncode),
            )
        assert observed == [None]
        assert time.monotonic() - before < 1
        assert proc.returncode is not None
    finally:
        proc.close(timeout=0.1)
    assert not proc._stdout_thread.is_alive() and not proc._write_thread.is_alive()


@pytest.mark.parametrize(
    "outcome, status",
    [
        (PlanInteractionOutcome.CANCELLED, HeadlessTerminalStatus.CANCELLED),
    ],
)
def test_handler_cancel_uses_native_abort_and_preserves_status(peer, outcome, status):
    result, frames = peer("question", lambda _: SessionInteractionResponse(outcome=outcome))
    assert result.status is status
    assert sum(f.get("method") == "session.abort" for f in frames) == 1
    assert not any(f.get("id") == 101 for f in frames)


def test_handler_timeout_late_answer_cannot_write(peer):
    release = threading.Event()

    def wait(_):
        release.wait(1)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="A")

    try:
        result, frames = peer("stall-question", wait, interaction_timeout_seconds=0.05)
        assert result.status is HeadlessTerminalStatus.TIMED_OUT
        assert sum(f.get("method") == "session.abort" for f in frames) == 1
        assert not any(f.get("id") == 101 for f in frames)
    finally:
        release.set()


def test_handler_error_keeps_native_denial_provenance(peer):
    result, _ = peer("denial")
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.denials


def test_startup_native_failure_keeps_status_metadata(peer, caplog):
    with pytest.raises(HeadlessTransportError) as error:
        peer("startup-error")
    assert error.value.partial.native_status == "failure"
    assert error.value.partial.status is HeadlessTerminalStatus.FAILED
    assert "private native refusal" not in caplog.text


def test_native_freeform(peer):
    result, frames = peer(
        "freeform",
        lambda _: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, answer="outside choices"
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert next(f for f in frames if f.get("id") == 101)["result"] == {
        "answer": "outside choices",
        "wasFreeform": True,
    }


def test_old_idle_before_admitted_turn_does_not_bind_result(peer):
    result, _ = peer("old-idle", resume_id="exact-native-session")
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.native_status == "success"
    assert result.final_text == "result ✓"


def test_unpermitted_freeform_and_handler_exception_fail_closed(peer):
    with pytest.raises(HeadlessTransportError):
        peer(
            "question",
            lambda _: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED, answer="not allowed"
            ),
        )

    def broken(_):
        raise RuntimeError("private task must not escape")

    with pytest.raises(HeadlessTransportError) as error:
        peer("permission", broken)
    assert "private task" not in str(error.value)


@pytest.mark.parametrize(
    "scenario, evidence", [("native-error", "failure"), ("denial", "denial"), ("usage", "usage")]
)
def test_native_accounting_precedes_fallible_observer(peer, scenario, evidence):
    def observer(event):
        if event.kind is HeadlessEventKind.ASSISTANT_MESSAGE:
            raise RuntimeError("observer failed")

    with pytest.raises(HeadlessTransportError) as error:
        peer(scenario, observer=observer)
    partial = error.value.partial
    assert partial.session_id
    if evidence == "failure":
        assert partial.native_status == "failure"
    elif evidence == "denial":
        assert partial.denials
    else:
        assert partial.usage.input_tokens == 3 and partial.usage.output_tokens == 4


def test_caller_cancellation_withdraws_answer_and_aborts(peer):
    cancel = threading.Event()

    def answer(_):
        cancel.set()
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="A")

    result, frames = peer("question", answer, cancel_event=cancel)
    assert result.status is HeadlessTerminalStatus.CANCELLED
    assert sum(f.get("method") == "session.abort" for f in frames) == 1
    assert not any(f.get("id") == 101 for f in frames)


def test_progress_never_extends_overall_deadline(peer):
    result, frames = peer("idle-progress-long", idle_timeout_seconds=0.25, timeout_seconds=0.6)
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert any(f.get("method") == "session.abort" for f in frames)


def test_version_probe_disables_update_and_uses_pinned_path(monkeypatch, tmp_path):
    from crossby.ai_tools import copilot_sdk

    binary = tmp_path / "copilot"
    binary.write_text(
        f"#!{sys.executable}\nimport sys\nassert sys.argv[1:] == "
        "['--version', '--no-auto-update']\nprint('GitHub Copilot CLI 1.0.97')\n"
    )
    binary.chmod(0o755)
    monkeypatch.setattr(copilot_sdk, "command", lambda: [str(binary)])
    version = CopilotAdapter()._detect_headless_version(
        timeout_seconds=1, mode=HeadlessInteractionMode.BROKERED
    )
    assert version.normalized == (1, 0, 97)


def test_non_copilot_context_fails_before_spawn(tmp_path):
    from crossby.ai_tools.codex import CodexAdapter

    with pytest.raises(HeadlessUnsupportedError, match="launch context"):
        CodexAdapter().preflight_headless_session(
            HeadlessSessionRequest(
                prompt="x",
                working_dir=tmp_path,
                launch_context=HeadlessLaunchContext(allow_tools=("write",)),
            )
        )


def test_backpressured_reply_obeys_moving_idle_without_partial_retry(peer):
    result, frames = peer(
        "backpressure",
        lambda _: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, answer="x" * 200000
        ),
        idle_timeout_seconds=0.25,
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    replies = [f for f in frames if f.get("id") == 101]
    assert len(replies) == 1 and len(replies[0]["result"]["answer"]) == 200000
    assert not any(f.get("method") == "session.abort" for f in frames)


def test_sdk_request_wait_survives_native_progress_before_admission(peer):
    result, frames = peer("send-delay", idle_timeout_seconds=0.25)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "result ✓"
    assert sum(f.get("method") == "session.send" for f in frames) == 1


def test_resume_response_identity_must_match(peer):
    with pytest.raises(HeadlessTransportError):
        peer("resume-mismatch", resume_id="exact-native-session")


def recorded_frames():
    return [
        json.loads(line)
        for line in Path(os.environ["CROSSBY_COPILOT_LOG"]).read_text().splitlines()
    ]


@pytest.mark.parametrize("scenario", ["child-permission", "child-form", "child-question"])
def test_child_interactions_never_reach_root_handler(peer, scenario):
    seen = []

    def approve(interaction):
        seen.append(interaction)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    with pytest.raises(HeadlessTransportError):
        peer(scenario, approve)
    assert not seen
    assert not any(
        f.get("id") == 101
        or f.get("method")
        in {
            "session.permissions.handlePendingPermissionRequest",
            "session.ui.handlePendingElicitation",
        }
        for f in recorded_frames()
    )


def test_resumed_autopilot_is_explicitly_changed_to_interactive(peer):
    result, frames = peer("resume-autopilot", resume_id="exact-native-session")
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    mode = next(f for f in frames if f.get("method") == "session.mode.set")
    assert mode["params"] == {"mode": "interactive", "sessionId": "exact-native-session"}
    assert (
        next(f for f in frames if f.get("method") == "session.send")["params"]["agentMode"]
        == "interactive"
    )


@pytest.mark.parametrize("scenario", ["resume-no-id", "mode-refusal"])
def test_native_admission_requires_identity_and_interactive_mode(peer, scenario):
    with pytest.raises(HeadlessTransportError):
        peer(scenario, resume_id="exact-native-session")
    assert not any(f.get("method") == "session.send" for f in recorded_frames())


def test_unmatched_substantive_root_output_after_idle_fails(peer):
    with pytest.raises(HeadlessTransportError):
        peer("unmatched-after-idle")


def test_native_terminal_fence_precedes_sdk_interaction_dispatch(peer):
    seen = []

    def approve(interaction):
        seen.append(interaction)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    with pytest.raises(HeadlessTransportError):
        peer("permission-after-idle", approve)
    assert not seen
    assert not any(
        f.get("method") == "session.permissions.handlePendingPermissionRequest"
        for f in recorded_frames()
    )


def test_native_model_turns_can_advance_within_one_submitted_message(peer):
    result, _ = peer("multi-turn")
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "result ✓"
    assert result.turn_id == "0"


def test_large_sdk_events_are_bounded_before_send_admission(peer, monkeypatch):
    from crossby.ai_tools.copilot_sdk import Bridge
    from crossby.ai_tools.plan_process import _QUEUED_OUTPUT_BYTES_LIMIT

    original = Bridge.reserve_event
    retained = []
    rejected = []

    def reserve(self, params):
        try:
            original(self, params)
        except ValueError as exc:
            rejected.append(str(exc))
            raise
        retained.append(self.event_bytes)

    monkeypatch.setattr(Bridge, "reserve_event", reserve)
    with pytest.raises(HeadlessTransportError):
        peer("large-event-burst")
    assert rejected == ["Copilot SDK event buffering exceeded the shared byte limit"]
    assert 7 * 1024 * 1024 < max(retained) <= _QUEUED_OUTPUT_BYTES_LIMIT
    assert any(f.get("method") == "session.abort" for f in recorded_frames())
