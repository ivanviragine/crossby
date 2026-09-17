"""Deterministic fake-adapter coverage for the managed headless lifecycle."""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import (
    HeadlessCleanupHooks,
    HeadlessPreflightError,
    HeadlessRequestError,
    HeadlessRuntimeContext,
    HeadlessSchemaError,
    HeadlessTransportError,
)
from crossby.ai_tools.vscode import VSCodeAdapter
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    HeadlessCapability,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessNativeTransport,
    HeadlessPromptTransport,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionResponse,
)
from crossby.utils.versioning import BinaryVersion

Behavior = Callable[[HeadlessRuntimeContext], HeadlessSessionResult | None]


with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

    class FakeHeadlessAdapter(AbstractAITool):
        TOOL_ID: ClassVar[AIToolID] = AIToolID.VSCODE

        def __init__(
            self,
            behavior: Behavior,
            *,
            capability: HeadlessCapability | None = None,
        ) -> None:
            self.behavior = behavior
            self.capability = capability or _capability()
            self.started = False
            self.version_probes = 0

        def capabilities(self) -> AIToolCapabilities:
            return AIToolCapabilities(
                tool_id=self.TOOL_ID,
                display_name="Fake",
                binary="fake",
                tool_type=AIToolType.TERMINAL,
                headless=self.capability,
            )

        def _detect_headless_version(self, **_kwargs: Any) -> BinaryVersion:
            self.version_probes += 1
            return BinaryVersion((1, 0, 0), "fake 1.0.0")

        def _run_headless_session(
            self,
            request: HeadlessSessionRequest,
            version: str,
            context: HeadlessRuntimeContext,
        ) -> HeadlessSessionResult | None:
            self.started = True
            return self.behavior(context)


# Do not let this deterministic fixture replace the real registry entry used by
# unrelated adapter tests collected in the same process.
AbstractAITool._registry[AIToolID.VSCODE] = VSCodeAdapter


def _capability(**updates: Any) -> HeadlessCapability:
    values: dict[str, Any] = {
        "transport": HeadlessNativeTransport.SUBPROCESS,
        "prompt_transport": HeadlessPromptTransport.STDIN,
        "native_outputs": (
            HeadlessNativeOutput.TEXT,
            HeadlessNativeOutput.JSON,
            HeadlessNativeOutput.JSONL,
        ),
        "interaction_modes": tuple(HeadlessInteractionMode),
        "supports_response_schema": True,
        "supports_resume": True,
        "verified_version": "1.0.0",
    }
    values.update(updates)
    return HeadlessCapability(**values)


def _request(tmp_path: Path, **updates: Any) -> HeadlessSessionRequest:
    values: dict[str, Any] = {"prompt": "private session prompt", "working_dir": tmp_path}
    values.update(updates)
    return HeadlessSessionRequest(**values)


def test_success_dispatches_ordered_events_and_one_terminal(tmp_path: Path) -> None:
    observed: list[HeadlessEventKind] = []

    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(HeadlessEventKind.STARTED)
        context.emit(HeadlessEventKind.PROGRESS, message="working")
        return context.complete(
            final_text="done",
            exit_code=0,
            session_id="session-1",
        )

    result = FakeHeadlessAdapter(run).run_headless_session(
        _request(tmp_path),
        event_handler=lambda event: observed.append(event.kind),
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "done"
    assert [event.sequence for event in result.events] == [1, 2, 3]
    assert observed == [
        HeadlessEventKind.STARTED,
        HeadlessEventKind.PROGRESS,
        HeadlessEventKind.TERMINAL,
    ]


def test_invalid_schema_is_rejected_before_adapter_start(tmp_path: Path) -> None:
    adapter = FakeHeadlessAdapter(lambda _context: None)

    with pytest.raises(HeadlessSchemaError, match="valid JSON Schema"):
        adapter.run_headless_session(
            _request(tmp_path, response_schema={"type": "not-a-json-schema-type"})
        )

    assert not adapter.started


def test_brokered_mode_requires_handler_before_adapter_start(tmp_path: Path) -> None:
    adapter = FakeHeadlessAdapter(lambda _context: None)

    with pytest.raises(HeadlessRequestError, match="requires an interaction handler"):
        adapter.run_headless_session(
            _request(tmp_path, interaction_mode=HeadlessInteractionMode.BROKERED)
        )

    assert not adapter.started


def test_version_probe_timeout_uses_original_deadline_before_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeHeadlessAdapter(lambda _context: None)

    def blocked_probe(**_kwargs: Any) -> BinaryVersion:
        time.sleep(1)
        return BinaryVersion((1, 0, 0), "late 1.0.0")

    monkeypatch.setattr(adapter, "_detect_headless_version", blocked_probe)
    started = time.monotonic()
    with pytest.raises(HeadlessPreflightError, match="version probing"):
        adapter.run_headless_session(_request(tmp_path, timeout_seconds=0.05))

    assert time.monotonic() - started < 0.5
    assert not adapter.started


@pytest.mark.parametrize(
    ("native_output", "text"),
    [
        (HeadlessNativeOutput.JSON, "not json"),
        (HeadlessNativeOutput.JSONL, '{"ok":true}\nnot json'),
    ],
)
def test_malformed_structured_output_returns_invalid_output(
    tmp_path: Path,
    native_output: HeadlessNativeOutput,
    text: str,
) -> None:
    adapter = FakeHeadlessAdapter(lambda context: context.complete(final_text=text))

    result = adapter.run_headless_session(_request(tmp_path, native_output=native_output))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None
    assert result.final_json is None
    assert result.events[-1].terminal_status is HeadlessTerminalStatus.INVALID_OUTPUT


@pytest.mark.parametrize(
    "status",
    (
        HeadlessTerminalStatus.CANCELLED,
        HeadlessTerminalStatus.TIMED_OUT,
        HeadlessTerminalStatus.FAILED,
    ),
)
def test_non_success_status_is_not_replaced_by_missing_structured_output(
    tmp_path: Path,
    status: HeadlessTerminalStatus,
) -> None:
    result = FakeHeadlessAdapter(lambda context: context.complete(status)).run_headless_session(
        _request(
            tmp_path,
            native_output=HeadlessNativeOutput.JSON,
            response_schema={"type": "object"},
        )
    )

    assert result.status is status
    assert result.final_json is None
    assert not result.final_json_present


def test_schema_constrained_json_null_is_a_present_final_output(tmp_path: Path) -> None:
    result = FakeHeadlessAdapter(
        lambda context: context.complete(final_json=None)
    ).run_headless_session(
        _request(
            tmp_path,
            native_output=HeadlessNativeOutput.JSON,
            response_schema={"type": "null"},
        )
    )

    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json is None
    assert result.final_json_present


def test_schema_invalid_output_keeps_safe_events_and_provenance(tmp_path: Path) -> None:
    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(HeadlessEventKind.PROGRESS, message="safe progress")
        return context.complete(final_json={"answer": 3}, session_id="session-1")

    result = FakeHeadlessAdapter(run).run_headless_session(
        _request(
            tmp_path,
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        )
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_json is None
    assert result.session_id == "session-1"
    assert result.events[0].message == "safe progress"


def test_missing_required_terminal_event_is_invalid(tmp_path: Path) -> None:
    capability = _capability(terminal_event_required=True)
    result = FakeHeadlessAdapter(
        lambda context: context.complete(final_text="untrusted"),
        capability=capability,
    ).run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.final_text is None


def test_authoritative_native_failure_wins_over_zero_exit(tmp_path: Path) -> None:
    capability = _capability(
        terminal_event_required=True,
        terminal_event_authoritative=True,
    )

    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(
            HeadlessEventKind.TERMINAL,
            terminal_status=HeadlessTerminalStatus.FAILED,
        )
        return context.complete(
            HeadlessTerminalStatus.SUCCEEDED,
            final_text="partial",
            exit_code=0,
        )

    result = FakeHeadlessAdapter(run, capability=capability).run_headless_session(
        _request(tmp_path)
    )
    assert result.status is HeadlessTerminalStatus.FAILED


def test_conflicting_native_terminal_events_are_invalid(tmp_path: Path) -> None:
    capability = _capability(terminal_event_required=True)

    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(
            HeadlessEventKind.TERMINAL,
            terminal_status=HeadlessTerminalStatus.SUCCEEDED,
        )
        context.emit(
            HeadlessEventKind.TERMINAL,
            terminal_status=HeadlessTerminalStatus.FAILED,
        )
        return context.complete(final_text="untrusted")

    result = FakeHeadlessAdapter(run, capability=capability).run_headless_session(
        _request(tmp_path)
    )
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert len([event for event in result.events if event.kind is HeadlessEventKind.TERMINAL]) == 1


def test_terminal_reconciliation_renumbers_surviving_events(tmp_path: Path) -> None:
    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(
            HeadlessEventKind.STARTED,
            session_id="session-1",
        )
        context.emit(
            HeadlessEventKind.TERMINAL,
            terminal_status=HeadlessTerminalStatus.SUCCEEDED,
        )
        context.emit(HeadlessEventKind.PROGRESS, message="late native progress")
        return context.complete(final_text="untrusted")

    result = FakeHeadlessAdapter(run).run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert [event.sequence for event in result.events] == [1, 2, 3]
    assert result.events[-1].terminal_status is HeadlessTerminalStatus.INVALID_OUTPUT


def test_overall_timeout_interrupts_adapter_and_runs_cleanup_in_order(tmp_path: Path) -> None:
    cleanup: list[str] = []
    capability = _capability(supports_native_abort=True)

    def run(context: HeadlessRuntimeContext) -> None:
        context.register_cleanup(
            HeadlessCleanupHooks(
                native_abort=lambda _cleanup: cleanup.append("abort"),
                close_input=lambda _cleanup: cleanup.append("close"),
                terminate=lambda _cleanup: cleanup.append("terminate"),
                force_kill=lambda _cleanup: cleanup.append("kill"),
                reap=lambda _cleanup: cleanup.append("reap"),
                join_workers=lambda _cleanup: cleanup.append("join"),
            )
        )
        time.sleep(2)

    started = time.monotonic()
    result = FakeHeadlessAdapter(run, capability=capability).run_headless_session(
        _request(tmp_path, timeout_seconds=0.1)
    )

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert time.monotonic() - started < 1
    assert cleanup == ["abort", "close", "terminate", "kill", "reap", "join"]


@pytest.mark.parametrize("stalled_hook", ["native_abort", "close_input"])
def test_cancelled_cooperative_cleanup_still_kills_and_reaps(
    tmp_path: Path, stalled_hook: str
) -> None:
    cleanup: list[str] = []
    hook_finished = threading.Event()

    def stall(cleanup_context: Any) -> None:
        cleanup.append(stalled_hook)
        cleanup_context.cancel_event.wait()
        hook_finished.set()

    cleanup_hooks: dict[str, Callable[[Any], object]] = {
        stalled_hook: stall,
        "force_kill": lambda _cleanup: cleanup.append("kill"),
        "reap": lambda _cleanup: cleanup.append("reap"),
    }
    capability = _capability(supports_native_abort=stalled_hook == "native_abort")

    def run(context: HeadlessRuntimeContext) -> None:
        context.register_cleanup(HeadlessCleanupHooks(**cleanup_hooks))
        time.sleep(2)

    result = FakeHeadlessAdapter(run, capability=capability).run_headless_session(
        _request(tmp_path, timeout_seconds=0.05)
    )

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert cleanup == [stalled_hook, "kill", "reap"]
    assert hook_finished.wait(timeout=0.1)


def test_idle_timeout_can_only_shorten_overall_deadline(tmp_path: Path) -> None:
    result = FakeHeadlessAdapter(lambda _context: time.sleep(2)).run_headless_session(
        _request(tmp_path, timeout_seconds=2, idle_timeout_seconds=0.05)
    )
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert "idle deadline" in result.warnings[0]


def test_external_cancellation_returns_cancelled_result(tmp_path: Path) -> None:
    cancel = threading.Event()

    def run(_context: HeadlessRuntimeContext) -> None:
        cancel.set()
        time.sleep(1)

    result = FakeHeadlessAdapter(run).run_headless_session(_request(tmp_path), cancel_event=cancel)
    assert result.status is HeadlessTerminalStatus.CANCELLED


def test_adapter_stop_finalizes_terminal_before_cleanup(tmp_path: Path) -> None:
    lifecycle: list[str] = []
    question = PlanInteraction(
        kind=PlanInteractionKind.QUESTION,
        question_id="question-1",
        prompt="private native question",
        session_id="session-1",
    )

    def run(context: HeadlessRuntimeContext) -> None:
        context.register_cleanup(
            HeadlessCleanupHooks(close_input=lambda _cleanup: lifecycle.append("cleanup"))
        )
        context.interact(question)

    result = FakeHeadlessAdapter(run).run_headless_session(
        _request(tmp_path),
        event_handler=lambda event: lifecycle.append(event.kind.value),
    )

    assert result.status is HeadlessTerminalStatus.FAILED
    assert lifecycle == ["interaction", "terminal", "cleanup"]


def test_runtime_stop_finalizes_terminal_before_cleanup(tmp_path: Path) -> None:
    lifecycle: list[str] = []

    def run(context: HeadlessRuntimeContext) -> None:
        context.register_cleanup(
            HeadlessCleanupHooks(close_input=lambda _cleanup: lifecycle.append("cleanup"))
        )
        time.sleep(1)

    result = FakeHeadlessAdapter(run).run_headless_session(
        _request(tmp_path, timeout_seconds=0.05),
        event_handler=lambda event: lifecycle.append(event.kind.value),
    )

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert lifecycle == ["terminal", "cleanup"]


def test_runtime_stop_overrides_completion_still_validating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_normalize = HeadlessRuntimeContext._normalize_output
    normalization_started = threading.Event()

    def slow_normalize(
        context: HeadlessRuntimeContext,
        final_text: str | None,
        final_json: Any,
        *,
        final_json_present: bool,
    ) -> tuple[str | None, Any, bool]:
        normalization_started.set()
        time.sleep(0.15)
        return original_normalize(
            context,
            final_text,
            final_json,
            final_json_present=final_json_present,
        )

    monkeypatch.setattr(HeadlessRuntimeContext, "_normalize_output", slow_normalize)
    result = FakeHeadlessAdapter(
        lambda context: context.complete(final_json={"answer": "done"})
    ).run_headless_session(
        _request(
            tmp_path,
            timeout_seconds=0.05,
            native_output=HeadlessNativeOutput.JSON,
        )
    )

    assert normalization_started.is_set()
    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert result.final_json is None


def test_handler_failure_raises_transport_error_with_safe_partial(tmp_path: Path) -> None:
    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        interaction = PlanInteraction(
            kind=PlanInteractionKind.QUESTION,
            question_id="question-1",
            prompt="native private question",
            session_id="session-1",
        )
        context.interact(interaction)
        raise AssertionError("unreachable")

    def fail(_interaction: PlanInteraction) -> PlanInteractionResponse:
        raise RuntimeError("private callback detail")

    with pytest.raises(HeadlessTransportError, match="interaction handler") as raised:
        FakeHeadlessAdapter(run).run_headless_session(
            _request(tmp_path, interaction_mode=HeadlessInteractionMode.BROKERED),
            interaction_handler=fail,
        )

    assert raised.value.partial_result.is_partial
    assert raised.value.partial_result.session_id == "session-1"
    assert "private callback detail" not in str(raised.value)
    assert "native private question" not in str(raised.value.partial_result)


def test_interaction_timeout_never_forwards_a_late_answer(tmp_path: Path) -> None:
    forwarded: list[PlanInteractionResponse] = []

    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        interaction = PlanInteraction(
            kind=PlanInteractionKind.QUESTION,
            question_id="question-1",
            prompt="native private question",
            session_id="session-1",
        )
        response = context.interact(interaction)
        forwarded.append(response)
        return context.complete(final_text="unreachable")

    def late(_interaction: PlanInteraction) -> PlanInteractionResponse:
        time.sleep(0.2)
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            answer="late private answer",
        )

    result = FakeHeadlessAdapter(run).run_headless_session(
        _request(
            tmp_path,
            interaction_mode=HeadlessInteractionMode.BROKERED,
            interaction_timeout_seconds=0.05,
        ),
        interaction_handler=late,
    )
    time.sleep(0.25)

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert forwarded == []


def test_unattended_question_fails_and_permission_denies(tmp_path: Path) -> None:
    question = PlanInteraction(
        kind=PlanInteractionKind.QUESTION,
        question_id="q",
        prompt="question",
        session_id="s",
    )
    failed = FakeHeadlessAdapter(lambda context: context.interact(question)).run_headless_session(
        _request(tmp_path)
    )
    assert failed.status is HeadlessTerminalStatus.FAILED

    permission = question.model_copy(
        update={"kind": PlanInteractionKind.PERMISSION, "question_id": "p"}
    )

    def deny(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        response = context.interact(permission)
        assert response.outcome is PlanInteractionOutcome.DENIED
        return context.complete(final_text="continued")

    denied = FakeHeadlessAdapter(deny).run_headless_session(_request(tmp_path))
    assert denied.status is HeadlessTerminalStatus.SUCCEEDED
    assert denied.denials == ("permission denied by unattended policy",)


def test_event_callback_failure_is_a_transport_error(tmp_path: Path) -> None:
    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(HeadlessEventKind.PROGRESS, message="safe")
        return context.complete(final_text="done")

    with pytest.raises(HeadlessTransportError, match="event handler"):
        FakeHeadlessAdapter(run).run_headless_session(
            _request(tmp_path),
            event_handler=lambda _event: (_ for _ in ()).throw(RuntimeError("private")),
        )


def test_terminal_callback_timeout_stays_timed_out(tmp_path: Path) -> None:
    def block_terminal(event: Any) -> None:
        if event.kind is HeadlessEventKind.TERMINAL:
            time.sleep(1)

    result = FakeHeadlessAdapter(
        lambda context: context.complete(final_text="done")
    ).run_headless_session(
        _request(tmp_path, timeout_seconds=0.05),
        event_handler=block_terminal,
    )

    assert result.status is HeadlessTerminalStatus.TIMED_OUT


def test_short_prompt_is_redacted_from_diagnostics_and_rejected_from_events(tmp_path: Path) -> None:
    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.add_warning("failed: secret")
        context.emit(HeadlessEventKind.WARNING, message="failed: secret")
        raise AssertionError("unreachable")

    result = FakeHeadlessAdapter(run).run_headless_session(_request(tmp_path, prompt="secret"))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.warnings[0] == "failed: <redacted>"
    assert "secret" not in str(result)


@pytest.mark.parametrize("prompt", ('secret"value', r"secret\value", "secret\nvalue"))
def test_json_escaped_prompt_is_rejected_from_event_payloads(tmp_path: Path, prompt: str) -> None:
    def run(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(HeadlessEventKind.PROGRESS, payload={"nested": [prompt]})
        raise AssertionError("unreachable")

    result = FakeHeadlessAdapter(run).run_headless_session(_request(tmp_path, prompt=prompt))

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert prompt not in str(result)


def test_event_count_payload_and_final_output_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crossby.ai_tools.headless._MAX_EVENTS", 3)

    def too_many(context: HeadlessRuntimeContext) -> HeadlessSessionResult:
        context.emit(HeadlessEventKind.PROGRESS, message="one")
        context.emit(HeadlessEventKind.PROGRESS, message="two")
        context.emit(HeadlessEventKind.PROGRESS, message="three")
        return context.complete(final_text="unreachable")

    count = FakeHeadlessAdapter(too_many).run_headless_session(_request(tmp_path))
    assert count.status is HeadlessTerminalStatus.INVALID_OUTPUT

    monkeypatch.setattr("crossby.ai_tools.headless._MAX_EVENT_PAYLOAD_BYTES", 8)
    payload = FakeHeadlessAdapter(
        lambda context: context.emit(HeadlessEventKind.PROGRESS, payload={"value": "long"})
    ).run_headless_session(_request(tmp_path))
    assert payload.status is HeadlessTerminalStatus.INVALID_OUTPUT

    monkeypatch.setattr("crossby.ai_tools.headless._MAX_FINAL_PAYLOAD_BYTES", 8)
    output = FakeHeadlessAdapter(
        lambda context: context.complete(final_text="long final output")
    ).run_headless_session(_request(tmp_path))
    assert output.status is HeadlessTerminalStatus.INVALID_OUTPUT


def test_preflight_normalizes_paths_and_runtime_repeats_checks(tmp_path: Path) -> None:
    adapter = FakeHeadlessAdapter(lambda context: context.complete(final_text="done"))
    request = _request(tmp_path / "missing" / "..")

    preflight = adapter.preflight_headless_session(request)
    assert preflight.working_dir == tmp_path.resolve()
    assert adapter.version_probes == 1

    (tmp_path / "missing").mkdir()
    result = adapter.run_headless_session(request)
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert adapter.version_probes == 2
