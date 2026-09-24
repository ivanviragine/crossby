"""Managed ordinary headless-session orchestration and typed failures."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from crossby.models.ai import (
    AIToolID,
    HeadlessCapability,
    HeadlessEvent,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    PlanInteractionKind,
    PlanInteractionOutcome,
    SessionInteraction,
    SessionInteractionResponse,
    TokenUsage,
)

SessionInteractionHandler = Callable[[SessionInteraction], SessionInteractionResponse]
HeadlessInteractionHandler = SessionInteractionHandler
HeadlessEventHandler = Callable[[HeadlessEvent], None]

_MAX_EVENTS = 4096
_MAX_EVENT_MESSAGE_BYTES = 64 * 1024
_MAX_EVENT_PAYLOAD_BYTES = 256 * 1024
_MAX_FINAL_PAYLOAD_BYTES = 8 * 1024 * 1024
_MAX_DIAGNOSTIC_BYTES = 16 * 1024
_MAX_PROVENANCE_ID_BYTES = 512
_MAX_NATIVE_STATUS_BYTES = 512
_CALLBACK_POLL_SECONDS = 0.05
_CLEANUP_GRACE_SECONDS = 0.25
_TERMINAL_CALLBACK_GRACE_SECONDS = 0.25
_MISSING_FINAL_JSON = object()
_PROVENANCE_ID_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~:@+=/"
)
_SENSITIVE_EVENT_KEYS = {
    "answer",
    "answers",
    "argv",
    "command",
    "frame",
    "prompt",
    "raw",
    "request",
}


class HeadlessSessionError(RuntimeError):
    """Base class for managed ordinary-session failures."""


class HeadlessPreflightError(HeadlessSessionError):
    """Base class for failures detected before the adapter transport starts."""

    def __init__(
        self,
        message: str,
        *,
        tool_id: AIToolID,
        capability: HeadlessCapability,
    ) -> None:
        super().__init__(message)
        self.tool_id = tool_id
        self.capability = capability


class HeadlessRequestError(HeadlessPreflightError):
    """A portable request is internally contradictory or lacks a handler."""


class HeadlessSchemaError(HeadlessRequestError):
    """A caller-provided response schema is not a valid JSON Schema."""


class HeadlessUnsupportedError(HeadlessPreflightError):
    """An adapter, version, or declared transport cannot honor a request."""

    @classmethod
    def for_tool(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: HeadlessCapability,
    ) -> HeadlessUnsupportedError:
        remediation = capability.remediation or "Use an adapter with managed headless support."
        return cls(
            f"{display_name} does not support managed headless sessions. "
            f"Remediation: {remediation}",
            tool_id=tool_id,
            capability=capability,
        )

    @classmethod
    def for_installed_version(
        cls,
        *,
        tool_id: AIToolID,
        display_name: str,
        capability: HeadlessCapability,
        installed_version: str | None,
    ) -> HeadlessUnsupportedError:
        return cls(
            f"{display_name} cannot provide a managed headless session for installed version "
            f"{installed_version or 'unknown'}. Crossby requires "
            f"{capability.version_requirement} The oldest adapter-verified release is "
            f"{capability.verified_version or 'not declared'}.",
            tool_id=tool_id,
            capability=capability,
        )


class HeadlessAdapterContractError(HeadlessPreflightError):
    """An adapter's declaration and protected runtime hook disagree."""


class HeadlessTransportError(HeadlessSessionError):
    """An unrecoverable spawn, protocol, or callback failure after startup."""

    def __init__(
        self,
        message: str,
        *,
        tool_id: AIToolID,
        capability: HeadlessCapability,
        partial_result: HeadlessSessionResult,
    ) -> None:
        super().__init__(message)
        self.tool_id = tool_id
        self.capability = capability
        self.partial_result = partial_result

    @property
    def partial(self) -> HeadlessSessionResult:
        """Compatibility spelling for the safe partial result snapshot."""
        return self.partial_result


class _HeadlessStopError(Exception):
    """Internal control flow for expected cancellation/timeout/output outcomes."""

    def __init__(self, status: HeadlessTerminalStatus, warning: str) -> None:
        super().__init__(warning)
        self.status = status
        self.warning = warning


@dataclass(frozen=True)
class HeadlessCleanupContext:
    """Deadline and cancellation signal supplied to one cleanup operation.

    Cleanup hooks run in bounded daemon workers because process teardown must
    continue if a cooperative operation stalls. Hooks must watch
    :attr:`cancel_event` and use :meth:`remaining_seconds` for every blocking
    operation so they exit when their cleanup stage expires.
    """

    deadline: float
    cancel_event: threading.Event

    def remaining_seconds(self) -> float:
        """Return the remaining budget for this cleanup stage."""
        return max(0.0, self.deadline - time.monotonic())


CleanupOperation = Callable[[HeadlessCleanupContext], object]


@dataclass(frozen=True)
class HeadlessCleanupHooks:
    """Adapter cleanup operations invoked by the runtime's one cleanup path.

    Every operation receives :class:`HeadlessCleanupContext` and must be
    cooperative: it may not block past the supplied deadline or after the
    cancellation signal is set.
    """

    native_abort: CleanupOperation | None = None
    close_input: CleanupOperation | None = None
    terminate: CleanupOperation | None = None
    force_kill: CleanupOperation | None = None
    reap: CleanupOperation | None = None
    join_workers: CleanupOperation | None = None


def validate_response_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a detached JSON-only schema after metaschema validation."""
    try:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("response_schema must contain only finite JSON values") from exc
    if len(encoded) > _MAX_EVENT_PAYLOAD_BYTES:
        raise ValueError("response_schema exceeded the fixed size limit")
    detached = cast(dict[str, Any], json.loads(encoded))
    try:
        Draft202012Validator.check_schema(detached)
    except SchemaError as exc:
        raise ValueError("response_schema is not a valid JSON Schema") from exc
    return detached


class HeadlessRuntimeContext:
    """Managed adapter boundary for deadlines, callbacks, events, and cleanup.

    Adapters retain their native parsing logic but never transfer a raw process
    or protocol stream to callers.  Every blocking stage should obtain its
    budget from :meth:`remaining_seconds` and report normalized progress with
    :meth:`emit`.
    """

    def __init__(
        self,
        *,
        tool_id: AIToolID,
        version: str,
        capability: HeadlessCapability,
        request: HeadlessSessionRequest,
        started_at: float,
        deadline: float,
        interaction_handler: SessionInteractionHandler | None,
        event_handler: HeadlessEventHandler | None,
        cancel_event: threading.Event | None,
    ) -> None:
        self.tool_id = tool_id
        self.version = version
        self.capability = capability
        self.request = request
        self.started_at = started_at
        self.deadline = deadline
        self._interaction_handler = interaction_handler
        self._event_handler = event_handler
        self._cancel_event = cancel_event
        self._events: list[HeadlessEvent] = []
        self._warnings: list[str] = []
        self._denials: list[str] = []
        self._last_progress = time.monotonic()
        self._cleanup_hooks = HeadlessCleanupHooks()
        self._cleanup_started = False
        self._cleanup_abort = False
        self._closed = False
        self._native_abort_called = False
        self._cleanup_lock = threading.Lock()
        self._completion_lock = threading.Lock()
        self._terminal_state_lock = threading.Lock()
        self._runtime_stop: _HeadlessStopError | None = None
        self._result: HeadlessSessionResult | None = None
        self._provenance: dict[str, Any] = {
            "native_status": None,
            "exit_code": None,
            "session_id": None,
            "thread_id": None,
            "turn_id": None,
            "conversation_id": None,
            "usage": None,
        }

    @property
    def events(self) -> tuple[HeadlessEvent, ...]:
        return tuple(self._events)

    @property
    def result(self) -> HeadlessSessionResult | None:
        return self._result

    @property
    def cancel_event(self) -> threading.Event | None:
        """Return the caller cancellation signal for bounded adapter helpers."""
        return self._cancel_event

    def mark_progress(self) -> None:
        """Reset the idle deadline after a semantically valid native milestone."""
        self._last_progress = time.monotonic()

    def _stop_if_needed(self) -> None:
        with self._terminal_state_lock:
            runtime_stop = self._runtime_stop
        if runtime_stop is not None:
            raise runtime_stop
        now = time.monotonic()
        if self._closed and self._result is None:
            raise _HeadlessStopError(
                HeadlessTerminalStatus.CANCELLED,
                "The managed headless runtime has already entered cleanup.",
            )
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise _HeadlessStopError(
                HeadlessTerminalStatus.CANCELLED,
                "The caller cancelled the managed headless session.",
            )
        if now >= self.deadline:
            raise _HeadlessStopError(
                HeadlessTerminalStatus.TIMED_OUT,
                "The managed headless session exceeded its overall deadline.",
            )
        idle = self.request.idle_timeout_seconds
        if idle is not None and now >= self._last_progress + idle:
            raise _HeadlessStopError(
                HeadlessTerminalStatus.TIMED_OUT,
                "The managed headless session exceeded its idle deadline.",
            )

    def checkpoint(self) -> None:
        """Raise internal terminal control flow when cancellation/deadlines fire."""
        self._stop_if_needed()

    def remaining_seconds(self) -> float:
        """Return the shortest active deadline budget for one blocking operation."""
        self._stop_if_needed()
        deadline = self.deadline
        if self.request.idle_timeout_seconds is not None:
            deadline = min(deadline, self._last_progress + self.request.idle_timeout_seconds)
        return max(0.0, deadline - time.monotonic())

    def wait(self, seconds: float) -> None:
        """Cancellation-aware bounded wait useful to SDK/server adapter hooks."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("wait duration must be finite and non-negative")
        end = min(self.deadline, time.monotonic() + seconds)
        while time.monotonic() < end:
            self._stop_if_needed()
            time.sleep(min(_CALLBACK_POLL_SECONDS, max(0.0, end - time.monotonic())))
        self._stop_if_needed()

    def emit(
        self,
        kind: HeadlessEventKind,
        *,
        message: str | None = None,
        payload: dict[str, Any] | None = None,
        terminal_status: HeadlessTerminalStatus | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
        conversation_id: str | None = None,
    ) -> HeadlessEvent:
        """Append and synchronously dispatch one bounded normalized event."""
        self._stop_if_needed()
        self.set_provenance(
            session_id=session_id,
            thread_id=thread_id,
            turn_id=turn_id,
            conversation_id=conversation_id,
        )
        if len(self._events) >= _MAX_EVENTS - 1:
            raise _HeadlessStopError(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                "The native session emitted too many normalized events.",
            )
        if message is not None:
            if not message.strip():
                raise ValueError("normalized event messages must be non-blank")
            if len(message.encode("utf-8")) > _MAX_EVENT_MESSAGE_BYTES:
                raise _HeadlessStopError(
                    HeadlessTerminalStatus.INVALID_OUTPUT,
                    "A normalized event message exceeded the fixed size limit.",
                )
            if _contains_prompt(message, self.request.prompt):
                raise _HeadlessStopError(
                    HeadlessTerminalStatus.INVALID_OUTPUT,
                    "A normalized event attempted to expose the session prompt.",
                )
        if payload is not None:
            try:
                encoded_payload = _json_bytes(
                    payload,
                    limit=_MAX_EVENT_PAYLOAD_BYTES,
                    label="event payload",
                )
            except ValueError as exc:
                raise _HeadlessStopError(
                    HeadlessTerminalStatus.INVALID_OUTPUT,
                    str(exc),
                ) from None
            if _contains_sensitive_key(payload) or _contains_prompt_value(
                payload, self.request.prompt
            ):
                raise _HeadlessStopError(
                    HeadlessTerminalStatus.INVALID_OUTPUT,
                    "A normalized event attempted to expose sensitive native data.",
                )
            payload = cast(dict[str, Any], json.loads(encoded_payload))
        event = HeadlessEvent(
            sequence=len(self._events) + 1,
            kind=kind,
            message=message,
            payload=payload,
            terminal_status=terminal_status,
            elapsed_seconds=max(0.0, time.monotonic() - self.started_at),
            session_id=self._provenance["session_id"],
            thread_id=self._provenance["thread_id"],
            turn_id=self._provenance["turn_id"],
            conversation_id=self._provenance["conversation_id"],
        )
        self._events.append(event)
        if kind in {
            HeadlessEventKind.STARTED,
            HeadlessEventKind.PROGRESS,
            HeadlessEventKind.OUTPUT,
            HeadlessEventKind.INTERACTION,
            HeadlessEventKind.TERMINAL,
        }:
            self.mark_progress()
        if self._event_handler is not None and kind is not HeadlessEventKind.TERMINAL:
            self._invoke_callback(
                self._event_handler,
                event,
                timeout_seconds=self.remaining_seconds(),
                label="event handler",
            )
        return event

    def emit_event(self, event: HeadlessEvent) -> HeadlessEvent:
        """Emit an adapter-created event while assigning runtime sequence/time."""
        if event.sequence != len(self._events) + 1:
            raise self.adapter_contract_error(
                "The adapter emitted a non-contiguous headless event sequence."
            )
        return self.emit(
            event.kind,
            message=event.message,
            payload=event.payload,
            terminal_status=event.terminal_status,
            session_id=event.session_id,
            thread_id=event.thread_id,
            turn_id=event.turn_id,
            conversation_id=event.conversation_id,
        )

    def interact(self, interaction: SessionInteraction) -> SessionInteractionResponse:
        """Resolve one interaction under unattended or bounded brokered policy."""
        self._stop_if_needed()
        self.set_provenance(
            session_id=interaction.session_id,
            thread_id=interaction.thread_id,
            turn_id=interaction.turn_id,
            conversation_id=interaction.conversation_id,
        )
        # Prompts and answer content deliberately never enter normalized events.
        self.emit(HeadlessEventKind.INTERACTION, message=interaction.kind.value)
        if self.request.interaction_mode is HeadlessInteractionMode.UNATTENDED:
            if interaction.kind is PlanInteractionKind.QUESTION:
                raise _HeadlessStopError(
                    HeadlessTerminalStatus.FAILED,
                    "An unattended session encountered an unexpected native question.",
                )
            self.add_denial(f"{interaction.kind.value} denied by unattended policy")
            return SessionInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

        handler = self._interaction_handler
        if handler is None:
            raise self.adapter_contract_error(
                "A brokered headless session reached the adapter without an interaction handler."
            )
        interaction_deadline = min(
            self.deadline,
            time.monotonic() + self.request.interaction_timeout_seconds,
        )
        if self.request.idle_timeout_seconds is not None:
            interaction_deadline = min(
                interaction_deadline,
                self._last_progress + self.request.idle_timeout_seconds,
            )
        response = self._invoke_callback(
            handler,
            interaction,
            timeout_seconds=max(0.0, interaction_deadline - time.monotonic()),
            label="interaction handler",
        )
        if not isinstance(response, SessionInteractionResponse):
            raise self.transport_error("The interaction handler returned an invalid response.")
        from crossby.ai_tools.plan_mode import validate_plan_option_selection

        try:
            validate_plan_option_selection(interaction, response)
        except ValueError:
            raise self.transport_error(
                "The interaction handler returned an invalid native option selection."
            ) from None
        if response.outcome is PlanInteractionOutcome.CANCELLED:
            raise _HeadlessStopError(
                HeadlessTerminalStatus.CANCELLED,
                "The interaction handler cancelled the managed headless session.",
            )
        if response.outcome in {
            PlanInteractionOutcome.DENIED,
            PlanInteractionOutcome.SKIPPED,
        }:
            self.add_denial(f"{interaction.kind.value} denied by interaction handler")
        return response

    # Short compatibility spelling used by some adapter prototypes.
    request_interaction = interact

    def _invoke_callback(
        self,
        callback: Callable[[Any], Any],
        value: Any,
        *,
        timeout_seconds: float,
        label: str,
    ) -> Any:
        if timeout_seconds <= 0:
            raise _HeadlessStopError(
                HeadlessTerminalStatus.TIMED_OUT,
                f"The managed headless session timed out waiting for the {label}.",
            )
        responses: queue.Queue[Any] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                result: Any = callback(value)
            except BaseException as exc:  # caller callbacks are an isolation boundary
                result = exc
            with suppress(queue.Full):
                responses.put_nowait(result)
                # The runtime timed out/cancelled and will never forward a late answer.

        threading.Thread(
            target=invoke,
            daemon=True,
            name=f"crossby-headless-{label.replace(' ', '-')}",
        ).start()
        callback_deadline = time.monotonic() + timeout_seconds
        while True:
            self._stop_if_needed()
            remaining = callback_deadline - time.monotonic()
            if remaining <= 0:
                raise _HeadlessStopError(
                    HeadlessTerminalStatus.TIMED_OUT,
                    f"The managed headless session timed out waiting for the {label}.",
                )
            try:
                result = responses.get(timeout=min(_CALLBACK_POLL_SECONDS, remaining))
            except queue.Empty:
                continue
            self._stop_if_needed()
            if isinstance(result, BaseException):
                raise self.transport_error(f"The managed {label} failed.") from None
            return result

    def set_provenance(
        self,
        *,
        native_status: str | None = None,
        exit_code: int | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
        conversation_id: str | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """Accumulate safe session IDs, status, exit code, and normalized usage."""
        self.validate_provenance(
            native_status=native_status,
            session_id=session_id,
            thread_id=thread_id,
            turn_id=turn_id,
            conversation_id=conversation_id,
            usage=usage,
        )
        updates = {
            "native_status": native_status,
            "exit_code": exit_code,
            "session_id": session_id,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "conversation_id": conversation_id,
            "usage": usage,
        }
        for name, value in updates.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                raise ValueError("headless-session provenance IDs must be non-blank")
            current = self._provenance[name]
            if current is not None and current != value:
                raise self.adapter_contract_error(
                    f"The adapter changed authoritative {name.replace('_', ' ')} provenance."
                )
            self._provenance[name] = value

    def validate_provenance(
        self,
        *,
        native_status: str | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
        conversation_id: str | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """Reject unsafe native metadata before it is retained or exposed."""
        identifiers = (session_id, thread_id, turn_id, conversation_id)
        if any(
            identifier is not None and not _is_safe_provenance_id(identifier, self.request.prompt)
            for identifier in identifiers
        ) or (
            usage is not None
            and usage.session_id is not None
            and not _is_safe_provenance_id(usage.session_id, self.request.prompt)
        ):
            raise _HeadlessStopError(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                "The native transport emitted unsafe session provenance.",
            )
        if native_status is not None and not _is_safe_native_status(
            native_status, self.request.prompt
        ):
            raise _HeadlessStopError(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                "The native transport emitted an unsafe native status.",
            )
        observed = {
            "session_id": session_id,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "conversation_id": conversation_id,
        }
        if any(
            value is not None
            and self._provenance[name] is not None
            and self._provenance[name] != value
            for name, value in observed.items()
        ):
            raise _HeadlessStopError(
                HeadlessTerminalStatus.INVALID_OUTPUT,
                "The native transport emitted conflicting session provenance.",
            )

    def add_warning(self, warning: str) -> None:
        self._warnings.append(_bounded_diagnostic(_redact_prompt(warning, self.request.prompt)))

    def add_denial(self, denial: str) -> None:
        self._denials.append(_bounded_diagnostic(_redact_prompt(denial, self.request.prompt)))

    def register_cleanup(self, hooks: HeadlessCleanupHooks | None = None, **kwargs: Any) -> None:
        """Register exactly one adapter cleanup sequence before blocking I/O."""
        candidate = hooks or HeadlessCleanupHooks(**kwargs)
        if candidate.native_abort is not None and not self.capability.supports_native_abort:
            raise self.adapter_contract_error(
                "The adapter registered native abort without declaring support."
            )
        with self._cleanup_lock:
            if self._cleanup_hooks != HeadlessCleanupHooks():
                raise self.adapter_contract_error("The adapter registered cleanup more than once.")
            self._cleanup_hooks = candidate
            cleanup_started = self._cleanup_started
            abort = self._cleanup_abort
        # The monitor can enter cleanup between the adapter's initial checkpoint
        # and this registration. Run late hooks synchronously so a process that
        # finishes spawning after that stop is still torn down.
        if cleanup_started:
            self._run_cleanup(candidate, abort=abort)

    def _claim_runtime_stop(self, stop: _HeadlessStopError) -> bool:
        """Atomically give a detected runtime stop terminal ownership.

        Completion can spend time parsing or validating output. The monitor
        must be able to claim a deadline/cancellation while that work is in
        progress so a late success cannot publish over the runtime outcome.
        """
        with self._terminal_state_lock:
            if self._result is not None:
                return False
            if self._runtime_stop is None:
                self._runtime_stop = stop
            return True

    def cleanup(self, *, abort: bool) -> None:
        """Run native abort then close/terminate/kill/reap/join, at most once."""
        with self._cleanup_lock:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            self._cleanup_abort = abort
            self._closed = True
            hooks = self._cleanup_hooks
        self._run_cleanup(hooks, abort=abort)

    def _run_cleanup(self, hooks: HeadlessCleanupHooks, *, abort: bool) -> None:
        """Run one registered hook sequence after cleanup ownership is claimed."""
        cooperative_operations: list[CleanupOperation | None] = []
        if abort and hooks.native_abort is not None and not self._native_abort_called:
            self._native_abort_called = True
            cooperative_operations.append(hooks.native_abort)
        cooperative_operations.extend(
            (
                hooks.close_input,
                hooks.terminate,
            )
        )
        _run_cleanup_stage(cooperative_operations)
        # A stalled cooperative hook must not leave an owned child alive or
        # unreaped. Each hard stage receives an independent fixed grace.
        _run_cleanup_stage((hooks.force_kill,))
        _run_cleanup_stage((hooks.reap,))
        _run_cleanup_stage((hooks.join_workers,))

    def complete(
        self,
        status: HeadlessTerminalStatus = HeadlessTerminalStatus.SUCCEEDED,
        *,
        final_text: str | None = None,
        final_json: Any = _MISSING_FINAL_JSON,
        native_status: str | None = None,
        exit_code: int | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
        turn_id: str | None = None,
        conversation_id: str | None = None,
        usage: TokenUsage | None = None,
        warnings: tuple[str, ...] = (),
        denials: tuple[str, ...] = (),
        _terminal_finalization: bool = False,
    ) -> HeadlessSessionResult:
        """Reconcile native evidence and construct the terminal result once."""
        with self._completion_lock:
            if self._result is not None:
                if _terminal_finalization:
                    return self._result
                raise self.adapter_contract_error("The adapter completed a headless session twice.")
            if not _terminal_finalization:
                self._stop_if_needed()
            self.set_provenance(
                native_status=native_status,
                exit_code=exit_code,
                session_id=session_id,
                thread_id=thread_id,
                turn_id=turn_id,
                conversation_id=conversation_id,
                usage=usage,
            )
            for warning in warnings:
                self.add_warning(warning)
            for denial in denials:
                self.add_denial(denial)

            terminal_events = [
                event for event in self._events if event.kind is HeadlessEventKind.TERMINAL
            ]
            forced_runtime_status = _terminal_finalization or status in {
                HeadlessTerminalStatus.CANCELLED,
                HeadlessTerminalStatus.TIMED_OUT,
                HeadlessTerminalStatus.INVALID_OUTPUT,
            }
            if forced_runtime_status and terminal_events:
                self._drop_terminal_events()
                terminal_events = []
            elif len(terminal_events) > 1:
                status = HeadlessTerminalStatus.INVALID_OUTPUT
                self.add_warning("The native transport emitted conflicting terminal events.")
            elif terminal_events:
                native_terminal = terminal_events[0]
                if native_terminal is not self._events[-1]:
                    status = HeadlessTerminalStatus.INVALID_OUTPUT
                    self.add_warning("The native terminal event was not final.")
                elif self.capability.terminal_event_authoritative:
                    assert native_terminal.terminal_status is not None
                    status = native_terminal.terminal_status
                elif native_terminal.terminal_status is not status:
                    status = HeadlessTerminalStatus.INVALID_OUTPUT
                    self.add_warning("The native terminal event conflicted with adapter status.")
            elif self.capability.terminal_event_required:
                status = HeadlessTerminalStatus.INVALID_OUTPUT
                self.add_warning("The native transport omitted its required terminal event.")

            if status is HeadlessTerminalStatus.SUCCEEDED and self._provenance["exit_code"] not in (
                None,
                0,
            ):
                status = HeadlessTerminalStatus.FAILED
                self.add_warning("The native process exited unsuccessfully.")
            successful_native_statuses = self.capability.successful_native_statuses
            if status is HeadlessTerminalStatus.SUCCEEDED and successful_native_statuses:
                native = self._provenance["native_status"]
                if native is None:
                    status = HeadlessTerminalStatus.INVALID_OUTPUT
                    self.add_warning("The native transport omitted its authoritative status.")
                elif native not in successful_native_statuses:
                    status = HeadlessTerminalStatus.FAILED
                    self.add_warning("The native transport reported failure.")

            final_json_present = final_json is not _MISSING_FINAL_JSON
            if status is HeadlessTerminalStatus.SUCCEEDED:
                try:
                    final_text, final_json, final_json_present = self._normalize_output(
                        final_text,
                        final_json,
                        final_json_present=final_json_present,
                    )
                except ValueError as exc:
                    status = HeadlessTerminalStatus.INVALID_OUTPUT
                    final_text = None
                    final_json = _MISSING_FINAL_JSON
                    final_json_present = False
                    self.add_warning(str(exc))
                if final_text is None and not final_json_present:
                    status = HeadlessTerminalStatus.INVALID_OUTPUT
                    self.add_warning("The native transport produced no final output.")
            else:
                # Non-success terminal states retain their cause rather than
                # being overwritten by absent or malformed final output.
                final_text = None
                final_json = _MISSING_FINAL_JSON
                final_json_present = False
            if status is HeadlessTerminalStatus.INVALID_OUTPUT:
                final_text = None
                final_json = _MISSING_FINAL_JSON
                final_json_present = False

            # Replace any malformed/conflicting native terminal evidence with one
            # normalized terminal event.  Raw frames are never retained.
            if status is HeadlessTerminalStatus.INVALID_OUTPUT and terminal_events:
                self._drop_terminal_events()
            if not self._events or self._events[-1].kind is not HeadlessEventKind.TERMINAL:
                if len(self._events) >= _MAX_EVENTS:
                    self._events = self._events[: _MAX_EVENTS - 1]
                event = HeadlessEvent(
                    sequence=len(self._events) + 1,
                    kind=HeadlessEventKind.TERMINAL,
                    terminal_status=status,
                    elapsed_seconds=max(0.0, time.monotonic() - self.started_at),
                    session_id=self._provenance["session_id"],
                    thread_id=self._provenance["thread_id"],
                    turn_id=self._provenance["turn_id"],
                    conversation_id=self._provenance["conversation_id"],
                )
                self._events.append(event)
            elif self._events[-1].terminal_status is not status:
                # Authoritative reconciliation may have changed status (for example
                # schema-invalid output after a native success terminal).
                self._events[-1] = self._events[-1].model_copy(update={"terminal_status": status})

            if _terminal_finalization:
                self._result = self._build_result(
                    status=status,
                    final_text=final_text,
                    final_json=final_json,
                    final_json_present=final_json_present,
                )
                self._notify_finalized_terminal(self._events[-1])
                return self._result

            if self._event_handler is not None:
                # Terminal events are withheld until all reconciliation and schema
                # validation succeeds, so a caller can never observe two terminal
                # outcomes for one session.
                self._invoke_callback(
                    self._event_handler,
                    self._events[-1],
                    timeout_seconds=self.remaining_seconds(),
                    label="event handler",
                )

            # Re-check immediately before publishing, then claim publication
            # atomically with the runtime monitor. This lets a deadline or
            # cancellation that fires during parsing override the adapter's
            # otherwise successful completion.
            if not _terminal_finalization:
                self._stop_if_needed()
            result = self._build_result(
                status=status,
                final_text=final_text,
                final_json=final_json,
                final_json_present=final_json_present,
            )
            with self._terminal_state_lock:
                if self._runtime_stop is not None and not _terminal_finalization:
                    raise self._runtime_stop
                self._result = result
                return result

    # A natural adapter spelling for successful completion.
    finish = complete

    def _drop_terminal_events(self) -> None:
        """Remove native terminal evidence and preserve contiguous sequences."""
        events = [event for event in self._events if event.kind is not HeadlessEventKind.TERMINAL]
        self._events = [
            event.model_copy(update={"sequence": sequence})
            for sequence, event in enumerate(events, start=1)
        ]

    def _normalize_output(
        self,
        final_text: str | None,
        final_json: Any,
        *,
        final_json_present: bool,
    ) -> tuple[str | None, Any, bool]:
        if final_text is not None:
            if not final_text.strip():
                raise ValueError("The native transport produced blank final output.")
            if len(final_text.encode("utf-8")) > _MAX_FINAL_PAYLOAD_BYTES:
                raise ValueError("The native final output exceeded the fixed size limit.")
        if final_json_present:
            encoded = _json_bytes(
                final_json,
                limit=_MAX_FINAL_PAYLOAD_BYTES,
                label="final JSON output",
            )
            final_json = json.loads(encoded)
        if final_text is not None and final_json_present:
            raise ValueError("The adapter returned both text and JSON final output.")

        requires_json = (
            self.request.response_schema is not None
            or self.request.native_output is HeadlessNativeOutput.JSON
        )
        if requires_json and not final_json_present:
            if final_text is None:
                raise ValueError("The native transport omitted required structured output.")
            try:
                final_json = json.loads(final_text)
            except json.JSONDecodeError as exc:
                raise ValueError("The native transport returned malformed JSON output.") from exc
            _json_bytes(
                final_json,
                limit=_MAX_FINAL_PAYLOAD_BYTES,
                label="final JSON output",
            )
            final_text = None
            final_json_present = True
        elif self.request.native_output is HeadlessNativeOutput.JSONL and not final_json_present:
            if final_text is None:
                raise ValueError("The native transport omitted required JSONL output.")
            values: list[Any] = []
            for number, line in enumerate(final_text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"The native transport returned malformed JSONL at line {number}."
                    ) from exc
                _json_bytes(
                    value,
                    limit=_MAX_FINAL_PAYLOAD_BYTES,
                    label="final JSONL output",
                )
                values.append(value)
            if not values:
                raise ValueError("The native transport returned empty JSONL output.")
            final_json = values[-1]
            final_text = None
            final_json_present = True

        if self.request.response_schema is not None:
            if not final_json_present:
                raise ValueError("The native transport omitted schema-constrained output.")
            try:
                Draft202012Validator(self.request.response_schema).validate(final_json)
            except ValidationError as exc:
                raise ValueError(
                    "The native structured output did not satisfy response_schema."
                ) from exc
        return final_text, final_json, final_json_present

    def _build_result(
        self,
        *,
        status: HeadlessTerminalStatus,
        final_text: str | None,
        final_json: Any,
        final_json_present: bool,
    ) -> HeadlessSessionResult:
        """Construct the immutable result after terminal reconciliation."""
        return HeadlessSessionResult(
            tool=self.tool_id,
            version=self.version,
            status=status,
            events=tuple(self._events),
            final_text=final_text,
            final_json=None if final_json is _MISSING_FINAL_JSON else final_json,
            final_json_present=final_json_present,
            duration_seconds=max(0.0, time.monotonic() - self.started_at),
            denials=tuple(self._denials),
            warnings=tuple(self._warnings),
            **self._provenance,
        )

    def _notify_finalized_terminal(self, event: HeadlessEvent) -> None:
        """Best-effort terminal notification that cannot re-enter stop control flow."""
        handler = self._event_handler
        if handler is None:
            return
        responses: queue.Queue[object] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                handler(event)
            except BaseException:
                pass
            finally:
                with suppress(queue.Full):
                    responses.put_nowait(object())

        threading.Thread(
            target=invoke,
            daemon=True,
            name="crossby-headless-terminal-event-handler",
        ).start()
        with suppress(queue.Empty):
            responses.get(timeout=_TERMINAL_CALLBACK_GRACE_SECONDS)

    def partial_snapshot(self) -> HeadlessSessionResult:
        """Build a safe, non-terminal snapshot without exposing native buffers."""
        events = [event for event in self._events if event.kind is not HeadlessEventKind.TERMINAL]
        safe_events = tuple(
            event.model_copy(update={"sequence": sequence})
            for sequence, event in enumerate(events, start=1)
        )
        return HeadlessSessionResult(
            tool=self.tool_id,
            version=self.version,
            status=HeadlessTerminalStatus.FAILED,
            events=safe_events,
            duration_seconds=max(0.0, time.monotonic() - self.started_at),
            denials=tuple(self._denials),
            warnings=tuple(self._warnings),
            is_partial=True,
            **self._provenance,
        )

    def transport_error(self, message: str) -> HeadlessTransportError:
        return HeadlessTransportError(
            message,
            tool_id=self.tool_id,
            capability=self.capability,
            partial_result=self.partial_snapshot(),
        )

    def adapter_contract_error(self, message: str) -> HeadlessAdapterContractError:
        return HeadlessAdapterContractError(
            message,
            tool_id=self.tool_id,
            capability=self.capability,
        )


def _json_bytes(value: Any, *, limit: int, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The {label} was not a finite JSON value.") from exc
    if len(encoded) > limit:
        raise ValueError(f"The {label} exceeded the fixed size limit.")
    return encoded


def _bounded_diagnostic(value: str) -> str:
    compact = " ".join(value.split())
    if not compact:
        raise ValueError("headless-session diagnostics must be non-blank")
    encoded = compact.encode("utf-8")
    if len(encoded) <= _MAX_DIAGNOSTIC_BYTES:
        return compact
    return encoded[:_MAX_DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore")


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in _SENSITIVE_EVENT_KEYS or _contains_sensitive_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _contains_prompt(value: str, prompt: str) -> bool:
    return bool(prompt) and prompt in value


def _is_safe_provenance_id(value: Any, prompt: str) -> bool:
    """Whether native provenance is bounded identifier data, not session content."""
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= _MAX_PROVENANCE_ID_BYTES
        and not _contains_prompt(value, prompt)
        and all(character in _PROVENANCE_ID_CHARACTERS for character in value)
    )


def _is_safe_native_status(value: Any, prompt: str) -> bool:
    """Whether native status is a bounded token rather than response content."""
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= _MAX_NATIVE_STATUS_BYTES
        and not _contains_prompt(value, prompt)
        and all(character in _PROVENANCE_ID_CHARACTERS for character in value)
    )


def _contains_prompt_value(value: Any, prompt: str) -> bool:
    if isinstance(value, str):
        return _contains_prompt(value, prompt)
    if isinstance(value, dict):
        return any(
            _contains_prompt_value(key, prompt) or _contains_prompt_value(item, prompt)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_prompt_value(item, prompt) for item in value)
    return False


def _redact_prompt(value: str, prompt: str) -> str:
    return value.replace(prompt, "<redacted>") if _contains_prompt(value, prompt) else value


def _run_cleanup_stage(
    operations: tuple[CleanupOperation | None, ...] | list[CleanupOperation | None],
) -> None:
    cleanup_deadline = time.monotonic() + _CLEANUP_GRACE_SECONDS
    cancelled = threading.Event()
    context = HeadlessCleanupContext(deadline=cleanup_deadline, cancel_event=cancelled)
    try:
        for operation in operations:
            if operation is None:
                continue
            remaining = context.remaining_seconds()
            if remaining <= 0:
                return
            thread = threading.Thread(
                target=_ignore_cleanup_error,
                args=(operation, context),
                daemon=True,
            )
            thread.start()
            thread.join(timeout=remaining)
            if thread.is_alive():
                return
    finally:
        # Well-behaved hooks use this signal to leave their daemon worker as
        # soon as the bounded stage completes or expires.
        cancelled.set()


def _ignore_cleanup_error(operation: CleanupOperation, context: HeadlessCleanupContext) -> None:
    with suppress(BaseException):
        operation(context)


__all__ = [
    "HeadlessAdapterContractError",
    "HeadlessCleanupContext",
    "HeadlessCleanupHooks",
    "HeadlessEventHandler",
    "HeadlessInteractionHandler",
    "HeadlessPreflightError",
    "HeadlessRequestError",
    "HeadlessRuntimeContext",
    "HeadlessSchemaError",
    "HeadlessSessionError",
    "HeadlessTransportError",
    "HeadlessUnsupportedError",
    "SessionInteractionHandler",
    "validate_response_schema",
]
