"""Optional, pinned Claude SDK bridge with Crossby-owned native I/O.

The SDK callback omits the wire request ID. The exposed custom Transport
boundary retains it without replacing SDK control dispatch or using display
text as permission authority. No SDK import occurs at module import time.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import subprocess
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager, suppress
from importlib import metadata
from typing import Any

from crossby.ai_tools.headless import (
    HeadlessCleanupContext,
    HeadlessRuntimeContext,
    _HeadlessInteractionCancelledError,
    _HeadlessStopError,
)
from crossby.ai_tools.headless_cli import MISSING, complete_session, non_blank_text, usage_from
from crossby.ai_tools.session_protocol import ProtocolSessionController
from crossby.models.ai import (
    HeadlessEventKind,
    HeadlessModeCapability,
    HeadlessNativeOutput,
    HeadlessNativeTransport,
    HeadlessPromptTransport,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    PlanCommandPolicySupport,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanOperationKind,
    PlanPermissionTargetKind,
    PlanRequestBehavior,
    SessionInteraction,
    SessionNativeBindingID,
    SessionOperation,
    SessionPermissionTarget,
    SessionQuestionOption,
)

SDK_VERSION = "0.2.165"
SDK_REMEDIATION = 'Install the compatible optional SDK: pip install "crossby[claude-sdk]".'
_MAX_IDS = 4096


def load_sdk() -> tuple[Any, Any]:
    """Verify distribution and API before returning the optional SDK objects."""
    if metadata.version("claude-agent-sdk") != SDK_VERSION:
        raise ImportError("incompatible Claude Agent SDK version")
    sdk = importlib.import_module("claude_agent_sdk")
    builder = importlib.import_module(
        "claude_agent_sdk._internal.transport.subprocess_cli"
    ).SubprocessCLITransport
    required = {
        "ClaudeSDKClient",
        "ClaudeAgentOptions",
        "PermissionResultAllow",
        "PermissionResultDeny",
    }
    if not all(callable(getattr(sdk, name, None)) for name in required):
        raise ImportError("incompatible Claude Agent SDK API")
    fields = getattr(sdk.ClaudeAgentOptions, "__dataclass_fields__", {})
    if (
        not {"effort", "output_format", "resume", "can_use_tool", "setting_sources"}
        <= fields.keys()
    ):
        raise ImportError("incompatible Claude Agent SDK options")
    if not all(
        callable(getattr(sdk.ClaudeSDKClient, name, None))
        for name in ("connect", "query", "receive_messages", "interrupt", "disconnect")
    ):
        raise ImportError("incompatible Claude Agent SDK client")
    if not callable(getattr(builder, "_build_command", None)):
        raise ImportError("incompatible Claude Agent SDK command builder")
    return sdk, builder


def brokered_profile() -> HeadlessModeCapability | None:
    """Advertise the alternate transport only for the pinned, importable SDK."""
    try:
        load_sdk()
    except Exception:
        return None
    return HeadlessModeCapability(
        transport=HeadlessNativeTransport.SDK,
        prompt_transport=HeadlessPromptTransport.PROTOCOL,
        native_outputs=tuple(HeadlessNativeOutput),
        supports_response_schema=True,
        supports_resume=True,
        supports_native_abort=True,
        terminal_event_required=True,
        terminal_event_authoritative=True,
        successful_native_statuses=("success",),
        sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
        command_policy_support=PlanCommandPolicySupport.NATIVE,
        supported_interaction_kinds=(PlanInteractionKind.QUESTION, PlanInteractionKind.PERMISSION),
        version_requirement=f"Claude Code >= 2.1.263 and claude-agent-sdk=={SDK_VERSION}.",
        verified_version="2.1.263",
        limitation=(
            "SDK custom transport is pinned; sandbox is tool-managed. Native question texts "
            "and option labels are answer keys. Shared request has no scene or tool-list context."
        ),
    )


def sdk_options(request: HeadlessSessionRequest, callback: Any) -> Any:
    """Render only representable normal launch context; never widen approvals."""
    from crossby.config.claude_allowlist import canonical_to_claude

    sdk, _ = load_sdk()
    return sdk.ClaudeAgentOptions(
        cli_path="claude",  # Probe and launch the same installed CLI, never the bundled one.
        cwd=str(request.working_dir),
        model=request.model,
        effort=request.effort.value if request.effort else None,
        add_dirs=[str(path) for path in request.trusted_dirs],
        resume=request.resume_id,
        system_prompt={"type": "preset", "preset": "claude_code"},
        permission_mode="default",
        permission_prompt_tool_name=None,
        can_use_tool=callback,
        # AskUserQuestion must always reach the broker, including ambient allow rules.
        settings=json.dumps({"permissions": {"ask": ["AskUserQuestion"]}}),
        setting_sources=[] if request.command_policy is not None else None,
        allowed_tools=(
            [canonical_to_claude(cmd) for cmd in request.command_policy.allowed_commands]
            if request.command_policy is not None
            else []
        ),
        output_format=(
            {"type": "json_schema", "schema": request.response_schema}
            if request.response_schema is not None
            else None
        ),
        include_partial_messages=True,
    )


def sdk_command(request: HeadlessSessionRequest) -> list[str]:
    """Use the exact pinned builder without spawning its unowned subprocess."""
    _, builder = load_sdk()
    options = sdk_options(request, None)
    options.permission_prompt_tool_name = "stdio"
    return list(builder(prompt="", options=options)._build_command())


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > 4096:
        raise ValueError("Claude SDK omitted a bounded native identity")
    return value


class _SDKController(ProtocolSessionController):
    """Reuse ordered process cleanup; send abort through the SDK itself."""

    client: Any = None
    loop: asyncio.AbstractEventLoop | None = None
    bridge: SDKTransport | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._abort_lock = threading.Lock()
        self._aborted = False
        self._abort_done = threading.Event()
        super().__init__(*args, **kwargs)

    def _abort(self, cleanup: HeadlessCleanupContext) -> None:
        with self._abort_lock:
            already_started = self._aborted
            self._aborted = True
        if already_started:
            self._abort_done.wait(timeout=cleanup.remaining_seconds())
            return
        try:
            if self.bridge is not None:
                self.bridge.revoke_replies()
            if self.client is None or self.loop is None or not self.loop.is_running():
                return
            future = asyncio.run_coroutine_threadsafe(self.client.interrupt(), self.loop)
            try:
                future.result(timeout=cleanup.remaining_seconds())
            except Exception:
                future.cancel()
        finally:
            self._abort_done.set()


class SDKTransport:
    """The pinned SDK custom Transport interface, backed by bounded real pipes."""

    def __init__(self, context: HeadlessRuntimeContext, command: list[str]) -> None:
        self.context = context
        self.command = command
        self.controller: _SDKController | None = None
        self.client: Any = None
        self.accept_replies = True
        self.requested_resume_id = context.request.resume_id
        self.session_id: str | None = None
        self.native_initialized = False
        self.session_state: str | None = None
        self.pending: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self.request_cancellations: dict[str, threading.Event] = {}
        self.pending_lock = threading.RLock()
        self.seen: set[str] = set()
        self.tool_ids: set[str] = set()
        # Pinned SDK 0.2.165: a result ends a turn, not the run. Only
        # local agents/workflows defer EOF; shells/monitors may never settle.
        self.inflight_tasks: set[str] = set()
        self.turn_in_progress = False
        self.run_complete = False
        self.terminal: dict[str, Any] | None = None
        self.failure_terminal: dict[str, Any] | None = None
        self.query_submitted = False
        self.failure: BaseException | None = None
        self.ended = asyncio.Event()
        self.closed = False
        self.callback_lock = asyncio.Lock()

    async def connect(self) -> None:
        self.context.checkpoint()
        self.controller = _SDKController(
            self.context,
            self.command,
            cwd=self.context.request.working_dir,
            headerless=True,
            # Match the pinned SDK's subprocess environment. Its nested-session
            # marker and SDK entrypoint are launch metadata, not permission flags.
            env={
                **{key: value for key, value in os.environ.items() if key != "CLAUDECODE"},
                "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
                "CLAUDE_AGENT_SDK_VERSION": SDK_VERSION,
                "PWD": str(self.context.request.working_dir),
                "CLAUDE_CODE_SDK_READS_SESSION_STATE": os.environ.get(
                    "CLAUDE_CODE_SDK_READS_SESSION_STATE", "1"
                ),
            },
        )
        self.controller.bridge = self
        self.controller.client = self.client
        self.controller.loop = asyncio.get_running_loop()

    def is_ready(self) -> bool:
        return self.controller is not None and not self.closed

    def _read(self) -> dict[str, Any] | None:
        assert self.controller is not None
        self.context.checkpoint()
        try:
            line = self.controller.process.read_line(timeout=self.context.remaining_seconds())
        except EOFError:
            return None
        finally:
            self.context.checkpoint()
        frame = json.loads(line)
        if not isinstance(frame, dict):
            raise ValueError("Claude SDK emitted a non-object frame")
        return frame

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        try:
            while not self.closed:
                frame = await asyncio.to_thread(self._read)
                if frame is None:
                    if self.terminal is None:
                        raise ValueError("Claude SDK stream ended without a terminal result")
                    break
                self._observe(frame)
                yield frame
        except BaseException as exc:
            if not self.closed and self.failure is None:
                self.failure = exc
            native_failed = self.terminal is not None and (
                self.terminal.get("is_error") is True or self.terminal.get("subtype") != "success"
            )
            if (
                not isinstance(exc, asyncio.CancelledError)
                and self.controller is not None
                and not native_failed
            ):
                # connect() automatically disconnects on reader failure. Finish
                # the bounded native abort attempt before propagating that
                # failure, while the SDK control channel is still alive.
                await asyncio.to_thread(
                    self.controller._abort,
                    HeadlessCleanupContext(
                        deadline=asyncio.get_running_loop().time() + 0.25,
                        cancel_event=threading.Event(),
                    ),
                )
            raise
        finally:
            self.ended.set()

    def _observe(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        subtype = frame.get("subtype")
        if self.terminal is not None:
            lifecycle = kind == "system" and subtype in {
                "session_state_changed",
                "task_started",
                "task_progress",
                "task_notification",
                "task_updated",
                "background_tasks_changed",
            }
            continuation = (not self.run_complete or self.turn_in_progress) and kind in {
                "assistant",
                "user",
                "stream_event",
                "control_request",
                "control_response",
                "control_cancel_request",
                "result",
            }
            # Replies to our bounded interrupt remain valid after a turn ends.
            if not (lifecycle or continuation or kind == "control_response"):
                raise ValueError("Claude SDK emitted content outside its active run")
        if kind in {"system", "assistant", "user", "stream_event", "result"}:
            native_session = frame.get("session_id")
            if native_session is not None:
                native_session = _identity(native_session)
                if (
                    self.requested_resume_id is not None
                    and native_session != self.requested_resume_id
                ):
                    raise _HeadlessStopError(
                        HeadlessTerminalStatus.INVALID_OUTPUT,
                        "Claude SDK did not confirm the requested native resume session.",
                    )
                if self.session_id is not None and native_session != self.session_id:
                    raise ValueError("Claude SDK frame changed the session binding")
                self.context.set_provenance(session_id=native_session)
                self.session_id = native_session
        if kind == "system" and frame.get("subtype") == "init":
            if frame.get("session_id") is None:
                raise ValueError("Claude SDK initialization omitted its session identity")
            self.native_initialized = True
        if kind == "system" and frame.get("subtype") == "session_state_changed":
            if frame.get("state") not in {"idle", "running", "requires_action"}:
                raise ValueError("Claude SDK reported an invalid lifecycle state")
            self.session_state = frame["state"]
            if self.session_state != "idle":
                self.run_complete = False
            elif self.terminal is not None and not self.inflight_tasks:
                self.run_complete = True
        if kind == "system" and subtype in {"task_started", "task_notification", "task_updated"}:
            task_id = _identity(frame.get("task_id"))
            if subtype == "task_started" and frame.get("task_type") in {
                "local_agent",
                "local_workflow",
            }:
                if len(self.inflight_tasks) >= _MAX_IDS:
                    raise ValueError("Claude SDK exceeded the active task limit")
                self.inflight_tasks.add(task_id)
                self.run_complete = False
            elif subtype == "task_notification" or (
                subtype == "task_updated"
                and isinstance(frame.get("patch"), dict)
                and frame["patch"].get("status") in {"completed", "failed", "stopped", "killed"}
            ):
                self.inflight_tasks.discard(task_id)
                # Completion can wake a continuation. Only its result/idle
                # can settle the run, not the now-empty task ledger.
        if kind == "control_request":
            request = frame.get("request")
            if not isinstance(request, dict) or request.get("subtype") != "can_use_tool":
                raise ValueError("Claude SDK emitted an unsupported control request")
            request_id = _identity(frame.get("request_id"))
            tool_id = _identity(request.get("tool_use_id"))
            tool = _identity(request.get("tool_name"))
            native_input = request.get("input")
            if (
                self.session_id is None
                or not self.native_initialized
                or not isinstance(native_input, dict)
                or request_id in self.seen
                or tool_id in self.tool_ids
                or len(self.seen) >= _MAX_IDS
            ):
                raise ValueError("Claude SDK control request has invalid binding")
            if len(json.dumps(native_input, allow_nan=False).encode()) > 256 * 1024:
                raise ValueError("Claude SDK tool arguments exceeded the fixed size limit")
            with self.pending_lock:
                if self.closed or not self.accept_replies:
                    raise _HeadlessInteractionCancelledError("Claude SDK session was closed.")
                self.seen.add(request_id)
                self.tool_ids.add(tool_id)
                self.pending[request_id] = (tool_id, tool, native_input)
                self.request_cancellations[request_id] = threading.Event()
        elif kind == "control_cancel_request":
            self.cancel_request(_identity(frame.get("request_id")))
        elif kind == "result":
            if self.pending or self.session_id is None:
                raise ValueError("Claude SDK result has unresolved requests or no session")
            origin = frame.get("origin")
            if origin is not None and (
                not isinstance(origin, dict)
                or origin.get("kind")
                not in {
                    "human",
                    "channel",
                    "peer",
                    "task-notification",
                    "coordinator",
                    "unclassified",
                    "observer",
                    "auto-continuation",
                    "observer-activity",
                }
            ):
                raise ValueError("Claude SDK reported an unsupported result origin")
            human = origin is None or origin.get("kind") == "human"
            submitted_result = self.terminal is None and human
            if submitted_result:
                self.terminal = frame
            elif self.terminal is not None and human:
                raise ValueError("Claude SDK returned a second submitted-prompt result")
            self.turn_in_progress = False
            self.context.mark_progress()
            if self.failure_terminal is None and (
                frame.get("is_error") is True
                or frame.get("subtype") != "success"
                or frame.get("terminal_reason") in {"aborted_streaming", "aborted_tools"}
            ):
                self.failure_terminal = frame
                prompt_result = self.terminal or frame
                self.context.set_provenance(
                    native_status=_identity(frame.get("subtype")),
                    usage=usage_from(
                        prompt_result.get("usage"),
                        input_tokens="input_tokens",
                        output_tokens="output_tokens",
                        cached="cache_read_input_tokens",
                        session_id=self.session_id,
                    ),
                )
            self.run_complete = (
                self.terminal is not None
                and not self.inflight_tasks
                and self.session_state in (None, "idle")
            )
            # All native state/provenance must precede user callbacks: an
            # observer may cancel, expire or raise while receiving accounting.
            if submitted_result:
                self._record_prompt_accounting(frame)
            if not self.query_submitted:
                # Wake the SDK's pending initialize request immediately. It
                # otherwise waits its fixed minimum even after native refusal.
                raise ValueError("Claude SDK returned a terminal result before query startup")
        elif kind in {"assistant", "user"}:
            if kind == "assistant" and frame.get("parent_tool_use_id") is None:
                self.turn_in_progress = True
                self.run_complete = False
            message = frame.get("message", {})
            blocks = message.get("content", []) if isinstance(message, dict) else []
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}:
                        self.context.emit(
                            HeadlessEventKind.TOOL_STARTED
                            if block["type"] == "tool_use"
                            else HeadlessEventKind.TOOL_COMPLETED
                        )
            if kind == "assistant":
                self.context.emit(HeadlessEventKind.ASSISTANT_MESSAGE)
        elif kind == "stream_event":
            if frame.get("parent_tool_use_id") is None:
                self.turn_in_progress = True
                self.run_complete = False
            self.context.mark_progress()
        elif kind == "system":
            self.context.emit(HeadlessEventKind.PROGRESS)

    def _record_prompt_accounting(self, frame: dict[str, Any]) -> None:
        """Publish observed prompt accounting before background waits/cleanup."""
        usage = usage_from(
            frame.get("usage"),
            input_tokens="input_tokens",
            output_tokens="output_tokens",
            cached="cache_read_input_tokens",
            session_id=self.session_id,
        )
        self.context.set_provenance(usage=usage)
        cost = frame.get("total_cost_usd")
        payload = (
            {"cost_usd": cost}
            if isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and self.context.request.prompt not in "cost_usd"
            else None
        )
        if usage is not None or payload is not None:
            self.context.emit(HeadlessEventKind.USAGE, payload=payload)
        denials = frame.get("permission_denials")
        if isinstance(denials, list) and denials:
            self.context.add_denial("Claude reported native permission denials")

    async def write(self, data: str) -> None:
        assert self.controller is not None
        for line in data.splitlines():
            frame = json.loads(line)
            # Native abort is the only write allowed beyond the work deadline.
            abort = (
                frame.get("type") == "control_request"
                and frame.get("request", {}).get("subtype") == "interrupt"
            )
            if abort:
                await asyncio.to_thread(self.controller.process.send_cleanup, frame)
                continue
            self.context.checkpoint()
            if frame.get("type") == "control_response" and (self.closed or not self.accept_replies):
                # SDK may finish a cancelled callback; discard its late response.
                return
            if self.closed or not self.accept_replies:
                raise ValueError("Claude SDK session no longer accepts writes")
            if frame.get("type") == "control_response":
                request_id = frame.get("response", {}).get("request_id")
                if request_id not in self.pending:
                    return
                if frame.get("response", {}).get("subtype") == "error":
                    if self.failure is None:
                        self.failure = self.context.transport_error(
                            "Claude SDK could not dispatch a live native request."
                        )
                    self.revoke_replies()
                    return
                # No await between checking liveness and the bounded write worker.
                await asyncio.to_thread(self._reply, request_id, frame)
            else:
                await asyncio.to_thread(
                    self.controller.process.send, frame, timeout=self.context.remaining_seconds()
                )

    def _reply(self, request_id: str, frame: dict[str, Any]) -> None:
        assert self.controller is not None
        self.context.checkpoint()
        self.controller.process.send(
            frame,
            timeout=self.context.remaining_seconds(),
            write_guard=lambda: self._commit_reply(request_id),
        )

    @contextmanager
    def _commit_reply(self, request_id: str) -> Iterator[None]:
        with self.pending_lock:
            if self.closed or not self.accept_replies or request_id not in self.pending:
                raise _HeadlessInteractionCancelledError("Claude SDK request was withdrawn.")
            self.context.checkpoint()
            self.pending.pop(request_id)
            self.request_cancellations.pop(request_id, None)
        # Rights are committed immediately before the physical writer. Pipe I/O
        # must not hold this lock: withdrawal/abort must remain able to run when
        # a committed response encounters backpressure.
        yield

    def cancel_request(self, request_id: str) -> None:
        with self.pending_lock:
            self.pending.pop(request_id, None)
            signal = self.request_cancellations.pop(request_id, None)
            if signal is not None:
                signal.set()

    def revoke_replies(self) -> None:
        with self.pending_lock:
            self.accept_replies = False
            for signal in self.request_cancellations.values():
                signal.set()
            self.request_cancellations.clear()
            self.pending.clear()

    async def end_input(self) -> None:
        if self.controller is not None:
            self.controller.process.close_input()

    async def finish(self) -> int:
        """Keep callbacks alive through the run; drain EOF after settled input."""
        while (not self.run_complete or self.turn_in_progress) and not self.ended.is_set():
            if self.failure_terminal is not None:
                break
            if self.failure is not None:
                raise self.failure
            self.context.checkpoint()
            await asyncio.sleep(min(0.01, self.context.remaining_seconds()))
        assert self.controller is not None
        if self.failure_terminal is not None:
            # Native failure/abort is already authoritative. Idle and EOF are
            # not success gates here; use cleanup time, never the work deadline.
            budget = min(0.1, self.context.remaining_seconds() / 4)
            await asyncio.to_thread(
                self.controller._abort,
                HeadlessCleanupContext(
                    deadline=asyncio.get_running_loop().time() + budget,
                    cancel_event=threading.Event(),
                ),
            )
            await self.end_input()
            try:
                return await asyncio.to_thread(self.controller.process._proc.wait, timeout=budget)
            except subprocess.TimeoutExpired:
                return await asyncio.to_thread(self.controller.process.close, timeout=budget)
        await self.end_input()
        while not self.ended.is_set():
            self.context.checkpoint()
            await asyncio.sleep(min(0.01, self.context.remaining_seconds()))
        if self.failure is not None:
            raise self.failure
        return await asyncio.to_thread(
            self.controller.process._proc.wait, timeout=self.context.remaining_seconds()
        )

    async def close(self) -> None:
        self.revoke_replies()
        self.closed = True
        if self.controller is not None:
            await asyncio.to_thread(self.controller.process.close, timeout=0.1)

    def _binding(self, tool: str, native_input: dict[str, Any], permission: Any) -> tuple[str, str]:
        tool_id = _identity(permission.tool_use_id)
        with self.pending_lock:
            matches = [
                request_id
                for request_id, (native_tool_id, native_tool, raw) in self.pending.items()
                if native_tool_id == tool_id and native_tool == tool and raw == native_input
            ]
        if len(matches) != 1 or self.session_id is None:
            raise ValueError("Claude SDK callback does not match a live native request")
        return matches[0], tool_id

    async def can_use_tool(self, tool: str, native_input: dict[str, Any], permission: Any) -> Any:
        async with self.callback_lock:
            return await self._can_use_tool(tool, native_input, permission)

    async def _can_use_tool(self, tool: str, native_input: dict[str, Any], permission: Any) -> Any:
        sdk, _ = load_sdk()
        try:
            request_id, tool_id = self._binding(tool, native_input, permission)
            with self.pending_lock:
                signal = self.request_cancellations.get(request_id)
            if signal is None:
                raise _HeadlessInteractionCancelledError("Claude SDK request was withdrawn.")
            if tool == "AskUserQuestion":
                updated = await asyncio.to_thread(self._questions, native_input, request_id, signal)
                if updated is None:
                    return sdk.PermissionResultDeny(
                        message="Clarification declined", interrupt=True
                    )
                return sdk.PermissionResultAllow(updated_input=updated)
            assert self.session_id is not None
            interaction = SessionInteraction(
                kind=PlanInteractionKind.PERMISSION,
                question_id=tool_id,
                request_id=request_id,
                session_id=self.session_id,
                prompt=non_blank_text(permission.title) or f"Allow {tool}?",
                operation=self._operation(tool, native_input, request_id, tool_id, permission),
            )
            response = await asyncio.to_thread(
                self.context.native_interaction_response, interaction, request_cancel_event=signal
            )
            if response.outcome is PlanInteractionOutcome.APPROVED:
                return sdk.PermissionResultAllow(updated_input=native_input)
            return sdk.PermissionResultDeny(message="Permission declined")
        except asyncio.CancelledError:
            raise
        except _HeadlessInteractionCancelledError:
            raise asyncio.CancelledError from None
        except BaseException as exc:
            # SDK converts callback exceptions to control errors; retain the actual
            # managed stop/handler failure so it cannot become a successful turn.
            self.failure = exc
            self.revoke_replies()
            raise ValueError("Claude brokered callback failed") from None

    def _questions(
        self, native_input: dict[str, Any], request_id: str, signal: threading.Event
    ) -> dict[str, Any] | None:
        questions = native_input.get("questions")
        if not isinstance(questions, list) or not 1 <= len(questions) <= 4:
            raise ValueError("Claude SDK supplied an invalid question collection")
        assert self.session_id is not None
        answers: dict[str, Any] = {}
        for question in questions:
            if signal.is_set():
                raise _HeadlessInteractionCancelledError("Claude SDK question was withdrawn.")
            if not isinstance(question, dict):
                raise ValueError("Claude SDK supplied an invalid question")
            key = _identity(question.get("question"))
            raw_options = question.get("options")
            multiple = question.get("multiSelect")
            if (
                key in answers
                or not isinstance(raw_options, list)
                or not isinstance(multiple, bool)
            ):
                raise ValueError("Claude SDK supplied ambiguous question keys")
            options = tuple(
                SessionQuestionOption(
                    option_id=_identity(option.get("label")),
                    label=option["label"],
                    description=option.get("description"),
                )
                for option in raw_options
                if isinstance(option, dict)
            )
            if (
                len(options) != len(raw_options)
                or not 2 <= len(options) <= 4
                or len({option.option_id for option in options}) != len(options)
            ):
                raise ValueError("Claude SDK supplied ambiguous native options")
            response = self.context.native_interaction_response(
                SessionInteraction(
                    kind=PlanInteractionKind.QUESTION,
                    question_id=key,
                    request_id=request_id,
                    session_id=self.session_id,
                    prompt=key,
                    options=options,
                    allow_multiple=multiple,
                    allow_other=True,
                ),
                request_cancel_event=signal,
            )
            if signal.is_set():
                raise _HeadlessInteractionCancelledError("Claude SDK question was withdrawn.")
            selected = [
                *((response.option_id,) if response.option_id else ()),
                *response.option_ids,
            ]
            if response.outcome in {PlanInteractionOutcome.DENIED, PlanInteractionOutcome.SKIPPED}:
                return None
            if (
                response.content is not None
                or (selected and response.answer)
                or not (selected or (response.answer and response.answer.strip()))
            ):
                raise ValueError("Claude question requires explicit native labels or free text")
            answers[key] = (selected if multiple else selected[0]) if selected else response.answer
        # Native `response` takes precedence over answers in AskUserQuestion.
        # Only the handler's validated answer map may authorize this reply.
        return {
            **{key: value for key, value in native_input.items() if key != "response"},
            "answers": answers,
        }

    def _operation(
        self, tool: str, raw: dict[str, Any], request_id: str, tool_id: str, permission: Any
    ) -> SessionOperation:
        targets = []
        path = raw.get("notebook_path") if tool == "NotebookEdit" else raw.get("file_path")
        if isinstance(path, str) and path.strip():
            targets.append(
                SessionPermissionTarget(
                    kind=(
                        PlanPermissionTargetKind.FILESYSTEM_READ
                        if tool == "Read"
                        else PlanPermissionTargetKind.FILESYSTEM_WRITE
                        if tool in {"Edit", "Write", "NotebookEdit"}
                        else PlanPermissionTargetKind.RESOURCE
                    ),
                    value=path,
                )
            )
        if permission.blocked_path:
            targets.append(
                SessionPermissionTarget(
                    kind=PlanPermissionTargetKind.RESOURCE, value=permission.blocked_path
                )
            )
        bindings = [
            SessionNativeBindingID(name="request_id", value=request_id),
            SessionNativeBindingID(name="tool_use_id", value=tool_id),
            SessionNativeBindingID(name="tool_name", value=tool),
        ]
        if permission.agent_id:
            bindings.append(SessionNativeBindingID(name="agent_id", value=permission.agent_id))
        return SessionOperation(
            kind=(
                PlanOperationKind.COMMAND
                if tool == "Bash"
                else PlanOperationKind.FILE_CHANGE
                if tool in {"Edit", "Write", "NotebookEdit"}
                else PlanOperationKind.MCP
                if tool.startswith("mcp__")
                else PlanOperationKind.OTHER
            ),
            shell_expression=non_blank_text(raw.get("command")) if tool == "Bash" else None,
            # Claude's Bash cwd can persist across commands; launch cwd is not
            # evidence of the current native execution directory.
            execution_dir=None,
            permission_targets=tuple(targets),
            native_binding_ids=tuple(bindings),
            native_input=raw,
        )


def run_sdk_session(
    request: HeadlessSessionRequest, context: HeadlessRuntimeContext
) -> HeadlessSessionResult:
    """Run SDK dispatch under the existing managed deadline and output contract."""
    sdk, _ = load_sdk()
    transport = SDKTransport(context, sdk_command(request))

    async def run() -> tuple[dict[str, Any], int]:
        client = sdk.ClaudeSDKClient(
            options=sdk_options(request, transport.can_use_tool), transport=transport
        )
        transport.client = client
        succeeded = False
        try:
            await client.connect()
            transport.query_submitted = True
            await client.query(request.prompt)
            response = client.receive_messages()
            pending = asyncio.create_task(anext(response, None))
            finishing: asyncio.Task[int] | None = None
            while True:
                if transport.failure_terminal is None:
                    if transport.failure is not None:
                        raise transport.failure
                    context.checkpoint()
                if finishing is None and (
                    transport.terminal is not None or transport.failure_terminal
                ):
                    finishing = asyncio.create_task(transport.finish())
                if finishing is not None and finishing.done():
                    code = finishing.result()
                    break
                await asyncio.wait({pending}, timeout=0.01)
                if pending.done():
                    if pending.result() is None:
                        if finishing is None:
                            raise ValueError("Claude SDK omitted the submitted-prompt result")
                        code = await finishing
                        break
                    pending = asyncio.create_task(anext(response, None))
            if transport.failure_terminal is None and transport.failure is not None:
                raise transport.failure
            frame = transport.terminal or transport.failure_terminal
            if frame is None:
                raise ValueError("Claude SDK omitted the native result")
            succeeded = True
            return frame, code
        finally:
            # Same task owns SDK connect/disconnect (anyio task-group invariant).
            # Process operations are bounded even when SDK teardown is shielded.
            if not succeeded and transport.controller is not None:
                cleanup = HeadlessCleanupContext(
                    deadline=asyncio.get_running_loop().time() + 0.2,
                    cancel_event=threading.Event(),
                )
                await asyncio.to_thread(transport.controller._abort, cleanup)
            if "finishing" in locals() and finishing is not None:
                if not finishing.done():
                    finishing.cancel()
                with suppress(BaseException):
                    await finishing
            if "pending" in locals():
                if not pending.done():
                    pending.cancel()
                with suppress(BaseException):
                    await pending
            await client.disconnect()

    try:
        frame, code = asyncio.run(run())
    except BaseException:
        if isinstance(transport.failure, _HeadlessStopError):
            raise transport.failure from None
        context.checkpoint()
        if (
            not transport.query_submitted
            and transport.terminal is not None
            and (
                transport.terminal.get("is_error") is True
                or transport.terminal.get("subtype") != "success"
            )
        ):
            frame = transport.terminal
            assert transport.controller is not None
            code = transport.controller.process._proc.wait(timeout=context.remaining_seconds())
        else:
            if transport.failure is not None and not isinstance(transport.failure, ValueError):
                raise transport.failure from None
            raise context.transport_error(
                "Claude SDK session failed during native dispatch."
            ) from None
    terminal_frame = transport.failure_terminal or frame
    native_status = _identity(terminal_frame.get("subtype"))
    status = (
        HeadlessTerminalStatus.CANCELLED
        if terminal_frame.get("terminal_reason") in {"aborted_streaming", "aborted_tools"}
        else HeadlessTerminalStatus.SUCCEEDED
        if native_status == "success"
        and terminal_frame.get("is_error") is False
        and not frame.get("deferred_tool_use")
        and transport.run_complete
        and not transport.turn_in_progress
        and transport.session_state in (None, "idle")
        else HeadlessTerminalStatus.FAILED
    )
    usage = usage_from(
        frame.get("usage"),
        input_tokens="input_tokens",
        output_tokens="output_tokens",
        cached="cache_read_input_tokens",
        session_id=transport.session_id,
    )
    if transport.terminal is None:
        transport._record_prompt_accounting(frame)
    context.emit(HeadlessEventKind.TERMINAL, terminal_status=status)
    return complete_session(
        context,
        request,
        status=status,
        exit_code=code,
        response_text=non_blank_text(frame.get("result")),
        native_object=frame,
        structured_output=frame.get("structured_output", MISSING),
        native_status=native_status,
        session_id=transport.session_id,
        usage=usage,
        warnings=(
            ("Claude SDK exited before its active lifecycle became idle.",)
            if transport.session_state not in (None, "idle")
            else ()
        ),
    )
