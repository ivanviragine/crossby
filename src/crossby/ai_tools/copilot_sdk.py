"""Pinned optional Copilot SDK dispatch over bounded, Crossby-owned pipes.

Private seams are release-specific and checked at preflight. The actual SDK
owns session/RPC conversion; Crossby owns subprocesses and response commitment.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import re
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from importlib import metadata
from types import SimpleNamespace
from typing import Any

from crossby.ai_tools.headless import (
    HeadlessCleanupContext,
    HeadlessRuntimeContext,
    HeadlessTransportError,
    _HeadlessInteractionCancelledError,
    _HeadlessStopError,
)
from crossby.ai_tools.session_protocol import ProtocolSessionController
from crossby.models.ai import (
    HeadlessEventKind,
    HeadlessLaunchContext,
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
    TokenUsage,
)

SDK_VERSION = "1.0.21"
NATIVE_VERSION = "1.0.97"
SDK_REMEDIATION = 'Install "crossby[copilot-sdk]" and Copilot CLI 1.0.97.'
_MAX_IDS = 4096
_MAX_PENDING_BYTES = 1024 * 1024


def load_sdk() -> tuple[Any, Any, Any]:
    """Import only the exact verified wheel, including its private bridge API."""
    if metadata.version("github-copilot-sdk") != SDK_VERSION:
        raise ImportError("unsupported Copilot SDK version")
    sdk = importlib.import_module("copilot.client")
    rpc = importlib.import_module("copilot.generated.rpc")
    wire = importlib.import_module("copilot._jsonrpc")
    session = importlib.import_module("copilot.session")
    required = {
        sdk.CopilotClient: {
            "_start_cli_server",
            "_connect_via_stdio",
            "_register_session",
            "create_session",
            "resume_session",
            "get_status",
        },
        session.CopilotSession: {
            "_execute_permission_and_respond",
            "_handle_elicitation_request",
            "_handle_user_input_request",
            "abort",
            "disconnect",
            "send",
        },
        wire.JsonRpcClient: {
            "_handle_message",
            "set_raw_request_handler",
            "_dispatch_notification",
        },
    }
    if any(
        not all(callable(getattr(cls, name, None)) for name in names)
        for cls, names in required.items()
    ):
        raise ImportError("unsupported Copilot SDK API")
    fields = set(inspect.signature(sdk.CopilotClient.create_session).parameters)
    if (
        not {
            "on_permission_request",
            "on_user_input_request",
            "on_elicitation_request",
            "enable_managed_settings",
            "disabled_mcp_servers",
            "on_event",
        }
        <= fields
    ):
        raise ImportError("unsupported Copilot SDK session options")
    for name in (
        "PermissionsConfigureParams",
        "PermissionsModifyRulesParams",
        "PermissionRule",
        "PermissionDecisionApproveOnce",
        "PermissionDecisionReject",
        "PermissionDecisionUserNotAvailable",
        "PermissionDecisionRequest",
        "UIHandlePendingElicitationRequest",
        "UIElicitationResponse",
    ):
        if not callable(getattr(rpc, name, None)):
            raise ImportError("unsupported Copilot generated API")
    return sdk, rpc, wire


def brokered_profile() -> HeadlessModeCapability | None:
    try:
        load_sdk()
    except Exception:
        return None
    return HeadlessModeCapability(
        transport=HeadlessNativeTransport.SDK,
        prompt_transport=HeadlessPromptTransport.PROTOCOL,
        native_outputs=(HeadlessNativeOutput.TEXT,),
        supports_resume=True,
        supports_native_abort=True,
        terminal_event_required=True,
        terminal_event_authoritative=True,
        successful_native_statuses=("success",),
        sandbox_behavior=PlanRequestBehavior.TOOL_MANAGED,
        command_policy_support=PlanCommandPolicySupport.NATIVE,
        supported_interaction_kinds=(PlanInteractionKind.QUESTION, PlanInteractionKind.PERMISSION),
        version_requirement=f"Copilot CLI=={NATIVE_VERSION}, SDK=={SDK_VERSION}, protocol 3.",
        verified_version=NATIVE_VERSION,
        limitation="Pinned owned SDK; text only, native manual approval, tool-managed sandbox; "
        "URL elicitation, effort, network overrides and other approval postures are unsupported.",
    )


def launch_context(
    *, scene: Any = None, allow_tools: tuple[str, ...] = (), allow_urls: tuple[str, ...] = ()
) -> HeadlessLaunchContext:
    """Resolve the existing profile/scene inputs without mutating native config."""
    return HeadlessLaunchContext(
        allow_tools=allow_tools,
        allow_urls=allow_urls,
        excluded_mcp_servers=tuple(sorted(scene.deselected_mcp())) if scene is not None else (),
    )


def approval_rules(request: HeadlessSessionRequest) -> list[tuple[str, str | None]]:
    from crossby.ai_tools.copilot import CopilotAdapter, _allow_entry_excluded

    launch = request.launch_context or HeadlessLaunchContext()
    entries = list(launch.allow_tools)
    if request.command_policy is not None:
        rendered = CopilotAdapter().allowed_commands_args(
            list(request.command_policy.allowed_commands)
        )
        entries += rendered[1::2]
    rules: list[tuple[str, str | None]] = []
    for entry in entries:
        if _allow_entry_excluded(entry, set(launch.excluded_mcp_servers)):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\(([^()\r\n]+)\))?", entry)
        if not match or "__" in match[1]:
            raise ValueError("Unsupported native Copilot rule; use kind or kind(argument).")
        rules.append((match[1], match[2]))
    rules.extend(("url", url) for url in launch.allow_urls)
    return rules


def command() -> list[str]:
    # Explicit installed path prevents SDK runtime downloads and transport env overrides.
    return [
        shutil.which("copilot") or "copilot",
        "--headless",
        "--no-auto-update",
        "--log-level",
        "none",
        "--stdio",
    ]


def identity(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > 4096:
        raise ValueError("Copilot omitted a bounded native identity")
    return value


class SDKController(ProtocolSessionController):
    """One cleanup chain; even expired writes attempt the real SDK abort first."""

    def __init__(self, bridge: Bridge) -> None:
        self.bridge = bridge
        self.abort_lock = threading.Lock()
        self.aborted = False
        self.abort_done = threading.Event()
        super().__init__(
            bridge.context, command(), cwd=bridge.context.request.working_dir, content_length=True
        )

    def _abort(self, cleanup: HeadlessCleanupContext) -> None:
        with self.abort_lock:
            first = not self.aborted
            self.aborted = True
        if not first:
            self.abort_done.wait(timeout=cleanup.remaining_seconds())
            return
        try:
            self.bridge.revoke()
            if self.bridge.session is None or not self.bridge.loop.is_running():
                return
            self.bridge.cleanup_deadline = cleanup.deadline
            future = asyncio.run_coroutine_threadsafe(self.bridge.session.abort(), self.bridge.loop)
            try:
                future.result(timeout=cleanup.remaining_seconds())
            except Exception:
                future.cancel()
        finally:
            self.abort_done.set()


class Bridge:
    def __init__(self, context: HeadlessRuntimeContext) -> None:
        self.context = context
        self.loop = asyncio.get_running_loop()
        self.controller: SDKController | None = None
        self.session: Any = None
        self.question_key: ContextVar[tuple[str, int | str]] = ContextVar("copilot_question_key")
        self.session_id: str | None = None
        self.rpc: Any = None
        self.closed = threading.Event()
        self.eof = threading.Event()
        self.cleanup_deadline: float | None = None
        self.failure: BaseException | None = None
        self.native_failed = False
        self.native_status: str | None = None
        self.native_aborted = False
        self.terminal = False
        self.sending = False
        self.message_id: str | None = None
        self.turn_id: str | None = None
        self.started = False
        self.final: str | None = None
        self.pending: dict[tuple[str, int | str], tuple[dict[str, Any], threading.Event]] = {}
        self.seen: set[tuple[str, int | str]] = set()
        self.lock = threading.RLock()
        self.accept = True
        self.frames = 0
        self.events: asyncio.Queue[tuple[dict[str, Any], Any]] = asyncio.Queue(_MAX_IDS)
        self.done = asyncio.Event()
        self.usage: TokenUsage | None = None
        self.usage_calls: set[str] = set()
        self.usage_events: set[str] = set()
        self.processing = False

    def fail(self, exc: BaseException) -> None:
        if self.failure is None:
            self.failure = exc
        self.loop.call_soon_threadsafe(self.done.set)

    def record_status(self, status: str) -> None:
        if self.native_status is None:
            self.context.set_provenance(native_status=status)
            self.native_status = status

    def track(self, key: tuple[str, int | str], data: dict[str, Any]) -> None:
        with self.lock:
            if not self.accept or key in self.seen or len(self.seen) >= _MAX_IDS:
                raise ValueError("Copilot reused a request or exceeded the request limit")
            total = sum(len(json.dumps(item[0]).encode()) for item in self.pending.values())
            if total + len(json.dumps(data, allow_nan=False).encode()) > _MAX_PENDING_BYTES:
                raise ValueError("Copilot pending requests exceeded the byte limit")
            self.pending[key] = (data, threading.Event())
            self.seen.add(key)

    def withdraw(self, key: tuple[str, int | str]) -> None:
        with self.lock:
            item = self.pending.pop(key, None)
            if item:
                item[1].set()

    def revoke(self) -> None:
        with self.lock:
            self.accept = False
            for _, signal in self.pending.values():
                signal.set()
            self.pending.clear()

    @contextmanager
    def commit(self, key: tuple[str, int | str]) -> Iterator[None]:
        with self.lock:
            if not self.accept or key not in self.pending:
                raise _HeadlessInteractionCancelledError("Copilot request was withdrawn.")
            self.context.checkpoint()
            self.pending.pop(key)
        # Never retain the cancellation registry lock across physical pipe I/O.
        yield

    def before_stop(self) -> None:
        assert self.controller is not None
        self.controller._abort(
            HeadlessCleanupContext(deadline=time.monotonic() + 0.2, cancel_event=threading.Event())
        )

    def observe(self, frame: dict[str, Any]) -> None:
        """Run on the reader before SDK observers; retain evidence and withdrawal."""
        self.frames += 1
        if self.frames > 16384:
            raise ValueError("Copilot exceeded the session frame limit")
        if "error" in frame and "id" in frame:
            self.native_failed = True
            self.record_status("failure")
            error = frame["error"]
            if not isinstance(error, dict) or not isinstance(error.get("code"), int):
                raise ValueError("Copilot emitted an invalid native RPC error")
            # SDK request logging includes its exception message. Preserve the
            # protocol code while withholding native diagnostics/session text.
            frame["error"] = {"code": error["code"], "message": "Copilot native RPC failed."}
        method = frame.get("method")
        params = frame.get("params", {})
        if method == "$/cancelRequest":
            self.withdraw(("rpc", params.get("id")))
        elif method == "userInput.request":
            rid = frame.get("id")
            if not isinstance(rid, int) or isinstance(rid, bool):
                raise ValueError("Copilot legacy question requires a numeric native RPC ID")
            if params.get("sessionId") != self.session_id or not self.sending:
                raise ValueError("Copilot question has no active session binding")
            self.track(("rpc", rid), params)
        elif method is not None and "id" in frame:
            raise ValueError("Copilot requested an unsupported native callback")
        elif method == "session.event":
            if params.get("sessionId") != self.session_id:
                raise ValueError("Copilot changed the native session binding")
            self.context.set_provenance(session_id=self.session_id)
            event = params["event"]
            data = event["data"]
            kind = event["type"]
            if not isinstance(data, dict):
                raise ValueError("Copilot event data is not an object")
            if kind in {"permission.completed", "elicitation.completed"}:
                if kind == "permission.completed" and data.get("result", {}).get(
                    "kind", ""
                ).startswith("denied"):
                    self.context.add_denial("Copilot denied a native permission request.")
                self.withdraw((kind.split(".")[0], identity(data.get("requestId"))))
            if kind in {"permission.requested", "elicitation.requested"}:
                if not self.sending or self.terminal:
                    raise ValueError("Copilot requested interaction outside the submitted turn")
                if data.get("resolvedByHook") is not True:
                    self.track((kind.split(".")[0], identity(data.get("requestId"))), data)
            if event.get("agentId"):
                return
            if kind == "session.error":
                self.native_failed = True
                self.record_status("failure")
            if kind == "session.idle" and self.started and data.get("aborted") is True:
                self.native_aborted = True
                self.record_status("aborted")
            if kind == "assistant.usage" and self.started:
                self.record_usage(data, identity(event.get("id")))

    def record_usage(self, data: dict[str, Any], event_id: str) -> None:
        if event_id in self.usage_events:
            return
        self.usage_events.add(event_id)
        call_id = data.get("apiCallId")
        if call_id is not None:
            call_id = identity(call_id)
            if call_id in self.usage_calls:
                return
            self.usage_calls.add(call_id)
        values = [data.get(k, 0) or 0 for k in ("inputTokens", "outputTokens", "cacheReadTokens")]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or int(v) != v
            for v in values
        ):
            raise ValueError("Copilot reported invalid token accounting")
        prior = self.usage or TokenUsage()
        self.usage = TokenUsage(
            input_tokens=(prior.input_tokens or 0) + int(values[0]),
            output_tokens=(prior.output_tokens or 0) + int(values[1]),
            cached_tokens=(prior.cached_tokens or 0) + int(values[2]),
            session_id=self.session_id,
        )
        self.context.record_usage(self.usage)

    async def interact(self, key: tuple[str, int | str], interaction: SessionInteraction) -> Any:
        with self.lock:
            item = self.pending.get(key)
        if item is None:
            raise _HeadlessInteractionCancelledError("Copilot request was withdrawn.")
        return await asyncio.to_thread(
            self.context.native_interaction_response, interaction, request_cancel_event=item[1]
        )

    async def raw_question(self, frame: dict[str, Any]) -> None:
        token = self.question_key.set(("rpc", frame["id"]))
        try:
            result = await self.session._handle_user_input_request(frame["params"])
            await self.rpc._send_message({"jsonrpc": "2.0", "id": frame["id"], "result": result})
        except _HeadlessInteractionCancelledError:
            pass
        except BaseException as exc:
            self.fail(exc)
        finally:
            self.question_key.reset(token)

    async def answer_question(self, _request: Any, invocation: Any) -> dict[str, Any]:
        key = self.question_key.get()
        with self.lock:
            item = self.pending.get(key)
        if item is None:
            raise _HeadlessInteractionCancelledError("Copilot question was withdrawn.")
        params = item[0]
        if invocation.get("session_id") != params.get("sessionId"):
            raise ValueError("Copilot user-input callback changed session binding")

        choices = params.get("choices") or []
        freeform = params.get("allowFreeform", True)
        if (
            not isinstance(choices, list)
            or len(set(choices)) != len(choices)
            or not isinstance(freeform, bool)
        ):
            raise ValueError("Copilot question choices are ambiguous")
        interaction = SessionInteraction(
            kind=PlanInteractionKind.QUESTION,
            question_id=str(key[1]),
            request_id=str(key[1]),
            session_id=identity(params.get("sessionId")),
            prompt=identity(params.get("question")),
            options=tuple(
                SessionQuestionOption(option_id=identity(c), label=identity(c)) for c in choices
            ),
            allow_other=freeform,
        )
        response = await self.interact(key, interaction)
        selected = response.option_id or (response.option_ids[0] if response.option_ids else None)
        if response.content is not None or (selected is not None and response.answer is not None):
            raise ValueError("Copilot question requires one choice or freeform answer")
        if selected is not None:
            result = {"answer": selected, "wasFreeform": False}
        elif freeform and response.answer is not None and response.answer.strip():
            result = {"answer": response.answer, "wasFreeform": True}
        else:
            raise ValueError("Copilot question was not answered")
        return result

    async def permission(self, rid: str, native: Any, _handler: Any) -> None:
        key = ("permission", rid)
        try:
            with self.lock:
                item = self.pending.get(key)
            if item is None:
                return
            data = item[0]
            raw = data["permissionRequest"]
            if data.get("ownerSessionId") not in (None, self.session_id) or data.get(
                "ownerAgentId"
            ):
                raise ValueError("Copilot permission belongs to an unsupported child owner")
            interaction = SessionInteraction(
                kind=PlanInteractionKind.PERMISSION,
                question_id=rid,
                request_id=rid,
                session_id=identity(self.session_id),
                prompt="Copilot requests permission for a native operation.",
                operation=operation(raw, rid, self.session_id),
            )
            response = await self.interact(key, interaction)
            _, rpc, _ = load_sdk()
            if response.outcome is PlanInteractionOutcome.APPROVED:
                decision = rpc.PermissionDecisionApproveOnce()
            elif response.outcome is PlanInteractionOutcome.DENIED:
                decision = rpc.PermissionDecisionReject()
            elif response.outcome is PlanInteractionOutcome.SKIPPED:
                decision = rpc.PermissionDecisionUserNotAvailable()
            else:
                raise ValueError("unsupported Copilot permission decision")
            applied = await self.session.rpc.permissions.handle_pending_permission_request(
                rpc.PermissionDecisionRequest(request_id=rid, result=decision)
            )
            if not applied.success:
                self.context.add_warning("Copilot permission request was already resolved.")
        except _HeadlessInteractionCancelledError:
            pass
        except BaseException as exc:
            self.fail(exc)

    async def elicitation(self, _context: Any, rid: str) -> None:
        key = ("elicitation", rid)
        try:
            with self.lock:
                item = self.pending.get(key)
            if item is None:
                return
            data = item[0]
            if data.get("mode") not in (None, "form") or data.get("url"):
                raise ValueError("Copilot URL elicitation is unsupported; use a native UI host.")
            if data.get("elicitationSource") is not None:
                raise ValueError("MCP elicitation dialect requires a native UI host.")
            schema = form_schema(data.get("requestedSchema"))
            response = await self.interact(
                key,
                SessionInteraction(
                    kind=PlanInteractionKind.QUESTION,
                    question_id=rid,
                    request_id=rid,
                    session_id=identity(self.session_id),
                    prompt=identity(data.get("message")),
                    form_schema=schema,
                ),
            )
            _, rpc, _ = load_sdk()
            content = response.content
            if response.outcome is PlanInteractionOutcome.ANSWERED:
                if (
                    content is None
                    or response.answer is not None
                    or response.option_id
                    or response.option_ids
                ):
                    raise ValueError("Copilot form requires explicit content")
                if any(
                    not isinstance(v, (str, bool, int, float, list))
                    or (isinstance(v, list) and any(not isinstance(x, str) for x in v))
                    for v in content.values()
                ):
                    raise ValueError("unsupported Copilot form value")
                if set(content) - set(schema["properties"]):
                    raise ValueError("Copilot form returned unknown fields")
                action = "accept"
            else:
                action, content = "decline", None
            applied = await self.session.rpc.ui.handle_pending_elicitation(
                rpc.UIHandlePendingElicitationRequest(
                    request_id=rid,
                    result=rpc.UIElicitationResponse(
                        action=rpc.UIElicitationResponseAction(action), content=content
                    ),
                )
            )
            if not applied.success:
                self.context.add_warning("Copilot elicitation request was already resolved.")
        except _HeadlessInteractionCancelledError:
            pass
        except BaseException as exc:
            self.fail(exc)

    def notification(self, method: str, params: dict[str, Any]) -> None:
        if method != "session.event":
            return
        try:
            sdk, _, _ = load_sdk()
            event = sdk.session_event_from_dict(params["event"])
            self.session._dispatch_event(event)
            kind = params["event"]["type"]
            if (
                not event.agent_id
                and not self.terminal
                and (
                    kind in {"session.mcp_servers_loaded", "session.tools_updated"}
                    or (
                        self.sending
                        and kind
                        in {
                            "user.message",
                            "assistant.message",
                            "assistant.message_delta",
                            "assistant.streaming_delta",
                            "assistant.turn_start",
                            "assistant.turn_end",
                            "tool.execution_start",
                            "tool.execution_complete",
                            "assistant.usage",
                            "permission.requested",
                            "elicitation.requested",
                        }
                    )
                )
            ):
                # Admission can still be awaiting its native message ID. Keep
                # the idle clock moving on validated session-scoped progress,
                # while ordered result processing waits for exact turn binding.
                self.context.checkpoint()
                self.context.mark_progress()
            self.events.put_nowait((params["event"], event))
        except BaseException as exc:
            self.fail(exc)

    async def process_events(self) -> None:
        try:
            while True:
                raw, _event = await self.events.get()
                self.processing = True
                try:
                    # Buffer processing until the SDK admits and identifies our send.
                    while self.sending and self.message_id is None and not self.failure:
                        self.context.checkpoint()
                        await asyncio.sleep(0.005)
                    if self.failure:
                        return
                    await asyncio.to_thread(self.event, raw)
                finally:
                    self.processing = False
                    self.events.task_done()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self.fail(exc)

    async def drain_events(self) -> None:
        await asyncio.sleep(0)
        while not self.events.empty() or self.processing:
            if self.failure:
                raise self.failure
            self.context.checkpoint()
            await asyncio.sleep(0.005)
        if self.failure:
            raise self.failure

    def event(self, raw: dict[str, Any]) -> None:
        kind, data = raw["type"], raw["data"]
        if raw.get("agentId") or not self.sending:
            return
        if kind == "user.message" and data.get("messageId") == self.message_id:
            self.started = True
            if data.get("turnId") is not None:
                self.turn_id = identity(data["turnId"])
                self.context.set_provenance(turn_id=self.turn_id)
        if kind == "assistant.message":
            if data.get("originatingMessageId") != self.message_id:
                return
            self.started = True
            if self.terminal:
                raise ValueError("Copilot emitted a second final response after idle")
            self.final = None if data.get("toolRequests") else data.get("content")
            if self.final is not None and not isinstance(self.final, str):
                raise ValueError("invalid Copilot final response")
            self.context.emit(HeadlessEventKind.ASSISTANT_MESSAGE)
        elif kind == "session.error":
            self.loop.call_soon_threadsafe(self.done.set)
        elif kind == "session.idle" and self.started and data.get("mode") != "autopilot":
            self.native_aborted = data.get("aborted") is True
            if self.native_aborted:
                self.record_status("aborted")
            with self.lock:
                pending = bool(self.pending)
            if pending or (not self.final and not self.native_aborted):
                self.native_failed = True
            self.terminal = True
            self.loop.call_soon_threadsafe(self.done.set)
        elif self.started and kind in {
            "assistant.message_delta",
            "assistant.streaming_delta",
            "assistant.turn_start",
            "assistant.turn_end",
            "tool.execution_start",
            "tool.execution_complete",
            "assistant.usage",
        }:
            if self.terminal:
                raise ValueError("Copilot emitted active output after terminal idle")
            if kind == "assistant.usage":
                self.record_usage(data, identity(raw.get("id")))
            normalized = {
                "tool.execution_start": HeadlessEventKind.TOOL_STARTED,
                "tool.execution_complete": HeadlessEventKind.TOOL_COMPLETED,
                "assistant.usage": HeadlessEventKind.USAGE,
            }.get(kind, HeadlessEventKind.PROGRESS)
            self.context.emit(normalized)


def operation(raw: dict[str, Any], rid: str, sid: str | None) -> SessionOperation:
    from pathlib import Path

    kind = raw.get("kind")
    mapping = {
        "shell": PlanOperationKind.COMMAND,
        "write": PlanOperationKind.FILE_CHANGE,
        "read": PlanOperationKind.OTHER,
        "url": PlanOperationKind.NETWORK,
        "mcp": PlanOperationKind.MCP,
        "custom-tool": PlanOperationKind.OTHER,
    }
    if kind not in mapping:
        raise ValueError("Copilot permission kind is unsupported")
    bindings = [
        SessionNativeBindingID(name="request_id", value=rid),
        SessionNativeBindingID(name="session_id", value=identity(sid)),
    ]
    if raw.get("toolCallId") is not None:
        bindings.append(
            SessionNativeBindingID(name="tool_call_id", value=identity(raw["toolCallId"]))
        )
    targets: list[SessionPermissionTarget] = []
    path = raw.get("resolvedPath")
    if path is not None:
        path = identity(path)
        if not Path(path).is_absolute():
            raise ValueError("Copilot resolved path is not absolute")
        targets.append(
            SessionPermissionTarget(
                kind=PlanPermissionTargetKind.FILESYSTEM_WRITE
                if kind == "write"
                else PlanPermissionTargetKind.FILESYSTEM_READ,
                value=path,
            )
        )
    cwd = raw.get("resolvedWorkingDirectory") if kind == "shell" else None
    if cwd is not None and not Path(identity(cwd)).is_absolute():
        raise ValueError("Copilot resolved cwd is not absolute")
    return SessionOperation(
        kind=mapping[kind],
        native_input=raw,
        shell_expression=identity(raw.get("fullCommandText")) if kind == "shell" else None,
        execution_dir=Path(cwd) if cwd else None,
        permission_targets=tuple(targets),
        native_binding_ids=tuple(bindings),
    )


def form_schema(value: Any) -> dict[str, Any]:
    """Keep SDK presentation hints intact; never reinterpret enums as restrictions."""
    if (
        not isinstance(value, dict)
        or value.get("type") != "object"
        or not isinstance(value.get("properties"), dict)
    ):
        raise ValueError("Unsupported Copilot elicitation form; use a native UI host.")
    if set(value) - {"type", "properties", "required"}:
        raise ValueError("Unsupported Copilot elicitation schema dialect.")
    for field in value["properties"].values():
        if not isinstance(field, dict) or field.get("type") not in {
            "string",
            "boolean",
            "number",
            "integer",
            "array",
        }:
            raise ValueError("Unsupported Copilot elicitation field.")
        if set(field) - {
            "type",
            "title",
            "description",
            "default",
            "enum",
            "enumNames",
            "oneOf",
            "items",
            "format",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
        }:
            raise ValueError("Unsupported Copilot elicitation field constraint.")
    return value


def client_for(bridge: Bridge) -> Any:
    sdk, _, wire = load_sdk()

    class OwnedRPC(wire.JsonRpcClient):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            assert bridge.controller is not None
            super().__init__(
                SimpleNamespace(stderr=None, poll=lambda: bridge.controller.process.returncode)
            )

        def _get_process_exit_error(self) -> str:
            return "Copilot owned SDK connection closed."

        def _wait_for_owned_close(self, exc: BaseException) -> None:
            bridge.fail(exc)
            # Keep SDK request admission open for the bounded abort attempt.
            # Returning EOF first makes SDK latch transport closure before abort.
            while not bridge.closed.wait(0.01):
                pass

        def _read_message(self) -> Any:
            assert bridge.controller is not None
            while not bridge.closed.is_set():
                try:
                    bridge.context.checkpoint()
                    frame = bridge.controller.process.read(
                        timeout=min(0.05, bridge.context.remaining_seconds())
                    )
                    bridge.observe(frame)
                    return frame
                except TimeoutError:
                    try:
                        # Idle may have moved; an expired overall stop must reach the owner.
                        bridge.context.checkpoint()
                    except BaseException as exc:
                        self._wait_for_owned_close(exc)
                        return None
                except EOFError:
                    bridge.eof.set()
                    if not bridge.terminal and not bridge.native_failed:
                        bridge.fail(ValueError("Copilot SDK reached EOF before completion"))
                    return None
                except BaseException as exc:
                    self._wait_for_owned_close(exc)
                    return None
            return None

        def _dispatch_notification(self, handler: Any, params: Any) -> None:
            try:
                handler(params)
            except BaseException as exc:
                bridge.fail(exc)

        async def request(self, method: str, params: Any = None, **kwargs: Any) -> Any:
            cleanup = method == "session.abort" and bridge.cleanup_deadline is not None
            deadline = bridge.cleanup_deadline if cleanup else bridge.context.deadline
            assert deadline is not None
            remaining = max(0.001, deadline - time.monotonic())
            kwargs["timeout"] = min(kwargs.get("timeout") or remaining, remaining)
            pending = asyncio.create_task(super().request(method, params, **kwargs))
            try:
                while not pending.done():
                    if not cleanup:
                        if bridge.failure is not None:
                            raise bridge.failure
                        bridge.context.checkpoint()
                    await asyncio.wait((pending,), timeout=0.01)
                response = await pending
                if method in {"session.create", "session.resume"}:
                    if not isinstance(response, dict):
                        raise ValueError("Copilot omitted the native session response")
                    confirmed = response.get("sessionId")
                    if confirmed is not None and confirmed != params.get("sessionId"):
                        raise ValueError("Copilot returned a different native session identity")
                return response
            finally:
                if not pending.done():
                    pending.cancel()
                with suppress(BaseException):
                    await pending

        async def _send_message(self, message: dict[str, Any]) -> None:
            assert bridge.controller is not None
            if message.get("method") == "session.abort" and bridge.cleanup_deadline is not None:
                await asyncio.to_thread(bridge.controller.process.send_cleanup, message)
                return
            bridge.context.checkpoint()
            key: tuple[str, int | str] | None = None
            if "result" in message or "error" in message:
                key = ("rpc", message["id"])
            method = message.get("method", "")
            if method in {
                "session.permissions.handlePendingPermissionRequest",
                "session.ui.handlePendingElicitation",
            }:
                key = (
                    "permission" if "Permission" in method else "elicitation",
                    message["params"]["requestId"],
                )
            await asyncio.to_thread(
                bridge.controller.process.send,
                message,
                # Physical I/O cannot safely restart a partial frame. Use the
                # fixed overall limit; readers/runtime monitor current idle.
                timeout=max(0.0, bridge.context.deadline - time.monotonic()),
                write_guard=(lambda: bridge.commit(key)) if key is not None else None,
                before_stop=bridge.before_stop,
            )

    class OwnedClient(sdk.CopilotClient):  # type: ignore[misc, name-defined]
        async def _start_cli_server(self) -> None:
            bridge.controller = SDKController(bridge)

        async def _connect_via_stdio(self) -> None:
            self._client = OwnedRPC()
            bridge.rpc = self._client
            self._rpc = sdk.ServerRpc(self._client)
            self._client.set_notification_handler(bridge.notification)
            self._client.set_raw_request_handler("userInput.request", bridge.raw_question)
            self._client.start(asyncio.get_running_loop())

        def _register_session(self, session: Any) -> Any:
            bridge.session, bridge.session_id = session, identity(session.session_id)
            session._execute_permission_and_respond = bridge.permission
            session._handle_elicitation_request = bridge.elicitation
            return super()._register_session(session)

    return OwnedClient(
        connection=sdk.RuntimeConnection.for_stdio(path=command()[0]),
        working_directory=str(bridge.context.request.working_dir),
        log_level="none",
    )


def run_session(
    request: HeadlessSessionRequest, context: HeadlessRuntimeContext
) -> HeadlessSessionResult:
    async def run() -> tuple[Bridge, int]:
        bridge = Bridge(context)
        client = client_for(bridge)
        event_task = asyncio.create_task(bridge.process_events())
        succeeded = False
        try:
            await client.start()
            status = await client.get_status()
            if status.version != NATIVE_VERSION or status.protocol_version != 3:
                raise ValueError("Unsupported Copilot runtime/protocol; " + SDK_REMEDIATION)
            launch = request.launch_context or HeadlessLaunchContext()
            from crossby.ai_tools.copilot import CopilotAdapter

            options = dict(
                model=CopilotAdapter().normalize_model_format(request.model)
                if request.model
                else None,
                working_directory=str(request.working_dir),
                additional_directories=[str(p) for p in request.trusted_dirs],
                streaming=True,
                disabled_mcp_servers=list(launch.excluded_mcp_servers),
                enable_managed_settings=True,
                enable_session_store=True,
                on_permission_request=lambda *_: None,
                on_user_input_request=bridge.answer_question,
                on_elicitation_request=lambda *_: None,
            )
            session = (
                await client.resume_session(
                    request.resume_id, continue_pending_work=False, **options
                )
                if request.resume_id
                else await client.create_session(**options)
            )
            context.set_provenance(session_id=identity(session.session_id))
            if request.resume_id and session.session_id != request.resume_id:
                raise ValueError("Copilot did not confirm the exact resume identity")
            _, rpc, _ = load_sdk()
            configured = await session.rpc.permissions.configure(
                rpc.PermissionsConfigureParams(
                    approve_all_read_permission_requests=False,
                    approve_all_tool_permission_requests=False,
                )
            )
            if not configured.success:
                raise ValueError("Copilot refused scoped permission configuration")
            rules = approval_rules(request)
            if rules:
                added = await session.rpc.permissions.modify_rules(
                    rpc.PermissionsModifyRulesParams(
                        scope=rpc.PermissionsModifyRulesScope.SESSION,
                        add=[rpc.PermissionRule(kind=k, argument=a) for k, a in rules],
                    )
                )
                if not added.success:
                    raise ValueError("Copilot refused session permission rules")
            mode = await session.rpc.permissions.set_mode(
                rpc.PermissionsSetModeRequest(mode=rpc.PermissionMode.MANUAL)
            )
            if not mode.success or mode.mode != rpc.PermissionMode.MANUAL:
                raise ValueError("Copilot refused manual permission mode")
            await bridge.drain_events()
            bridge.sending = True
            bridge.message_id = identity(await session.send(request.prompt))
            while not bridge.done.is_set():
                context.checkpoint()
                await asyncio.sleep(min(0.01, context.remaining_seconds()))
            if bridge.failure:
                raise bridge.failure
            await bridge.drain_events()
            if bridge.failure:
                raise bridge.failure
            await session.disconnect()
            await client._rpc.runtime.shutdown(
                timeout=max(0.0, context.deadline - time.monotonic())
            )
            assert bridge.controller is not None
            code = await asyncio.to_thread(
                bridge.controller.process.close, timeout=context.remaining_seconds()
            )
            # Give scheduled SDK parsing/observer work its final chance before completion.
            await asyncio.sleep(0)
            await bridge.drain_events()
            if bridge.failure:
                raise bridge.failure
            succeeded = True
            return bridge, code
        finally:
            if bridge.controller is not None:
                if not succeeded:
                    await asyncio.to_thread(bridge.before_stop)
                bridge.revoke()
                bridge.closed.set()  # Wake polling readers before shutting down the executor.
                await asyncio.to_thread(bridge.controller.process.close, timeout=0.2)
            event_task.cancel()
            with suppress(BaseException):
                await event_task
            if bridge.rpc is not None:
                await bridge.rpc.stop()

    try:
        bridge, code = asyncio.run(run())
    except (_HeadlessStopError, HeadlessTransportError):
        raise
    except BaseException:
        context.checkpoint()
        raise context.transport_error(
            "Copilot SDK dispatch failed; unsupported native forms/options "
            "require a compatible native UI host. " + SDK_REMEDIATION
        ) from None
    status = (
        HeadlessTerminalStatus.FAILED
        if bridge.native_failed or code != 0
        else HeadlessTerminalStatus.CANCELLED
        if bridge.native_aborted
        else HeadlessTerminalStatus.SUCCEEDED
    )
    context.emit(HeadlessEventKind.TERMINAL, terminal_status=status)
    return context.complete(
        status,
        final_text=bridge.final,
        exit_code=code,
        native_status=bridge.native_status or ("failure" if bridge.native_failed else "success"),
        session_id=bridge.session_id,
        turn_id=bridge.turn_id,
        usage=bridge.usage,
    )
