"""Run OpenCode planning through its native HTTP question and session APIs."""

from __future__ import annotations

import base64
import http.client
import json
import os
import queue
import re
import secrets
import signal
import socket
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from crossby.ai_tools.headless import (
    HeadlessCleanupContext,
    HeadlessCleanupHooks,
    HeadlessRuntimeContext,
)
from crossby.ai_tools.plan_mode import (
    PlanArtifactMalformedError,
    PlanBindingMismatchError,
    PlanInteractionHandler,
    PlanInteractionRequiredError,
    PlanSessionError,
    PlanSessionUnsupportedError,
    PlanTransportError,
    parse_plan_question_options,
    validate_plan_option_selection,
)
from crossby.ai_tools.plan_policy import command_pattern_to_shell_pattern
from crossby.ai_tools.plan_process import JsonRpcProcess, kill_process_group
from crossby.models.ai import (
    AIToolID,
    HeadlessEventKind,
    HeadlessSessionRequest,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanModeCapability,
    PlanNativeBindingID,
    PlanOperation,
    PlanOperationKind,
    PlanPermissionTarget,
    PlanPermissionTargetKind,
    PlanQuestionOption,
    PlanSessionRequest,
)

_HTTP_BODY_LIMIT = 8 * 1024 * 1024
_HTTP_BODY_CHUNK_SIZE = 64 * 1024
_SSE_EVENT_LIMIT = 1024 * 1024
_SSE_QUEUE_LIMIT = 256


class BoundedSSEParser:
    """Decode fragmented SSE data records without retaining an unbounded line."""

    def __init__(self) -> None:
        self._pending = bytearray()
        self._data: list[bytes] = []
        self._data_size = 0

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self._pending.extend(chunk)
        if len(self._pending) > _SSE_EVENT_LIMIT:
            raise ValueError("OpenCode SSE line exceeded its size limit")
        events: list[dict[str, Any]] = []
        while (newline := self._pending.find(b"\n")) >= 0:
            line = bytes(self._pending[:newline]).removesuffix(b"\r")
            del self._pending[: newline + 1]
            if not line:
                if self._data:
                    encoded = b"\n".join(self._data)
                    try:
                        value = json.loads(encoded.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise ValueError("OpenCode SSE contained malformed JSON") from exc
                    if not isinstance(value, dict):
                        raise ValueError("OpenCode SSE event was not an object")
                    events.append(value)
                    self._data.clear()
                    self._data_size = 0
                continue
            if line.startswith(b"data:"):
                data = line[5:].removeprefix(b" ")
                self._data_size += len(data)
                if self._data_size > _SSE_EVENT_LIMIT:
                    raise ValueError("OpenCode SSE event exceeded its size limit")
                self._data.append(data)
        return events

    def finish(self) -> None:
        if self._pending or self._data:
            raise ValueError("OpenCode SSE ended with an incomplete event")


class OpenCodeServer:
    """A run-owned, authenticated loopback server with a shared deadline."""

    def __init__(
        self,
        request: PlanSessionRequest | HeadlessSessionRequest,
        deadline: float,
        runtime: HeadlessRuntimeContext | None = None,
    ) -> None:
        self.deadline = deadline
        self.runtime = runtime
        self.directory = request.working_dir
        self.port: int | None = None
        self.active_session_id: str | None = None
        self._event_stop = threading.Event()
        self._event_connection: http.client.HTTPConnection | None = None
        self._event_thread: threading.Thread | None = None
        self._events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_SSE_QUEUE_LIMIT)
        self._event_error: ValueError | None = None
        self._bound_message_ids: set[str] = set()
        self._tool_states: dict[str, str] = {}
        password = secrets.token_urlsafe(32)
        self.authorization = "Basic " + base64.b64encode(f"opencode:{password}".encode()).decode()
        # The CLI run command disables questions. The native server exposes
        # them without changing the plan agent's sandbox/approval policy.
        self.process = JsonRpcProcess(
            ["opencode", "serve", "--hostname", "127.0.0.1", "--port", "0"],
            cwd=request.working_dir,
            timeout=self.remaining(),
            env={
                **os.environ,
                "OPENCODE_SERVER_USERNAME": "opencode",
                "OPENCODE_SERVER_PASSWORD": password,
                "OPENCODE_ENABLE_QUESTION_TOOL": "true",
            },
        )
        if runtime is not None:
            runtime.register_cleanup(
                HeadlessCleanupHooks(
                    native_abort=self._abort_active_session,
                    close_input=self._close_input,
                    terminate=self._terminate,
                    force_kill=lambda _ctx: kill_process_group(self.process._proc),
                    reap=self._reap,
                    join_workers=self._join_workers,
                )
            )

    def _close_input(self, _ctx: HeadlessCleanupContext) -> None:
        self._event_stop.set()
        if self._event_connection is not None:
            with suppress(OSError):
                self._event_connection.close()
        self.process.close_input()

    def _join_workers(self, ctx: HeadlessCleanupContext) -> None:
        self.process.join_workers(timeout=ctx.remaining_seconds())
        thread = self._event_thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout=ctx.remaining_seconds())

    def open_events(self) -> None:
        """Subscribe before prompting; polling remains terminal authority."""
        if self.port is None:
            raise OSError("OpenCode event stream requires a started server")

        def receive() -> None:
            parser = BoundedSSEParser()
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=max(0.01, self.deadline - time.monotonic())
            )
            self._event_connection = connection
            try:
                if self._event_stop.is_set():
                    return
                connection.request(
                    "GET",
                    "/event?" + urlencode({"directory": str(self.directory)}),
                    headers={
                        "Authorization": self.authorization,
                        "Accept": "text/event-stream",
                    },
                )
                response = connection.getresponse()
                if response.status != 200 or "text/event-stream" not in response.getheader(
                    "Content-Type", ""
                ):
                    return
                while not self._event_stop.is_set() and time.monotonic() < self.deadline:
                    chunk = response.read1(_HTTP_BODY_CHUNK_SIZE)
                    if not chunk:
                        parser.finish()
                        return
                    for event in parser.feed(chunk):
                        try:
                            self._events.put_nowait(event)
                        except queue.Full:
                            # Polling remains the terminal authority when progress is dropped.
                            continue
            except ValueError as exc:
                self._event_error = exc
            except (OSError, http.client.HTTPException):
                # Some verified 1.18.x builds expose the endpoint but do not
                # deliver live events. Bound HTTP polling still proves terminal
                # state and exact message binding.
                return
            finally:
                connection.close()

        self._event_thread = threading.Thread(
            target=receive, daemon=True, name="crossby-opencode-sse"
        )
        self._event_thread.start()

    def drain_events(
        self, context: HeadlessRuntimeContext, session_id: str, prompt_id: str
    ) -> None:
        """Normalize only bounded events for the active native prompt."""
        if self._event_error is not None:
            raise self._event_error
        for _ in range(_SSE_QUEUE_LIMIT):
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            kind = event.get("type")
            properties = event.get("properties")
            if not isinstance(kind, str) or not isinstance(properties, dict):
                raise ValueError("OpenCode SSE event had malformed native fields")
            if kind == "message.updated":
                info = properties.get("info")
                if not isinstance(info, dict):
                    raise ValueError("OpenCode SSE message update omitted info")
                if (
                    info.get("sessionID") != session_id
                    or info.get("role") != "assistant"
                    or info.get("parentID") != prompt_id
                ):
                    continue
                self._bound_message_ids.add(_required_text(info, "id"))
                context.emit(HeadlessEventKind.ASSISTANT_MESSAGE, message="message update")
                if isinstance(info.get("tokens"), dict):
                    context.emit(HeadlessEventKind.USAGE, message="usage updated")
            elif kind == "message.part.updated":
                part = properties.get("part")
                if not isinstance(part, dict):
                    raise ValueError("OpenCode SSE part update omitted its part")
                if part.get("sessionID") != session_id or part.get("type") != "tool":
                    continue
                if part.get("messageID") not in self._bound_message_ids:
                    continue
                part_id = _required_text(part, "id")
                state = part.get("state")
                status = state.get("status") if isinstance(state, dict) else None
                if not isinstance(status, str) or not status.strip():
                    raise ValueError("OpenCode SSE tool update omitted its status")
                if self._tool_states.get(part_id) == status:
                    continue
                self._tool_states[part_id] = status
                context.emit(
                    HeadlessEventKind.TOOL_COMPLETED
                    if status in {"completed", "error"}
                    else HeadlessEventKind.TOOL_STARTED,
                    payload={"tool_call_id": part_id},
                )
            elif kind == "message.part.delta":
                if (
                    properties.get("sessionID") != session_id
                    or properties.get("messageID") not in self._bound_message_ids
                ):
                    continue
                _required_text(properties, "partID")
                _required_text(properties, "field")
                _required_text(properties, "delta")
                context.mark_progress()

    def _abort_active_session(self, ctx: HeadlessCleanupContext) -> None:
        session_id = self.active_session_id
        if session_id is None or self.port is None:
            return
        budget = ctx.remaining_seconds()
        if budget <= 0 or ctx.cancel_event.is_set():
            return
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=min(0.2, budget))
        try:
            connection.request(
                "POST",
                f"/session/{quote(session_id, safe='')}/abort?"
                + urlencode({"directory": str(self.directory)}),
                headers={"Authorization": self.authorization},
            )
            response = connection.getresponse()
            response.read(1024)
        finally:
            connection.close()

    def _terminate(self, _ctx: HeadlessCleanupContext) -> None:
        proc = self.process._proc
        if proc.poll() is not None:
            return
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()

    def _reap(self, ctx: HeadlessCleanupContext) -> None:
        with suppress(subprocess.TimeoutExpired):
            self.process._proc.wait(timeout=ctx.remaining_seconds())

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenCode plan session exceeded its timeout")
        return remaining

    def start(self) -> None:
        while self.port is None:
            line = self.process.read_line(timeout=self.remaining())
            match = re.fullmatch(r"opencode server listening on http://127\.0\.0\.1:(\d+)\s*", line)
            if match is not None:
                port = int(match[1])
                if not 0 < port < 65536:
                    raise ValueError("OpenCode advertised an invalid loopback port")
                self.port = port
                self.process.discard_stdout()

    def request(self, method: str, path: str, payload: Any = None) -> Any:
        if self.runtime is not None:
            self.runtime.checkpoint()
        if self.port is None:
            raise OSError("OpenCode server has not started")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.remaining())
        try:
            connection.request(
                method,
                path + "?" + urlencode({"directory": str(self.directory)}),
                body=json.dumps(payload) if payload is not None else None,
                headers={"Authorization": self.authorization, "Content-Type": "application/json"},
            )
            response_socket = connection.sock
            if response_socket is None:
                raise OSError("OpenCode native API connection closed before its response")
            response_socket.settimeout(self.remaining())
            # ``HTTPResponse.begin()`` can perform multiple receives while parsing
            # a trickled status line or header block. A socket timeout bounds only
            # each inactive receive, so interrupt that phase at the absolute
            # request deadline as well.
            header_timer = threading.Timer(
                self.remaining(),
                lambda: _shutdown_socket(response_socket),
            )
            header_timer.daemon = True
            header_timer.start()
            try:
                response = connection.getresponse()
                self.remaining()
            finally:
                header_timer.cancel()
            if not 200 <= response.status < 300:
                raise OSError(f"OpenCode native API returned HTTP {response.status}")
            content = bytearray()
            while len(content) <= _HTTP_BODY_LIMIT:
                response_socket.settimeout(self.remaining())
                chunk = response.read1(
                    min(_HTTP_BODY_CHUNK_SIZE, _HTTP_BODY_LIMIT + 1 - len(content))
                )
                self.remaining()
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > _HTTP_BODY_LIMIT:
                    raise ValueError("OpenCode native API exceeded the response size limit")
            if self.runtime is not None:
                self.runtime.checkpoint()
            return json.loads(content) if content else None
        finally:
            connection.close()

    def close(self) -> int:
        self._event_stop.set()
        if self._event_connection is not None:
            with suppress(OSError):
                self._event_connection.close()
        if self._event_thread is not None and self._event_thread.ident is not None:
            self._event_thread.join(timeout=min(0.2, max(0.0, self.deadline - time.monotonic())))
        return self.process.close()


def run_native_plan(
    request: PlanSessionRequest,
    handler: PlanInteractionHandler | None,
    capability: PlanModeCapability,
    *,
    deadline: float,
) -> tuple[str, str]:
    """Finish one fresh plan session, returning its session and terminal message IDs."""
    prompt: dict[str, Any] = {
        "agent": "plan",
        "parts": [{"type": "text", "text": request.prompt}],
    }
    if request.model is not None:
        provider, separator, model = request.model.partition("/")
        if not separator or not provider.strip() or not model.strip():
            raise PlanSessionUnsupportedError(
                "OpenCode requires a provider/model identifier for collected plan sessions.",
                tool_id=AIToolID.OPENCODE,
                capability=capability,
            )
        prompt["model"] = {"providerID": provider, "modelID": model}
    if request.effort is not None:
        prompt["variant"] = request.effort.value
    server: OpenCodeServer | None = None
    session_id: str | None = None
    try:
        server = OpenCodeServer(request, deadline)
        server.start()
        if request.effort is not None:
            prompt["model"] = _validated_effort_model(server, request, capability)
        session_request: dict[str, Any] = {}
        if request.command_policy is not None:
            session_request["permission"] = [
                {"permission": "bash", "pattern": "*", "action": "ask"},
                *(
                    {
                        "permission": "bash",
                        "pattern": command_pattern_to_shell_pattern(pattern),
                        "action": "allow",
                    }
                    for pattern in request.command_policy.allowed_commands
                ),
            ]
        session = server.request("POST", "/session", session_request)
        session_id = _required_text(session, "id")
        session_path = "/session/" + quote(session_id, safe="")
        server.request("POST", session_path + "/prompt_async", prompt)
        while True:
            for question in _bound_requests(server.request("GET", "/question"), session_id):
                _answer_question(server, question, session_id, handler, capability)
            for permission in _bound_requests(server.request("GET", "/permission"), session_id):
                _answer_permission(
                    server,
                    permission,
                    session_id,
                    handler,
                    capability,
                    execution_dir=request.working_dir,
                )
            messages = server.request("GET", session_path + "/message")
            if not isinstance(messages, list):
                raise ValueError("OpenCode returned malformed session messages")
            if messages:
                info = messages[-1].get("info") if isinstance(messages[-1], dict) else None
                if not isinstance(info, dict) or info.get("sessionID") != session_id:
                    raise ValueError("OpenCode returned unbound session messages")
                if info.get("role") == "assistant":
                    if info.get("error") is not None:
                        raise PlanTransportError(
                            "OpenCode's native planning turn failed.",
                            tool_id=AIToolID.OPENCODE,
                            capability=capability,
                            session_id=session_id,
                        )
                    finish = info.get("finish")
                    if finish is not None and finish != "tool-calls":
                        if finish != "stop" or info.get("agent") != "plan":
                            raise ValueError("OpenCode did not finish successfully in plan mode")
                        completed = info.get("time")
                        if isinstance(completed, dict) and isinstance(
                            completed.get("completed"), (int, float)
                        ):
                            return session_id, _required_text(info, "id")
            time.sleep(min(0.1, server.remaining()))
    except PlanSessionError:
        raise
    except (OSError, ValueError, EOFError, TimeoutError, http.client.HTTPException) as exc:
        # API error bodies and JSON decode snippets can contain prompt text.
        raise PlanTransportError(
            f"OpenCode native planning session failed ({type(exc).__name__}).",
            tool_id=AIToolID.OPENCODE,
            capability=capability,
            session_id=session_id,
        ) from None
    finally:
        if server is not None:
            server.close()


def run_native_session(
    request: HeadlessSessionRequest,
    context: HeadlessRuntimeContext,
    capability: PlanModeCapability,
) -> tuple[str, str, str, Any]:
    """Run an ordinary session through the verified loopback question APIs."""
    # Match OpenCode's ascending message ID format so its native turn ordering
    # remains valid even when the prompt and reply are created in one millisecond.
    timestamp = (time.time_ns() // 1_000_000) & ((1 << 36) - 1)
    prompt_id = f"msg_{timestamp << 12:012x}{secrets.token_hex(7)}"
    prompt: dict[str, Any] = {
        "messageID": prompt_id,
        "parts": [{"type": "text", "text": request.prompt}],
    }
    if request.model is not None:
        provider, separator, model = request.model.partition("/")
        if not separator or not provider.strip() or not model.strip():
            raise ValueError("OpenCode requires a provider/model identifier")
        prompt["model"] = {"providerID": provider, "modelID": model}
    if request.effort is not None:
        prompt["variant"] = request.effort.value
    server = OpenCodeServer(request, context.deadline, runtime=context)
    server.start()
    if request.effort is not None:
        prompt["model"] = _validated_effort_model(server, request, capability, plan_agent=False)
    session_request: dict[str, Any] = {}
    if request.command_policy is not None:
        session_request["permission"] = [
            {"permission": "bash", "pattern": "*", "action": "ask"},
            *(
                {
                    "permission": "bash",
                    "pattern": command_pattern_to_shell_pattern(pattern),
                    "action": "allow",
                }
                for pattern in request.command_policy.allowed_commands
            ),
        ]
    if request.resume_id is None:
        session = server.request("POST", "/session", session_request)
        session_id = _required_text(session, "id")
        # Only a session created for this run is safe to abort as a whole.
        server.active_session_id = session_id
    else:
        session_id = request.resume_id
        context.validate_provenance(session_id=session_id)
        session = server.request("GET", "/session/" + quote(session_id, safe=""))
        if _required_text(session, "id") != session_id:
            raise ValueError("OpenCode resumed a different native session")
        native_directory = session.get("directory")
        if native_directory is not None and native_directory != str(request.working_dir):
            raise ValueError("OpenCode resumed a session from a different working directory")
    context.set_provenance(session_id=session_id)
    session_path = "/session/" + quote(session_id, safe="")
    existing_ids: set[str] = set()
    if request.resume_id is not None:
        history = server.request("GET", session_path + "/message")
        if not isinstance(history, list):
            raise ValueError("OpenCode returned malformed resumed session history")
        for message in history:
            info = message.get("info") if isinstance(message, dict) else None
            if not isinstance(info, dict) or info.get("sessionID") != session_id:
                raise ValueError("OpenCode resumed history from a different session")
            message_id = _required_text(info, "id")
            if message_id in existing_ids:
                raise ValueError("OpenCode resumed duplicate native message IDs")
            existing_ids.add(message_id)
    if prompt_id in existing_ids:
        raise ValueError("OpenCode generated a duplicate native prompt ID")
    if hasattr(server, "open_events"):
        server.open_events()
    server.request("POST", session_path + "/prompt_async", prompt)
    context.emit(HeadlessEventKind.STARTED, message="OpenCode session started")
    observed: set[str] = set(existing_ids)
    prompt_seen = False
    answered_requests: dict[str, str] = {}
    while True:
        context.checkpoint()
        if hasattr(server, "drain_events"):
            server.drain_events(context, session_id, prompt_id)
        for question in _prompt_bound_requests(
            server, server.request("GET", "/question"), session_id, prompt_id
        ):
            request_id = _required_text(question, "id")
            if request_id in answered_requests:
                if answered_requests[request_id] != "question":
                    raise ValueError("OpenCode reused a native request ID across interaction kinds")
                continue
            _answer_question(
                server, question, session_id, context.native_interaction_response, capability
            )
            answered_requests[request_id] = "question"
        for permission in _prompt_bound_requests(
            server, server.request("GET", "/permission"), session_id, prompt_id
        ):
            request_id = _required_text(permission, "id")
            if request_id in answered_requests:
                if answered_requests[request_id] != "permission":
                    raise ValueError("OpenCode reused a native request ID across interaction kinds")
                continue
            _answer_permission(
                server,
                permission,
                session_id,
                context.native_interaction_response,
                capability,
            )
            answered_requests[request_id] = "permission"
        messages = server.request("GET", session_path + "/message")
        if not isinstance(messages, list):
            raise ValueError("OpenCode returned malformed session messages")
        for message in messages:
            info = message.get("info") if isinstance(message, dict) else None
            if not isinstance(info, dict) or info.get("sessionID") != session_id:
                raise ValueError("OpenCode returned a message for another session")
            message_id = _required_text(info, "id")
            if message_id == prompt_id:
                if info.get("role") != "user":
                    raise ValueError("OpenCode prompt ID belongs to a non-user message")
                prompt_seen = True
            if message_id not in observed:
                observed.add(message_id)
                if info.get("role") == "assistant" and info.get("parentID") == prompt_id:
                    context.emit(HeadlessEventKind.ASSISTANT_MESSAGE, message="message")
        if prompt_seen:
            bound = [
                message["info"]
                for message in messages
                if message["info"].get("role") == "assistant"
                and message["info"].get("parentID") == prompt_id
            ]
            completed_bound = []
            for info in bound:
                if info.get("error") is not None:
                    raise ValueError("OpenCode assistant message failed")
                finish = info.get("finish")
                completed = info.get("time")
                if finish is not None and finish != "tool-calls":
                    if finish != "stop" or not (
                        isinstance(completed, dict)
                        and isinstance(completed.get("completed"), (int, float))
                    ):
                        raise ValueError("OpenCode assistant message ended unsuccessfully")
                    completed_bound.append(_required_text(info, "id"))
            if len(completed_bound) > 1:
                raise ValueError("OpenCode completed multiple messages for one prompt")
            if completed_bound:
                message_id = completed_bound[0]
                exact = server.request(
                    "GET", session_path + "/message/" + quote(message_id, safe="")
                )
                exact_info = exact.get("info") if isinstance(exact, dict) else None
                if not isinstance(exact_info, dict) or (
                    exact_info.get("sessionID") != session_id
                    or exact_info.get("id") != message_id
                    or exact_info.get("role") != "assistant"
                    or exact_info.get("parentID") != prompt_id
                ):
                    raise ValueError("OpenCode completed message binding changed")
                parts = exact.get("parts")
                if not isinstance(parts, list):
                    raise ValueError("OpenCode completed message omitted parts")
                texts = [
                    part["text"]
                    for part in parts
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ]
                text = "\n".join(texts)
                if not text.strip():
                    raise ValueError("OpenCode completed message was blank")
                return session_id, message_id, text, exact_info.get("tokens")
        context.wait(0.1)


def _validated_effort_model(
    server: OpenCodeServer,
    request: PlanSessionRequest | HeadlessSessionRequest,
    capability: PlanModeCapability,
    *,
    plan_agent: bool = True,
) -> dict[str, str]:
    """Pin the native selected model and require its advertised effort variant."""
    assert request.effort is not None
    selected: dict[str, str] | None = None
    model_name = request.model
    if model_name is None:
        agent_model = None
        if plan_agent:
            agents = server.request("GET", "/agent")
            if not isinstance(agents, list) or any(not isinstance(agent, dict) for agent in agents):
                raise ValueError("OpenCode returned malformed native agents")
            planning = [agent for agent in agents if agent.get("name") == "plan"]
            if len(planning) != 1:
                raise ValueError("OpenCode did not advertise one native plan agent")
            agent_model = planning[0].get("model")
        if agent_model is not None:
            selected = {
                "providerID": _required_text(agent_model, "providerID"),
                "modelID": _required_text(agent_model, "modelID"),
            }
        else:
            config = server.request("GET", "/config")
            model_name = config.get("model") if isinstance(config, dict) else None
            if not isinstance(model_name, str) or not model_name.strip():
                raise PlanSessionUnsupportedError(
                    "OpenCode effort validation requires an explicit model or a configured "
                    "native default model. Supply model='provider/model'; Crossby will "
                    "not guess from private recent-model state.",
                    tool_id=AIToolID.OPENCODE,
                    capability=capability,
                )
    if selected is None:
        assert model_name is not None
        provider, separator, model = model_name.partition("/")
        if not separator or not provider.strip() or not model.strip():
            raise ValueError("OpenCode configured model is not a provider/model identifier")
        selected = {"providerID": provider, "modelID": model}

    advertised = server.request("GET", "/provider")
    providers = advertised.get("all") if isinstance(advertised, dict) else None
    connected = advertised.get("connected") if isinstance(advertised, dict) else None
    if (
        not isinstance(providers, list)
        or any(not isinstance(provider, dict) for provider in providers)
        or not isinstance(connected, list)
        or any(not isinstance(provider, str) for provider in connected)
    ):
        raise ValueError("OpenCode returned malformed native providers")
    matches = [provider for provider in providers if provider.get("id") == selected["providerID"]]
    if len(matches) > 1:
        raise ValueError("OpenCode advertised duplicate native provider IDs")
    models = matches[0].get("models") if matches else None
    metadata = models.get(selected["modelID"]) if isinstance(models, dict) else None
    variants = metadata.get("variants") if isinstance(metadata, dict) else None
    variant = variants.get(request.effort.value) if isinstance(variants, dict) else None
    if (
        selected["providerID"] not in connected
        or not isinstance(variant, dict)
        or variant.get("disabled", False) is not False
    ):
        raise PlanSessionUnsupportedError(
            f"OpenCode model {selected['providerID']}/{selected['modelID']} does not advertise "
            f"an available effort={request.effort.value!r} variant. Choose an advertised "
            "model/effort combination or omit effort.",
            tool_id=AIToolID.OPENCODE,
            capability=capability,
        )
    return selected


def _required_text(value: Any, field: str) -> str:
    result = value.get(field) if isinstance(value, dict) else None
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"OpenCode omitted a non-blank {field}")
    return result


def _bound_requests(value: Any, session_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("OpenCode returned malformed pending requests")
    # The API lists pending requests, never artifacts. Service only the fresh
    # session returned by POST /session; other sessions cannot supply its plan.
    bound = [item for item in value if _required_text(item, "sessionID") == session_id]
    request_ids = [_required_text(item, "id") for item in bound]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("OpenCode returned duplicate pending request IDs")
    return bound


def _prompt_bound_requests(
    server: OpenCodeServer, value: Any, session_id: str, prompt_id: str
) -> list[dict[str, Any]]:
    """Select pending tool calls whose assistant message belongs to this prompt."""
    bound = []
    for request in _bound_requests(value, session_id):
        tool = request.get("tool")
        message_id = _required_text(tool, "messageID")
        call_id = _required_text(tool, "callID")
        message = server.request(
            "GET", f"/session/{quote(session_id, safe='')}/message/{quote(message_id, safe='')}"
        )
        info = message.get("info") if isinstance(message, dict) else None
        if not isinstance(info, dict) or (
            info.get("id") != message_id or info.get("sessionID") != session_id
        ):
            raise ValueError("OpenCode pending request has an unbound tool message")
        if info.get("role") != "assistant" or info.get("parentID") != prompt_id:
            continue
        parts = message.get("parts")
        if (
            not isinstance(parts, list)
            or sum(
                isinstance(part, dict)
                and part.get("type") == "tool"
                and part.get("callID") == call_id
                for part in parts
            )
            != 1
        ):
            raise ValueError("OpenCode pending request has no matching tool call")
        bound.append(request)
    return bound


def _shutdown_socket(transport: socket.socket) -> None:
    """Interrupt a header parser that reached the request-wide deadline."""
    with suppress(OSError):
        transport.shutdown(socket.SHUT_RDWR)


def _answer_question(
    server: OpenCodeServer,
    request: dict[str, Any],
    session_id: str,
    handler: PlanInteractionHandler | None,
    capability: PlanModeCapability,
) -> None:
    request_id = _required_text(request, "id")
    questions = request.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("OpenCode emitted no native questions")
    tool = request.get("tool")
    tool_name = "question"
    artifact_id: str | None = None
    if tool is not None:
        message_id = _required_text(tool, "messageID")
        artifact_id = _required_text(tool, "callID")
        message = server.request(
            "GET", f"/session/{quote(session_id, safe='')}/message/{quote(message_id, safe='')}"
        )
        info = message.get("info") if isinstance(message, dict) else None
        if not isinstance(info, dict) or info.get("sessionID") != session_id:
            raise PlanBindingMismatchError(
                "OpenCode question tool belonged to a different session.",
                tool_id=AIToolID.OPENCODE,
                capability=capability,
                session_id=session_id,
            )
        parts = message.get("parts")
        matches = (
            [part for part in parts if isinstance(part, dict) and part.get("callID") == artifact_id]
            if isinstance(parts, list)
            else []
        )
        if len(matches) != 1:
            raise ValueError("OpenCode question omitted its originating tool call")
        tool_name = _required_text(matches[0], "tool")
    answers: list[list[str]] = []
    for index, question in enumerate(questions):
        prompt = _required_text(question, "question")
        multiple = question.get("multiple", False)
        custom = question.get("custom", True)
        if not isinstance(multiple, bool) or not isinstance(custom, bool):
            raise PlanArtifactMalformedError(
                "OpenCode question contained a non-boolean multiple or custom field.",
                tool_id=AIToolID.OPENCODE,
                capability=capability,
                session_id=session_id,
            )
        options = parse_plan_question_options(question.get("options"))
        final_approval = tool_name == "plan_exit"
        interaction = PlanInteraction(
            kind=(
                PlanInteractionKind.PLAN_APPROVAL
                if final_approval
                else PlanInteractionKind.QUESTION
            ),
            question_id=request_id if len(questions) == 1 else f"{request_id}:{index}",
            prompt=prompt,
            options=tuple(option for option in options if option.option_id == "No")
            if final_approval
            else options,
            allow_multiple=multiple,
            allow_other=custom if not final_approval else False,
            session_id=session_id,
            artifact_id=artifact_id,
            request_id=request_id,
        )
        if handler is None:
            raise _interaction_required(interaction, capability)
        response = handler(interaction)
        if final_approval:
            if response.outcome is PlanInteractionOutcome.APPROVED:
                raise _interaction_required(interaction, capability)
            if response.outcome is PlanInteractionOutcome.ANSWERED:
                try:
                    if validate_plan_option_selection(interaction, response) != ("No",):
                        raise ValueError("implementation approval is not a planning answer")
                except ValueError:
                    raise _interaction_required(interaction, capability) from None
            server.request("POST", f"/question/{quote(request_id, safe='')}/reject")
            return
        if response.outcome is not PlanInteractionOutcome.ANSWERED:
            raise _interaction_required(interaction, capability)
        try:
            selected = validate_plan_option_selection(interaction, response)
            custom_answer = response.answer if response.answer and response.answer.strip() else None
            if custom_answer and not custom:
                raise ValueError("native question does not allow a custom answer")
            if selected and custom_answer and not multiple:
                raise ValueError("single-select native question requires exactly one answer")
            labels = {option.option_id: option.label for option in options}
            answer_values = [labels[option_id] for option_id in selected]
            if custom_answer:
                answer_values.append(custom_answer)
            if not answer_values:
                raise ValueError("native question was not answered")
            answers.append(answer_values)
        except ValueError:
            raise _interaction_required(interaction, capability) from None
    server.request("POST", f"/question/{quote(request_id, safe='')}/reply", {"answers": answers})


def _answer_permission(
    server: OpenCodeServer,
    request: dict[str, Any],
    session_id: str,
    handler: PlanInteractionHandler | None,
    capability: PlanModeCapability,
    *,
    execution_dir: Path | None = None,
) -> None:
    request_id = _required_text(request, "id")
    permission = _required_text(request, "permission")
    patterns = request.get("patterns")
    if patterns is None:
        patterns = []
    if not isinstance(patterns, list) or any(
        not isinstance(pattern, str) or not pattern.strip() for pattern in patterns
    ):
        raise ValueError("OpenCode permission request contained malformed native patterns")
    target_kind = {
        "bash": PlanPermissionTargetKind.COMMAND_PATTERN,
        "shell": PlanPermissionTargetKind.COMMAND_PATTERN,
        "read": PlanPermissionTargetKind.FILESYSTEM_READ,
        "edit": PlanPermissionTargetKind.FILESYSTEM_WRITE,
        "webfetch": PlanPermissionTargetKind.NETWORK_HOST,
        "websearch": PlanPermissionTargetKind.NETWORK_HOST,
    }.get(permission, PlanPermissionTargetKind.RESOURCE)
    operation_kind = {
        "bash": PlanOperationKind.COMMAND,
        "shell": PlanOperationKind.COMMAND,
        "edit": PlanOperationKind.FILE_CHANGE,
        "webfetch": PlanOperationKind.NETWORK,
        "websearch": PlanOperationKind.NETWORK,
    }.get(permission, PlanOperationKind.OTHER)
    tool = request.get("tool")
    native_binding_ids = [PlanNativeBindingID(name="request_id", value=request_id)]
    if isinstance(tool, dict):
        for field, name in (("messageID", "message_id"), ("callID", "call_id")):
            value = tool.get(field)
            if isinstance(value, str) and value.strip():
                native_binding_ids.append(PlanNativeBindingID(name=name, value=value))
    interaction = PlanInteraction(
        kind=PlanInteractionKind.PERMISSION,
        question_id=request_id,
        prompt=f"OpenCode requests {permission}: {patterns}",
        options=(
            PlanQuestionOption(option_id="once", label="Approve once"),
            PlanQuestionOption(option_id="reject", label="Deny"),
        ),
        session_id=session_id,
        request_id=request_id,
        operation=PlanOperation(
            kind=operation_kind,
            execution_dir=execution_dir.resolve() if execution_dir is not None else None,
            permission_targets=tuple(
                PlanPermissionTarget(kind=target_kind, value=pattern) for pattern in patterns
            ),
            native_binding_ids=tuple(native_binding_ids),
        ),
    )
    if handler is None:
        raise _interaction_required(interaction, capability)
    response = handler(interaction)
    if (
        response.outcome
        in {
            PlanInteractionOutcome.APPROVED,
            PlanInteractionOutcome.ANSWERED,
        }
        and response.answer is not None
    ):
        raise _interaction_required(interaction, capability)
    if response.outcome is PlanInteractionOutcome.APPROVED:
        reply = "once"
    elif response.outcome is PlanInteractionOutcome.ANSWERED:
        try:
            selected = validate_plan_option_selection(interaction, response)
            if len(selected) != 1:
                raise ValueError("permission requires one native option")
            reply = selected[0]
        except ValueError:
            raise _interaction_required(interaction, capability) from None
    else:
        reply = "reject"
    server.request("POST", f"/permission/{quote(request_id, safe='')}/reply", {"reply": reply})


def _interaction_required(
    interaction: PlanInteraction, capability: PlanModeCapability
) -> PlanInteractionRequiredError:
    return PlanInteractionRequiredError(
        "OpenCode requires a valid native answer; plan collection cannot authorize implementation.",
        interaction=interaction,
        tool_id=AIToolID.OPENCODE,
        capability=capability,
    )
