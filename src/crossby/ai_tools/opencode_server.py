"""Run OpenCode planning through its native HTTP question and session APIs."""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import secrets
import time
from typing import Any
from urllib.parse import quote, urlencode

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
from crossby.ai_tools.plan_process import JsonRpcProcess
from crossby.models.ai import (
    AIToolID,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanModeCapability,
    PlanQuestionOption,
    PlanSessionRequest,
)

_HTTP_BODY_LIMIT = 8 * 1024 * 1024


class OpenCodeServer:
    """A run-owned, authenticated loopback server with a shared deadline."""

    def __init__(self, request: PlanSessionRequest, deadline: float) -> None:
        self.deadline = deadline
        self.directory = request.working_dir
        self.port: int | None = None
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

    def request(self, method: str, path: str, payload: Any = None) -> Any:
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
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                raise OSError(f"OpenCode native API returned HTTP {response.status}")
            content = response.read(_HTTP_BODY_LIMIT + 1)
            if len(content) > _HTTP_BODY_LIMIT:
                raise ValueError("OpenCode native API exceeded the response size limit")
            self.remaining()
            return json.loads(content) if content else None
        finally:
            connection.close()

    def close(self) -> None:
        self.process.close()


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
        session = server.request("POST", "/session", {})
        session_id = _required_text(session, "id")
        session_path = "/session/" + quote(session_id, safe="")
        server.request("POST", session_path + "/prompt_async", prompt)
        while True:
            for question in _bound_requests(server.request("GET", "/question"), session_id):
                _answer_question(server, question, session_id, handler, capability)
            for permission in _bound_requests(server.request("GET", "/permission"), session_id):
                _answer_permission(server, permission, session_id, handler, capability)
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
    return [item for item in value if _required_text(item, "sessionID") == session_id]


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
            session_id=session_id,
            artifact_id=artifact_id,
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
            if selected:
                labels = {option.option_id: option.label for option in options}
                answers.append([labels[option_id] for option_id in selected])
            elif custom and response.answer and response.answer.strip():
                answers.append([response.answer])
            else:
                raise ValueError("native question was not answered")
        except ValueError:
            raise _interaction_required(interaction, capability) from None
    server.request("POST", f"/question/{quote(request_id, safe='')}/reply", {"answers": answers})


def _answer_permission(
    server: OpenCodeServer,
    request: dict[str, Any],
    session_id: str,
    handler: PlanInteractionHandler | None,
    capability: PlanModeCapability,
) -> None:
    request_id = _required_text(request, "id")
    interaction = PlanInteraction(
        kind=PlanInteractionKind.PERMISSION,
        question_id=request_id,
        prompt=f"OpenCode requests {_required_text(request, 'permission')}: "
        f"{request.get('patterns', [])}",
        options=(
            PlanQuestionOption(option_id="once", label="Approve once"),
            PlanQuestionOption(option_id="reject", label="Deny"),
        ),
        session_id=session_id,
    )
    if handler is None:
        raise _interaction_required(interaction, capability)
    response = handler(interaction)
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
