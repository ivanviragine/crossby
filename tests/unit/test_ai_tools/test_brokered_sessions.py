"""Deterministic ordinary-session coverage for the verified native bindings."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

from crossby.ai_tools import AbstractAITool
from crossby.ai_tools.headless import HeadlessRequestError, HeadlessTransportError
from crossby.ai_tools.headless_cli import HeadlessCommandOutput
from crossby.models.ai import (
    AIToolID,
    HeadlessCapability,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessModeCapability,
    HeadlessNativeOutput,
    HeadlessNativeTransport,
    HeadlessPromptTransport,
    HeadlessSessionRequest,
    HeadlessTerminalStatus,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanQuestionOption,
    SessionInteraction,
    SessionInteractionResponse,
)
from crossby.utils.versioning import BinaryVersion


class _FakeProcess:
    def poll(self) -> int:
        return 0

    def wait(self, *, timeout: float) -> int:
        assert timeout >= 0
        return 0


class _FakeRpc:
    frames: ClassVar[list[dict[str, Any]]] = []
    instances: ClassVar[list[_FakeRpc]] = []

    def __init__(self, command: list[str], *, cwd: Path, **_kwargs: Any) -> None:
        self.command = command
        self.cwd = cwd
        self.messages = list(self.frames)
        self.sent: list[tuple[str, Any]] = []
        self.stderr = ""
        self.returncode: int | None = None
        self._proc = _FakeProcess()
        self.instances.append(self)

    def request(self, request_id: int, method: str, params: dict[str, Any]) -> None:
        self.sent.append((method, params))

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.sent.append((method, params))

    def respond(self, request_id: object, result: Any) -> None:
        self.sent.append(("respond", {"id": request_id, "result": result}))

    def read(self, *, timeout: float) -> dict[str, Any]:
        assert timeout > 0
        if not self.messages:
            raise EOFError("fixture exhausted")
        return self.messages.pop(0)

    def close(self) -> int:
        self.returncode = 0
        return 0

    def close_input(self) -> None:
        return None

    def join_workers(self, *, timeout: float) -> None:
        return None

    def send_cleanup(self, payload: dict[str, Any]) -> None:
        self.sent.append(("abort", payload))


@pytest.fixture(autouse=True)
def _version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda _binary, **_kwargs: BinaryVersion((9999, 0, 0), "fixture 9999.0.0"),
    )


def _request(tmp_path: Path, **updates: Any) -> HeadlessSessionRequest:
    values: dict[str, Any] = {
        "prompt": "Implement the ordinary task",
        "working_dir": tmp_path,
        "interaction_mode": HeadlessInteractionMode.BROKERED,
    }
    values.update(updates)
    return HeadlessSessionRequest(**values)


def _install_rpc(monkeypatch: pytest.MonkeyPatch, frames: list[dict[str, Any]]) -> None:
    _FakeRpc.frames = frames
    _FakeRpc.instances = []
    monkeypatch.setattr("crossby.ai_tools.session_protocol.HeaderlessJsonRpcProcess", _FakeRpc)
    monkeypatch.setattr("crossby.ai_tools.session_protocol.JsonRpcProcess", _FakeRpc)


@pytest.mark.parametrize("prompt", ["Implement the ordinary task", "a", "Codex", "question"])
def test_codex_question_uses_exact_thread_turn_request_and_normal_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            {
                "id": 71,
                "method": "item/tool/requestUserInput",
                "params": {
                    "threadId": "thr-exact",
                    "turnId": "turn-exact",
                    "itemId": "item-exact",
                    "questions": [
                        {
                            "id": "scope",
                            "question": "Choose scope",
                            "options": [{"id": "api", "label": "API"}],
                        }
                    ],
                },
            },
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turnId": "turn-exact",
                    "item": {"type": "agentMessage", "id": "msg-exact", "text": "Completed."},
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turn": {"id": "turn-exact", "status": "completed"},
                },
            },
            {"id": 4, "result": {}},
        ],
    )
    seen: list[SessionInteraction] = []

    def answer(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="api")

    result = AbstractAITool.get(AIToolID.CODEX).run_headless_session(
        _request(tmp_path, prompt=prompt), answer
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert (result.thread_id, result.turn_id, result.final_text) == (
        "thr-exact",
        "turn-exact",
        "Completed.",
    )
    assert seen[0].request_id == "71"
    assert seen[0].kind is PlanInteractionKind.QUESTION
    sent = _FakeRpc.instances[0].sent
    assert any(method == "thread/start" and params["ephemeral"] is False for method, params in sent)
    assert ("respond", {"id": 71, "result": {"answers": {"scope": {"answers": ["API"]}}}}) in sent
    assert not any("collaborationMode" in params for _, params in sent)
    assert any(event.kind is HeadlessEventKind.ASSISTANT_MESSAGE for event in result.events)


@pytest.mark.parametrize("prompt", ["a", "delta", "completed"])
def test_codex_deltas_refresh_progress_and_completed_messages_accumulate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            *[
                {
                    "method": "item/agentMessage/delta",
                    "params": {"threadId": "thr-exact", "turnId": "turn-exact", "delta": "x"},
                }
                for _ in range(4200)
            ],
            *[
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thr-exact",
                        "turnId": "turn-exact",
                        "item": {"type": "agentMessage", "id": f"msg-{i}", "text": value},
                    },
                }
                for i, value in enumerate(("First.", "Second."))
            ],
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turn": {"id": "turn-exact", "status": "completed"},
                },
            },
            {"id": 4, "result": {}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CODEX).run_headless_session(
        _request(tmp_path, prompt=prompt),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "First.\nSecond."
    assert sum(event.kind is HeadlessEventKind.ASSISTANT_MESSAGE for event in result.events) == 512


def test_codex_rejects_aggregate_assistant_text_before_retaining_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crossby.ai_tools.codex._CODEX_MESSAGE_LIMIT", 10)
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            *[
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thr-exact",
                        "turnId": "turn-exact",
                        "item": {"type": "agentMessage", "id": f"msg-{i}", "text": text},
                    },
                }
                for i, text in enumerate(("ááá", "éé"))
            ],
        ],
    )
    with pytest.raises(HeadlessTransportError, match="assistant messages exceeded"):
        AbstractAITool.get(AIToolID.CODEX).run_headless_session(
            _request(tmp_path),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )


@pytest.mark.parametrize("over_limit", [False, True])
@pytest.mark.parametrize("method", ["item/agentMessage/delta", "unknown/notification"])
def test_codex_pending_notifications_share_byte_budget_across_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, over_limit: bool, method: str
) -> None:
    notifications = [
        {
            "method": method,
            "params": {"threadId": "thr-exact", "turnId": "turn-exact", "delta": "é" * 100},
        }
        for _ in range(3)
    ]
    byte_count = sum(
        len(json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        for frame in notifications
    )
    monkeypatch.setattr(
        "crossby.ai_tools.codex._CODEX_NOTIFICATION_BYTES_LIMIT", byte_count - int(over_limit)
    )
    _install_rpc(
        monkeypatch,
        [
            notifications[0],
            {"id": 1, "result": {}},
            notifications[1],
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            notifications[2],
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turnId": "turn-exact",
                    "item": {"type": "agentMessage", "id": "msg-exact", "text": "Done."},
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turn": {"id": "turn-exact", "status": "completed"},
                },
            },
            {"id": 4, "result": {}},
        ],
    )

    def run() -> Any:
        return AbstractAITool.get(AIToolID.CODEX).run_headless_session(
            _request(tmp_path),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )

    if over_limit:
        with pytest.raises(HeadlessTransportError, match="pending notifications exceeded"):
            run()
        # Reject while waiting for turn/start, before reading its response.
        assert _FakeRpc.instances[0].messages[0]["id"] == 3
    else:
        result = run()
        assert result.status is HeadlessTerminalStatus.SUCCEEDED
        assert result.final_text == "Done."


def test_codex_item_lifecycle_events_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            *[
                {
                    "method": "item/started",
                    "params": {
                        "threadId": "thr-exact",
                        "turnId": "turn-exact",
                        "item": {"type": "commandExecution"},
                    },
                }
                for _ in range(4200)
            ],
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turnId": "turn-exact",
                    "item": {"type": "agentMessage", "id": "msg-exact", "text": "Done."},
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turn": {"id": "turn-exact", "status": "completed"},
                },
            },
            {"id": 4, "result": {}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CODEX).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "Done."
    assert sum(event.kind is HeadlessEventKind.TOOL_STARTED for event in result.events) == 512


@pytest.mark.parametrize(
    "method",
    [
        "item/plan/delta",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/textDelta",
        "item/commandExecution/outputDelta",
        "item/fileChange/outputDelta",
        "item/mcpToolCall/progress",
    ],
)
def test_codex_bound_native_streams_refresh_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            {
                "method": method,
                "params": {"threadId": "thr-exact", "turnId": "turn-exact", "delta": "x"},
            },
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turnId": "turn-exact",
                    "item": {"type": "agentMessage", "id": "msg-exact", "text": "Done."},
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turn": {"id": "turn-exact", "status": "completed"},
                },
            },
            {"id": 4, "result": {}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CODEX).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert any(event.kind is HeadlessEventKind.PROGRESS for event in result.events)


@pytest.mark.parametrize("binding", [{"threadId": "thr-foreign"}, {"turnId": "turn-foreign"}])
def test_codex_native_stream_rejects_foreign_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: dict[str, str]
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            {
                "method": "item/reasoning/textDelta",
                "params": {"threadId": "thr-exact", "turnId": "turn-exact", **binding},
            },
        ],
    )
    with pytest.raises(HeadlessTransportError):
        AbstractAITool.get(AIToolID.CODEX).run_headless_session(
            _request(tmp_path),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )


def _cursor_state(
    model: str, effort: str, *, thinking: str | None = None, fast: str | None = None
) -> dict[str, Any]:
    options: list[dict[str, Any]] = [
        {
            "id": "model",
            "currentValue": model,
            "options": [{"value": "other"}, {"value": "claude-opus-5-5"}],
        },
        {
            "id": "effort",
            "category": "thought_level",
            "currentValue": effort,
            "options": [{"value": "low"}, {"value": "high"}],
        },
    ]
    if thinking is not None:
        options.append(
            {
                "id": "thinking",
                "currentValue": thinking,
                "options": [{"value": "false"}, {"value": "true"}],
            }
        )
    if fast is not None:
        options.append(
            {
                "id": "fast",
                "currentValue": fast,
                "options": [{"value": "false"}, {"value": "true"}],
            }
        )
    return {"sessionId": "ses-exact", "configOptions": options}


def test_cursor_brokered_model_encoded_effort_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": _cursor_state("other", "low")},
            {"id": 4, "result": _cursor_state("claude-opus-5-5", "low")},
            {"id": 5, "result": _cursor_state("claude-opus-5-5", "high")},
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-exact",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-exact",
                        "content": {"type": "text", "text": "Completed."},
                    },
                },
            },
            {"id": 6, "result": {"stopReason": "end_turn"}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path, model="claude-opus-5-5-high"),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert (
        "session/set_config_option",
        {"sessionId": "ses-exact", "configId": "effort", "value": "high"},
    ) in _FakeRpc.instances[0].sent


@pytest.mark.parametrize(
    ("final_model", "final_effort"),
    [("other", "high"), ("claude-opus-5-5", "low")],
)
def test_cursor_brokered_rejects_model_or_effort_reset_by_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, final_model: str, final_effort: str
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": _cursor_state("other", "low", thinking="true")},
            {"id": 4, "result": _cursor_state("claude-opus-5-5", "low", thinking="true")},
            {"id": 5, "result": _cursor_state("claude-opus-5-5", "high", thinking="true")},
            {"id": 6, "result": _cursor_state(final_model, final_effort, thinking="false")},
        ],
    )
    with pytest.raises(HeadlessTransportError):
        AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
            _request(tmp_path, model="claude-opus-5-5-high"),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )
    assert not any(method == "session/prompt" for method, _ in _FakeRpc.instances[0].sent)


def test_cursor_brokered_rejects_earlier_variant_reset_by_later_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": _cursor_state("other", "low", thinking="true", fast="true")},
            {
                "id": 4,
                "result": _cursor_state("claude-opus-5-5", "low", thinking="true", fast="true"),
            },
            {
                "id": 5,
                "result": _cursor_state("claude-opus-5-5", "high", thinking="true", fast="true"),
            },
            {
                "id": 6,
                "result": _cursor_state("claude-opus-5-5", "high", thinking="false", fast="true"),
            },
            {
                "id": 7,
                "result": _cursor_state("claude-opus-5-5", "high", thinking="true", fast="false"),
            },
        ],
    )
    with pytest.raises(HeadlessTransportError):
        AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
            _request(tmp_path, model="claude-opus-5-5-high"),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )
    assert not any(method == "session/prompt" for method, _ in _FakeRpc.instances[0].sent)


@pytest.mark.parametrize(
    "prompt", ["Implement the ordinary task", "a", "Cursor", "chunk", "question"]
)
def test_cursor_question_preserves_native_option_id_and_omits_plan_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            {
                "id": 90,
                "method": "cursor/ask_question",
                "params": {
                    "toolCallId": "call-exact",
                    "questions": [
                        {
                            "id": "scope",
                            "prompt": "Choose scope",
                            "options": [{"id": "api", "label": "API"}],
                        }
                    ],
                },
            },
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-exact",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-exact",
                        "content": {"type": "text", "text": "Completed."},
                    },
                },
            },
            {"id": 4, "result": {"stopReason": "end_turn"}},
        ],
    )
    seen: list[SessionInteraction] = []

    def answer(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.ANSWERED, option_id="api")

    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path, prompt=prompt), answer
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.session_id == "ses-exact"
    assert result.final_text == "Completed."
    assert seen[0].request_id == "90"
    sent = _FakeRpc.instances[0].sent
    assert not any(method == "session/set_mode" for method, _ in sent)
    assert (
        "respond",
        {
            "id": 90,
            "result": {
                "outcome": {
                    "outcome": "answered",
                    "answers": [{"questionId": "scope", "selectedOptionIds": ["api"]}],
                }
            },
        },
    ) in sent


def test_cursor_many_message_chunks_keep_response_without_exhausting_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [
        {
            "method": "session/update",
            "params": {
                "sessionId": "ses-exact",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "messageId": "msg-exact",
                    "content": {"type": "text", "text": "x"},
                },
            },
        }
        for _ in range(4200)
    ]
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            *chunks,
            {"id": 4, "result": {"stopReason": "end_turn"}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "x" * 4200
    assert sum(event.kind is HeadlessEventKind.ASSISTANT_MESSAGE for event in result.events) == 512


def test_cursor_brokered_preserves_all_assistant_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            *[
                {
                    "method": "session/update",
                    "params": {
                        "sessionId": "ses-exact",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": message_id,
                            "content": {"type": "text", "text": text},
                        },
                    },
                }
                for message_id, text in (("first", "Before tool."), ("second", "After tool."))
            ],
            {"id": 4, "result": {"stopReason": "end_turn"}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.final_text == "Before tool.\nAfter tool."


def test_cursor_context_usage_does_not_become_turn_token_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-exact",
                    "update": {"sessionUpdate": "usage_update", "used": 53000, "size": 200000},
                },
            },
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-exact",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-exact",
                        "content": {"type": "text", "text": "Completed."},
                    },
                },
            },
            {"id": 4, "result": {"stopReason": "end_turn"}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.usage is None
    assert any(event.kind is HeadlessEventKind.USAGE for event in result.events)


def test_cursor_brokered_bounds_message_bytes_before_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crossby.ai_tools.cursor._CURSOR_MESSAGE_LIMIT", 4)
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            *[
                {
                    "method": "session/update",
                    "params": {
                        "sessionId": "ses-exact",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "messageId": "msg-exact",
                            "content": {"type": "text", "text": "é"},
                        },
                    },
                }
                for _ in range(3)
            ],
        ],
    )
    with pytest.raises(HeadlessTransportError, match="size limit"):
        AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
            _request(tmp_path),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )


@pytest.mark.parametrize("text", [None, " \t "])
def test_cursor_brokered_empty_end_turn_is_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str | None
) -> None:
    frames: list[dict[str, Any]] = [
        {"id": 1, "result": {"authMethods": []}},
        {"id": 3, "result": {"sessionId": "ses-exact"}},
    ]
    if text is not None:
        frames.append(
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-exact",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-exact",
                        "content": {"type": "text", "text": text},
                    },
                },
            }
        )
    frames.append({"id": 4, "result": {"stopReason": "end_turn"}})
    _install_rpc(monkeypatch, frames)
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.events[-1].terminal_status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert any("without assistant text" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("model", "error"),
    [
        ("claude-opus-5-5[foo=1]", "invalid effort overrides"),
        ("claude-opus-5-5-high[effort=low]", "conflicting effort encodings"),
    ],
)
def test_cursor_brokered_rejects_bad_model_effort_before_session_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, error: str
) -> None:
    _install_rpc(monkeypatch, [])
    with pytest.raises(HeadlessRequestError, match=error):
        AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
            _request(tmp_path, model=model),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )
    assert _FakeRpc.instances == []


def test_cursor_many_non_text_updates_and_tools_remain_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    updates = [
        {
            "method": "session/update",
            "params": {
                "sessionId": "ses-exact",
                "update": (
                    {"sessionUpdate": "tool_call_update", "toolCallId": f"tool-{i}"}
                    if i % 2
                    else {"sessionUpdate": "agent_thought_chunk"}
                ),
            },
        }
        for i in range(4400)
    ]
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            *updates,
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-exact",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-exact",
                        "content": {"type": "text", "text": "Completed."},
                    },
                },
            },
            {"id": 4, "result": {"stopReason": "end_turn"}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "Completed."
    assert sum(event.kind is HeadlessEventKind.TOOL_STARTED for event in result.events) == 512
    assert len(result.events) < 1100


def test_cursor_brokered_accepts_cursor_permission_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            {
                "id": 90,
                "method": "cursor/request_permission",
                "params": {
                    "sessionId": "ses-exact",
                    "toolCall": {"toolCallId": "tool-exact", "title": "Run a command?"},
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once"},
                        {"optionId": "deny-once", "name": "Deny once"},
                    ],
                },
            },
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-exact",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-exact",
                        "content": {"type": "text", "text": "Completed."},
                    },
                },
            },
            {"id": 4, "result": {"stopReason": "end_turn"}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.APPROVED, option_id="allow-once"
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert (
        "respond",
        {"id": 90, "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}}},
    ) in _FakeRpc.instances[0].sent


@pytest.mark.parametrize("tool_id", [AIToolID.CODEX, AIToolID.CURSOR])
def test_brokered_native_failure_emits_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_id: AIToolID
) -> None:
    frames = (
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turn": {"id": "turn-exact", "status": "failed"},
                },
            },
        ]
        if tool_id is AIToolID.CODEX
        else [
            {"id": 1, "result": {"authMethods": []}},
            {"id": 3, "result": {"sessionId": "ses-exact"}},
            {"id": 4, "result": {"stopReason": "error"}},
        ]
    )
    _install_rpc(monkeypatch, frames)

    result = AbstractAITool.get(tool_id).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.events[-1].kind is HeadlessEventKind.TERMINAL
    assert result.events[-1].terminal_status is HeadlessTerminalStatus.FAILED


@pytest.mark.parametrize("prompt", ["Implement the ordinary task", "a", "OpenCode", "message"])
def test_opencode_ordinary_session_does_not_select_plan_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    class FakeServer:
        instances: ClassVar[list[FakeServer]] = []

        def __init__(self, _request: Any, _deadline: float, runtime: Any = None) -> None:
            self.calls: list[tuple[str, str, Any]] = []
            self.active_session_id: str | None = None
            self.prompt_id: str | None = None
            self._bound_message_ids: set[str] = set()
            self.instances.append(self)

        def start(self) -> None:
            return None

        def request(self, method: str, path: str, payload: Any = None) -> Any:
            self.calls.append((method, path, payload))
            if method == "POST" and path == "/session":
                return {"id": "ses-exact"}
            if path == "/session/ses-exact/prompt_async":
                self.prompt_id = payload["messageID"]
                return None
            if path in {"/question", "/permission"}:
                return []
            if path == "/session/ses-exact/message":
                return [
                    {"info": {"id": self.prompt_id, "sessionID": "ses-exact", "role": "user"}},
                    {
                        "info": {
                            "id": "msg-exact",
                            "sessionID": "ses-exact",
                            "role": "assistant",
                            "parentID": self.prompt_id,
                            "finish": "stop",
                            "time": {"completed": 1},
                        }
                    },
                ]
            if path == "/session/ses-exact/message/msg-exact":
                return {
                    "info": {
                        "id": "msg-exact",
                        "sessionID": "ses-exact",
                        "role": "assistant",
                        "parentID": self.prompt_id,
                    },
                    "parts": [{"type": "text", "text": "Completed."}],
                }
            return None

    monkeypatch.setattr("crossby.ai_tools.opencode_server.OpenCodeServer", FakeServer)
    result = AbstractAITool.get(AIToolID.OPENCODE).run_headless_session(
        _request(tmp_path, prompt=prompt),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "Completed."
    prompts = [
        payload
        for method, path, payload in FakeServer.instances[0].calls
        if method == "POST" and path.endswith("/prompt_async")
    ]
    assert len(prompts) == 1 and "agent" not in prompts[0]
    assert FakeServer.instances[0].active_session_id == "ses-exact"


def test_antigravity_continuation_reuses_exact_conversation_without_plan_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeOwner:
        def __init__(self, _context: Any) -> None:
            return None

    outputs = iter(
        [
            HeadlessCommandOutput(
                0,
                json.dumps(
                    {
                        "status": "WAITING_FOR_INPUT",
                        "conversation_id": "conv-exact",
                        "question": {
                            "id": "scope",
                            "prompt": "Choose scope",
                            "options": [{"id": "api", "label": "API"}],
                        },
                    }
                ),
                "",
            ),
            HeadlessCommandOutput(
                0,
                json.dumps(
                    {"status": "SUCCESS", "conversation_id": "conv-exact", "response": "Done."}
                ),
                "",
            ),
        ]
    )
    commands: list[list[str]] = []

    def run(_context: Any, *, argv: list[str], **_kwargs: Any) -> HeadlessCommandOutput:
        commands.append(argv)
        return next(outputs)

    monkeypatch.setattr("crossby.ai_tools.headless_cli.ResumableManagedCLIController", FakeOwner)
    monkeypatch.setattr("crossby.ai_tools.headless_cli.run_managed_command", run)
    result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON),
        lambda interaction: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_id="api"
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.conversation_id == "conv-exact"
    assert len(commands) == 2
    assert commands[1][1:5] == ["--conversation", "conv-exact", "--print", "API"]
    assert all("--mode" not in argv for argv in commands)


def test_codex_builtin_permission_grants_exact_profile_only_after_explicit_approval(
    tmp_path: Path,
) -> None:
    from crossby.ai_tools.codex import _answer_codex_builtin_permission

    rpc = _FakeRpc([], cwd=tmp_path)
    message = {
        "id": 91,
        "method": "item/permissions/requestApproval",
        "params": {
            "threadId": "thr-exact",
            "turnId": "turn-exact",
            "itemId": "item-exact",
            "cwd": str(tmp_path),
            "permissions": {"network": {"enabled": True}},
        },
    }
    seen: list[SessionInteraction] = []

    def approve(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        return SessionInteractionResponse(outcome=PlanInteractionOutcome.APPROVED)

    _answer_codex_builtin_permission(
        rpc, message, thread_id="thr-exact", turn_id="turn-exact", handler=approve
    )
    assert seen[0].request_id == "91"
    assert seen[0].kind is PlanInteractionKind.PERMISSION
    assert rpc.sent == [
        (
            "respond",
            {
                "id": 91,
                "result": {"permissions": {"network": {"enabled": True}}, "scope": "turn"},
            },
        )
    ]

    rpc.sent.clear()
    for outcome in (PlanInteractionOutcome.DENIED, PlanInteractionOutcome.SKIPPED):
        _answer_codex_builtin_permission(
            rpc,
            message,
            thread_id="thr-exact",
            turn_id="turn-exact",
            handler=lambda _interaction, outcome=outcome: SessionInteractionResponse(
                outcome=outcome
            ),
        )
    assert (
        rpc.sent
        == [
            ("respond", {"id": 91, "result": {"permissions": {}, "scope": "turn"}}),
        ]
        * 2
    )
    rpc.sent.clear()
    with pytest.raises(ValueError, match="refusal included answer fields"):
        _answer_codex_builtin_permission(
            rpc,
            message,
            thread_id="thr-exact",
            turn_id="turn-exact",
            handler=lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED, answer="approve"
            ),
        )
    assert rpc.sent == []
    _answer_codex_builtin_permission(
        rpc,
        message,
        thread_id="thr-exact",
        turn_id="turn-exact",
        handler=approve,
        deny_automatically=True,
    )
    assert len(seen) == 1
    assert rpc.sent == [("respond", {"id": 91, "result": {"permissions": {}, "scope": "turn"}})]


@pytest.mark.parametrize(
    "outcome", [None, PlanInteractionOutcome.DENIED, PlanInteractionOutcome.SKIPPED]
)
def test_codex_builtin_permission_denial_completes_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: PlanInteractionOutcome | None
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-exact"}}},
            {"id": 3, "result": {"turn": {"id": "turn-exact"}}},
            {
                "id": 91,
                "method": "item/permissions/requestApproval",
                "params": {
                    "threadId": "thr-exact",
                    "turnId": "turn-exact",
                    "itemId": "item-exact",
                    "cwd": str(tmp_path),
                    "permissions": {"network": {"enabled": True}},
                },
            },
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turnId": "turn-exact",
                    "item": {
                        "type": "agentMessage",
                        "id": "msg-exact",
                        "text": "Permission denied.",
                    },
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-exact",
                    "turn": {"id": "turn-exact", "status": "completed"},
                },
            },
            {"id": 4, "result": {}},
        ],
    )
    from crossby.models.ai import PlanApprovalPolicy

    seen: list[SessionInteraction] = []

    def deny(interaction: SessionInteraction) -> SessionInteractionResponse:
        if outcome is None:
            pytest.fail("approval_policy=never must not ask for a permission decision")
        seen.append(interaction)
        return SessionInteractionResponse(outcome=outcome)

    result = AbstractAITool.get(AIToolID.CODEX).run_headless_session(
        _request(
            tmp_path,
            approval_policy=(
                PlanApprovalPolicy.NEVER if outcome is None else PlanApprovalPolicy.ON_REQUEST
            ),
        ),
        deny,
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "Permission denied."
    assert len(seen) == (0 if outcome is None else 1)
    if outcome is not None:
        assert result.denials
    assert (
        "respond",
        {"id": 91, "result": {"permissions": {}, "scope": "turn"}},
    ) in _FakeRpc.instances[0].sent


@pytest.mark.parametrize(
    ("failure", "expected_native_status"),
    [
        ({"error": {"name": "ProviderAuthError"}}, "ProviderAuthError"),
        ({"finish": "length"}, "length"),
    ],
)
def test_opencode_assistant_failure_returns_failed_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: dict[str, Any],
    expected_native_status: str,
) -> None:
    class FakeServer:
        def __init__(self, _request: Any, _deadline: float, runtime: Any = None) -> None:
            self.active_session_id: str | None = None
            self.prompt_id: str | None = None
            self._bound_message_ids: set[str] = set()

        def start(self) -> None:
            return None

        def request(self, method: str, path: str, payload: Any = None) -> Any:
            if method == "POST" and path == "/session":
                return {"id": "ses-exact"}
            if path == "/session/ses-exact/prompt_async":
                self.prompt_id = payload["messageID"]
                return None
            if path in {"/question", "/permission"}:
                return []
            if path == "/session/ses-exact/message":
                return [
                    {"info": {"id": self.prompt_id, "sessionID": "ses-exact", "role": "user"}},
                    {
                        "info": {
                            "id": "msg-failed",
                            "sessionID": "ses-exact",
                            "role": "assistant",
                            "parentID": self.prompt_id,
                            **failure,
                        }
                    },
                ]
            raise AssertionError((method, path))

    monkeypatch.setattr("crossby.ai_tools.opencode_server.OpenCodeServer", FakeServer)
    result = AbstractAITool.get(AIToolID.OPENCODE).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.native_status == expected_native_status
    assert result.final_text is None
    assert result.events[-1].terminal_status is HeadlessTerminalStatus.FAILED
    assert result.session_id == "ses-exact"


def test_codex_mcp_form_requires_exact_schema_content(tmp_path: Path) -> None:
    from crossby.ai_tools.codex import _answer_codex_mcp_elicitation

    rpc = _FakeRpc([], cwd=tmp_path)
    message = {
        "id": 92,
        "method": "mcpServer/elicitation/request",
        "params": {
            "threadId": "thr-exact",
            "turnId": "turn-exact",
            "serverName": "fixture-server",
            "mode": "form",
            "message": "Choose a value",
            "requestedSchema": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": ["safe"]}},
                "required": ["choice"],
            },
        },
    }
    seen: list[SessionInteraction] = []

    def accept(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        return SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_id="accept",
            content={"choice": "safe"},
        )

    _answer_codex_mcp_elicitation(
        rpc, message, thread_id="thr-exact", turn_id="turn-exact", handler=accept
    )
    assert seen[0].form_schema == message["params"]["requestedSchema"]
    assert rpc.sent == [
        ("respond", {"id": 92, "result": {"action": "accept", "content": {"choice": "safe"}}})
    ]

    rpc.sent.clear()
    with pytest.raises(ValueError, match="schema validation"):
        _answer_codex_mcp_elicitation(
            rpc,
            message,
            thread_id="thr-exact",
            turn_id="turn-exact",
            handler=lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                option_id="accept",
                content={"choice": "unsafe"},
            ),
        )
    assert rpc.sent == []


@pytest.mark.parametrize(
    ("outcome", "action"),
    [
        (PlanInteractionOutcome.DENIED, "decline"),
        (PlanInteractionOutcome.SKIPPED, "decline"),
        (PlanInteractionOutcome.CANCELLED, "cancel"),
    ],
)
def test_codex_mcp_form_forwards_refusal(
    tmp_path: Path, outcome: PlanInteractionOutcome, action: str
) -> None:
    from crossby.ai_tools.codex import _answer_codex_mcp_elicitation

    rpc = _FakeRpc([], cwd=tmp_path)
    message = {
        "id": 92,
        "params": {
            "threadId": "thr-exact",
            "turnId": "turn-exact",
            "serverName": "fixture-server",
            "mode": "form",
            "message": "Choose a value",
            "requestedSchema": {"type": "object"},
        },
    }
    _answer_codex_mcp_elicitation(
        rpc,
        message,
        thread_id="thr-exact",
        turn_id="turn-exact",
        handler=lambda _interaction: SessionInteractionResponse(outcome=outcome),
    )
    assert rpc.sent == [("respond", {"id": 92, "result": {"action": action}})]


def test_codex_mcp_url_preserves_elicitation_id(tmp_path: Path) -> None:
    from crossby.ai_tools.codex import _answer_codex_mcp_elicitation

    rpc = _FakeRpc([], cwd=tmp_path)
    seen: list[SessionInteraction] = []
    message = {
        "id": 93,
        "params": {
            "threadId": "thr-exact",
            "turnId": "turn-exact",
            "serverName": "fixture-server",
            "mode": "url",
            "message": "Open the authorization URL",
            "elicitationId": "elicit-exact",
            "url": "https://example.test/authorize",
        },
    }

    def accept(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        return SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_id="accept"
        )

    _answer_codex_mcp_elicitation(
        rpc, message, thread_id="thr-exact", turn_id="turn-exact", handler=accept
    )
    assert seen[0].url == "https://example.test/authorize"
    assert any(
        binding.name == "elicitation_id" and binding.value == "elicit-exact"
        for binding in seen[0].operation.native_binding_ids
    )
    assert rpc.sent == [("respond", {"id": 93, "result": {"action": "accept"}})]


def test_opencode_ordinary_effort_uses_default_model_without_plan_agent(
    tmp_path: Path,
) -> None:
    from crossby.ai_tools.opencode_server import _validated_effort_model
    from crossby.models.ai import EffortLevel

    class FakeServer:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def request(self, _method: str, path: str) -> Any:
            self.paths.append(path)
            if path == "/config":
                return {"model": "anthropic/claude-sonnet-4"}
            if path == "/provider":
                return {
                    "all": [
                        {
                            "id": "anthropic",
                            "models": {"claude-sonnet-4": {"variants": {"high": {}}}},
                        }
                    ],
                    "connected": ["anthropic"],
                }
            raise AssertionError(f"unexpected native endpoint: {path}")

    server = FakeServer()
    result = _validated_effort_model(
        server,
        _request(tmp_path, effort=EffortLevel.HIGH),
        AbstractAITool.get(AIToolID.OPENCODE).capabilities().plan_mode,
        plan_agent=False,
    )
    assert result == {"providerID": "anthropic", "modelID": "claude-sonnet-4"}
    assert server.paths == ["/config", "/provider"]


def test_canonical_interaction_aliases_and_profile_coherence() -> None:
    from crossby.models.ai import PlanInteractionResponse, SessionQuestionOption

    assert PlanInteraction is SessionInteraction
    assert PlanInteractionResponse is SessionInteractionResponse
    assert PlanQuestionOption is SessionQuestionOption
    profile = HeadlessModeCapability(
        transport=HeadlessNativeTransport.APP_SERVER,
        prompt_transport=HeadlessPromptTransport.PROTOCOL,
        native_outputs=(HeadlessNativeOutput.TEXT,),
        version_requirement="verified test protocol",
        verified_version="1.0.0",
        limitation="fixture",
    )
    with pytest.raises(ValueError, match="brokered mode and its profile"):
        HeadlessCapability(
            transport=HeadlessNativeTransport.HEADLESS_CLI,
            prompt_transport=HeadlessPromptTransport.STDIN,
            native_outputs=(HeadlessNativeOutput.TEXT,),
            interaction_modes=(HeadlessInteractionMode.UNATTENDED,),
            verified_version="1.0.0",
            brokered=profile,
        )


def test_protocol_controller_blocks_reused_and_late_native_replies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.session_protocol import ProtocolSessionController

    _install_rpc(monkeypatch, [])

    class FakeContext:
        deadline = 9999999999.0

        def __init__(self) -> None:
            self.hooks: Any = None

        def remaining_seconds(self) -> float:
            return 10.0

        def register_cleanup(self, hooks: Any) -> None:
            self.hooks = hooks

        def checkpoint(self) -> None:
            return None

    context = FakeContext()
    controller = ProtocolSessionController(context, ["fixture"], cwd=tmp_path)
    controller.track_native_request({"id": 17, "method": "question"})
    with pytest.raises(ValueError, match="reused"):
        controller.track_native_request({"id": 17, "method": "question"})
    controller.respond_native(17, {"answer": "one"})
    with pytest.raises(ValueError, match="no longer live"):
        controller.respond_native(17, {"answer": "late"})
    controller.track_native_request({"id": 18, "method": "question"})
    controller._resolved = True
    with pytest.raises(ValueError, match="no longer live"):
        controller.respond_native(18, {"answer": "late"})
    assert _FakeRpc.instances[-1].sent == [("respond", {"id": 17, "result": {"answer": "one"}})]


@pytest.mark.parametrize("notify", [False, True])
def test_protocol_controller_uses_notification_for_cursor_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notify: bool
) -> None:
    from crossby.ai_tools.session_protocol import ProtocolSessionController

    _install_rpc(monkeypatch, [])

    class FakeContext:
        deadline = 9999999999.0

        def remaining_seconds(self) -> float:
            return 10.0

        def register_cleanup(self, hooks: Any) -> None:
            self.hooks = hooks

    context = FakeContext()
    controller = ProtocolSessionController(context, ["fixture"], cwd=tmp_path)
    controller.set_abort("session/cancel", {"sessionId": "ses-exact"}, notify=notify)
    context.hooks.native_abort(None)
    payload = _FakeRpc.instances[-1].sent[-1][1]
    assert payload == {
        **({} if notify else {"id": 999_999}),
        "method": "session/cancel",
        "params": {"sessionId": "ses-exact"},
    }


def test_managed_cli_kills_new_child_when_rotation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.headless_cli import run_managed_command

    child = object()
    killed: list[object] = []

    class FakeContext:
        def checkpoint(self) -> None:
            return None

        def transport_error(self, message: str) -> RuntimeError:
            return RuntimeError(message)

    class FakeOwner:
        cleanup_requested = threading.Event()

        def rotate(self, _child: object) -> None:
            raise RuntimeError("previous child is still live")

    monkeypatch.setattr("crossby.ai_tools.headless_cli.subprocess.Popen", lambda *_a, **_k: child)
    monkeypatch.setattr("crossby.ai_tools.headless_cli.kill_process_group", killed.append)
    with pytest.raises(RuntimeError, match="could not rotate"):
        run_managed_command(
            FakeContext(),  # type: ignore[arg-type]
            argv=["fixture"],
            cwd=tmp_path,
            controller=FakeOwner(),  # type: ignore[arg-type]
        )
    assert killed == [child]


def test_opencode_abort_respects_cleanup_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.headless import HeadlessCleanupContext
    from crossby.ai_tools.opencode_server import OpenCodeServer

    timeouts: list[float] = []

    class FakeConnection:
        def __init__(self, _host: str, _port: int, *, timeout: float) -> None:
            timeouts.append(timeout)

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def getresponse(self) -> FakeConnection:
            return self

        def read(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "crossby.ai_tools.opencode_server.http.client.HTTPConnection", FakeConnection
    )
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.active_session_id = "ses-exact"
    server.port = 1234
    server.directory = tmp_path
    server.authorization = "fixture"
    cancelled = threading.Event()
    server._abort_active_session(HeadlessCleanupContext(time.monotonic() - 1, cancelled))
    cancelled.set()
    server._abort_active_session(HeadlessCleanupContext(time.monotonic() + 1, cancelled))
    assert timeouts == []
    cancelled.clear()
    server._abort_active_session(HeadlessCleanupContext(time.monotonic() + 0.05, cancelled))
    assert len(timeouts) == 1
    assert 0 < timeouts[0] <= 0.05


@pytest.mark.parametrize(
    "tool_id",
    [AIToolID.CODEX, AIToolID.CURSOR, AIToolID.OPENCODE, AIToolID.ANTIGRAVITY_CLI],
)
def test_brokered_preflight_requires_a_handler_before_native_start(
    tmp_path: Path, tool_id: AIToolID
) -> None:
    from crossby.ai_tools.headless import HeadlessRequestError

    with pytest.raises(HeadlessRequestError, match="requires an interaction handler"):
        AbstractAITool.get(tool_id).preflight_headless_session(_request(tmp_path))


def test_codex_brokered_preflight_uses_its_own_verified_version_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.headless import HeadlessUnsupportedError

    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda _binary, **_kwargs: BinaryVersion((0, 160, 0), "codex-cli 0.160.0"),
    )
    adapter = AbstractAITool.get(AIToolID.CODEX)
    with pytest.raises(HeadlessUnsupportedError, match=r"0\.162\.1"):
        adapter.preflight_headless_session(
            _request(tmp_path),
            lambda _interaction: SessionInteractionResponse(
                outcome=PlanInteractionOutcome.CANCELLED
            ),
        )


def test_opencode_sse_parser_handles_fragmentation_and_rejects_bad_frames() -> None:
    from crossby.ai_tools.opencode_server import BoundedSSEParser

    parser = BoundedSSEParser()
    assert parser.feed(b': heartbeat\n\ndata: {"type":"message.') == []
    assert parser.feed(b'updated","properties":{}}\n\n') == [
        {"type": "message.updated", "properties": {}}
    ]
    parser.finish()

    with pytest.raises(ValueError, match="malformed JSON"):
        BoundedSSEParser().feed(b"data: {broken}\n\n")
    with pytest.raises(ValueError, match="size limit"):
        BoundedSSEParser().feed(b"data: " + b"x" * (1024 * 1024 + 1))


def test_opencode_sse_progress_uses_exact_prompt_binding() -> None:
    from crossby.ai_tools.headless_cli import BoundedProgress
    from crossby.ai_tools.opencode_server import OpenCodeServer

    server = OpenCodeServer.__new__(OpenCodeServer)
    server._event_error = None
    server._events = queue.Queue()
    server._bound_message_ids = set()
    server._tool_states = {}
    server._events.put(
        {
            "type": "message.updated",
            "properties": {
                "info": {
                    "id": "msg-foreign",
                    "sessionID": "ses-exact",
                    "role": "assistant",
                    "parentID": "prompt-foreign",
                    "tokens": {},
                }
            },
        }
    )
    server._events.put(
        {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "id": "tool-foreign",
                    "sessionID": "ses-exact",
                    "messageID": "msg-foreign",
                    "type": "tool",
                    "state": {"status": "running"},
                }
            },
        }
    )
    server._events.put(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses-exact",
                "messageID": "msg-foreign",
                "partID": "part-foreign",
                "field": "text",
                "delta": "ignore",
            },
        }
    )
    server._events.put(
        {
            "type": "message.updated",
            "properties": {
                "info": {
                    "id": "msg-current",
                    "sessionID": "ses-exact",
                    "role": "assistant",
                    "parentID": "prompt-exact",
                    "tokens": {},
                }
            },
        }
    )
    server._events.put(
        {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "id": "tool-exact",
                    "sessionID": "ses-exact",
                    "messageID": "msg-current",
                    "type": "tool",
                    "state": {"status": "completed"},
                }
            },
        }
    )
    server._events.put(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses-exact",
                "messageID": "msg-current",
                "partID": "part-exact",
                "field": "text",
                "delta": "streamed",
            },
        }
    )
    server._events.put(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses-foreign",
                "messageID": "msg-current",
                "partID": "part-exact",
                "field": "text",
                "delta": "ignore",
            },
        }
    )

    class FakeContext:
        def __init__(self) -> None:
            self.events: list[tuple[HeadlessEventKind, str | None, Any]] = []
            self.progress_count = 0

        def emit(self, kind: HeadlessEventKind, **fields: Any) -> None:
            self.events.append((kind, fields.get("message"), fields.get("payload")))

        def mark_progress(self) -> None:
            self.progress_count += 1

    context = FakeContext()
    server.drain_events(context, "ses-exact", "prompt-exact", BoundedProgress(context))
    assert context.events == [
        (HeadlessEventKind.ASSISTANT_MESSAGE, None, None),
        (HeadlessEventKind.USAGE, None, None),
        (HeadlessEventKind.TOOL_COMPLETED, None, {"tool_call_id": "tool-exact"}),
    ]
    assert context.progress_count == 1


def test_opencode_sse_progress_stays_bounded_across_drains() -> None:
    from crossby.ai_tools.headless_cli import BoundedProgress
    from crossby.ai_tools.opencode_server import OpenCodeServer

    server = OpenCodeServer.__new__(OpenCodeServer)
    server._event_error = None
    server._events = queue.Queue()
    server._bound_message_ids = set()
    server._tool_states = {}

    class FakeContext:
        def __init__(self) -> None:
            self.events = 0
            self.progress = 0

        def emit(self, _kind: HeadlessEventKind, **_fields: Any) -> None:
            self.events += 1

        def mark_progress(self) -> None:
            self.progress += 1

    context = FakeContext()
    progress = BoundedProgress(context, limit=2)
    for _ in range(2):
        for index in range(3):
            server._events.put(
                {
                    "type": "message.updated",
                    "properties": {
                        "info": {
                            "id": f"msg-{index}",
                            "sessionID": "ses-exact",
                            "role": "assistant",
                            "parentID": "prompt-exact",
                        }
                    },
                }
            )
        server.drain_events(context, "ses-exact", "prompt-exact", progress)
    assert context.events == 2
    assert context.progress == 4


def test_opencode_poll_binding_keeps_later_sse_deltas_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.headless_cli import BoundedProgress
    from crossby.ai_tools.opencode_server import OpenCodeServer

    class FakeServer:
        instance: ClassVar[FakeServer]

        def __init__(self, _request: Any, _deadline: float, runtime: Any = None) -> None:
            self.active_session_id: str | None = None
            self.prompt_id: str | None = None
            self._bound_message_ids: set[str] = set()
            self._events: queue.Queue[dict[str, Any]] = queue.Queue()
            self._event_error = None
            self._tool_states: dict[str, str] = {}
            self.polls = 0
            FakeServer.instance = self

        def start(self) -> None:
            pass

        def open_events(self) -> None:
            pass

        def request(self, method: str, path: str, payload: Any = None) -> Any:
            if method == "POST" and path == "/session":
                return {"id": "ses-exact"}
            if path == "/session/ses-exact/prompt_async":
                self.prompt_id = payload["messageID"]
                return None
            if path in {"/question", "/permission"}:
                return []
            if path == "/session/ses-exact/message":
                self.polls += 1
                return [
                    {"info": {"id": self.prompt_id, "sessionID": "ses-exact", "role": "user"}},
                    {
                        "info": {
                            "id": "msg-bound",
                            "sessionID": "ses-exact",
                            "role": "assistant",
                            "parentID": self.prompt_id,
                            **(
                                {"finish": "stop", "time": {"completed": 1}}
                                if self.polls > 1
                                else {}
                            ),
                        }
                    },
                ]
            if path == "/session/ses-exact/message/msg-bound":
                return {
                    "info": {
                        "id": "msg-bound",
                        "sessionID": "ses-exact",
                        "role": "assistant",
                        "parentID": self.prompt_id,
                    },
                    "parts": [{"type": "text", "text": "Done."}],
                }
            raise AssertionError((method, path))

    monkeypatch.setattr("crossby.ai_tools.opencode_server.OpenCodeServer", FakeServer)
    result = AbstractAITool.get(AIToolID.OPENCODE).run_headless_session(
        _request(tmp_path),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    server = FakeServer.instance
    assert server._bound_message_ids == {"msg-bound"}
    server._events.put(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses-exact",
                "messageID": "msg-bound",
                "partID": "part-bound",
                "field": "text",
                "delta": "streaming",
            },
        }
    )

    class FakeContext:
        def __init__(self) -> None:
            self.progress = 0

        def mark_progress(self) -> None:
            self.progress += 1

    context = FakeContext()
    OpenCodeServer.drain_events(
        server, context, "ses-exact", server.prompt_id, BoundedProgress(context)
    )
    assert context.progress == 1


def test_opencode_sse_survives_quiet_period_longer_than_200ms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.opencode_server import OpenCodeServer

    connections: list[Any] = []

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.reads = 0

        def getheader(self, _name: str, _default: str = "") -> str:
            return "text/event-stream"

        def read1(self, _amount: int) -> bytes:
            self.reads += 1
            if self.reads == 1:
                time.sleep(0.25)
                if connections[0].timeout < 0.25:
                    raise TimeoutError("socket expired during quiet period")
                return b'data: {"type":"session.status","properties":{}}\n\n'
            return b""

    class FakeConnection:
        def __init__(self, _host: str, _port: int, *, timeout: float) -> None:
            self.timeout = timeout
            connections.append(self)

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return response

        def close(self) -> None:
            pass

    response = FakeResponse()
    monkeypatch.setattr(
        "crossby.ai_tools.opencode_server.http.client.HTTPConnection", FakeConnection
    )
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.port = 1234
    server.directory = tmp_path
    server.authorization = "Basic fixture"
    server.deadline = time.monotonic() + 2
    server._event_stop = threading.Event()
    server._event_error = None
    server._event_connection = None
    server._events = queue.Queue()
    server.open_events()
    assert server._event_thread is not None
    server._event_thread.join(timeout=1)
    assert not server._event_thread.is_alive()
    assert server._event_error is None
    assert server._events.get_nowait()["type"] == "session.status"


def test_opencode_sse_full_queue_does_not_fail_polling_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools.opencode_server import OpenCodeServer

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.reads = 0

        def getheader(self, _name: str, _default: str = "") -> str:
            return "text/event-stream"

        def read1(self, _amount: int) -> bytes:
            self.reads += 1
            if self.reads == 1:
                return b'data: {"type":"session.status","properties":{}}\n\n' * 3
            return b""

    class FakeConnection:
        def __init__(self, _host: str, _port: int, *, timeout: float) -> None:
            pass

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def getresponse(self) -> FakeResponse:
            return response

        def close(self) -> None:
            pass

    response = FakeResponse()
    monkeypatch.setattr(
        "crossby.ai_tools.opencode_server.http.client.HTTPConnection", FakeConnection
    )
    server = OpenCodeServer.__new__(OpenCodeServer)
    server.port = 1234
    server.directory = tmp_path
    server.authorization = "Basic fixture"
    server.deadline = time.monotonic() + 2
    server._event_stop = threading.Event()
    server._event_error = None
    server._event_connection = None
    server._events = queue.Queue(maxsize=1)
    server.open_events()
    assert server._event_thread is not None
    server._event_thread.join(timeout=1)
    assert not server._event_thread.is_alive()
    assert server._event_error is None
    assert server._events.qsize() == 1


def test_json_rpc_cleanup_retries_short_write(monkeypatch: pytest.MonkeyPatch) -> None:
    from crossby.ai_tools.plan_process import JsonRpcProcess

    class RunningProcess:
        def poll(self) -> None:
            return None

    class OpenInput:
        closed = False

        def fileno(self) -> int:
            return 10

    rpc = JsonRpcProcess.__new__(JsonRpcProcess)
    rpc._proc = RunningProcess()  # type: ignore[assignment]
    rpc._stdin = OpenInput()  # type: ignore[assignment]
    rpc._include_jsonrpc_version = True
    rpc._send_lock = threading.Lock()
    chunks: list[bytes] = []

    def short_write(_fd: int, data: bytes | memoryview) -> int:
        chunk = bytes(data[:3])
        chunks.append(chunk)
        return len(chunk)

    monkeypatch.setattr("crossby.ai_tools.plan_process.os.write", short_write)
    monkeypatch.setattr("crossby.ai_tools.plan_process.os.get_blocking", lambda _fd: True)
    monkeypatch.setattr("crossby.ai_tools.plan_process.os.set_blocking", lambda _fd, _mode: None)
    rpc.send_cleanup({"method": "session/cancel", "params": {"sessionId": "ses-exact"}})
    assert b"".join(chunks) == (
        b'{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"ses-exact"}}\n'
    )


def test_unattended_profile_requires_verified_version() -> None:
    capability = (
        AbstractAITool.get(AIToolID.CODEX)
        .capabilities()
        .headless.model_copy(update={"verified_version": None})
    )
    with pytest.raises(ValueError, match="verified managed version"):
        capability.profile_for(HeadlessInteractionMode.UNATTENDED)


def test_antigravity_brokered_continuation_stops_after_nine_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeOwner:
        def __init__(self, _context: Any) -> None:
            return None

    calls: list[list[str]] = []

    def run(_context: Any, *, argv: list[str], **_kwargs: Any) -> HeadlessCommandOutput:
        calls.append(argv)
        return HeadlessCommandOutput(
            0,
            json.dumps(
                {
                    "status": "WAITING_FOR_INPUT",
                    "conversation_id": "conv-exact",
                    "question": {
                        "id": "choice",
                        "prompt": "Continue?",
                        "options": [{"id": "yes", "label": "Yes"}],
                    },
                }
            ),
            "",
        )

    monkeypatch.setattr("crossby.ai_tools.headless_cli.ResumableManagedCLIController", FakeOwner)
    monkeypatch.setattr("crossby.ai_tools.headless_cli.run_managed_command", run)
    result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON),
        lambda _interaction: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_id="yes"
        ),
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.conversation_id == "conv-exact"
    assert len(calls) == 9
    assert "nine native continuation" in " ".join(result.warnings)


def test_antigravity_stream_progress_is_bounded_across_continuations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeOwner:
        def __init__(self, _context: Any) -> None:
            pass

    invocations = 0

    def run(_context: Any, *, on_stdout_lines: Any, **_kwargs: Any) -> HeadlessCommandOutput:
        nonlocal invocations
        invocations += 1
        envelope: dict[str, Any] = {
            "status": "WAITING_FOR_INPUT" if invocations < 9 else "SUCCESS",
            "conversation_id": "conv-exact",
        }
        if invocations < 9:
            envelope["question"] = {
                "id": "choice",
                "prompt": "Continue?",
                "options": [{"id": "yes", "label": "Yes"}],
            }
        else:
            envelope["response"] = "Done."
        lines = [json.dumps({"event": "step_update", "conversation_id": "conv-exact"})] * 520
        lines.append(json.dumps({"event": "result", "result": envelope}))
        on_stdout_lines(lines)
        return HeadlessCommandOutput(0, "\n".join(lines), "")

    monkeypatch.setattr("crossby.ai_tools.headless_cli.ResumableManagedCLIController", FakeOwner)
    monkeypatch.setattr("crossby.ai_tools.headless_cli.run_managed_command", run)
    result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSONL),
        lambda _interaction: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, option_id="yes"
        ),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_json == {
        "status": "SUCCESS",
        "conversation_id": "conv-exact",
        "response": "Done.",
    }
    assert invocations == 9
    assert sum(event.kind is HeadlessEventKind.PROGRESS for event in result.events) == 512


def test_antigravity_nonzero_exit_emits_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools import headless_cli

    class FakeOwner:
        def __init__(self, _context: Any) -> None:
            pass

    monkeypatch.setattr(headless_cli, "ResumableManagedCLIController", FakeOwner)
    monkeypatch.setattr(
        headless_cli,
        "run_managed_command",
        lambda *_args, **_kwargs: HeadlessCommandOutput(
            1,
            json.dumps({"status": "FAILED", "conversation_id": "conv-exact"}),
            "",
        ),
    )
    result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.events[-1].terminal_status is HeadlessTerminalStatus.FAILED


def test_antigravity_invalid_native_answer_emits_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.ai_tools import headless_cli

    class FakeOwner:
        def __init__(self, _context: Any) -> None:
            pass

    monkeypatch.setattr(headless_cli, "ResumableManagedCLIController", FakeOwner)
    monkeypatch.setattr(
        headless_cli,
        "run_managed_command",
        lambda *_args, **_kwargs: HeadlessCommandOutput(
            0,
            json.dumps(
                {
                    "status": "WAITING_FOR_INPUT",
                    "conversation_id": "conv-exact",
                    "question": {
                        "id": "choice",
                        "prompt": "Continue?",
                        "options": [{"id": "yes", "label": "Yes"}],
                    },
                }
            ),
            "",
        ),
    )
    result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON),
        lambda _interaction: SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED, answer="maybe"
        ),
    )
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.events[-1].terminal_status is HeadlessTerminalStatus.FAILED


def test_codex_resume_starts_normal_turn_on_exact_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "thr-existing"}}},
            {"id": 3, "result": {"turn": {"id": "turn-new"}}},
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-existing",
                    "turnId": "turn-new",
                    "item": {"id": "msg-new", "type": "agentMessage", "text": "New answer."},
                },
            },
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thr-existing",
                    "turn": {"id": "turn-new", "status": "completed"},
                },
            },
            {"id": 4, "result": {}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CODEX).run_headless_session(
        _request(tmp_path, resume_id="thr-existing"),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert (result.thread_id, result.turn_id, result.final_text) == (
        "thr-existing",
        "turn-new",
        "New answer.",
    )
    sent = _FakeRpc.instances[0].sent
    assert any(
        method == "thread/resume" and params["threadId"] == "thr-existing"
        for method, params in sent
    )
    assert not any(method == "thread/start" for method, _params in sent)


def test_cursor_load_validates_replay_and_prompts_exact_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_rpc(
        monkeypatch,
        [
            {
                "id": 1,
                "result": {
                    "authMethods": [],
                    "agentCapabilities": {"loadSession": True},
                },
            },
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-existing",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-old",
                        "content": {"type": "text", "text": "Old answer."},
                    },
                },
            },
            {"id": 3, "result": {}},
            {
                "method": "session/update",
                "params": {
                    "sessionId": "ses-existing",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "messageId": "msg-new",
                        "content": {"type": "text", "text": "New answer."},
                    },
                },
            },
            {"id": 4, "result": {"stopReason": "end_turn"}},
        ],
    )
    result = AbstractAITool.get(AIToolID.CURSOR).run_headless_session(
        _request(tmp_path, resume_id="ses-existing"),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.session_id == "ses-existing"
    assert result.final_text == "New answer."
    assert any(method == "session/load" for method, _params in _FakeRpc.instances[0].sent)


def test_opencode_resume_ignores_historical_assistant_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeServer:
        instances: ClassVar[list[FakeServer]] = []

        def __init__(self, _request: Any, _deadline: float, runtime: Any = None) -> None:
            self.calls: list[tuple[str, str, Any]] = []
            self.active_session_id: str | None = None
            self.prompted = False
            self.prompt_id: str | None = None
            self._bound_message_ids: set[str] = set()
            self.instances.append(self)

        def start(self) -> None:
            return None

        def request(self, method: str, path: str, payload: Any = None) -> Any:
            self.calls.append((method, path, payload))
            if path == "/session/ses-existing" and method == "GET":
                return {"id": "ses-existing", "directory": str(tmp_path)}
            if path in {"/question", "/permission"}:
                return []
            if path == "/session/ses-existing/prompt_async":
                self.prompted = True
                self.prompt_id = payload["messageID"]
                return None
            if path == "/session/ses-existing/message":
                old = {
                    "info": {
                        "id": "msg-old",
                        "sessionID": "ses-existing",
                        "role": "assistant",
                        "finish": "stop",
                        "time": {"completed": 1},
                    }
                }
                pending = {
                    "info": {"id": "msg-pending", "sessionID": "ses-existing", "role": "user"}
                }
                if not self.prompted:
                    return [old, pending]
                return [
                    old,
                    pending,
                    {
                        "info": {
                            "id": self.prompt_id,
                            "sessionID": "ses-existing",
                            "role": "user",
                        }
                    },
                    {
                        "info": {
                            "id": "msg-new",
                            "sessionID": "ses-existing",
                            "role": "assistant",
                            "parentID": self.prompt_id,
                            "finish": "stop",
                            "time": {"completed": 2},
                        }
                    },
                    {
                        "info": {
                            "id": "msg-foreign",
                            "sessionID": "ses-existing",
                            "role": "assistant",
                            "parentID": "msg-pending",
                            "finish": "stop",
                            "time": {"completed": 3},
                        }
                    },
                ]
            if path == "/session/ses-existing/message/msg-new":
                return {
                    "info": {
                        "id": "msg-new",
                        "sessionID": "ses-existing",
                        "role": "assistant",
                        "parentID": self.prompt_id,
                    },
                    "parts": [{"type": "text", "text": "New answer."}],
                }
            raise AssertionError((method, path))

    monkeypatch.setattr("crossby.ai_tools.opencode_server.OpenCodeServer", FakeServer)
    result = AbstractAITool.get(AIToolID.OPENCODE).run_headless_session(
        _request(tmp_path, resume_id="ses-existing"),
        lambda _interaction: SessionInteractionResponse(outcome=PlanInteractionOutcome.CANCELLED),
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "New answer."
    assert FakeServer.instances[0].active_session_id is None
    assert FakeServer.instances[0]._bound_message_ids == {"msg-new"}
    prompt = next(
        payload
        for method, path, payload in FakeServer.instances[0].calls
        if method == "POST" and path.endswith("/prompt_async")
    )
    assert prompt["messageID"].startswith("msg_")
    assert len(prompt["messageID"]) == 30
    assert not any(
        method == "POST" and path == "/session"
        for method, path, _payload in FakeServer.instances[0].calls
    )


@pytest.mark.parametrize("kind", ["question", "permission"])
def test_opencode_resume_answers_only_current_prompts_requests(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeServer:
        instance: ClassVar[FakeServer]

        def __init__(self, _request: Any, _deadline: float, runtime: Any = None) -> None:
            self.calls: list[tuple[str, str, Any]] = []
            self.active_session_id: str | None = None
            self.prompt_id: str | None = None
            self._bound_message_ids: set[str] = set()
            FakeServer.instance = self

        def start(self) -> None:
            return None

        def request(self, method: str, path: str, payload: Any = None) -> Any:
            self.calls.append((method, path, payload))
            if path == "/session/ses-existing" and method == "GET":
                return {"id": "ses-existing", "directory": str(tmp_path)}
            if path == "/session/ses-existing/prompt_async":
                self.prompt_id = payload["messageID"]
                return None
            if path == f"/{kind}":
                requests = []
                for owner in ("foreign", "current"):
                    pending = {
                        "id": f"{kind}-{owner}",
                        "sessionID": "ses-existing",
                        "tool": {"messageID": f"msg-{owner}", "callID": f"call-{owner}"},
                    }
                    if kind == "question":
                        pending["questions"] = [
                            {"question": "Continue?", "options": [{"label": "Yes"}]}
                        ]
                    else:
                        pending.update(permission="bash", patterns=["git status"])
                    requests.append(pending)
                return requests
            if path == f"/{'permission' if kind == 'question' else 'question'}":
                return []
            if path in {
                "/session/ses-existing/message/msg-foreign",
                "/session/ses-existing/message/msg-current",
            }:
                owner = path.rsplit("-", 1)[-1]
                return {
                    "info": {
                        "id": f"msg-{owner}",
                        "sessionID": "ses-existing",
                        "role": "assistant",
                        "parentID": "msg-old" if owner == "foreign" else self.prompt_id,
                    },
                    "parts": [{"type": "tool", "callID": f"call-{owner}", "tool": "question"}],
                }
            if path == "/session/ses-existing/message":
                history = [{"info": {"id": "msg-old", "sessionID": "ses-existing"}}]
                if self.prompt_id is None:
                    return history
                return [
                    *history,
                    {"info": {"id": self.prompt_id, "sessionID": "ses-existing", "role": "user"}},
                    {
                        "info": {
                            "id": "msg-done",
                            "sessionID": "ses-existing",
                            "role": "assistant",
                            "parentID": self.prompt_id,
                            "finish": "stop",
                            "time": {"completed": 1},
                        }
                    },
                ]
            if path == "/session/ses-existing/message/msg-done":
                return {
                    "info": {
                        "id": "msg-done",
                        "sessionID": "ses-existing",
                        "role": "assistant",
                        "parentID": self.prompt_id,
                    },
                    "parts": [{"type": "text", "text": "Current answer."}],
                }
            if path == f"/{kind}/{kind}-current/reply":
                return True
            raise AssertionError((method, path))

    monkeypatch.setattr("crossby.ai_tools.opencode_server.OpenCodeServer", FakeServer)
    seen: list[SessionInteraction] = []

    def answer(interaction: SessionInteraction) -> SessionInteractionResponse:
        seen.append(interaction)
        return SessionInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_id="Yes" if kind == "question" else "once",
        )

    result = AbstractAITool.get(AIToolID.OPENCODE).run_headless_session(
        _request(tmp_path, resume_id="ses-existing"), answer
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "Current answer."
    assert FakeServer.instance._bound_message_ids == {"msg-current", "msg-done"}
    assert [interaction.request_id for interaction in seen] == [f"{kind}-current"]
    if kind == "permission":
        assert seen[0].operation is not None
        assert seen[0].operation.execution_dir == tmp_path.resolve()
    replies = [path for method, path, _ in FakeServer.instance.calls if method == "POST"]
    assert f"/{kind}/{kind}-current/reply" in replies
    assert f"/{kind}/{kind}-foreign/reply" not in replies
