"""Opt-in native OpenCode test using only a local, deterministic model stub."""

from __future__ import annotations

import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools import (
    AbstractAITool,
    PlanCommandPolicy,
    PlanSessionRequest,
    PlanSessionUnsupportedError,
    PlanTransportError,
)
from crossby.ai_tools.opencode_server import OpenCodeServer
from crossby.models.ai import AIToolID, EffortLevel, PlanInteractionOutcome, PlanInteractionResponse

pytestmark = pytest.mark.skipif(
    os.environ.get("CROSSBY_OPENCODE_LOCAL_SMOKE") != "1" or shutil.which("opencode") is None,
    reason="opt-in native OpenCode test (local model stub; no credentials or paid calls)",
)


@pytest.mark.parametrize(
    ("model_source", "effort"),
    [
        (source, effort)
        for source in ["explicit", "default", "agent"]
        for effort in [None, EffortLevel.HIGH, EffortLevel.LOW]
    ]
    + [("blocked-callback", None)],
)
def test_real_opencode_forwards_native_multiselect_and_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_source: str,
    effort: EffortLevel | None,
) -> None:
    requests: list[dict[str, Any]] = []
    plan = "# Native plan\n\n1. Inspect the adapters.\n2. Add regression coverage."

    class ModelStub(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            has_question_tool = any(
                tool.get("function", {}).get("name") == "question"
                for tool in payload.get("tools", [])
            )
            has_answer = any(message.get("role") == "tool" for message in payload["messages"])
            if has_question_tool and not has_answer:
                delta = {
                    "content": "Before I finalize the plan, which platforms should I support?",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_fixture_question",
                            "type": "function",
                            "function": {
                                "name": "question",
                                "arguments": json.dumps(
                                    {
                                        "questions": [
                                            {
                                                "question": "Which platforms?",
                                                "header": "Platforms",
                                                "options": [
                                                    {
                                                        "label": "Linux",
                                                        "description": "Linux support",
                                                    },
                                                    {
                                                        "label": "macOS",
                                                        "description": "macOS support",
                                                    },
                                                ],
                                                "multiple": True,
                                            }
                                        ]
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta = {"content": plan}
                finish = "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk, reason in [(delta, None), ({}, finish)]:
                event = {
                    "id": "chatcmpl_fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "fixture-model",
                    "choices": [{"index": 0, "delta": chunk, "finish_reason": reason}],
                }
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    stub = ThreadingHTTPServer(("127.0.0.1", 0), ModelStub)
    worker = threading.Thread(target=stub.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENCODE_DISABLE_PROJECT_CONFIG", "true")
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps(
            {
                "enabled_providers": ["crossbyfixture"],
                "model": (
                    "crossbyfixture/default-model"
                    if model_source == "agent"
                    else "crossbyfixture/fixture-model"
                ),
                "agent": (
                    {"plan": {"model": "crossbyfixture/fixture-model"}}
                    if model_source == "agent"
                    else {}
                ),
                "small_model": "crossbyfixture/fixture-model",
                "share": "disabled",
                "provider": {
                    "crossbyfixture": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Local fixture",
                        "options": {
                            "baseURL": f"http://127.0.0.1:{stub.server_port}/v1",
                            "apiKey": "fixture",
                        },
                        "models": {
                            "fixture-model": {
                                "name": "Fixture model",
                                "tool_call": True,
                                "reasoning": True,
                                "variants": {
                                    "low": {"disabled": True},
                                    "high": {"reasoningEffort": "high"},
                                },
                            },
                            "default-model": {
                                "name": "Decoy default model",
                                "tool_call": True,
                                "reasoning": True,
                                "variants": {"high": {"disabled": True}},
                            },
                        },
                    }
                },
            }
        ),
    )
    interactions: list[Any] = []
    callback_entered = threading.Event()
    callback_release = threading.Event()
    closed_processes: list[Any] = []
    original_close = OpenCodeServer.close

    def observe_close(server: OpenCodeServer) -> None:
        original_close(server)
        closed_processes.append(server.process._proc)

    monkeypatch.setattr(OpenCodeServer, "close", observe_close)
    # Release even on a regression, so a broken deadline cannot hang the test.
    safety_release = threading.Timer(12, callback_release.set)
    safety_release.daemon = True

    def answer(interaction: Any) -> PlanInteractionResponse:
        interactions.append(interaction)
        if model_source == "blocked-callback":
            callback_entered.set()
            callback_release.wait()
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_ids=("Linux", "macOS"),
        )

    try:
        request = PlanSessionRequest(
            prompt="Ask which platforms to support, then return an implementation plan.",
            working_dir=tmp_path,
            model="crossbyfixture/fixture-model" if model_source == "explicit" else None,
            effort=effort,
            timeout_seconds=10 if model_source == "blocked-callback" else 30,
            command_policy=PlanCommandPolicy(allowed_commands=("git status",)),
        )
        adapter = AbstractAITool.get(AIToolID.OPENCODE)
        if model_source == "blocked-callback":
            safety_release.start()
            with pytest.raises(PlanTransportError, match=r"timed out.*interaction callback"):
                adapter.run_plan_session(request, answer)
            assert callback_entered.is_set()
            assert not callback_release.is_set()
            assert len(closed_processes) == 1
            assert closed_processes[0].returncode is not None
            return
        if effort is EffortLevel.LOW:
            with pytest.raises(PlanSessionUnsupportedError, match="does not advertise"):
                adapter.run_plan_session(request, answer)
            assert not requests
            assert not interactions
            return
        result = adapter.run_plan_session(request, answer)
        assert result.plan == plan
        assert result.session_id.startswith("ses")
        assert result.artifact_id and result.artifact_id.startswith("msg")
        assert len(interactions) == 1
        assert interactions[0].allow_multiple is True
        assert [option.option_id for option in interactions[0].options] == ["Linux", "macOS"]
        assert any(
            message.get("role") == "tool"
            and "Linux" in str(message.get("content"))
            and "macOS" in str(message.get("content"))
            for payload in requests
            for message in payload["messages"]
        )
        planning_requests = [payload for payload in requests if payload.get("tools")]
        assert planning_requests
        assert all(payload["model"] == "fixture-model" for payload in planning_requests)
        if effort is EffortLevel.HIGH:
            assert all(payload.get("reasoning_effort") == "high" for payload in planning_requests)
        else:
            assert all("reasoning_effort" not in payload for payload in planning_requests)
    finally:
        callback_release.set()
        safety_release.cancel()
        stub.shutdown()
        stub.server_close()
        worker.join(timeout=2)
