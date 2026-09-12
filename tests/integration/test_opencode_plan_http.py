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

from crossby.ai_tools import AbstractAITool, PlanSessionRequest
from crossby.models.ai import AIToolID, PlanInteractionOutcome, PlanInteractionResponse

pytestmark = pytest.mark.skipif(
    os.environ.get("CROSSBY_OPENCODE_LOCAL_SMOKE") != "1" or shutil.which("opencode") is None,
    reason="opt-in native OpenCode test (local model stub; no credentials or paid calls)",
)


def test_real_opencode_forwards_native_multiselect_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                "model": "crossbyfixture/fixture-model",
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
                        "models": {"fixture-model": {"name": "Fixture model", "tool_call": True}},
                    }
                },
            }
        ),
    )
    interactions: list[Any] = []

    def answer(interaction: Any) -> PlanInteractionResponse:
        interactions.append(interaction)
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_ids=("Linux", "macOS"),
        )

    try:
        result = AbstractAITool.get(AIToolID.OPENCODE).run_plan_session(
            PlanSessionRequest(
                prompt="Ask which platforms to support, then return an implementation plan.",
                working_dir=tmp_path,
                model="crossbyfixture/fixture-model",
                timeout_seconds=30,
            ),
            answer,
        )
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
    finally:
        stub.shutdown()
        stub.server_close()
        worker.join(timeout=2)
