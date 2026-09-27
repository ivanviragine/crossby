"""The chat loop against the real Anthropic SDK, with the HTTP layer mocked.

Checks that the SDK accepts the exact request the loop builds (tools, cached system
prompt, server-side fallbacks) and that its parsed responses drive the loop.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from carbrain.chat import ChatSession
from carbrain.tools import ToolContext

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")


def _message(stop_reason: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_sdk_round_trip(loaded, registry, resolver) -> None:  # type: ignore[no-untyped-def]
    replies = [
        _message(
            "tool_use",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "fuel_prices",
                    "input": {"place": "Curitiba, PR"},
                }
            ],
        ),
        _message("end_turn", [{"type": "text", "text": "O etanol está em 4,63 R$/l."}]),
    ]
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx2.Response(200, json=replies[len(requests) - 1])

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    ctx = ToolContext(loaded, registry, resolver)
    answer = ChatSession(client.beta.messages, ctx).ask("Quanto está o etanol em Curitiba?")

    assert answer.text == "O etanol está em 4,63 R$/l."
    assert answer.unsupported_numbers == []
    first = requests[0]
    assert first["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert first["body"]["fallbacks"] == "default"
    assert first["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    # The second request carries the tool result for the SDK's parsed tool_use block.
    tool_result = requests[1]["body"]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_1"
    assert json.loads(tool_result["content"])["data"]["place"] == "PR/CURITIBA"
