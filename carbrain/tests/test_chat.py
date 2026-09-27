from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from carbrain import chat
from carbrain.chat import ChatSession, unsupported_numbers
from carbrain.rights import RightsError
from carbrain.tools import ToolContext


def text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t)


def tool_use(name: str, args: dict[str, Any], id_: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id_)


def response(stop: str, *blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeApi:
    def __init__(self, *responses: SimpleNamespace) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def ctx(loaded, registry, resolver) -> ToolContext:  # type: ignore[no-untyped-def]
    return ToolContext(loaded, registry, resolver)


def test_tool_round_trip(ctx: ToolContext) -> None:
    api = FakeApi(
        response("tool_use", text("Vou consultar."), tool_use("loan_rate", {})),
        response("end_turn", text("A taxa média é 1,98% ao mês (Fonte: BCB, 2026-07).")),
    )
    answer = ChatSession(api, ctx).ask("Qual a taxa de juros para financiar carro?")
    assert answer.text.startswith("A taxa média é 1,98%")
    assert answer.unsupported_numbers == []
    assert [c[0] for c in answer.tool_calls] == ["loan_rate"]
    second = api.calls[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"
    assert json.loads(second[-1]["content"][0]["content"])["data"]["monthly_pct"] == 1.98


def test_request_shape(ctx: ToolContext) -> None:
    api = FakeApi(response("end_turn", text("Olá!")))
    ChatSession(api, ctx).ask("oi")
    call = api.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert {t["name"] for t in call["tools"]} >= {"find_vehicle", "ownership_cost"}


def test_invented_numbers_trigger_one_revision(ctx: ToolContext) -> None:
    api = FakeApi(
        response("end_turn", text("O Polo Track custa R$ 99.990 e consome 13,4 km/l.")),
        response("end_turn", text("Ainda não tenho preços de versões para mostrar.")),
    )
    answer = ChatSession(api, ctx).ask("Quanto custa um Polo Track?")
    assert answer.text.startswith("Ainda não tenho")
    revision = api.calls[1]["messages"][-1]["content"]
    assert "99.990" in revision and "13,4" in revision


def test_numbers_still_unsupported_after_revision_are_reported(ctx: ToolContext) -> None:
    api = FakeApi(
        response("end_turn", text("Custa R$ 99.990.")),
        response("end_turn", text("Custa R$ 99.990.")),
    )
    answer = ChatSession(api, ctx).ask("Quanto custa?")
    assert answer.unsupported_numbers == ["99.990"]


def test_refusal_keeps_history_valid(ctx: ToolContext) -> None:
    api = FakeApi(response("refusal"))
    session = ChatSession(api, ctx)
    answer = session.ask("algo proibido")
    assert answer.refused and session.messages == []


def test_bad_tool_input_is_returned_as_error(ctx: ToolContext) -> None:
    api = FakeApi(
        response("tool_use", tool_use("ownership_cost", {"purchase_price": -5})),
        response("end_turn", text("Preciso do preço do carro.")),
    )
    ChatSession(api, ctx).ask("custo?")
    result = api.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "Invalid input" in result["content"]


def test_sources_must_allow_ai_processing(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat, "TOOL_SOURCES", {"anp_weekly", "youtube"})
    with pytest.raises(RightsError, match="youtube"):
        ChatSession(FakeApi(), ctx)


@pytest.mark.parametrize(
    ("answer", "sources", "bad"),
    [
        ("R$ 74.594,00", ['{"value": 74594.0}'], []),
        ("1,98% ao mês", ['{"monthly_pct": 1.98}'], []),
        ("0,512 por km", ['{"cost": 0.5125}'], []),
        ("com R$ 120 mil de entrada", ["Tenho R$ 120 mil"], []),
        ("em 2027 muda o imposto", [], []),
        ("são 3 versões", [], []),
        ("custa 57.386", ['{"value": 57386}'], []),
        ("custa 3.500 por ano", ['{"value": 3200}'], ["3.500"]),
        ("inflação de 4,5%", ['{"inflation_pct": 0.045}'], []),
    ],
)
def test_unsupported_numbers(answer: str, sources: list[str], bad: list[str]) -> None:
    assert unsupported_numbers(answer, sources) == bad
