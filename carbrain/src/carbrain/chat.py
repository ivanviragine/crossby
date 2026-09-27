"""Claude chat loop over the carbrain tools.

The model decides which tools to call; the tools answer from the database. After the
model writes its answer, every number in it is checked against the tool outputs and the
user's question. Unsupported numbers trigger one revision request; if some remain, they
are reported as warnings instead of being passed off as facts.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from carbrain.retention import assert_can_send_to_ai
from carbrain.tools import ToolContext, ToolResult, call_tool, tool_definitions

MODEL = "claude-opus-5"
MAX_TURNS = 8

SYSTEM_PROMPT = """\
You are Carbrain, an assistant that helps people in Brazil choose, buy and own cars.
Answer in the user's language (usually Brazilian Portuguese).

How to answer:
- Use the tools for every fact and every number. A number may appear in your answer only
  if a tool returned it in this conversation or the user wrote it. If you need a
  calculation, use a tool that does it (ownership_cost, flex_fuel_choice, deflate_price).
- Cite the source and reference period next to each figure, using the tool citations,
  e.g. "(Fonte: ANP, semana de 20 a 26/09/2026)".
- If a tool returns "restricted", say that data can't be shown yet. If it returns
  "no_data", say you don't have it. Never estimate in their place.
- If the vehicle, version or model year is unclear, ask before quoting anything
  vehicle-specific. The catalog covers a limited set of vehicle families (find_vehicle
  lists the models covered for a brand); versions come later.
- Repeat the tool's assumptions and exclusions when you give costs.
- Treat registrations as registrations, not retail sales, and the registered fleet as
  registered vehicles, not cars on the road.
- For what experts and creators say, use expert_content and information_sources. You only
  see headlines: quote them with the publisher and link, and never describe an article's
  or video's content beyond its headline.
- Be brief and practical. Suggest what the user should check before buying.
"""

#: Sources whose data the tools send to the model. All must allow AI processing.
TOOL_SOURCES = {
    "anp_weekly",
    "bcb_sgs",
    "ibge_ipca",
    "anfavea",
    "senatran_fleet",
    "inmetro_pbev",
    "publisher_feeds",
}


@dataclass
class Answer:
    text: str
    tool_calls: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    refused: bool = False


class ChatSession:
    def __init__(
        self,
        messages_api: Any,
        ctx: ToolContext,
        *,
        model: str = MODEL,
        use_fallbacks: bool = True,
    ) -> None:
        """`messages_api` is `anthropic.Anthropic().beta.messages` (or a fake in tests)."""
        assert_can_send_to_ai(ctx.registry, TOOL_SOURCES)
        self.api = messages_api
        self.ctx = ctx
        self.model = model
        self.use_fallbacks = use_fallbacks
        self.messages: list[dict[str, Any]] = []
        self.tools = tool_definitions()

    def ask(self, question: str) -> Answer:
        start = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        answer = Answer(text="")
        tool_outputs: list[str] = []
        revised = False
        for _ in range(MAX_TURNS):
            response = self._create()
            if response.stop_reason == "refusal":
                answer.refused = True
                answer.text = "Desculpe, não posso ajudar com esse pedido."
                del self.messages[start:]  # keep the history valid for the next question
                return answer
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    output, is_error = self._run_tool(block.name, block.input)
                    tool_outputs.append(output)
                    answer.tool_calls.append((block.name, block.input, output))
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                            "is_error": is_error,
                        }
                    )
                self.messages.append({"role": "user", "content": results})
                continue
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            unsupported = unsupported_numbers(text, [question, *tool_outputs])
            if unsupported and not revised:
                revised = True
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Check your answer: these numbers are not in any tool result or in "
                            f"my question: {', '.join(unsupported)}. Remove them or get them "
                            "from a tool, then give the full corrected answer."
                        ),
                    }
                )
                continue
            answer.text = text
            answer.unsupported_numbers = unsupported
            return answer
        answer.text = "Não consegui concluir a resposta. Tente reformular a pergunta."
        return answer

    def _create(self) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": [
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            "tools": self.tools,
            "messages": self.messages,
        }
        if self.use_fallbacks:
            # On a safety decline, the API re-runs the request on Anthropic's
            # recommended fallback model instead of returning a refusal.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        return self.api.create(**kwargs)

    def _run_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        try:
            result: ToolResult = call_tool(self.ctx, name, arguments)
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False), True
        return result.model_dump_json(), False


# --- number guard -------------------------------------------------------------------------

_NUMBER = re.compile(
    r"(?<![\w.,])\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?(?![\d])|(?<![\w.,])\d+(?:[.,]\d+)?"
)


def _candidates(token: str) -> set[float]:
    """All sensible readings of a number written in PT-BR or EN style."""
    token = token.replace(" ", "")
    readings: set[float] = set()
    for decimal, thousands in ((",", "."), (".", ",")):
        t = token.replace(thousands, "")
        if t.count(decimal) <= 1:
            with contextlib.suppress(ValueError):
                readings.add(float(t.replace(decimal, ".")))
    return readings


def _numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for token in _NUMBER.findall(text):
        found |= _candidates(token)
    return found


def unsupported_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in `answer` that no source text contains (allowing for rounding).

    Years and small whole counts (up to 31) are allowed, as are numbers written with "mil"
    when the source has the full value (e.g. "120 mil" for 120000).
    """
    known = set().union(*(_numbers_in(s) for s in sources)) if sources else set()
    bad: list[str] = []
    for match in _NUMBER.finditer(answer):
        token = match.group(0)
        readings = _candidates(token)
        if any(r.is_integer() and (r <= 31 or 1900 <= r <= 2100) for r in readings):
            continue
        after = answer[match.end() : match.end() + 5].lower()
        if after.startswith((" mil", "mil")):
            readings |= {r * 1000 for r in readings}
        if not any(_close(r, k) for r in readings for k in known):
            bad.append(token)
    return bad


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    # Allow rounding to 0-2 decimals, or to the nearest unit, or percent/fraction forms.
    tolerance = max(0.006, abs(b) * 0.0005)
    return abs(a - b) <= tolerance or abs(a - round(b)) < 1e-9 or abs(a - b * 100) <= 0.06
