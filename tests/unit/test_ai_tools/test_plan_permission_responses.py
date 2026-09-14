"""Option-only permissions must never discard contradictory callback text."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from crossby.ai_tools import AbstractAITool, PlanInteraction, PlanInteractionRequiredError
from crossby.ai_tools.codex import _answer_codex_approval
from crossby.ai_tools.cursor import _answer_cursor_permission
from crossby.ai_tools.opencode_server import _answer_permission
from crossby.models.ai import AIToolID, PlanInteractionOutcome, PlanInteractionResponse


def _respond(tool: AIToolID, native: Mock, response: PlanInteractionResponse) -> None:
    capability = AbstractAITool.get(tool).capabilities().plan_mode

    def answer(interaction: PlanInteraction) -> PlanInteractionResponse:
        assert not interaction.allow_other
        return response

    if tool is AIToolID.CODEX:
        _answer_codex_approval(
            native,
            {"id": 42, "params": {"threadId": "thread", "turnId": "turn", "itemId": "item"}},
            thread_id="thread",
            turn_id="turn",
            handler=answer,
            tool_id=tool,
            capability=capability,
            deny_automatically=False,
        )
    elif tool is AIToolID.CURSOR:
        _answer_cursor_permission(
            native,
            {
                "id": 42,
                "params": {
                    "sessionId": "session",
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
                        {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
                    ],
                },
            },
            session_id="session",
            handler=answer,
            tool_id=tool,
            capability=capability,
            deny_automatically=False,
        )
    else:
        _answer_permission(
            native,
            {"id": "permission", "permission": "bash", "patterns": ["echo hi"]},
            "session",
            answer,
            capability,
        )


@pytest.mark.parametrize(
    ("tool", "option"),
    [(AIToolID.CODEX, "accept"), (AIToolID.CURSOR, "allow-once"), (AIToolID.OPENCODE, "once")],
)
@pytest.mark.parametrize(
    "outcome", [PlanInteractionOutcome.ANSWERED, PlanInteractionOutcome.APPROVED]
)
@pytest.mark.parametrize("text", ["deny", "", " "])
def test_permission_rejects_text_before_native_reply(
    tool: AIToolID, option: str, outcome: PlanInteractionOutcome, text: str
) -> None:
    native = Mock()
    response = PlanInteractionResponse(outcome=outcome, answer=text, option_id=option)

    with pytest.raises(PlanInteractionRequiredError):
        _respond(tool, native, response)

    assert not native.mock_calls


@pytest.mark.parametrize(
    ("tool", "option", "reply"),
    [
        (AIToolID.CODEX, "accept", {"decision": "decline"}),
        (
            AIToolID.CURSOR,
            "allow-once",
            {"outcome": {"outcome": "selected", "optionId": "reject-once"}},
        ),
        (AIToolID.OPENCODE, "once", {"reply": "reject"}),
    ],
)
def test_permission_denial_still_overrides_stale_selection(
    tool: AIToolID, option: str, reply: dict[str, object]
) -> None:
    native = Mock()
    _respond(
        tool,
        native,
        PlanInteractionResponse(
            outcome=PlanInteractionOutcome.DENIED,
            answer="stale",
            option_id=option,
        ),
    )
    if tool is AIToolID.OPENCODE:
        native.request.assert_called_once_with("POST", "/permission/permission/reply", reply)
    else:
        native.respond.assert_called_once_with(42, reply)
