"""Option-only permissions must never discard contradictory callback text."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from crossby.ai_tools import (
    AbstractAITool,
    PlanCommandPolicy,
    PlanCommandPolicyUnsupportedError,
    PlanInteraction,
    PlanInteractionRequiredError,
    PlanTransportError,
)
from crossby.ai_tools.codex import _answer_codex_approval
from crossby.ai_tools.cursor import _answer_cursor_permission
from crossby.ai_tools.opencode_server import _answer_permission
from crossby.models.ai import (
    AIToolID,
    PlanInteractionOutcome,
    PlanInteractionResponse,
    PlanOperationKind,
    PlanPermissionTargetKind,
)


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


def _codex_command_message(tmp_path: Path, command: str | None) -> dict[str, object]:
    params: dict[str, object] = {
        "threadId": "thread",
        "turnId": "turn",
        "approvalId": "approval",
        "itemId": "item",
        "cwd": str(tmp_path),
        "availableDecisions": ["accept", "decline", "cancel"],
    }
    if command is not None:
        params["command"] = command
    return {
        "id": 42,
        "method": "item/commandExecution/requestApproval",
        "params": params,
    }


def test_codex_policy_approves_only_a_matching_authoritative_command(tmp_path: Path) -> None:
    native = Mock()
    handler = Mock(side_effect=AssertionError("matched policy must not invoke callback"))

    _answer_codex_approval(
        native,
        _codex_command_message(tmp_path, "git diff --stat"),
        thread_id="thread",
        turn_id="turn",
        handler=handler,
        tool_id=AIToolID.CODEX,
        capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
        deny_automatically=True,
        command_policy=PlanCommandPolicy(allowed_commands=("git diff:*",)),
        allowed_execution_roots=(tmp_path,),
    )

    native.respond.assert_called_once_with(42, {"decision": "accept"})
    handler.assert_not_called()


def test_codex_unmatched_policy_uses_never_approval_posture(tmp_path: Path) -> None:
    native = Mock()

    _answer_codex_approval(
        native,
        _codex_command_message(tmp_path, "git status"),
        thread_id="thread",
        turn_id="turn",
        handler=None,
        tool_id=AIToolID.CODEX,
        capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
        deny_automatically=True,
        command_policy=PlanCommandPolicy(allowed_commands=("git diff:*",)),
        allowed_execution_roots=(tmp_path,),
    )

    native.respond.assert_called_once_with(42, {"decision": "decline"})


@pytest.mark.parametrize("command", [None, "git status", "git diff && curl example.invalid"])
def test_codex_policy_leaves_missing_unmatched_and_compound_commands_unapproved(
    command: str | None, tmp_path: Path
) -> None:
    native = Mock()
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_codex_approval(
        native,
        _codex_command_message(tmp_path, command),
        thread_id="thread",
        turn_id="turn",
        handler=deny,
        tool_id=AIToolID.CODEX,
        capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
        deny_automatically=False,
        command_policy=PlanCommandPolicy(allowed_commands=("git diff:*",)),
        allowed_execution_roots=(tmp_path,),
    )

    native.respond.assert_called_once_with(42, {"decision": "decline"})
    assert seen[0].operation is not None
    assert seen[0].operation.kind is PlanOperationKind.COMMAND
    assert seen[0].operation.shell_expression == command
    assert seen[0].operation.execution_dir == tmp_path
    assert [(binding.name, binding.value) for binding in seen[0].operation.native_binding_ids] == [
        ("item_id", "item"),
        ("approval_id", "approval"),
    ]


def test_codex_policy_rejects_a_match_without_native_approve_once(tmp_path: Path) -> None:
    native = Mock()
    message = _codex_command_message(tmp_path, "git status")
    assert isinstance(message["params"], dict)
    message["params"]["availableDecisions"] = ["decline", "cancel"]

    with pytest.raises(PlanCommandPolicyUnsupportedError, match="one-operation acceptance"):
        _answer_codex_approval(
            native,
            message,
            thread_id="thread",
            turn_id="turn",
            handler=None,
            tool_id=AIToolID.CODEX,
            capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
            deny_automatically=False,
            command_policy=PlanCommandPolicy(allowed_commands=("git status",)),
            allowed_execution_roots=(tmp_path,),
        )

    assert not native.mock_calls


def test_codex_rejects_structured_native_decisions_instead_of_hiding_them(
    tmp_path: Path,
) -> None:
    native = Mock()
    message = _codex_command_message(tmp_path, "git status")
    assert isinstance(message["params"], dict)
    message["params"]["availableDecisions"] = [
        "acceptForSession",
        {"acceptWithExecpolicyAmendment": {"execpolicy_amendment": {}}},
        "decline",
    ]
    with pytest.raises(
        PlanTransportError,
        match="unrepresentable native approval decisions",
    ):
        _answer_codex_approval(
            native,
            message,
            thread_id="thread",
            turn_id="turn",
            handler=lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED
            ),
            tool_id=AIToolID.CODEX,
            capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
            deny_automatically=False,
        )

    assert not native.mock_calls


def test_codex_unknown_operation_kind_never_matches_policy(tmp_path: Path) -> None:
    native = Mock()
    message = _codex_command_message(tmp_path, "git status")
    assert isinstance(message["params"], dict)
    message["params"]["kind"] = "futureOperation"
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_codex_approval(
        native,
        message,
        thread_id="thread",
        turn_id="turn",
        handler=deny,
        tool_id=AIToolID.CODEX,
        capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
        deny_automatically=False,
        command_policy=PlanCommandPolicy(allowed_commands=("git status",)),
        allowed_execution_roots=(tmp_path,),
    )

    assert seen[0].operation is not None
    assert seen[0].operation.kind is PlanOperationKind.OTHER
    native.respond.assert_called_once_with(42, {"decision": "decline"})


def test_codex_extra_permission_target_prevents_policy_approval(tmp_path: Path) -> None:
    native = Mock()
    message = _codex_command_message(tmp_path, "git status")
    assert isinstance(message["params"], dict)
    message["params"]["networkApprovalContext"] = {"host": "example.invalid"}
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_codex_approval(
        native,
        message,
        thread_id="thread",
        turn_id="turn",
        handler=deny,
        tool_id=AIToolID.CODEX,
        capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
        deny_automatically=False,
        command_policy=PlanCommandPolicy(allowed_commands=("git status",)),
        allowed_execution_roots=(tmp_path,),
    )

    assert seen[0].operation is not None
    assert seen[0].operation.permission_targets[0].kind is PlanPermissionTargetKind.NETWORK_HOST
    native.respond.assert_called_once_with(42, {"decision": "decline"})


def test_codex_special_filesystem_target_prevents_policy_approval(tmp_path: Path) -> None:
    native = Mock()
    message = _codex_command_message(tmp_path, "git status")
    assert isinstance(message["params"], dict)
    message["params"]["additionalPermissions"] = {
        "fileSystem": {
            "read": None,
            "write": None,
            "entries": [
                {
                    "access": "write",
                    "path": {"type": "special", "value": {"kind": "project_roots"}},
                }
            ],
        },
        "network": None,
    }
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_codex_approval(
        native,
        message,
        thread_id="thread",
        turn_id="turn",
        handler=deny,
        tool_id=AIToolID.CODEX,
        capability=AbstractAITool.get(AIToolID.CODEX).capabilities().plan_mode,
        deny_automatically=False,
        command_policy=PlanCommandPolicy(allowed_commands=("git status",)),
        allowed_execution_roots=(tmp_path,),
    )

    assert seen[0].operation is not None
    assert seen[0].operation.permission_targets[0].value == "special:project_roots"
    native.respond.assert_called_once_with(42, {"decision": "decline"})


def test_cursor_forwards_raw_operation_fields_without_parsing_title(tmp_path: Path) -> None:
    native = Mock()
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_cursor_permission(
        native,
        {
            "id": 42,
            "params": {
                "sessionId": "session",
                "toolCall": {
                    "toolCallId": "call",
                    "title": "Run display-only command text",
                    "kind": "execute",
                    "rawInput": {"argv": ["git", "status"], "cwd": str(tmp_path)},
                },
                "options": [
                    {"optionId": "allow-once", "name": "Allow once"},
                    {"optionId": "deny-once", "name": "Deny once"},
                ],
            },
        },
        session_id="session",
        handler=deny,
        tool_id=AIToolID.CURSOR,
        capability=AbstractAITool.get(AIToolID.CURSOR).capabilities().plan_mode,
        deny_automatically=False,
    )

    assert seen[0].operation is not None
    assert seen[0].operation.kind is PlanOperationKind.COMMAND
    assert seen[0].operation.argv == ("git", "status")
    assert seen[0].operation.shell_expression is None
    assert seen[0].operation.execution_dir == tmp_path
    assert [(binding.name, binding.value) for binding in seen[0].operation.native_binding_ids] == [
        ("request_id", "42"),
        ("tool_call_id", "call"),
    ]
    native.respond.assert_called_once_with(
        42,
        {"outcome": {"outcome": "selected", "optionId": "deny-once"}},
    )


def test_cursor_rejects_conflicting_native_command_representations(tmp_path: Path) -> None:
    native = Mock()

    with pytest.raises(PlanTransportError, match="conflicting command representations"):
        _answer_cursor_permission(
            native,
            {
                "id": 42,
                "params": {
                    "sessionId": "session",
                    "toolCall": {
                        "kind": "execute",
                        "rawInput": {
                            "argv": ["git", "status"],
                            "command": "git status && curl example.invalid",
                            "cwd": str(tmp_path),
                        },
                    },
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once"},
                        {"optionId": "deny-once", "name": "Deny once"},
                    ],
                },
            },
            session_id="session",
            handler=lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED
            ),
            tool_id=AIToolID.CURSOR,
            capability=AbstractAITool.get(AIToolID.CURSOR).capabilities().plan_mode,
            deny_automatically=False,
        )

    assert not native.mock_calls


def test_cursor_rejects_malformed_argv_alongside_native_command(tmp_path: Path) -> None:
    native = Mock()
    handler = Mock()

    with pytest.raises(PlanTransportError, match="conflicting command representations"):
        _answer_cursor_permission(
            native,
            {
                "id": 42,
                "params": {
                    "sessionId": "session",
                    "toolCall": {
                        "kind": "execute",
                        "rawInput": {
                            "argv": ["git", 1],
                            "command": "git status",
                            "cwd": str(tmp_path),
                        },
                    },
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once"},
                        {"optionId": "deny-once", "name": "Deny once"},
                    ],
                },
            },
            session_id="session",
            handler=handler,
            tool_id=AIToolID.CURSOR,
            capability=AbstractAITool.get(AIToolID.CURSOR).capabilities().plan_mode,
            deny_automatically=False,
        )

    handler.assert_not_called()
    assert not native.mock_calls


def test_cursor_rejects_conflicting_native_working_directories(tmp_path: Path) -> None:
    native = Mock()
    handler = Mock()

    with pytest.raises(PlanTransportError, match="conflicting working directories"):
        _answer_cursor_permission(
            native,
            {
                "id": 42,
                "params": {
                    "sessionId": "session",
                    "toolCall": {
                        "kind": "execute",
                        "rawInput": {
                            "argv": ["git", "status"],
                            "cwd": str(tmp_path),
                            "workingDirectory": str(tmp_path / "outside"),
                        },
                    },
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once"},
                        {"optionId": "deny-once", "name": "Deny once"},
                    ],
                },
            },
            session_id="session",
            handler=handler,
            tool_id=AIToolID.CURSOR,
            capability=AbstractAITool.get(AIToolID.CURSOR).capabilities().plan_mode,
            deny_automatically=False,
        )

    handler.assert_not_called()
    assert not native.mock_calls


@pytest.mark.parametrize(
    "locations",
    [
        {"path": "/valid"},
        [{"path": "/valid"}, {"uri": "file:///unrepresentable"}],
        [{"path": "/valid"}, {"path": " "}],
        [{"path": "/valid"}, "unrepresentable"],
    ],
)
def test_cursor_rejects_malformed_native_locations(locations: object) -> None:
    native = Mock()
    handler = Mock()

    with pytest.raises(PlanTransportError, match="malformed locations"):
        _answer_cursor_permission(
            native,
            {
                "id": 42,
                "params": {
                    "sessionId": "session",
                    "toolCall": {
                        "kind": "edit",
                        "locations": locations,
                    },
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once"},
                        {"optionId": "deny-once", "name": "Deny once"},
                    ],
                },
            },
            session_id="session",
            handler=handler,
            tool_id=AIToolID.CURSOR,
            capability=AbstractAITool.get(AIToolID.CURSOR).capabilities().plan_mode,
            deny_automatically=False,
        )

    handler.assert_not_called()
    assert not native.mock_calls


def test_cursor_keeps_request_binding_distinct_without_tool_call_id(tmp_path: Path) -> None:
    native = Mock()
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_cursor_permission(
        native,
        {
            "id": 42,
            "params": {
                "sessionId": "session",
                "toolCall": {
                    "toolCallId": " ",
                    "kind": "execute",
                    "rawInput": {"argv": ["git", "status"], "cwd": str(tmp_path)},
                },
                "options": [
                    {"optionId": "allow-once", "name": "Allow once"},
                    {"optionId": "deny-once", "name": "Deny once"},
                ],
            },
        },
        session_id="session",
        handler=deny,
        tool_id=AIToolID.CURSOR,
        capability=AbstractAITool.get(AIToolID.CURSOR).capabilities().plan_mode,
        deny_automatically=False,
    )

    assert seen[0].artifact_id == "42"
    assert seen[0].operation is not None
    assert [(binding.name, binding.value) for binding in seen[0].operation.native_binding_ids] == [
        ("request_id", "42"),
    ]


def test_opencode_external_directory_does_not_invent_write_authority(
    tmp_path: Path,
) -> None:
    native = Mock()
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_permission(
        native,
        {
            "id": "permission",
            "permission": "external_directory",
            "patterns": [str(tmp_path / "external")],
        },
        "session",
        deny,
        AbstractAITool.get(AIToolID.OPENCODE).capabilities().plan_mode,
        execution_dir=tmp_path,
    )

    assert seen[0].operation is not None
    assert seen[0].operation.kind is PlanOperationKind.OTHER
    assert [target.kind for target in seen[0].operation.permission_targets] == [
        PlanPermissionTargetKind.RESOURCE
    ]


def test_opencode_forwards_permission_targets_and_native_bindings(tmp_path: Path) -> None:
    native = Mock()
    seen: list[PlanInteraction] = []

    def deny(interaction: PlanInteraction) -> PlanInteractionResponse:
        seen.append(interaction)
        return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

    _answer_permission(
        native,
        {
            "id": "permission",
            "permission": "bash",
            "patterns": ["git status"],
            "tool": {"messageID": "message", "callID": "call"},
        },
        "session",
        deny,
        AbstractAITool.get(AIToolID.OPENCODE).capabilities().plan_mode,
        execution_dir=tmp_path,
    )

    assert seen[0].operation is not None
    assert seen[0].operation.kind is PlanOperationKind.COMMAND
    assert seen[0].operation.execution_dir == tmp_path.resolve()
    assert seen[0].operation.shell_expression is None
    assert [target.value for target in seen[0].operation.permission_targets] == ["git status"]
    assert [(binding.name, binding.value) for binding in seen[0].operation.native_binding_ids] == [
        ("request_id", "permission"),
        ("message_id", "message"),
        ("call_id", "call"),
    ]
