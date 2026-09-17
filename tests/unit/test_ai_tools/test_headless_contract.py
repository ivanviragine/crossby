"""Public model and compatibility coverage for managed headless sessions."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from crossby.ai_tools import (
    AbstractAITool,
    HeadlessCapability,
    HeadlessEvent,
    HeadlessEventKind,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessNativeTransport,
    HeadlessPromptTransport,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    PlanInteraction,
    PlanQuestionOption,
    SessionInteraction,
    SessionQuestionOption,
)
from crossby.models.ai import AIToolID


def _terminal(status: HeadlessTerminalStatus) -> HeadlessEvent:
    return HeadlessEvent(
        sequence=1,
        kind=HeadlessEventKind.TERMINAL,
        terminal_status=status,
    )


def test_public_models_forbid_unknown_fields_and_are_frozen(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HeadlessSessionRequest(prompt="task", working_dir=tmp_path, unknown=True)

    request = HeadlessSessionRequest(prompt="task", working_dir=tmp_path)
    with pytest.raises(ValidationError, match="frozen"):
        request.prompt = "changed"


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_request_timeouts_must_be_positive_and_finite(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValidationError, match="positive and finite"):
        HeadlessSessionRequest(prompt="task", working_dir=tmp_path, timeout_seconds=value)


def test_native_output_and_response_schema_are_independent(tmp_path: Path) -> None:
    request = HeadlessSessionRequest(
        prompt="task",
        working_dir=tmp_path,
        native_output=HeadlessNativeOutput.TEXT,
        response_schema={"type": "object"},
    )

    assert request.native_output is HeadlessNativeOutput.TEXT
    assert request.response_schema == {"type": "object"}


def test_terminal_event_requires_status_and_nonterminal_forbids_it() -> None:
    with pytest.raises(ValidationError, match="only terminal events"):
        HeadlessEvent(sequence=1, kind=HeadlessEventKind.TERMINAL)
    with pytest.raises(ValidationError, match="only terminal events"):
        HeadlessEvent(
            sequence=1,
            kind=HeadlessEventKind.PROGRESS,
            terminal_status=HeadlessTerminalStatus.SUCCEEDED,
        )


def test_terminal_result_requires_one_final_matching_terminal_event() -> None:
    with pytest.raises(ValidationError, match="terminal headless result"):
        HeadlessSessionResult(
            tool=AIToolID.CODEX,
            version="1.0.0",
            status=HeadlessTerminalStatus.FAILED,
            duration_seconds=0,
        )

    with pytest.raises(ValidationError, match="must agree"):
        HeadlessSessionResult(
            tool=AIToolID.CODEX,
            version="1.0.0",
            status=HeadlessTerminalStatus.FAILED,
            events=(_terminal(HeadlessTerminalStatus.SUCCEEDED),),
            duration_seconds=0,
        )


def test_json_null_is_distinct_from_missing_final_output() -> None:
    result = HeadlessSessionResult(
        tool=AIToolID.CODEX,
        version="1.0.0",
        status=HeadlessTerminalStatus.SUCCEEDED,
        events=(_terminal(HeadlessTerminalStatus.SUCCEEDED),),
        final_json=None,
        duration_seconds=0,
    )

    assert result.final_json is None
    assert result.final_json_present


def test_partial_result_cannot_expose_terminal_or_final_output() -> None:
    partial = HeadlessSessionResult(
        tool=AIToolID.CODEX,
        version="1.0.0",
        status=HeadlessTerminalStatus.FAILED,
        duration_seconds=0,
        is_partial=True,
    )
    assert partial.events == ()

    with pytest.raises(ValidationError, match="safe partial"):
        HeadlessSessionResult(
            tool=AIToolID.CODEX,
            version="1.0.0",
            status=HeadlessTerminalStatus.FAILED,
            events=(_terminal(HeadlessTerminalStatus.FAILED),),
            duration_seconds=0,
            is_partial=True,
        )


def test_managed_capability_defaults_unavailable_and_rejects_false_advertising() -> None:
    assert not HeadlessCapability().managed_supported
    with pytest.raises(ValidationError, match="unavailable"):
        HeadlessCapability(supports_response_schema=True)

    capability = HeadlessCapability(
        transport=HeadlessNativeTransport.SUBPROCESS,
        prompt_transport=HeadlessPromptTransport.STDIN,
        native_outputs=(HeadlessNativeOutput.TEXT,),
        interaction_modes=(HeadlessInteractionMode.UNATTENDED,),
        verified_version="1.0.0",
    )
    assert capability.managed_supported


def test_real_adapters_do_not_advertise_managed_support_yet() -> None:
    capabilities = [
        AbstractAITool.get(tool).capabilities() for tool in AbstractAITool.available_tools()
    ]

    assert any(capability.supports_headless for capability in capabilities)
    assert all(not capability.supports_managed_headless_session for capability in capabilities)


def test_generalized_session_types_preserve_plan_import_identity() -> None:
    assert SessionInteraction is PlanInteraction
    assert SessionQuestionOption is PlanQuestionOption
