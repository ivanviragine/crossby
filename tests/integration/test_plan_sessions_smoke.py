"""Opt-in authenticated smoke tests for installed native planning harnesses.

Run only explicitly selected, already-authenticated CLIs in disposable Git
repositories, for example::

    CROSSBY_PLAN_SMOKE_TOOLS=codex,cursor \
    CROSSBY_PLAN_SMOKE_ANSWER="Use the existing public API" \
      uv run pytest -s tests/integration/test_plan_sessions_smoke.py

Each selected CLI must already be authenticated in the invoking user's normal
configuration. OpenCode additionally needs a configured provider/model. These
tests can consume paid model tokens and are therefore never selected by the
default suite.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools import (
    AbstractAITool,
    PlanInteraction,
    PlanInteractionKind,
    PlanInteractionOutcome,
    PlanInteractionResponse,
    PlanQuestionOption,
    PlanSessionRequest,
)
from crossby.models.ai import AIToolID, EffortLevel, PlanApprovalPolicy

SELECTED = {
    value.strip()
    for value in os.environ.get("CROSSBY_PLAN_SMOKE_TOOLS", "").split(",")
    if value.strip()
}

TOOLS = (
    AIToolID.CLAUDE,
    AIToolID.CODEX,
    AIToolID.CURSOR,
    AIToolID.COPILOT,
    AIToolID.OPENCODE,
    AIToolID.ANTIGRAVITY_CLI,
)


def _answer_interaction(interaction: PlanInteraction) -> PlanInteractionResponse:
    if interaction.kind in {
        PlanInteractionKind.PLAN_APPROVAL,
        PlanInteractionKind.PERMISSION,
    }:
        rejecting = next(
            (
                option.option_id
                for option in interaction.options
                if any(
                    word in f"{option.option_id} {option.label}".lower()
                    for word in ("deny", "reject", "cancel", "decline")
                )
            ),
            None,
        )
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.DENIED,
            option_id=rejecting,
        )
    if interaction.options:
        return PlanInteractionResponse(
            outcome=PlanInteractionOutcome.ANSWERED,
            option_id=interaction.options[0].option_id,
        )
    configured = os.environ.get("CROSSBY_PLAN_SMOKE_ANSWER", "").strip()
    if not configured:
        pytest.fail(
            "the harness asked a planning question; set CROSSBY_PLAN_SMOKE_ANSWER to an "
            "explicit operator-provided response"
        )
    return PlanInteractionResponse(
        outcome=PlanInteractionOutcome.ANSWERED,
        answer=configured,
    )


def test_smoke_callback_selects_first_native_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CROSSBY_PLAN_SMOKE_ANSWER", "free text is not a native option")
    response = _answer_interaction(
        PlanInteraction(
            kind=PlanInteractionKind.QUESTION,
            question_id="scope",
            prompt="Which scope should the plan cover?",
            options=(
                PlanQuestionOption(option_id="api", label="Public API"),
                PlanQuestionOption(option_id="cli", label="CLI only"),
            ),
            session_id="smoke-session",
        )
    )

    assert response == PlanInteractionResponse(
        outcome=PlanInteractionOutcome.ANSWERED,
        option_id="api",
    )


@pytest.mark.parametrize("tool_id", TOOLS)
def test_authenticated_native_plan_collection(tool_id: AIToolID, tmp_path: Path) -> None:
    if tool_id.value not in SELECTED:
        pytest.skip("set CROSSBY_PLAN_SMOKE_TOOLS to opt into paid authenticated smoke tests")
    adapter = AbstractAITool.get(tool_id)
    if shutil.which(adapter.capabilities().binary) is None:
        pytest.skip(f"{adapter.capabilities().binary} is not installed")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "crossby-smoke@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Crossby Smoke"], cwd=tmp_path, check=True)

    sentinel = tmp_path / "copilot-must-not-write.txt"
    prompt = "Produce a short Markdown implementation plan for adding a README heading."
    if tool_id is AIToolID.COPILOT:
        prompt = (
            f"Attempt to create {sentinel.name} containing 'unsafe', then produce the plan. "
            "Do not implement any other change."
        )
    else:
        prompt += " Do not edit files or implement the plan."

    request_options: dict[str, Any] = {}
    if tool_id is AIToolID.COPILOT:
        request_options["approval_policy"] = PlanApprovalPolicy.NEVER
    elif tool_id is AIToolID.CURSOR:
        request_options.update(model="sonnet-4.6", effort=EffortLevel.MEDIUM)
    elif tool_id is AIToolID.ANTIGRAVITY_CLI:
        request_options.update(model="gemini-3.8-flash", effort=EffortLevel.MEDIUM)

    result = adapter.run_plan_session(
        PlanSessionRequest(
            prompt=prompt,
            working_dir=tmp_path,
            timeout_seconds=300,
            **request_options,
        ),
        _answer_interaction,
    )

    assert result.plan.strip()
    assert result.version.strip()
    assert result.native_mode.strip()
    assert result.session_id.strip()
    assert not sentinel.exists()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if tool_id is AIToolID.CLAUDE:
        assert status and all(".crossby/plan-sessions/" in line for line in status)
    else:
        assert status == []
