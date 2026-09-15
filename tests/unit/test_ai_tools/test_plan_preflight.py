"""Public collected-session preflight boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from crossby.ai_tools import (
    AbstractAITool,
    PlanCommandPolicy,
    PlanCommandPolicyUnsupportedError,
    PlanPreflightCheck,
    PlanPreflightDeferredCheck,
    PlanSessionPreflight,
    PlanSessionRequest,
    PlanSessionResult,
    PlanSessionUnsupportedError,
    PlanTransportError,
    preflight_plan_session,
)
from crossby.models.ai import (
    AIToolID,
    EffortLevel,
    PlanArtifactSource,
    PlanSessionBinding,
)
from crossby.utils.versioning import BinaryVersion

VERSION = BinaryVersion((9999, 0, 0), "fixture-tool 9999.0.0+exact")


def _request(path: Path, **updates: Any) -> PlanSessionRequest:
    return PlanSessionRequest(prompt="Create a plan", working_dir=path, **updates)


@pytest.fixture(autouse=True)
def _version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda _binary, **_kwargs: VERSION,
    )


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_preflight_rejects_non_positive_or_non_finite_timeout(
    timeout_seconds: float,
    tmp_path: Path,
) -> None:
    adapter = AbstractAITool.get(AIToolID.CODEX)

    with (
        patch("crossby.utils.versioning.detect_binary_version_info") as probe,
        pytest.raises(ValueError, match="timeout must be positive and finite"),
    ):
        adapter.preflight_plan_session(
            _request(tmp_path / "future"),
            timeout_seconds=timeout_seconds,
        )

    probe.assert_not_called()


@pytest.mark.parametrize("tool_id", [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.OPENCODE])
def test_policy_preflight_succeeds_without_creating_future_directory(
    tool_id: AIToolID, tmp_path: Path
) -> None:
    future = tmp_path / "future-worktree"
    output = future / "plans" if tool_id is AIToolID.CLAUDE else None

    result = preflight_plan_session(
        tool_id,
        _request(
            future,
            plan_output_dir=output,
            command_policy=PlanCommandPolicy(allowed_commands=("git status:*",)),
        ),
    )

    assert isinstance(result, PlanSessionPreflight)
    assert result.tool is tool_id
    assert result.detected_version == VERSION.text
    assert result.normalized_version == VERSION.normalized
    assert set(result.checked) == set(PlanPreflightCheck)
    assert set(result.deferred) == set(PlanPreflightDeferredCheck)
    assert not future.exists()


@pytest.mark.parametrize(
    ("tool_id", "updates"),
    [
        (
            AIToolID.CURSOR,
            {"model": "gpt-5.3-codex", "effort": EffortLevel.MEDIUM},
        ),
        (
            AIToolID.ANTIGRAVITY_CLI,
            {"model": "gemini-3.8-flash", "effort": EffortLevel.MEDIUM},
        ),
    ],
)
def test_unsupported_explicit_policy_fails_before_probe(
    tool_id: AIToolID,
    updates: dict[str, Any],
    tmp_path: Path,
) -> None:
    with (
        patch("crossby.utils.versioning.detect_binary_version_info") as probe,
        pytest.raises(PlanCommandPolicyUnsupportedError, match="command_policy"),
    ):
        preflight_plan_session(
            tool_id,
            _request(
                tmp_path / "future",
                command_policy=PlanCommandPolicy(allowed_commands=("git status",)),
                **updates,
            ),
        )
    probe.assert_not_called()


def test_static_model_requirements_fail_before_probe(tmp_path: Path) -> None:
    with (
        patch("crossby.utils.versioning.detect_binary_version_info") as probe,
        pytest.raises(PlanSessionUnsupportedError, match="explicit model and effort"),
    ):
        preflight_plan_session(AIToolID.CURSOR, _request(tmp_path / "future"))
    probe.assert_not_called()


@pytest.mark.parametrize(
    "detected",
    [None, BinaryVersion((0, 1, 0), "codex-cli 0.1.0")],
    ids=("unknown", "below-floor"),
)
def test_unknown_or_below_floor_version_fails_without_artifacts(
    detected: BinaryVersion | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = tmp_path / "future"
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda _binary, **_kwargs: detected,
    )

    with pytest.raises(PlanSessionUnsupportedError, match="installed version"):
        preflight_plan_session(AIToolID.CODEX, _request(future))
    assert not future.exists()


def test_probe_that_exhausts_preflight_deadline_is_transport_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((100.0, 105.0))
    monkeypatch.setattr("crossby.ai_tools.base.monotonic", lambda: next(clock))
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda _binary, **_kwargs: None,
    )

    with pytest.raises(PlanTransportError, match="timed out during version probing"):
        preflight_plan_session(
            AIToolID.CODEX,
            _request(tmp_path / "future"),
            timeout_seconds=5.0,
        )


def test_successful_probe_that_exhausts_preflight_deadline_is_transport_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((100.0, 105.0))
    monkeypatch.setattr("crossby.ai_tools.base.monotonic", lambda: next(clock))

    with pytest.raises(PlanTransportError, match="timed out during version probing"):
        preflight_plan_session(
            AIToolID.CODEX,
            _request(tmp_path / "future"),
            timeout_seconds=5.0,
        )


def test_preflight_does_not_replace_mandatory_runtime_validation(
    tmp_path: Path,
) -> None:
    future = tmp_path / "future"
    adapter = AbstractAITool.get(AIToolID.CODEX)
    request = _request(future)
    adapter.preflight_plan_session(request)

    with (
        patch.object(adapter, "_run_plan_session") as collect,
        pytest.raises(PlanSessionUnsupportedError, match="existing working directory"),
    ):
        adapter.run_plan_session(request)
    collect.assert_not_called()


def test_consumer_can_preflight_then_create_workspace_and_submit_public_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = tmp_path / "future"
    request = _request(
        future,
        command_policy=PlanCommandPolicy(allowed_commands=("git diff:*",)),
    )
    probes: list[str] = []

    def detect(binary: str, **_kwargs: Any) -> BinaryVersion:
        probes.append(binary)
        return VERSION

    monkeypatch.setattr("crossby.utils.versioning.detect_binary_version_info", detect)
    adapter = AbstractAITool.get(AIToolID.CODEX)
    adapter.preflight_plan_session(request)
    future.mkdir()
    provider_effect = future / "provider-effect"
    provider_effect.write_text("created after preflight", encoding="utf-8")
    native = PlanSessionResult(
        tool=AIToolID.CODEX,
        version=VERSION.text,
        plan="# Plan",
        session_id="thread",
        native_mode="collaborationMode.mode=plan",
        artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
        binding=PlanSessionBinding.THREAD_TURN_IDS,
        exit_code=0,
        thread_id="thread",
        turn_id="turn",
        artifact_id="plan",
    )

    with patch.object(adapter, "_run_plan_session", return_value=native) as collect:
        result = adapter.run_plan_session(request)

    assert result == native
    assert collect.call_args.args[0].command_policy == request.command_policy
    assert probes == ["codex", "codex"]
    assert provider_effect.read_text(encoding="utf-8") == "created after preflight"
