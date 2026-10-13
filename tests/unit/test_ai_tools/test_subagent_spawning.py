"""The readiness boundary must remain fail closed without consumer allowlists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from crossby.ai_tools import (
    AbstractAITool,
    HeadlessSessionRequest,
    SubagentSpawnEvidence,
    SubagentSpawnReason,
    SubagentSpawnState,
    SubagentSpawnStatus,
    SubagentSpawnSupport,
)
from crossby.ai_tools.subagent_context import SubagentLaunchContext
from crossby.models.ai import AIToolID


@pytest.mark.parametrize("state", list(SubagentSpawnState))
def test_only_available_is_truthy_and_serializes(state: SubagentSpawnState, tmp_path: Path) -> None:
    status = SubagentSpawnStatus(
        tool_id=AIToolID.CODEX,
        state=state,
        reason_code=SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED,
        reason="Readiness was checked.",
        evidence=SubagentSpawnEvidence.EFFECTIVE_CONFIG,
        working_dir=tmp_path,
        headless=True,
        context_fingerprint="opaque",
    )
    assert bool(status) is (state is SubagentSpawnState.AVAILABLE)
    assert status.is_available is bool(status)
    payload = json.loads(status.model_dump_json())
    assert payload["working_dir"] == str(tmp_path)
    assert payload["state"] == state.value
    assert SubagentSpawnStatus.model_validate(payload) == status
    with pytest.raises(ValidationError):
        status.state = SubagentSpawnState.AVAILABLE  # type: ignore[misc]


@pytest.mark.parametrize("tool", list(AIToolID))
def test_default_declaration_is_unaudited_without_a_headless_shortcut(
    tool: AIToolID, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AbstractAITool.get(tool)
    capabilities = adapter.capabilities().model_copy(
        update={"subagent_spawning": SubagentSpawnSupport.UNAUDITED, "supports_headless": False}
    )
    monkeypatch.setattr(adapter, "capabilities", lambda: capabilities)
    monkeypatch.setattr(
        SubagentLaunchContext,
        "binary_path",
        lambda *_: pytest.fail("unaudited adapters must not probe installed binaries"),
    )
    result = adapter.subagent_spawning_status(working_dir=tmp_path)
    assert result.state is SubagentSpawnState.UNKNOWN
    assert result.reason_code is SubagentSpawnReason.ADAPTER_NOT_AUDITED
    assert result.working_dir == tmp_path.resolve()
    assert not result


@pytest.mark.parametrize(
    ("support", "headless", "binary", "expected", "reason"),
    [
        (
            SubagentSpawnSupport.UNSUPPORTED,
            True,
            None,
            SubagentSpawnState.UNSUPPORTED,
            SubagentSpawnReason.ADAPTER_DECLARES_UNSUPPORTED,
        ),
        (
            SubagentSpawnSupport.SUPPORTED,
            False,
            None,
            SubagentSpawnState.UNSUPPORTED,
            SubagentSpawnReason.NO_HEADLESS_MODE,
        ),
        (
            SubagentSpawnSupport.SUPPORTED,
            True,
            None,
            SubagentSpawnState.UNKNOWN,
            SubagentSpawnReason.TOOL_NOT_INSTALLED,
        ),
        (
            SubagentSpawnSupport.SUPPORTED,
            True,
            "/fixture/codex",
            SubagentSpawnState.UNKNOWN,
            SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED,
        ),
    ],
)
def test_audited_base_never_assumes_runtime_availability(
    support: SubagentSpawnSupport,
    headless: bool,
    binary: str | None,
    expected: SubagentSpawnState,
    reason: SubagentSpawnReason,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AbstractAITool.get("vscode")
    caps = adapter.capabilities().model_copy(
        update={"subagent_spawning": support, "supports_headless": headless}
    )
    monkeypatch.setattr(adapter, "capabilities", lambda: caps)
    monkeypatch.setattr(SubagentLaunchContext, "binary_path", lambda *_: binary)
    result = adapter.subagent_spawning_status(working_dir=tmp_path)
    assert result.state is expected
    assert result.reason_code is reason
    assert not result


def test_context_uses_child_cwd_and_environment_without_exposing_prompt_or_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    bin_dir = child / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "codex"
    executable.write_text("fixture")
    executable.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    environment = {"PATH": "bin", "HOME": str(child), "SECRET_TOKEN": "private-credential"}
    request = HeadlessSessionRequest(prompt="private-prompt", working_dir=child)
    context = SubagentLaunchContext.normalize(
        working_dir=None,
        headless=True,
        launch_args=(),
        environment=environment,
        request=request,
    )
    assert context.binary_path("codex") == str(executable)
    changed_prompt = SubagentLaunchContext.normalize(
        working_dir=None,
        headless=True,
        launch_args=(),
        environment=environment,
        request=request.model_copy(update={"prompt": "another prompt"}),
    )
    assert context.fingerprint == changed_prompt.fingerprint
    changed_env = SubagentLaunchContext.normalize(
        working_dir=None,
        headless=True,
        launch_args=(),
        environment={**environment, "CODEX_HOME": "alternate"},
        request=request,
    )
    assert changed_env.fingerprint != context.fingerprint
    status = context.status(
        AIToolID.CODEX,
        SubagentSpawnState.UNKNOWN,
        SubagentSpawnReason.EFFECTIVE_CONTEXT_UNRESOLVED,
        "Unresolved.",
    )
    serialized = status.model_dump_json()
    assert "private-prompt" not in serialized
    assert "private-credential" not in serialized
    environment["SECRET_TOKEN"] = "changed"
    assert context.environment["SECRET_TOKEN"] == "private-credential"


@pytest.mark.parametrize("conflict", ["cwd", "args", "mode"])
def test_conflicting_managed_inputs_fail_closed(
    conflict: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AbstractAITool.get("codex")
    caps = adapter.capabilities().model_copy(
        update={"subagent_spawning": SubagentSpawnSupport.SUPPORTED}
    )
    monkeypatch.setattr(adapter, "capabilities", lambda: caps)
    result = adapter.subagent_spawning_status(
        request=HeadlessSessionRequest(prompt="task", working_dir=tmp_path),
        working_dir=tmp_path / "elsewhere" if conflict == "cwd" else None,
        launch_args=("--json",) if conflict == "args" else (),
        headless=conflict != "mode",
    )
    assert result.reason_code is SubagentSpawnReason.CONTEXT_UNSUPPORTED
    assert not result
