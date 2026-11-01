"""Unattended Claude Code session resume and explicit session IDs.

A ``claude`` started inside another Claude Code session inherits the parent's
session ID unless it is given one, so two agents could otherwise share a
session. Every new unattended session therefore gets ``--session-id`` and a
resumed one ``--resume``; the ID Claude reports back must match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools import claude as claude_module
from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import HeadlessRequestError
from crossby.models.ai import (
    AIToolID,
    HeadlessInteractionMode,
    HeadlessNativeOutput,
    HeadlessTerminalStatus,
)
from crossby.utils.versioning import BinaryVersion
from tests.unit.test_ai_tools.test_headless_adapters import (
    CLAUDE_RESULT,
    _adapter,
    _command,
    _fake_cli,
    _request,
)

SESSION = CLAUDE_RESULT["session_id"]
OTHER = "5f0c1a7e-2b7d-4c55-9a3e-0f6f7d2c9b11"


def _recording_adapter(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | None = None, *, text: str = ""
) -> tuple[AbstractAITool, list[dict[str, Any]]]:
    stdout = json.dumps(payload) if payload is not None else text
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=stdout))
    # Resume is verified from 2.1.316, later than the unattended floor.
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion((2, 1, 316), "2.1.316"),
        raising=True,
    )
    stub = type(adapter)._headless_command  # type: ignore[attr-defined]
    calls: list[dict[str, Any]] = []

    def record(self: AbstractAITool, request: Any, **kwargs: Any) -> list[str]:
        calls.append({"resume_id": request.resume_id, **kwargs})
        return list(stub(self, request, **kwargs))

    monkeypatch.setattr(type(adapter), "_headless_command", record, raising=True)
    return adapter, calls


def test_unattended_profile_declares_resume() -> None:
    headless = AbstractAITool.get(AIToolID.CLAUDE).capabilities().headless
    assert headless.profile_for(HeadlessInteractionMode.UNATTENDED).supports_resume


def test_new_session_argv_pins_the_session_id(tmp_path: Path) -> None:
    command = _command(AIToolID.CLAUDE, _request(tmp_path), session_id=SESSION)
    assert command[command.index("--session-id") + 1] == SESSION
    assert "--resume" not in command


def test_resumed_session_argv_uses_resume_and_no_new_id(tmp_path: Path) -> None:
    command = _command(AIToolID.CLAUDE, _request(tmp_path, resume_id=SESSION), session_id=OTHER)
    assert command[command.index("--resume") + 1] == SESSION
    assert "--session-id" not in command


def test_every_new_session_gets_a_fresh_pinned_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, calls = _recording_adapter(monkeypatch, CLAUDE_RESULT)
    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON)
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.session_id == SESSION
    # Validation sizes the argv with a same-length placeholder; the run pins the real ID.
    assert calls[-1] == {"resume_id": None, "session_id": SESSION}
    assert all(
        call["session_id"] in {SESSION, claude_module._PLACEHOLDER_SESSION_ID} for call in calls
    )


def test_fresh_ids_differ_between_sessions() -> None:
    assert claude_module._new_session_id() != claude_module._new_session_id()


def test_resume_continues_the_requested_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, calls = _recording_adapter(monkeypatch, CLAUDE_RESULT)
    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, resume_id=SESSION)
    )
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.session_id == SESSION
    assert calls[0]["resume_id"] == SESSION


def test_a_different_reported_session_is_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _calls = _recording_adapter(monkeypatch, CLAUDE_RESULT)
    result = adapter.run_headless_session(
        _request(tmp_path, native_output=HeadlessNativeOutput.JSON, resume_id=OTHER)
    )
    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.session_id is None
    assert any("different session" in warning for warning in result.warnings)


def test_text_output_reports_the_pinned_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _calls = _recording_adapter(monkeypatch, text="OK\n")
    result = adapter.run_headless_session(_request(tmp_path, resume_id=SESSION))
    assert result.status is HeadlessTerminalStatus.SUCCEEDED
    assert result.final_text == "OK"
    assert result.session_id == SESSION


@pytest.mark.parametrize("bad", ["latest", "not-a-uuid", "1492F340-106A-4548-B7BA-6C3ABA132018x"])
def test_a_non_uuid_resume_id_is_rejected_before_spawning(tmp_path: Path, bad: str) -> None:
    adapter = AbstractAITool.get(AIToolID.CLAUDE)
    with pytest.raises(HeadlessRequestError, match="UUID"):
        adapter._validate_headless_requirements(_request(tmp_path, resume_id=bad))


@pytest.mark.parametrize(
    ("version", "supported"),
    [("2.1.316", True), ("2.2.0", True), ("2.1.315", False), ("2.1.263", False), (None, False)],
)
def test_unattended_resume_is_limited_to_the_verified_build(
    version: str | None, supported: bool
) -> None:
    adapter = AbstractAITool.get(AIToolID.CLAUDE)
    assert adapter.headless_resume_supported(version) is supported


def test_resume_on_an_unverified_build_is_refused_before_spawning(tmp_path: Path) -> None:
    from crossby.ai_tools.headless import HeadlessUnsupportedError

    adapter = AbstractAITool.get(AIToolID.CLAUDE)
    old = BinaryVersion((2, 1, 300), "2.1.300")
    with pytest.raises(HeadlessUnsupportedError, match=r"verified on 2\.1\.316"):
        adapter._validate_headless_mode_version(_request(tmp_path, resume_id=SESSION), old)
    # The same build still runs new sessions, and a verified build resumes.
    adapter._validate_headless_mode_version(_request(tmp_path), old)
    adapter._validate_headless_mode_version(
        _request(tmp_path, resume_id=SESSION), BinaryVersion((2, 1, 316), "2.1.316")
    )


def test_tools_without_resume_report_none() -> None:
    assert not AbstractAITool.get(AIToolID.CURSOR).headless_resume_supported("9.9.9")
