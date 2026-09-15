"""Tests for :mod:`crossby.web.sessions` — request parsing and capability gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from crossby.models.ai import AIToolID, EffortLevel
from crossby.utils.pty_runner import WindowSize
from crossby.web.sessions import (
    MAX_CONCURRENT_SESSIONS,
    LaunchRequest,
    LaunchValidationError,
    SessionManager,
    SessionNotFoundError,
    describe_tools,
    embeddable_tools,
    window_size_from_payload,
)


class TestLaunchRequestParsing:
    def test_minimal_payload(self) -> None:
        request = LaunchRequest.from_payload({"tool": "claude"})
        assert request.tool is AIToolID.CLAUDE
        assert request.model is None
        assert request.effort is None
        assert request.yolo is False

    def test_full_payload(self) -> None:
        request = LaunchRequest.from_payload(
            {
                "tool": "claude",
                "model": "claude-opus-5",
                "effort": "high",
                "yolo": True,
                "initial_message": "hi",
                "cols": 120,
                "rows": 40,
            }
        )
        assert request.model == "claude-opus-5"
        assert request.effort is EffortLevel.HIGH
        assert request.yolo is True
        assert request.initial_message == "hi"
        assert request.size == WindowSize(cols=120, rows=40)

    def test_missing_tool_is_rejected(self) -> None:
        with pytest.raises(LaunchValidationError, match="'tool' is required"):
            LaunchRequest.from_payload({})

    def test_unknown_tool_is_rejected(self) -> None:
        with pytest.raises(LaunchValidationError, match="unknown tool"):
            LaunchRequest.from_payload({"tool": "not-a-tool"})

    def test_unknown_effort_is_rejected(self) -> None:
        with pytest.raises(LaunchValidationError, match="unknown effort level"):
            LaunchRequest.from_payload({"tool": "claude", "effort": "turbo"})

    def test_empty_strings_become_none(self) -> None:
        request = LaunchRequest.from_payload(
            {"tool": "claude", "model": "", "effort": "", "initial_message": ""}
        )
        assert request.model is None
        assert request.effort is None
        assert request.initial_message is None

    @pytest.mark.parametrize("payload", [{"cols": "80"}, {"rows": None}, {"cols": 1.5}])
    def test_non_integer_dimensions_are_rejected(self, payload: dict[str, Any]) -> None:
        with pytest.raises(LaunchValidationError, match="must be integers"):
            LaunchRequest.from_payload({"tool": "claude", **payload})

    def test_out_of_range_dimensions_are_rejected(self) -> None:
        with pytest.raises(LaunchValidationError, match="between 1 and 10000"):
            window_size_from_payload({"cols": 0, "rows": 24})


class TestToolDiscovery:
    def test_gui_tools_are_never_embeddable(self) -> None:
        """VS Code and the Antigravity IDE open a window; there is nothing to embed."""
        with patch(
            "crossby.web.sessions.AbstractAITool.detect_installed",
            return_value=[AIToolID.VSCODE, AIToolID.CLAUDE],
        ):
            assert embeddable_tools() == [AIToolID.CLAUDE]

    def test_description_drives_the_launch_form(self) -> None:
        with patch(
            "crossby.web.sessions.AbstractAITool.detect_installed",
            return_value=[AIToolID.CLAUDE],
        ):
            described = describe_tools()
        assert len(described) == 1
        entry = described[0]
        assert entry["id"] == "claude"
        assert entry["display_name"]
        assert entry["models"]
        assert set(entry) == {
            "id",
            "display_name",
            "supports_model_flag",
            "models",
            "supports_effort",
            "supported_efforts",
            "supports_yolo",
            "supports_initial_message",
        }


class TestSessionManagerValidation:
    """Capability gating happens before any process is created."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> SessionManager:
        return SessionManager(tmp_path)

    def test_gui_tool_is_refused(self, manager: SessionManager) -> None:
        with pytest.raises(LaunchValidationError, match="GUI tool"):
            manager.create(LaunchRequest(tool=AIToolID.VSCODE))

    def test_unsupported_effort_is_refused(self, manager: SessionManager) -> None:
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="effort"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.COPILOT, effort=EffortLevel.HIGH))
        spawn.assert_not_called()

    def test_unsupported_yolo_is_refused_before_spawn(self, manager: SessionManager) -> None:
        """OpenCode is a terminal tool that declares no YOLO mode."""
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="YOLO"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.OPENCODE, yolo=True))
        spawn.assert_not_called()

    def test_out_of_range_effort_is_refused(self, manager: SessionManager) -> None:
        """Copilot supports no effort levels at all."""
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="effort"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.COPILOT, effort=EffortLevel.LOW))
        spawn.assert_not_called()

    def test_session_cap_is_enforced(self, manager: SessionManager) -> None:
        live = [_FakeSession(str(i)) for i in range(MAX_CONCURRENT_SESSIONS)]
        for session in live:
            manager._sessions[session.id] = session  # type: ignore[assignment]
        with pytest.raises(LaunchValidationError, match="too many live sessions"):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE))

    def test_unknown_session_lookup_raises(self, manager: SessionManager) -> None:
        with pytest.raises(SessionNotFoundError):
            manager.get("nope")

    def test_sessions_are_pinned_to_the_project_root(self, tmp_path: Path) -> None:
        """The browser never chooses a working directory."""
        manager = SessionManager(tmp_path)
        assert manager.project_root == tmp_path.resolve()


class _FakeSession:
    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.running = True
