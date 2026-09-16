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
    Autonomy,
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
        assert request.autonomy is Autonomy.DEFAULT
        assert request.yolo is False

    def test_full_payload(self) -> None:
        request = LaunchRequest.from_payload(
            {
                "tool": "claude",
                "model": "claude-opus-5",
                "effort": "high",
                "autonomy": "yolo",
                "initial_message": "hi",
                "cols": 120,
                "rows": 40,
            }
        )
        assert request.model == "claude-opus-5"
        assert request.effort is EffortLevel.HIGH
        assert request.autonomy is Autonomy.YOLO
        assert request.yolo is True
        assert request.plan_mode is False
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
            "supports_initial_message",
            "autonomy",
            "supports_network_access",
            "supports_sandbox_toggle",
        }
        assert "default" in entry["autonomy"]


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

    def test_unsupported_autonomy_is_refused_before_spawn(self, manager: SessionManager) -> None:
        """OpenCode is a terminal tool that declares no YOLO mode."""
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="autonomy"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.OPENCODE, autonomy=Autonomy.YOLO))
        spawn.assert_not_called()

    def test_plan_mode_refused_where_unsupported(self, manager: SessionManager) -> None:
        """Codex declares no native plan mode."""
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="autonomy"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.CODEX, autonomy=Autonomy.PLAN))
        spawn.assert_not_called()

    def test_network_access_refused_where_unsupported(self, manager: SessionManager) -> None:
        """Only Codex has a sandbox network opt-in."""
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="network"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE, network_access=True))
        spawn.assert_not_called()

    def test_disabling_sandbox_refused_where_unsupported(self, manager: SessionManager) -> None:
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="sandbox"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE, sandbox=False))
        spawn.assert_not_called()

    def test_autonomy_reaches_the_command(self, manager: SessionManager) -> None:
        """A selected rung must actually appear in the tool's argv."""
        with patch("crossby.web.sessions.PtySession") as spawn:
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE, autonomy=Autonomy.PLAN))
        argv = spawn.call_args.args[0]
        assert any("plan" in part for part in argv), argv

    def test_unknown_autonomy_is_rejected(self) -> None:
        with pytest.raises(LaunchValidationError, match="unknown autonomy"):
            LaunchRequest.from_payload({"tool": "claude", "autonomy": "chaos"})

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
