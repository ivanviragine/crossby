"""Tests for :mod:`crossby.web.sessions` — request parsing and capability gating."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crossby.models.ai import AIToolID, EffortLevel
from crossby.utils.pty_runner import WindowSize
from crossby.web.sessions import (
    MAX_CONCURRENT_SESSIONS,
    Autonomy,
    LaunchRequest,
    LaunchValidationError,
    MultiplexedStream,
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
        """OpenCode declares no accept-edits tier."""
        with (
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="autonomy"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.OPENCODE, autonomy=Autonomy.ACCEPT_EDITS))
        spawn.assert_not_called()

    def test_autonomy_offered_matches_the_adapter(self) -> None:
        """The form is generated from capabilities, so it must track them.

        Pinning a tool's rungs in a test goes stale the moment an adapter gains
        one — this replaced a test asserting OpenCode had no YOLO mode, which it
        has since gained. Assert the derivation instead of the current values.
        """
        from crossby.ai_tools.base import AbstractAITool
        from crossby.web.sessions import _supported_autonomy

        for tool_id in (AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.OPENCODE):
            caps = AbstractAITool.get(tool_id).capabilities()
            offered = _supported_autonomy(caps)
            assert Autonomy.DEFAULT in offered, tool_id
            assert (Autonomy.PLAN in offered) is caps.plan_mode.supported, tool_id
            assert (Autonomy.ACCEPT_EDITS in offered) is caps.supports_accept_edits, tool_id
            assert (Autonomy.AUTO in offered) is caps.supports_auto, tool_id
            assert (Autonomy.YOLO in offered) is caps.supports_yolo, tool_id

    def test_rung_the_adapter_does_not_declare_is_refused(self, manager: SessionManager) -> None:
        """Capabilities are stubbed rather than taken from whatever is installed.

        Asserting against a real adapter made this pass locally and fail on CI,
        where no tool is on PATH and the adapter's own version gate answered
        first with a different error.
        """
        from crossby.ai_tools.base import AbstractAITool

        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        without_yolo = adapter.capabilities().model_copy(update={"supports_yolo": False})
        with (
            patch.object(type(adapter), "capabilities", lambda _self: without_yolo),
            patch("crossby.web.sessions.PtySession") as spawn,
            pytest.raises(LaunchValidationError, match="autonomy"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE, autonomy=Autonomy.YOLO))
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

    @pytest.mark.parametrize(
        "autonomy,expected",
        [
            (Autonomy.DEFAULT, {}),
            (Autonomy.PLAN, {"plan_mode": True}),
            (Autonomy.ACCEPT_EDITS, {"accept_edits": True}),
            (Autonomy.AUTO, {"auto": True}),
            (Autonomy.YOLO, {"yolo": True}),
        ],
    )
    def test_autonomy_reaches_the_adapter(
        self, manager: SessionManager, autonomy: Autonomy, expected: dict[str, bool]
    ) -> None:
        """Exactly one rung is set, and it is the one that was asked for.

        This asserts the flags handed to the adapter rather than the resulting
        argv: turning flags into argv is the adapter's contract, and asserting on
        argv made the test depend on a real tool of a sufficient version being
        installed — green locally, red on CI.
        """
        from crossby.ai_tools.base import AbstractAITool

        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        rungs = ("plan_mode", "accept_edits", "auto", "yolo")
        with (
            patch.object(type(adapter), "build_launch_command", return_value=["true"]) as build,
            patch("crossby.web.sessions.PtySession"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE, autonomy=autonomy))

        kwargs = build.call_args.kwargs
        assert {rung: kwargs[rung] for rung in rungs} == {
            rung: expected.get(rung, False) for rung in rungs
        }

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


class TestWorkingDirectoryResolution:
    """Sessions may run anywhere at or below an allowed root — and nowhere else.

    Only the operator sets the roots (``--path`` plus ``--allow-dir``); the
    browser picks within them. Containment is checked against the *resolved*
    path so a symlink cannot be used to step outside.
    """

    @pytest.fixture
    def tree(self, tmp_path: Path) -> Path:
        (tmp_path / "root" / "project" / "nested").mkdir(parents=True)
        (tmp_path / "root" / ".hidden").mkdir()
        (tmp_path / "root" / "file.txt").write_text("x", encoding="utf-8")
        (tmp_path / "outside" / "secret").mkdir(parents=True)
        return tmp_path

    @pytest.fixture
    def manager(self, tree: Path) -> SessionManager:
        return SessionManager(tree / "root")

    def test_default_is_the_project_root(self, manager: SessionManager, tree: Path) -> None:
        assert manager.resolve_workdir(None) == (tree / "root").resolve()

    def test_subdirectory_is_allowed(self, manager: SessionManager, tree: Path) -> None:
        target = tree / "root" / "project" / "nested"
        assert manager.resolve_workdir(str(target)) == target.resolve()

    def test_directory_outside_the_root_is_refused(
        self, manager: SessionManager, tree: Path
    ) -> None:
        with pytest.raises(LaunchValidationError, match="outside the directories"):
            manager.resolve_workdir(str(tree / "outside" / "secret"))

    def test_parent_traversal_is_refused(self, manager: SessionManager, tree: Path) -> None:
        with pytest.raises(LaunchValidationError, match="outside the directories"):
            manager.resolve_workdir(str(tree / "root" / ".." / "outside"))

    def test_symlink_escape_is_refused(self, manager: SessionManager, tree: Path) -> None:
        """Resolving first is what makes this a refusal rather than a hole."""
        link = tree / "root" / "escape"
        link.symlink_to(tree / "outside")
        with pytest.raises(LaunchValidationError, match="outside the directories"):
            manager.resolve_workdir(str(link))

    def test_missing_directory_is_refused(self, manager: SessionManager, tree: Path) -> None:
        with pytest.raises(LaunchValidationError, match="no such directory"):
            manager.resolve_workdir(str(tree / "root" / "nope"))

    def test_file_is_refused(self, manager: SessionManager, tree: Path) -> None:
        with pytest.raises(LaunchValidationError, match="not a directory"):
            manager.resolve_workdir(str(tree / "root" / "file.txt"))

    def test_extra_allowed_root_is_reachable(self, tree: Path) -> None:
        """``--allow-dir`` widens the boundary; nothing else does."""
        manager = SessionManager(tree / "root", [tree / "outside"])
        target = tree / "outside" / "secret"
        assert manager.resolve_workdir(str(target)) == target.resolve()

    def test_session_runs_in_the_requested_directory(
        self, manager: SessionManager, tree: Path
    ) -> None:
        from crossby.ai_tools.base import AbstractAITool

        target = tree / "root" / "project"
        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        # Stub the builder so this asserts directory forwarding only, and cannot
        # be broken by an unrelated change to Claude's command line.
        with (
            patch.object(type(adapter), "build_launch_command", return_value=["true"]),
            patch("crossby.web.sessions.PtySession") as spawn,
        ):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE, cwd=str(target)))
        assert spawn.call_args.kwargs["cwd"] == target.resolve()


class TestDirectoryBrowsing:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> Path:
        (tmp_path / "root" / "alpha").mkdir(parents=True)
        (tmp_path / "root" / "beta").mkdir()
        (tmp_path / "root" / ".git").mkdir()
        (tmp_path / "root" / "readme.md").write_text("x", encoding="utf-8")
        (tmp_path / "outside").mkdir()
        return tmp_path

    def test_lists_only_directories(self, tree: Path) -> None:
        listing = SessionManager(tree / "root").browse(None)
        assert listing["children"] == ["alpha", "beta"], "files and dotdirs are noise here"

    def test_parent_is_withheld_at_a_root(self, tree: Path) -> None:
        """Offering the parent would let the picker walk out of the boundary."""
        listing = SessionManager(tree / "root").browse(None)
        assert listing["parent"] is None

    def test_parent_is_offered_below_a_root(self, tree: Path) -> None:
        manager = SessionManager(tree / "root")
        listing = manager.browse(str(tree / "root" / "alpha"))
        assert listing["parent"] == str((tree / "root").resolve())

    def test_browsing_outside_a_root_is_refused(self, tree: Path) -> None:
        with pytest.raises(LaunchValidationError, match="outside the directories"):
            SessionManager(tree / "root").browse(str(tree / "outside"))

    def test_roots_are_reported(self, tree: Path) -> None:
        manager = SessionManager(tree / "root", [tree / "outside"])
        listing = manager.browse(None)
        assert listing["roots"] == [
            str((tree / "root").resolve()),
            str((tree / "outside").resolve()),
        ]


class TestConcurrencyInvariants:
    """Races that a single-threaded test would never reach."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> SessionManager:
        return SessionManager(tmp_path)

    def test_failed_creation_releases_its_reserved_slot(self, manager: SessionManager) -> None:
        """A slot is claimed before spawning; a failed spawn must give it back.

        Without this the limit ratchets down on every failure until no session
        can start at all.
        """
        from crossby.ai_tools.base import AbstractAITool

        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        with (
            patch.object(type(adapter), "build_launch_command", return_value=["true"]),
            patch("crossby.web.sessions.PtySession", side_effect=FileNotFoundError("boom")),
            pytest.raises(FileNotFoundError),
        ):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE))
        assert manager._reserved == 0

    def test_reaped_sessions_are_closed_not_just_forgotten(self, manager: SessionManager) -> None:
        """Dropping the reference alone leaks the master fd and reader thread."""
        from crossby.ai_tools.base import AbstractAITool

        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        finished = MagicMock()
        finished.running = False
        finished.id = "dead-session"
        manager._sessions["dead-session"] = finished

        with (
            patch.object(type(adapter), "build_launch_command", return_value=["true"]),
            patch("crossby.web.sessions.PtySession"),
        ):
            manager.create(LaunchRequest(tool=AIToolID.CLAUDE))

        finished.close.assert_called_once()
        assert "dead-session" not in manager._sessions

    def test_a_session_is_pumped_once_even_if_claimed_twice(self, manager: SessionManager) -> None:
        """`__enter__`'s snapshot and the creation listener can both claim one
        session; two pumps would deliver its output twice."""
        stream = MultiplexedStream(manager)
        session = MagicMock()
        session.id = "session-1"
        session.attach.return_value = (b"", None)

        stream._start_pump(session)
        stream._start_pump(session)
        for pump in stream._pumps:
            pump.join(timeout=2)

        assert len(stream._pumps) == 1
        assert session.attach.call_count == 1
