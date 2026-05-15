"""Tests for the top-level ``crossby`` wizard menu in ``cli.main``.

Mocking strategy: the menu dispatches by calling each subcommand's top-level
Python function directly (not ``ctx.invoke``), so tests patch the imported
function reference inside ``crossby.cli.main`` — e.g.
``crossby.cli.main.launch`` — and assert on the captured keyword arguments.
``crossby.ui.prompts.is_tty`` and ``prompts.menu`` / ``prompts.select`` /
``prompts.input_prompt`` are patched at their point of definition so lazy
imports inside the menu resolve to the patches.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app

runner = CliRunner()


@pytest.fixture()
def _tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: True)


@pytest.fixture()
def _no_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)


@pytest.fixture()
def _no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_config_file → None, so Init appears in the menu."""
    monkeypatch.setattr("crossby.config.loader.find_config_file", lambda _p: None)


@pytest.fixture()
def _has_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """find_config_file → an existing path, so Init is omitted."""
    fake = tmp_path / ".crossby.yml"
    fake.write_text("version: 1\n")
    monkeypatch.setattr("crossby.config.loader.find_config_file", lambda _p: fake)


def _patch_subcommands(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace every subcommand function the menu dispatches to with a mock."""
    mocks = {
        "launch": MagicMock(return_value=None),
        "sync": MagicMock(return_value=None),
        "handoff": MagicMock(return_value=None),
        "convert": MagicMock(return_value=None),
        "stats": MagicMock(return_value=None),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(f"crossby.cli.main.{name}", mock)
    return mocks


class _MenuRecorder:
    """Stand-in for ``prompts.menu`` / ``prompts.select`` driven by a queue."""

    def __init__(self, responses: list[int]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(
        self,
        title: str,
        items: list[str],
        default: int = 0,
        hints: list[str] | None = None,
        version: str | None = None,
    ) -> int:
        _ = (default, hints, version)
        self.calls.append((title, list(items)))
        if not self.responses:
            raise AssertionError(f"menu called past queue: title={title!r}")
        return self.responses.pop(0)


class TestNoSubcommandDispatch:
    def test_non_tty_prints_help(self, _no_tty: None) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "CROSSBY" in result.output or "Usage" in result.output

    def test_tty_invokes_menu(
        self,
        _tty: None,
        _has_config: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TTY + no subcommand → menu is rendered."""
        mocks = _patch_subcommands(monkeypatch)
        # Choose "Launch" (index 0) — first entry.
        recorder = _MenuRecorder([0])
        monkeypatch.setattr("crossby.ui.prompts.menu", recorder)

        result = runner.invoke(app, [])

        assert result.exit_code == 0, result.output
        assert len(recorder.calls) == 1
        mocks["launch"].assert_called_once()


class TestInitMenuVisibility:
    def test_init_hidden_when_config_exists(
        self,
        _tty: None,
        _has_config: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_subcommands(monkeypatch)
        recorder = _MenuRecorder([0])
        monkeypatch.setattr("crossby.ui.prompts.menu", recorder)

        runner.invoke(app, [])

        _title, items = recorder.calls[0]
        assert "Init" not in items
        assert items == ["Launch", "Sync", "Handoff", "Convert", "Stats"]

    def test_init_shown_when_no_config(
        self,
        _tty: None,
        _no_config: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_subcommands(monkeypatch)
        # Menu: pick Launch (index 0) so we don't enter the Init branch yet.
        recorder = _MenuRecorder([0])
        monkeypatch.setattr("crossby.ui.prompts.menu", recorder)

        runner.invoke(app, [])

        _title, items = recorder.calls[0]
        assert items[-1] == "Init"


class TestMenuDispatch:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        _tty: None,
        _has_config: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.mocks = _patch_subcommands(monkeypatch)
        self.monkeypatch = monkeypatch

    def _run(self, menu_idx: int, *, extras: list[int] | None = None) -> None:
        """Drive the menu + optional downstream select/input prompts."""
        self.monkeypatch.setattr("crossby.ui.prompts.menu", _MenuRecorder([menu_idx]))
        if extras is not None:
            select_recorder = _MenuRecorder(extras)
            self.monkeypatch.setattr("crossby.ui.prompts.select", select_recorder)
            self.monkeypatch.setattr("crossby.ui.prompts.input_prompt", lambda *_a, **_kw: "x")
        runner.invoke(app, [])

    def test_launch(self) -> None:
        self._run(0)
        self.mocks["launch"].assert_called_once()
        kwargs = self.mocks["launch"].call_args.kwargs
        assert kwargs["tool"] is None
        assert kwargs["model"] is None
        assert kwargs["path"] == Path(".")

    def test_sync(self) -> None:
        self._run(1)
        self.mocks["sync"].assert_called_once()
        kwargs = self.mocks["sync"].call_args.kwargs
        assert kwargs["from_tool"] is None
        assert kwargs["to_tool"] is None
        assert kwargs["concern"] is None

    def test_handoff(self) -> None:
        self._run(2)
        self.mocks["handoff"].assert_called_once()
        kwargs = self.mocks["handoff"].call_args.kwargs
        assert kwargs["from_tool"] is None
        assert kwargs["to_tool"] is None
        assert kwargs["prompt_preset"] == "default"
        assert kwargs["token_budget"] == 32_000

    def test_convert(self) -> None:
        # Convert flow issues: menu pick (3) → input_prompt(pattern)
        # → select(source, index 0 = canonical) → select(target, index 1 = claude).
        self._run(3, extras=[0, 1])
        self.mocks["convert"].assert_called_once()
        kwargs = self.mocks["convert"].call_args.kwargs
        assert kwargs["pattern"] == "x"
        assert kwargs["from_tool"] == "canonical"
        assert kwargs["to_tool"] == "claude"

    def test_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No detect_installed needed — stub it for determinism.
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: []),
        )
        # Menu pick (4) → select(tool, index 0 = auto-detect) → input("session path").
        self._run(4, extras=[0])
        self.mocks["stats"].assert_called_once()
        kwargs = self.mocks["stats"].call_args.kwargs
        assert kwargs["tool"] is None
        assert kwargs["transcript_path"] == Path("x")


class TestPromptHelpers:
    def test_prompt_convert_args_maps_selection_to_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.cli.main import _prompt_convert_args

        monkeypatch.setattr("crossby.ui.prompts.input_prompt", lambda *_a, **_kw: "Bash(ls)")
        recorder = _MenuRecorder([2, 1])  # cursor, copilot (canonical is idx 0)
        monkeypatch.setattr("crossby.ui.prompts.select", recorder)

        pattern, from_tool, to_tool = _prompt_convert_args()
        assert pattern == "Bash(ls)"
        tool_choices = ["canonical", "claude", "copilot", "cursor", "gemini"]
        assert from_tool == tool_choices[2]
        assert to_tool == tool_choices[1]

    def test_prompt_stats_args_auto_detect_returns_none_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.cli.main import _prompt_stats_args

        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: []),
        )
        monkeypatch.setattr("crossby.ui.prompts.select", _MenuRecorder([0]))
        monkeypatch.setattr(
            "crossby.ui.prompts.input_prompt",
            lambda *_a, **_kw: "/tmp/session.txt",
        )

        path, tool = _prompt_stats_args()
        assert tool is None
        assert path == Path("/tmp/session.txt")

    def test_prompt_stats_args_with_installed_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from crossby.ai_tools.base import AbstractAITool
        from crossby.cli.main import _prompt_stats_args
        from crossby.models.ai import AIToolID

        monkeypatch.setattr(
            AbstractAITool,
            "detect_installed",
            classmethod(lambda _cls: [AIToolID.CLAUDE]),
        )
        # Menu: "(auto-detect)" [0], "claude" [1] → pick claude (idx 1).
        monkeypatch.setattr("crossby.ui.prompts.select", _MenuRecorder([1]))
        monkeypatch.setattr("crossby.ui.prompts.input_prompt", lambda *_a, **_kw: "/t.txt")

        path, tool = _prompt_stats_args()
        assert tool == "claude"
        assert path == Path("/t.txt")
