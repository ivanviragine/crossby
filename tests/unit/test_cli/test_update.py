"""CLI tests for ``crossby tools update`` via ``CliRunner``.

``detect_installed`` and ``is_tty`` are pinned per test; ``run_update`` is
patched to a recorder so no real updater runs. ``updatable_tools`` runs for real
against the pinned inventory, exercising the actual filtering.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from crossby.ai_tools.base import AbstractAITool
from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.services.tool_update import UpdateResult

runner = CliRunner()


def _pin_installed(monkeypatch: pytest.MonkeyPatch, ids: list[AIToolID]) -> None:
    monkeypatch.setattr(AbstractAITool, "detect_installed", classmethod(lambda _cls: list(ids)))


def _pin_tty(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: value)


def _result(
    tool_id: AIToolID,
    *,
    display_name: str,
    command: tuple[str, ...],
    success: bool = True,
    before: str | None = "1.0.0",
    after: str | None = "1.1.0",
    unchanged: bool = False,
    error: str | None = None,
    output_tail: str = "",
) -> UpdateResult:
    return UpdateResult(
        tool_id=tool_id,
        display_name=display_name,
        command=command,
        success=success,
        exit_code=0 if success else 1,
        output_tail=output_tail,
        before_version=before,
        after_version=after,
        unchanged=unchanged,
        error=error,
    )


class TestDryRun:
    def test_dry_run_prints_commands_and_runs_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE, AIToolID.CODEX])
        _pin_tty(monkeypatch, False)
        run_mock = MagicMock()
        monkeypatch.setattr("crossby.services.tool_update.run_update", run_mock)

        result = runner.invoke(app, ["tools", "update", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "claude update" in result.output
        assert "codex update" in result.output
        run_mock.assert_not_called()


class TestExplicitToolValidation:
    def test_unknown_id_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE])
        result = runner.invoke(app, ["tools", "update", "--tool", "bogus"])
        assert result.exit_code == 1
        assert "Unknown tool" in result.output

    def test_not_installed_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [])
        result = runner.invoke(app, ["tools", "update", "--tool", "claude"])
        assert result.exit_code == 1
        assert "not installed" in result.output

    def test_installed_but_not_updatable_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # vscode is a GUI tool with no update command.
        _pin_installed(monkeypatch, [AIToolID.VSCODE])
        result = runner.invoke(app, ["tools", "update", "--tool", "vscode"])
        assert result.exit_code == 1
        assert "has no update command" in result.output

    def test_reason_precedence_is_distinct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same empty-ish inventory, two different ids → two different reasons.
        _pin_installed(monkeypatch, [AIToolID.VSCODE])
        not_installed = runner.invoke(app, ["tools", "update", "--tool", "claude"])
        not_updatable = runner.invoke(app, ["tools", "update", "--tool", "vscode"])
        assert "not installed" in not_installed.output
        assert "has no update command" in not_updatable.output

    def test_validation_runs_even_when_inventory_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No tools installed at all: explicit --tool still errors (validation
        # precedes the "nothing to update" early return).
        _pin_installed(monkeypatch, [])
        result = runner.invoke(app, ["tools", "update", "--tool", "vscode"])
        assert result.exit_code == 1
        assert "not installed" in result.output

    def test_duplicate_tool_values_deduped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE])
        _pin_tty(monkeypatch, False)
        result = runner.invoke(
            app, ["tools", "update", "--tool", "claude", "--tool", "claude", "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        # One line only, despite the duplicate flag.
        assert result.output.count("claude update") == 1


class TestSelection:
    def test_no_updatable_tools_exits_zero_with_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only a GUI tool installed → nothing updatable.
        _pin_installed(monkeypatch, [AIToolID.VSCODE])
        _pin_tty(monkeypatch, False)
        result = runner.invoke(app, ["tools", "update"])
        assert result.exit_code == 0, result.output
        assert "No updatable AI tools detected" in result.output

    def test_zero_selection_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE, AIToolID.CODEX])
        _pin_tty(monkeypatch, True)
        monkeypatch.setattr("crossby.ui.prompts.multi_select", lambda *_a, **_k: [])
        run_mock = MagicMock()
        monkeypatch.setattr("crossby.services.tool_update.run_update", run_mock)

        result = runner.invoke(app, ["tools", "update"])

        assert result.exit_code == 0, result.output
        assert "No tools selected" in result.output
        run_mock.assert_not_called()

    def test_non_tty_selects_all_updatable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE, AIToolID.CODEX])
        _pin_tty(monkeypatch, False)
        ran: list[AIToolID] = []

        def fake_run(tool_id: AIToolID) -> UpdateResult:
            ran.append(tool_id)
            return _result(
                tool_id,
                display_name=str(tool_id),
                command=("x", "update"),
            )

        monkeypatch.setattr("crossby.services.tool_update.run_update", fake_run)
        result = runner.invoke(app, ["tools", "update"])

        assert result.exit_code == 0, result.output
        assert ran == [AIToolID.CLAUDE, AIToolID.CODEX]


class TestConfirmation:
    def test_declined_confirmation_runs_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE])
        _pin_tty(monkeypatch, True)
        monkeypatch.setattr("crossby.ui.prompts.multi_select", lambda *_a, **_k: [0])
        monkeypatch.setattr("crossby.ui.prompts.confirm", lambda *_a, **_k: False)
        run_mock = MagicMock()
        monkeypatch.setattr("crossby.services.tool_update.run_update", run_mock)

        result = runner.invoke(app, ["tools", "update"])

        assert result.exit_code == 0, result.output
        assert "Aborted" in result.output
        run_mock.assert_not_called()

    def test_yes_skips_confirmation_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE])
        _pin_tty(monkeypatch, True)
        confirm_mock = MagicMock(return_value=False)
        monkeypatch.setattr("crossby.ui.prompts.confirm", confirm_mock)

        def fake_run(tool_id: AIToolID) -> UpdateResult:
            return _result(tool_id, display_name="Claude Code", command=("claude", "update"))

        monkeypatch.setattr("crossby.services.tool_update.run_update", fake_run)
        result = runner.invoke(app, ["tools", "update", "--tool", "claude", "--yes"])

        assert result.exit_code == 0, result.output
        confirm_mock.assert_not_called()


class TestReport:
    def test_success_report_renders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE])
        _pin_tty(monkeypatch, False)

        def fake_run(tool_id: AIToolID) -> UpdateResult:
            return _result(
                tool_id,
                display_name="Claude Code",
                command=("claude", "update"),
                before="1.0.0",
                after="1.1.0",
            )

        monkeypatch.setattr("crossby.services.tool_update.run_update", fake_run)
        result = runner.invoke(app, ["tools", "update"])

        assert result.exit_code == 0, result.output
        assert "Claude Code" in result.output
        assert "1.0.0" in result.output
        assert "1.1.0" in result.output

    def test_unchanged_warning_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE])
        _pin_tty(monkeypatch, False)

        def fake_run(tool_id: AIToolID) -> UpdateResult:
            return _result(
                tool_id,
                display_name="Claude Code",
                command=("claude", "update"),
                before="1.0.0",
                after="1.0.0",
                unchanged=True,
            )

        monkeypatch.setattr("crossby.services.tool_update.run_update", fake_run)
        result = runner.invoke(app, ["tools", "update"])

        assert result.exit_code == 0, result.output
        assert "version did not change" in result.output

    def test_continue_after_failure_nonzero_exit_deterministic_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _pin_installed(monkeypatch, [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.CURSOR])
        _pin_tty(monkeypatch, False)
        ran: list[AIToolID] = []
        names = {
            AIToolID.CLAUDE: ("Claude Code", ("claude", "update")),
            AIToolID.CODEX: ("Codex CLI", ("codex", "update")),
            AIToolID.CURSOR: ("Cursor", ("agent", "update")),
        }

        def fake_run(tool_id: AIToolID) -> UpdateResult:
            ran.append(tool_id)
            display_name, command = names[tool_id]
            # Codex fails; the others succeed — the run must not stop at codex.
            failed = tool_id is AIToolID.CODEX
            return _result(
                tool_id,
                display_name=display_name,
                command=command,
                success=not failed,
                error="exited with code 1" if failed else None,
                output_tail="boom" if failed else "",
            )

        monkeypatch.setattr("crossby.services.tool_update.run_update", fake_run)
        result = runner.invoke(app, ["tools", "update"])

        # All three ran, in registry order, despite codex failing in the middle.
        assert ran == [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.CURSOR]
        assert result.exit_code == 1
        assert "Codex CLI update failed" in result.output
        assert "boom" in result.output
