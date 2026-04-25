"""Tests for ``crossby init`` — scaffold a ``.crossby.yml`` file.

The on-disk format is validated by round-tripping through
``parse_config_file`` — we never assert on raw YAML text because PyYAML's
output shape is not part of the contract (only the parsed result is).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from crossby.cli.init import _render_init_yaml
from crossby.cli.main import app
from crossby.config.loader import parse_config_file
from crossby.models.ai import AIToolID

runner = CliRunner()


def _answers(**overrides: object) -> dict[str, object]:
    """Build a full answer dict with sensible Nones as defaults."""
    base: dict[str, object] = {
        "ai_tool": None,
        "ai_model": None,
        "ai_effort": None,
        "ai_yolo": False,
        "sync_from": None,
        "sync_to": None,
        "handoff_from": None,
        "handoff_to": None,
        "handoff_preset": None,
        "handoff_budget": None,
    }
    base.update(overrides)
    return base


class TestInitNonInteractive:
    def test_writes_file_with_first_installed_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: [AIToolID.CLAUDE, AIToolID.CURSOR]),
        )

        result = runner.invoke(
            app, ["init", "--path", str(tmp_path), "--non-interactive"]
        )

        assert result.exit_code == 0, result.output
        target = tmp_path / ".crossby.yml"
        assert target.exists()

        parsed = parse_config_file(target)
        assert parsed.ai.default_tool == "claude"

    def test_zero_tools_produces_valid_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: []),
        )

        result = runner.invoke(
            app, ["init", "--path", str(tmp_path), "--non-interactive"]
        )

        assert result.exit_code == 0, result.output
        parsed = parse_config_file(tmp_path / ".crossby.yml")
        assert parsed.ai.default_tool is None

    def test_refuses_tty_required_without_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --non-interactive in a non-TTY session, we abort with a friendly error."""
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: []),
        )

        result = runner.invoke(app, ["init", "--path", str(tmp_path)])

        assert result.exit_code == 1
        assert not (tmp_path / ".crossby.yml").exists()


class TestInitExistingFileGuard:
    def test_refuses_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / ".crossby.yml"
        existing = "version: 1\nai:\n  default_tool: claude\n"
        target.write_text(existing, encoding="utf-8")
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: []),
        )

        result = runner.invoke(
            app, ["init", "--path", str(tmp_path), "--non-interactive"]
        )

        assert result.exit_code == 1
        assert str(target) in result.output or "overwrite" in result.output.lower()
        assert target.read_text(encoding="utf-8") == existing

    def test_force_overwrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / ".crossby.yml"
        target.write_text("garbage that is not valid config\n", encoding="utf-8")
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: [AIToolID.CLAUDE]),
        )

        result = runner.invoke(
            app,
            ["init", "--path", str(tmp_path), "--non-interactive", "--force"],
        )

        assert result.exit_code == 0, result.output
        parsed = parse_config_file(target)
        assert parsed.ai.default_tool == "claude"


class TestRenderInitYaml:
    def test_empty_answers_round_trip(self, tmp_path: Path) -> None:
        yaml_text = _render_init_yaml(_answers())
        target = tmp_path / ".crossby.yml"
        target.write_text(yaml_text, encoding="utf-8")

        parsed = parse_config_file(target)
        assert parsed.version == 1
        assert parsed.ai.default_tool is None
        assert parsed.sync_defaults.from_tool is None
        assert parsed.handoff_defaults.from_tool is None

    def test_full_answers_round_trip_preserves_every_field(
        self, tmp_path: Path
    ) -> None:
        yaml_text = _render_init_yaml(
            _answers(
                ai_tool="claude",
                ai_model="claude-sonnet-4.6",
                ai_effort="medium",
                ai_yolo=True,
                sync_from="claude",
                sync_to="cursor",
                handoff_from="claude",
                handoff_to="codex",
                handoff_preset="cc-compact",
                handoff_budget=16_000,
            )
        )
        target = tmp_path / ".crossby.yml"
        target.write_text(yaml_text, encoding="utf-8")

        parsed = parse_config_file(target)

        assert parsed.ai.default_tool == "claude"
        assert parsed.ai.default_model == "claude-sonnet-4.6"
        assert parsed.ai.effort == "medium"
        assert parsed.ai.yolo is True
        assert parsed.sync_defaults.from_tool is AIToolID.CLAUDE
        assert parsed.sync_defaults.to is AIToolID.CURSOR
        assert parsed.handoff_defaults.from_tool is AIToolID.CLAUDE
        assert parsed.handoff_defaults.to is AIToolID.CODEX
        assert parsed.handoff_defaults.prompt_preset == "cc-compact"
        assert parsed.handoff_defaults.token_budget == 16_000

    def test_from_alias_emitted_not_python_name(self) -> None:
        """YAML must use the alias ``from`` — not ``from_tool`` — so humans can read it."""
        yaml_text = _render_init_yaml(_answers(sync_from="claude", handoff_from="claude"))

        assert "from: claude" in yaml_text
        assert "from_tool" not in yaml_text

    def test_unset_fields_are_excluded(self) -> None:
        """exclude_none keeps the file minimal — only chosen values appear."""
        yaml_text = _render_init_yaml(_answers(ai_tool="claude"))

        assert "default_tool: claude" in yaml_text
        assert "default_model" not in yaml_text
        assert "effort" not in yaml_text
        assert "yolo" not in yaml_text

    def test_yolo_false_omitted(self) -> None:
        """When yolo is False we store None so the file doesn't serialise `yolo: false`."""
        yaml_text = _render_init_yaml(_answers(ai_tool="claude", ai_yolo=False))

        assert "yolo" not in yaml_text


class TestPickTokenBudget:
    def test_non_integer_input_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid input warns the user and leaves the budget unset."""
        from crossby.cli.init import _pick_token_budget

        monkeypatch.setattr(
            "crossby.ui.prompts.input_prompt", lambda *_a, **_kw: "not-an-int"
        )
        assert _pick_token_budget() is None

    def test_blank_input_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.cli.init import _pick_token_budget

        monkeypatch.setattr(
            "crossby.ui.prompts.input_prompt", lambda *_a, **_kw: ""
        )
        assert _pick_token_budget() is None

    def test_integer_input_returns_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from crossby.cli.init import _pick_token_budget

        monkeypatch.setattr(
            "crossby.ui.prompts.input_prompt", lambda *_a, **_kw: "16000"
        )
        assert _pick_token_budget() == 16_000
