"""Tests for AI resolution and confirmation behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import crossby.ai_tools  # noqa: F401 - register adapters
from crossby.models.ai import AIToolID, EffortLevel
from crossby.models.config import AIConfig, CommandConfig, CrossbyConfig
from crossby.services.ai_resolution import (
    confirm_ai_selection,
    resolve_ai_tool,
    resolve_effort,
    resolve_model,
    resolve_yolo,
)

_IS_TTY = "crossby.ui.prompts.is_tty"
_SELECT = "crossby.ui.prompts.select"
_INPUT_PROMPT = "crossby.ui.prompts.input_prompt"
_DETECT = "crossby.services.ai_resolution.AbstractAITool.detect_installed"
_MODELS_FOR_TOOL = "crossby.data.get_models_for_tool"
_CONSOLE_KV = "crossby.ui.console.console.kv"


def _make_installed(*names: str) -> list[AIToolID]:
    return [AIToolID(name) for name in names]


class TestResolveAiTool:
    def test_explicit_tool_wins(self) -> None:
        config = CrossbyConfig(ai=AIConfig(default_tool="gemini"))
        assert resolve_ai_tool("claude", config, "default") == "claude"

    def test_command_config_beats_global_default(self) -> None:
        config = CrossbyConfig(
            ai=AIConfig(
                default_tool="claude",
                commands={"review": CommandConfig(tool="copilot")},
            )
        )
        assert resolve_ai_tool(None, config, "review") == "copilot"

    def test_auto_detect_used_when_no_explicit_or_config(self) -> None:
        config = CrossbyConfig()
        with patch(_DETECT, return_value=_make_installed("gemini", "claude")):
            assert resolve_ai_tool(None, config, "default") == "gemini"


class TestResolveModel:
    def test_complexity_mapping_used_when_no_explicit_or_command_model(self) -> None:
        config = CrossbyConfig(
            models={"claude": {"complex": "claude-opus-4.6"}},  # type: ignore[dict-item]
        )
        result = resolve_model(None, config, "plan", tool="claude", complexity="complex")
        assert result == "claude-opus-4.6"

    def test_strict_incompatible_model_raises(self) -> None:
        with patch(_DETECT, return_value=_make_installed("claude")), patch(_CONSOLE_KV):
            config = CrossbyConfig()
            try:
                resolve_model("gpt-4o", config, tool="claude", strict=True)
            except ValueError as err:
                assert "not compatible" in str(err)
            else:
                raise AssertionError("resolve_model should have raised ValueError")

    def test_nonstrict_incompatible_model_returns_none(self) -> None:
        config = CrossbyConfig()
        assert resolve_model("gpt-4o", config, tool="claude", strict=False) is None

    def test_strict_compatible_model_returns_value(self) -> None:
        config = CrossbyConfig()
        assert (
            resolve_model("claude-sonnet-4.6", config, tool="claude", strict=True)
            == "claude-sonnet-4.6"
        )

    def test_strict_unsupported_model_flag_raises(self) -> None:
        config = CrossbyConfig()
        with pytest.raises(ValueError, match="does not support explicit model"):
            resolve_model("some-model", config, tool="vscode", strict=True)

    def test_nonstrict_unsupported_model_flag_returns_model(self) -> None:
        config = CrossbyConfig()
        assert resolve_model("some-model", config, tool="vscode", strict=False) == "some-model"


class TestResolveEffort:
    def test_env_var_used_when_flag_missing(self, monkeypatch) -> None:
        monkeypatch.setenv("CROSSBY_EFFORT", "high")
        config = CrossbyConfig()
        assert resolve_effort(None, config) == EffortLevel.HIGH

    def test_strict_unsupported_tool_raises(self) -> None:
        config = CrossbyConfig()
        try:
            resolve_effort("high", config, tool="copilot", strict=True)
        except ValueError as err:
            assert "does not support effort" in str(err)
        else:
            raise AssertionError("resolve_effort should have raised ValueError")

    def test_strict_invalid_level_raises(self) -> None:
        config = CrossbyConfig()
        try:
            resolve_effort("ultra", config, strict=True)
        except ValueError as err:
            assert "Invalid effort level" in str(err)
        else:
            raise AssertionError("resolve_effort should have raised ValueError")

    def test_nonstrict_invalid_level_returns_none(self) -> None:
        config = CrossbyConfig()
        assert resolve_effort("ultra", config, strict=False) is None

    def test_strict_supported_tool_returns_level(self) -> None:
        config = CrossbyConfig()
        assert resolve_effort("high", config, tool="claude", strict=True) == EffortLevel.HIGH


class TestResolveYolo:
    def test_strict_unsupported_tool_raises(self) -> None:
        config = CrossbyConfig()
        try:
            resolve_yolo(True, config, tool="opencode", strict=True)
        except ValueError as err:
            assert "does not support YOLO" in str(err)
        else:
            raise AssertionError("resolve_yolo should have raised ValueError")

    def test_nonstrict_unsupported_returns_false(self) -> None:
        config = CrossbyConfig()
        assert resolve_yolo(True, config, tool="opencode", strict=False) is False

    def test_strict_supported_tool_returns_true(self) -> None:
        config = CrossbyConfig()
        assert resolve_yolo(True, config, tool="claude", strict=True) is True

    def test_config_value_used_when_flag_missing(self) -> None:
        config = CrossbyConfig(ai=AIConfig(yolo=True))
        assert resolve_yolo(None, config, tool="claude") is True


class TestConfirmAiSelection:
    def test_non_tty_returns_unchanged(self) -> None:
        with patch(_IS_TTY, return_value=False), patch(_SELECT) as mock_select:
            result = confirm_ai_selection(
                "claude",
                "claude-sonnet-4.6",
                tool_explicit=False,
                model_explicit=False,
            )
        assert result == ("claude", "claude-sonnet-4.6", None, False)
        mock_select.assert_not_called()

    def test_all_explicit_skips_prompts(self) -> None:
        with patch(_IS_TTY, return_value=True), patch(_SELECT) as mock_select:
            result = confirm_ai_selection(
                "claude",
                "claude-sonnet-4.6",
                tool_explicit=True,
                model_explicit=True,
                resolved_effort=EffortLevel.HIGH,
                effort_explicit=True,
                resolved_yolo=True,
                yolo_explicit=True,
            )
        assert result == ("claude", "claude-sonnet-4.6", EffortLevel.HIGH, True)
        mock_select.assert_not_called()

    def test_change_tool_reprompts_for_model(self) -> None:
        call_count = 0

        def fake_select(title: str, items: list[str], **kwargs) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return items.index("Change AI tool")
            if call_count == 2:
                return items.index("copilot")
            if call_count == 3:
                return 0
            return items.index("Proceed")

        with (
            patch(_IS_TTY, return_value=True),
            patch(_SELECT, side_effect=fake_select),
            patch(_DETECT, return_value=_make_installed("claude", "copilot")),
            patch(_MODELS_FOR_TOOL, return_value=["gpt-5"]),
            patch(_CONSOLE_KV),
        ):
            tool, model, effort, yolo = confirm_ai_selection(
                "claude",
                "claude-sonnet-4.6",
                tool_explicit=False,
                model_explicit=False,
            )

        assert tool == "copilot"
        assert model == "gpt-5"
        assert effort is None
        assert yolo is False

    def test_custom_model_prompt_can_clear_model(self) -> None:
        call_count = 0

        def fake_select(title: str, items: list[str], **kwargs) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return items.index("Change model")
            if call_count == 2:
                return items.index("Custom…")
            return items.index("Proceed")

        with (
            patch(_IS_TTY, return_value=True),
            patch(_SELECT, side_effect=fake_select),
            patch(_INPUT_PROMPT, return_value=""),
            patch(_DETECT, return_value=_make_installed("claude", "copilot")),
            patch(_MODELS_FOR_TOOL, return_value=["claude-sonnet-4.6"]),
            patch(_CONSOLE_KV),
        ):
            tool, model, effort, yolo = confirm_ai_selection(
                "claude",
                "claude-sonnet-4.6",
                tool_explicit=False,
                model_explicit=False,
            )

        assert tool == "claude"
        assert model is None
        assert effort is None
        assert yolo is False
