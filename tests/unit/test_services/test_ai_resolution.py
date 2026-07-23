"""Tests for strict mode in AI resolution functions."""

from __future__ import annotations

import pytest

from crossby.models.ai import EffortLevel
from crossby.models.config import (
    AIConfig,
    CommandConfig,
    ComplexityModelMapping,
    CrossbyConfig,
)
from crossby.services.ai_resolution import (
    resolve_accept_edits,
    resolve_auto,
    resolve_effort,
    resolve_model,
    resolve_yolo,
)


class TestResolveModelStrict:
    """Tests for resolve_model() strict mode."""

    def test_strict_incompatible_raises(self) -> None:
        config = CrossbyConfig()
        with pytest.raises(ValueError, match="not compatible with claude"):
            resolve_model("gpt-4o", config, tool="claude", strict=True)

    def test_nonstrict_incompatible_returns_none(self) -> None:
        config = CrossbyConfig()
        result = resolve_model("gpt-4o", config, tool="claude", strict=False)
        assert result is None

    def test_strict_compatible_returns_model(self) -> None:
        config = CrossbyConfig()
        result = resolve_model("claude-sonnet-4.6", config, tool="claude", strict=True)
        assert result == "claude-sonnet-4.6"

    def test_strict_unsupported_model_flag_raises(self) -> None:
        """Tools that don't support --model should reject explicit model in strict mode."""
        config = CrossbyConfig()
        with pytest.raises(ValueError, match="does not support explicit model"):
            resolve_model("some-model", config, tool="vscode", strict=True)

    def test_nonstrict_unsupported_model_flag_returns_model(self) -> None:
        """Non-strict mode still returns the model even if tool doesn't support --model."""
        config = CrossbyConfig()
        result = resolve_model("some-model", config, tool="vscode", strict=False)
        assert result == "some-model"


class TestResolveEffortStrict:
    """Tests for resolve_effort() strict mode."""

    def test_strict_unsupported_tool_raises(self) -> None:
        config = CrossbyConfig()
        with pytest.raises(ValueError, match="does not support effort"):
            resolve_effort("high", config, tool="copilot", strict=True)

    def test_strict_invalid_level_raises(self) -> None:
        config = CrossbyConfig()
        with pytest.raises(ValueError, match="Invalid effort level"):
            resolve_effort("ultra", config, strict=True)

    def test_nonstrict_invalid_level_returns_none(self) -> None:
        config = CrossbyConfig()
        result = resolve_effort("ultra", config, strict=False)
        assert result is None

    def test_strict_supported_tool_returns_level(self) -> None:
        config = CrossbyConfig()
        result = resolve_effort("high", config, tool="claude", strict=True)
        assert result == EffortLevel.HIGH


class TestResolveEffortComplexityAndEnvVar:
    """Per-complexity-tier effort resolution and a configurable effort env var."""

    def test_per_tier_effort_used_when_complexity_given(self) -> None:
        config = CrossbyConfig(models={"claude": ComplexityModelMapping(complex_effort="high")})
        result = resolve_effort(None, config, tool="claude", complexity="complex")
        assert result == EffortLevel.HIGH

    def test_per_tier_skipped_without_complexity(self) -> None:
        config = CrossbyConfig(models={"claude": ComplexityModelMapping(complex_effort="high")})
        assert resolve_effort(None, config, tool="claude") is None

    def test_command_effort_takes_precedence_over_per_tier(self) -> None:
        config = CrossbyConfig(
            ai=AIConfig(commands={"plan": CommandConfig(effort="low")}),
            models={"claude": ComplexityModelMapping(complex_effort="high")},
        )
        result = resolve_effort(None, config, command="plan", tool="claude", complexity="complex")
        assert result == EffortLevel.LOW

    def test_global_effort_used_when_no_per_tier(self) -> None:
        config = CrossbyConfig(ai=AIConfig(effort="medium"))
        result = resolve_effort(None, config, tool="claude", complexity="easy")
        assert result == EffortLevel.MEDIUM

    def test_custom_env_var_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CROSSBY_EFFORT", raising=False)
        monkeypatch.setenv("WADE_EFFORT", "high")
        result = resolve_effort(None, CrossbyConfig(), tool="claude", env_var="WADE_EFFORT")
        assert result == EffortLevel.HIGH

    def test_default_env_var_not_read_when_custom_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CROSSBY_EFFORT", "high")
        monkeypatch.delenv("WADE_EFFORT", raising=False)
        # Consumer asked only for WADE_EFFORT; CROSSBY_EFFORT must not leak in.
        assert resolve_effort(None, CrossbyConfig(), tool="claude", env_var="WADE_EFFORT") is None


class TestResolveYoloStrict:
    """Tests for resolve_yolo() strict mode."""

    def test_strict_unsupported_tool_raises(self) -> None:
        config = CrossbyConfig()
        with pytest.raises(ValueError, match="does not support YOLO"):
            resolve_yolo(True, config, tool="opencode", strict=True)

    def test_nonstrict_unsupported_returns_false(self) -> None:
        config = CrossbyConfig()
        result = resolve_yolo(True, config, tool="opencode", strict=False)
        assert result is False

    def test_strict_supported_tool_returns_true(self) -> None:
        config = CrossbyConfig()
        result = resolve_yolo(True, config, tool="claude", strict=True)
        assert result is True


class TestResolveAcceptEdits:
    """resolve_accept_edits() fallback chain: arg -> command -> global -> False."""

    def test_explicit_arg_wins(self) -> None:
        assert resolve_accept_edits(True, CrossbyConfig()) is True

    def test_explicit_false_overrides_config(self) -> None:
        config = CrossbyConfig(ai=AIConfig(accept_edits=True))
        assert resolve_accept_edits(False, config) is False

    def test_command_config_used(self) -> None:
        config = CrossbyConfig(ai=AIConfig(commands={"plan": CommandConfig(accept_edits=True)}))
        assert resolve_accept_edits(None, config, command="plan") is True

    def test_global_config_used(self) -> None:
        config = CrossbyConfig(ai=AIConfig(accept_edits=True))
        assert resolve_accept_edits(None, config) is True

    def test_command_overrides_global(self) -> None:
        config = CrossbyConfig(
            ai=AIConfig(accept_edits=True, commands={"plan": CommandConfig(accept_edits=False)})
        )
        assert resolve_accept_edits(None, config, command="plan") is False

    def test_default_false(self) -> None:
        assert resolve_accept_edits(None, CrossbyConfig()) is False

    def test_unsupported_tool_is_not_dropped(self) -> None:
        # Unlike yolo, accept-edits is passed through (builder degrades it).
        assert resolve_accept_edits(True, CrossbyConfig()) is True


class TestResolveAuto:
    """resolve_auto() fallback chain mirrors resolve_accept_edits()."""

    def test_explicit_arg_wins(self) -> None:
        assert resolve_auto(True, CrossbyConfig()) is True

    def test_explicit_false_overrides_config(self) -> None:
        config = CrossbyConfig(ai=AIConfig(auto=True))
        assert resolve_auto(False, config) is False

    def test_command_config_used(self) -> None:
        config = CrossbyConfig(ai=AIConfig(commands={"plan": CommandConfig(auto=True)}))
        assert resolve_auto(None, config, command="plan") is True

    def test_global_config_used(self) -> None:
        config = CrossbyConfig(ai=AIConfig(auto=True))
        assert resolve_auto(None, config) is True

    def test_default_false(self) -> None:
        assert resolve_auto(None, CrossbyConfig()) is False
