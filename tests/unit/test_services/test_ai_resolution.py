"""Tests for strict mode in AI resolution functions."""

from __future__ import annotations

import pytest

from crossby.models.config import CrossbyConfig
from crossby.services.ai_resolution import (
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
