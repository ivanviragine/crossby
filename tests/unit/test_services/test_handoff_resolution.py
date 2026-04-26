"""Tests for ``crossby.services.handoff_resolution`` resolvers.

The contract for ``prompt_preset`` and ``token_budget`` is None-based:
``None`` means "the user did not pass this flag," so config wins. Any
non-None value (including the fallback string/int) is treated as an
explicit user choice and overrides config.
"""

from __future__ import annotations

import pytest

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig, HandoffDefaults
from crossby.services.handoff_resolution import (
    resolve_handoff_from,
    resolve_handoff_preset,
    resolve_handoff_to,
    resolve_handoff_token_budget,
)


class TestResolveHandoffFromTo:
    def test_explicit_wins(self) -> None:
        cfg = CrossbyConfig(
            handoff_defaults=HandoffDefaults(from_tool=AIToolID.CURSOR, to=AIToolID.CODEX)
        )
        assert resolve_handoff_from("claude", cfg) is AIToolID.CLAUDE
        assert resolve_handoff_to("copilot", cfg) is AIToolID.COPILOT

    def test_config_fallback(self) -> None:
        cfg = CrossbyConfig(
            handoff_defaults=HandoffDefaults(from_tool=AIToolID.CURSOR, to=AIToolID.CODEX)
        )
        assert resolve_handoff_from(None, cfg) is AIToolID.CURSOR
        assert resolve_handoff_to(None, cfg) is AIToolID.CODEX

    def test_missing_returns_none(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_from(None, cfg) is None
        assert resolve_handoff_to(None, cfg) is None

    def test_invalid_raises(self) -> None:
        cfg = CrossbyConfig()
        with pytest.raises(ValueError):
            resolve_handoff_from("nosuchtool", cfg)


class TestResolveHandoffPreset:
    def test_none_lets_config_win(self) -> None:
        """Flag not provided (None) → use config value if present."""
        cfg = CrossbyConfig(
            handoff_defaults=HandoffDefaults(prompt_preset="cc-compact")
        )
        assert resolve_handoff_preset(None, cfg) == "cc-compact"

    def test_explicit_default_overrides_config(self) -> None:
        """User passing ``--prompt-preset default`` must override the config."""
        cfg = CrossbyConfig(
            handoff_defaults=HandoffDefaults(prompt_preset="cc-compact")
        )
        assert resolve_handoff_preset("default", cfg) == "default"

    def test_explicit_non_default_overrides_config(self) -> None:
        cfg = CrossbyConfig(
            handoff_defaults=HandoffDefaults(prompt_preset="cc-compact")
        )
        assert resolve_handoff_preset("other", cfg) == "other"

    def test_falls_back_to_default_when_unset(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_preset(None, cfg) == "default"

    def test_custom_fallback(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_preset(None, cfg, fallback="cc-compact") == "cc-compact"


class TestResolveHandoffTokenBudget:
    def test_none_lets_config_win(self) -> None:
        cfg = CrossbyConfig(handoff_defaults=HandoffDefaults(token_budget=16_000))
        assert resolve_handoff_token_budget(None, cfg) == 16_000

    def test_explicit_default_overrides_config(self) -> None:
        """User passing ``--token-budget 32000`` must override the config."""
        cfg = CrossbyConfig(handoff_defaults=HandoffDefaults(token_budget=16_000))
        assert resolve_handoff_token_budget(32_000, cfg) == 32_000

    def test_explicit_non_default_overrides_config(self) -> None:
        cfg = CrossbyConfig(handoff_defaults=HandoffDefaults(token_budget=16_000))
        assert resolve_handoff_token_budget(8_000, cfg) == 8_000

    def test_falls_back_to_default_when_unset(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_token_budget(None, cfg) == 32_000

    def test_custom_fallback(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_token_budget(None, cfg, fallback=8_000) == 8_000
