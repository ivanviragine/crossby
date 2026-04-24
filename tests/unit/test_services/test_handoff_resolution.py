"""Tests for ``crossby.services.handoff_resolution`` resolvers.

The CLI-default handling for ``prompt_preset`` and ``token_budget`` is
asymmetric — the flags have real default values (``"default"`` / ``32000``)
that should *not* override config. Tests pin that contract.
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
    def test_cli_default_lets_config_win(self) -> None:
        """Flag left at the CLI default → use config value if present."""
        cfg = CrossbyConfig(
            handoff_defaults=HandoffDefaults(prompt_preset="cc-compact")
        )
        assert resolve_handoff_preset("default", cfg) == "cc-compact"

    def test_explicit_non_default_overrides_config(self) -> None:
        cfg = CrossbyConfig(
            handoff_defaults=HandoffDefaults(prompt_preset="cc-compact")
        )
        assert resolve_handoff_preset("default", cfg, cli_default="default") == "cc-compact"
        assert resolve_handoff_preset("other", cfg, cli_default="default") == "other"

    def test_falls_back_to_cli_default(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_preset("default", cfg) == "default"

    def test_none_from_cli_uses_config_or_default(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_preset(None, cfg) == "default"


class TestResolveHandoffTokenBudget:
    def test_cli_default_lets_config_win(self) -> None:
        cfg = CrossbyConfig(handoff_defaults=HandoffDefaults(token_budget=16_000))
        assert resolve_handoff_token_budget(32_000, cfg) == 16_000

    def test_explicit_non_default_overrides_config(self) -> None:
        cfg = CrossbyConfig(handoff_defaults=HandoffDefaults(token_budget=16_000))
        assert resolve_handoff_token_budget(8_000, cfg) == 8_000

    def test_falls_back_to_cli_default(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_token_budget(32_000, cfg) == 32_000

    def test_none_input_uses_config_or_default(self) -> None:
        cfg = CrossbyConfig(handoff_defaults=HandoffDefaults(token_budget=16_000))
        assert resolve_handoff_token_budget(None, cfg) == 16_000

    def test_none_input_and_no_config_uses_default(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_handoff_token_budget(None, cfg) == 32_000
