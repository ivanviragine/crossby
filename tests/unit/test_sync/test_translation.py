"""Tests for cross-provider translation primitives."""

from __future__ import annotations

import pytest

from crossby.models.ai import EffortLevel
from crossby.sync.translation import (
    CLAUDE_PERMISSION_MODES_UNMAPPED,
    find_claude_family,
    map_effort_claude_to_codex,
    map_effort_codex_to_claude,
    map_model_claude_to_codex,
    map_model_codex_to_claude,
    map_permission_mode_claude_to_codex,
    map_permission_mode_codex_to_claude,
)


class TestPermissionModeForward:
    @pytest.mark.parametrize(
        ("claude", "codex"),
        [
            ("acceptEdits", "workspace-write"),
            ("readOnly", "read-only"),
            ("bypassPermissions", "danger-full-access"),
        ],
    )
    def test_supported_modes(self, claude: str, codex: str) -> None:
        assert map_permission_mode_claude_to_codex(claude) == codex

    @pytest.mark.parametrize("mode", ["default", "dontAsk", "plan"])
    def test_unmapped_returns_none(self, mode: str) -> None:
        assert map_permission_mode_claude_to_codex(mode) is None
        assert mode in CLAUDE_PERMISSION_MODES_UNMAPPED

    def test_empty_returns_none(self) -> None:
        assert map_permission_mode_claude_to_codex(None) is None
        assert map_permission_mode_claude_to_codex("") is None


class TestPermissionModeReverse:
    @pytest.mark.parametrize(
        ("codex", "claude"),
        [
            ("workspace-write", "acceptEdits"),
            ("read-only", "readOnly"),
        ],
    )
    def test_supported_modes(self, codex: str, claude: str) -> None:
        assert map_permission_mode_codex_to_claude(codex) == claude

    def test_danger_mode_not_round_tripped(self) -> None:
        # Intentional: don't auto-promote a Codex agent to Claude bypass mode.
        assert map_permission_mode_codex_to_claude("danger-full-access") is None

    def test_empty_returns_none(self) -> None:
        assert map_permission_mode_codex_to_claude(None) is None
        assert map_permission_mode_codex_to_claude("") is None


class TestModelFamily:
    def test_find_opus(self) -> None:
        mapping = find_claude_family("claude-opus-4-7")
        assert mapping is not None
        assert mapping.codex_model == "gpt-5.4"

    def test_find_sonnet(self) -> None:
        mapping = find_claude_family("claude-sonnet-4.6")
        assert mapping is not None
        assert mapping.codex_model == "gpt-5.4-mini"

    def test_find_haiku(self) -> None:
        mapping = find_claude_family("claude-haiku-4-5")
        assert mapping is not None
        assert mapping.codex_model == "gpt-5.4-mini"

    def test_unknown_returns_none(self) -> None:
        assert find_claude_family("gpt-5.4") is None
        assert find_claude_family("") is None


class TestModelTranslation:
    @pytest.mark.parametrize(
        ("claude", "codex"),
        [
            ("claude-opus-4-7", "gpt-5.4"),
            ("claude-opus-4.6", "gpt-5.4"),
            ("claude-sonnet-4.6", "gpt-5.4-mini"),
            ("claude-haiku-4-5", "gpt-5.4-mini"),
        ],
    )
    def test_claude_to_codex(self, claude: str, codex: str) -> None:
        assert map_model_claude_to_codex(claude) == codex

    def test_unknown_passes_through(self) -> None:
        assert map_model_claude_to_codex("o3-mini") == "o3-mini"

    @pytest.mark.parametrize(
        ("codex", "claude"),
        [
            ("gpt-5.4", "claude-opus-4.7"),
            ("gpt-5.4-mini", "claude-sonnet-4.6"),
        ],
    )
    def test_codex_to_claude(self, codex: str, claude: str) -> None:
        assert map_model_codex_to_claude(codex) == claude

    def test_codex_unknown_passes_through(self) -> None:
        assert map_model_codex_to_claude("o3") == "o3"


class TestEffortClaudeToCodex:
    @pytest.mark.parametrize(
        ("model", "effort_in", "effort_out"),
        [
            # Opus: 1:1 except max → xhigh
            ("claude-opus-4.7", EffortLevel.LOW, EffortLevel.LOW),
            ("claude-opus-4.7", EffortLevel.MEDIUM, EffortLevel.MEDIUM),
            ("claude-opus-4.7", EffortLevel.HIGH, EffortLevel.HIGH),
            ("claude-opus-4.7", EffortLevel.MAX, EffortLevel.XHIGH),
            # Sonnet: shift up one tier
            ("claude-sonnet-4.6", EffortLevel.LOW, EffortLevel.MEDIUM),
            ("claude-sonnet-4.6", EffortLevel.MEDIUM, EffortLevel.HIGH),
            ("claude-sonnet-4.6", EffortLevel.HIGH, EffortLevel.XHIGH),
            ("claude-sonnet-4.6", EffortLevel.MAX, EffortLevel.XHIGH),
            # Haiku: 1:1 except max
            ("claude-haiku-4-5", EffortLevel.LOW, EffortLevel.LOW),
            ("claude-haiku-4-5", EffortLevel.MAX, EffortLevel.XHIGH),
        ],
    )
    def test_known_pairs(self, model: str, effort_in: EffortLevel, effort_out: EffortLevel) -> None:
        assert map_effort_claude_to_codex(model, effort_in) == effort_out

    def test_string_effort_accepted(self) -> None:
        assert map_effort_claude_to_codex("claude-opus-4.7", "high") == EffortLevel.HIGH

    def test_unknown_string_returns_none(self) -> None:
        assert map_effort_claude_to_codex("claude-opus-4.7", "ludicrous") is None

    def test_no_model_passes_effort_through(self) -> None:
        assert map_effort_claude_to_codex(None, EffortLevel.HIGH) == EffortLevel.HIGH

    def test_unknown_model_passes_effort_through(self) -> None:
        assert map_effort_claude_to_codex("o3-mini", EffortLevel.HIGH) == EffortLevel.HIGH

    def test_none_effort_returns_none(self) -> None:
        assert map_effort_claude_to_codex("claude-opus-4.7", None) is None


class TestEffortCodexToClaude:
    @pytest.mark.parametrize(
        ("codex_model", "codex_effort", "claude_effort"),
        [
            # Opus default — same family, 1:1
            ("gpt-5.4", EffortLevel.LOW, EffortLevel.LOW),
            ("gpt-5.4", EffortLevel.MEDIUM, EffortLevel.MEDIUM),
            ("gpt-5.4", EffortLevel.HIGH, EffortLevel.HIGH),
            # Sonnet/Haiku default (gpt-5.4-mini) — Sonnet bias is reversed:
            # Codex MEDIUM came from Sonnet LOW, Codex HIGH came from Sonnet MEDIUM.
            ("gpt-5.4-mini", EffortLevel.MEDIUM, EffortLevel.LOW),
            ("gpt-5.4-mini", EffortLevel.HIGH, EffortLevel.MEDIUM),
        ],
    )
    def test_known_pairs(
        self, codex_model: str, codex_effort: EffortLevel, claude_effort: EffortLevel
    ) -> None:
        assert map_effort_codex_to_claude(codex_model, None, codex_effort) == claude_effort

    def test_target_model_overrides_codex_default(self) -> None:
        # Force Haiku so Codex MEDIUM maps to Haiku MEDIUM (1:1) rather than
        # Sonnet's default reverse mapping.
        result = map_effort_codex_to_claude("gpt-5.4-mini", "claude-haiku-4-5", EffortLevel.MEDIUM)
        assert result == EffortLevel.MEDIUM

    def test_xhigh_picks_lowest_source_tier(self) -> None:
        # Multiple Sonnet tiers map forward to xhigh; reverse should pick the
        # lowest one (HIGH) so we don't over-bill.
        result = map_effort_codex_to_claude("gpt-5.4-mini", "claude-sonnet-4.6", EffortLevel.XHIGH)
        assert result == EffortLevel.HIGH

    def test_none_effort_returns_none(self) -> None:
        assert map_effort_codex_to_claude("gpt-5.4", None, None) is None
