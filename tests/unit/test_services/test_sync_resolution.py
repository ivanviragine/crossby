"""Tests for ``crossby.services.sync_resolution`` resolvers.

Each resolver mirrors the fallback chain of :mod:`ai_resolution`
(CLI → ``.crossby.yml`` → auto-detect) so tests pin each layer's handoff
independently.
"""

from __future__ import annotations

import pytest

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig, SyncDefaults
from crossby.services.sync_resolution import (
    resolve_sync_concern,
    resolve_sync_from,
    resolve_sync_to,
)
from crossby.sync.base import SyncConcern


class TestResolveSyncFrom:
    def test_explicit_wins(self) -> None:
        cfg = CrossbyConfig(sync_defaults=SyncDefaults(from_tool=AIToolID.CURSOR))
        assert resolve_sync_from("claude", cfg) is AIToolID.CLAUDE

    def test_invalid_explicit_raises(self) -> None:
        cfg = CrossbyConfig()
        with pytest.raises(ValueError):
            resolve_sync_from("nosuchtool", cfg)

    def test_config_fallback(self) -> None:
        cfg = CrossbyConfig(sync_defaults=SyncDefaults(from_tool=AIToolID.CURSOR))
        assert resolve_sync_from(None, cfg) is AIToolID.CURSOR

    def test_returns_none_without_auto_detect(self) -> None:
        cfg = CrossbyConfig()
        assert resolve_sync_from(None, cfg, auto_detect=False) is None

    def test_auto_detect_returns_first_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            classmethod(lambda _cls: [AIToolID.CODEX, AIToolID.CLAUDE]),
        )
        cfg = CrossbyConfig()
        assert resolve_sync_from(None, cfg) is AIToolID.CODEX


class TestResolveSyncTo:
    def test_explicit_wins(self) -> None:
        cfg = CrossbyConfig(sync_defaults=SyncDefaults(to=AIToolID.COPILOT))
        assert resolve_sync_to("cursor", cfg) is AIToolID.CURSOR

    def test_config_fallback(self) -> None:
        cfg = CrossbyConfig(sync_defaults=SyncDefaults(to=AIToolID.COPILOT))
        assert resolve_sync_to(None, cfg) is AIToolID.COPILOT

    def test_missing_returns_none(self) -> None:
        assert resolve_sync_to(None, CrossbyConfig()) is None


class TestResolveSyncConcern:
    def test_explicit_wins(self) -> None:
        cfg = CrossbyConfig(sync_defaults=SyncDefaults(concern="mcp"))
        assert resolve_sync_concern("rules", cfg) is SyncConcern.RULES

    def test_config_fallback(self) -> None:
        cfg = CrossbyConfig(sync_defaults=SyncDefaults(concern="mcp"))
        assert resolve_sync_concern(None, cfg) is SyncConcern.MCP

    def test_missing_returns_none(self) -> None:
        assert resolve_sync_concern(None, CrossbyConfig()) is None

    def test_invalid_concern_raises(self) -> None:
        cfg = CrossbyConfig()
        with pytest.raises(ValueError):
            resolve_sync_concern("nosuch", cfg)
