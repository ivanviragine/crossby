"""Tests for run_sync() orchestrator — filtering, continue-on-error, dry-run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import patch

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig, SyncConfig
from crossby.sync import run_sync
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncRegistry, SyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer(
    tool_id: AIToolID,
    concern: SyncConcern,
    action: Literal["created", "updated", "skipped", "error"] = "created",
    raises: Exception | None = None,
) -> AbstractSyncWriter:
    """Create a fake writer that either returns a result or raises."""

    class _W(AbstractSyncWriter):
        def __init__(self) -> None:
            self.tool_id = tool_id
            self.concern = concern
            self.calls: list[bool] = []
            self.force_calls: list[bool] = []

        def sync(
            self,
            config: CrossbyConfig,
            project_root: Path,
            *,
            dry_run: bool = False,
            force: bool = False,
        ) -> SyncResult:
            self.calls.append(dry_run)
            self.force_calls.append(force)
            if raises is not None:
                raise raises
            return SyncResult(tool_id=self.tool_id, concern=self.concern, action=action)

    return _W()


def _registry_with(*writers: AbstractSyncWriter) -> SyncRegistry:
    reg = SyncRegistry()
    for w in writers:
        reg.register(w)
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunSyncFiltering:
    def test_filters_by_tool(self, tmp_path: Path) -> None:
        w_claude = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_cursor = _make_writer(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        reg = _registry_with(w_claude, w_cursor)
        config = CrossbyConfig()

        results = run_sync(
            config,
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=reg,
        )
        assert len(results) == 1
        assert results[0].tool_id == AIToolID.CLAUDE

    def test_filters_by_concern(self, tmp_path: Path) -> None:
        w_perms = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_rules = _make_writer(AIToolID.CLAUDE, SyncConcern.RULES)
        reg = _registry_with(w_perms, w_rules)
        config = CrossbyConfig()

        results = run_sync(
            config,
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.RULES,
            registry=reg,
        )
        assert len(results) == 1
        assert results[0].concern == SyncConcern.RULES

    def test_filters_uninstalled_tools(self, tmp_path: Path) -> None:
        w_claude = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_cursor = _make_writer(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        reg = _registry_with(w_claude, w_cursor)
        config = CrossbyConfig()

        results = run_sync(
            config,
            tmp_path,
            installed_tools=[AIToolID.CLAUDE],
            registry=reg,
        )
        assert len(results) == 1
        assert results[0].tool_id == AIToolID.CLAUDE

    def test_no_installed_tools_returns_empty(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        config = CrossbyConfig()

        results = run_sync(config, tmp_path, installed_tools=[], registry=reg)
        assert results == []

    def test_explicit_tool_id_bypasses_installed_filter(self, tmp_path: Path) -> None:
        """When tool_id is explicit, installed_tools filter is not applied."""
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        config = CrossbyConfig()

        # Claude is not in installed_tools, but tool_id is explicit
        results = run_sync(
            config,
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            installed_tools=[],  # ignored when tool_id is set
            registry=reg,
        )
        assert len(results) == 1

    def test_sync_tools_config_filter(self, tmp_path: Path) -> None:
        """sync.tools config restricts which installed tools are synced."""
        w_claude = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_cursor = _make_writer(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        reg = _registry_with(w_claude, w_cursor)

        config = CrossbyConfig(sync=SyncConfig(tools=["claude"]))

        results = run_sync(
            config,
            tmp_path,
            installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR],
            registry=reg,
        )
        assert len(results) == 1
        assert results[0].tool_id == AIToolID.CLAUDE

    def test_empty_sync_tools_uses_all_installed(self, tmp_path: Path) -> None:
        """sync.tools=[] (default) allows all installed tools."""
        w_claude = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_cursor = _make_writer(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        reg = _registry_with(w_claude, w_cursor)
        config = CrossbyConfig(sync=SyncConfig(tools=[]))

        results = run_sync(
            config,
            tmp_path,
            installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR],
            registry=reg,
        )
        assert len(results) == 2


class TestRunSyncContinueOnError:
    def test_error_recorded_other_writers_continue(self, tmp_path: Path) -> None:
        w_fail = _make_writer(
            AIToolID.CLAUDE, SyncConcern.PERMISSIONS, raises=RuntimeError("boom")
        )
        w_ok = _make_writer(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        reg = _registry_with(w_fail, w_ok)
        config = CrossbyConfig()

        results = run_sync(
            config,
            tmp_path,
            installed_tools=[AIToolID.CLAUDE, AIToolID.CURSOR],
            registry=reg,
        )
        assert len(results) == 2
        error_results = [r for r in results if r.action == "error"]
        ok_results = [r for r in results if r.action != "error"]
        assert len(error_results) == 1
        assert error_results[0].tool_id == AIToolID.CLAUDE
        assert "boom" in (error_results[0].message or "")
        assert len(ok_results) == 1
        assert ok_results[0].tool_id == AIToolID.CURSOR

    def test_error_result_has_message(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS, raises=ValueError("bad config"))
        reg = _registry_with(w)
        config = CrossbyConfig()

        results = run_sync(
            config, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg
        )
        assert results[0].action == "error"
        assert "bad config" in (results[0].message or "")


class TestRunSyncDryRun:
    def test_dry_run_flag_passed_to_writer(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        config = CrossbyConfig()

        run_sync(config, tmp_path, tool_id=AIToolID.CLAUDE, dry_run=True, registry=reg)
        assert w.calls == [True]  # type: ignore[attr-defined]

    def test_no_dry_run_by_default(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        config = CrossbyConfig()

        run_sync(config, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert w.calls == [False]  # type: ignore[attr-defined]


class TestRunSyncForce:
    def test_force_flag_passed_to_writer(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        config = CrossbyConfig()

        run_sync(config, tmp_path, tool_id=AIToolID.CLAUDE, force=True, registry=reg)
        assert w.force_calls == [True]  # type: ignore[attr-defined]

    def test_no_force_by_default(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        config = CrossbyConfig()

        run_sync(config, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert w.force_calls == [False]  # type: ignore[attr-defined]


class TestRunSyncAutoDetect:
    def test_auto_detects_installed_tools_when_not_provided(
        self, tmp_path: Path
    ) -> None:
        """When installed_tools is None, detect_installed() is called."""
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        config = CrossbyConfig()

        with patch(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            return_value=[AIToolID.CLAUDE],
        ):
            results = run_sync(config, tmp_path, registry=reg)

        assert len(results) == 1
        assert results[0].tool_id == AIToolID.CLAUDE
