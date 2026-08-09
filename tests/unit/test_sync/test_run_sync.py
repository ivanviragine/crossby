"""Tests for run_sync() orchestrator — filtering, continue-on-error, dry-run."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import patch

from crossby.models.ai import AIToolID
from crossby.sync import run_sync
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncRegistry, SyncResult

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
            data: SyncData,
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


def _make_whole_file_writer(
    tool_id: AIToolID,
    concern: SyncConcern,
    target_rel: str,
    action: Literal["created", "updated", "skipped", "error"] = "created",
) -> AbstractSyncWriter:
    """A whole-file overwrite writer (opts into target-path grouping)."""

    class _WF(AbstractSyncWriter):
        _owns_whole_file = True

        def __init__(self) -> None:
            self.tool_id = tool_id
            self.concern = concern
            self._target_rel = target_rel
            self.calls: list[bool] = []

        def sync(
            self,
            data: SyncData,
            project_root: Path,
            *,
            dry_run: bool = False,
            force: bool = False,
        ) -> SyncResult:
            self.calls.append(dry_run)
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=project_root / self._target_rel,
            )

    return _WF()


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
        data = SyncData()

        results = run_sync(
            data,
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
        data = SyncData()

        results = run_sync(
            data,
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
        data = SyncData()

        results = run_sync(
            data,
            tmp_path,
            installed_tools=[AIToolID.CLAUDE],
            registry=reg,
        )
        assert len(results) == 1
        assert results[0].tool_id == AIToolID.CLAUDE

    def test_no_installed_tools_returns_empty(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        data = SyncData()

        results = run_sync(data, tmp_path, installed_tools=[], registry=reg)
        assert results == []

    def test_explicit_tool_id_bypasses_installed_filter(self, tmp_path: Path) -> None:
        """When tool_id is explicit, installed_tools filter is not applied."""
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        data = SyncData()

        # Claude is not in installed_tools, but tool_id is explicit
        results = run_sync(
            data,
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            installed_tools=[],  # ignored when tool_id is set
            registry=reg,
        )
        assert len(results) == 1


class TestRunSyncWholeFileGrouping:
    """run_sync collapses whole-file writers that share one physical target."""

    def _rules_rows(self, results: list[SyncResult]) -> list[SyncResult]:
        return [r for r in results if r.concern == SyncConcern.RULES]

    def test_shared_target_collapses_to_first_registered_winner(self, tmp_path: Path) -> None:
        # Codex registered before Antigravity CLI → Codex is the deterministic
        # winner; agy is emitted covered-by without running its .sync().
        w_codex = _make_whole_file_writer(AIToolID.CODEX, SyncConcern.RULES, "AGENTS.md")
        w_agy = _make_whole_file_writer(AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES, "AGENTS.md")
        reg = _registry_with(w_codex, w_agy)

        results = run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI],
            registry=reg,
        )

        assert w_codex.calls == [False]  # type: ignore[attr-defined]  # winner ran
        assert w_agy.calls == []  # type: ignore[attr-defined]  # covered — never ran
        rows = self._rules_rows(results)
        ran = [r for r in rows if r.action != "skipped"]
        covered = [r for r in rows if r.action == "skipped"]
        assert [r.tool_id for r in ran] == [AIToolID.CODEX]
        assert len(covered) == 1
        assert covered[0].tool_id == AIToolID.ANTIGRAVITY_CLI
        assert covered[0].message == "covered by codex"
        assert covered[0].file_path == tmp_path / "AGENTS.md"

    def test_no_two_grouped_writers_run_for_one_path(self, tmp_path: Path) -> None:
        w_codex = _make_whole_file_writer(AIToolID.CODEX, SyncConcern.SKILLS, ".agents/skills")
        w_agy = _make_whole_file_writer(
            AIToolID.ANTIGRAVITY_CLI, SyncConcern.SKILLS, ".agents/skills"
        )
        reg = _registry_with(w_codex, w_agy)

        run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI],
            registry=reg,
        )
        # Exactly one of the two writers sharing the path actually ran.
        ran = [w for w in (w_codex, w_agy) if w.calls]  # type: ignore[attr-defined]
        assert len(ran) == 1

    def test_covered_row_keeps_original_position_when_interleaved(self, tmp_path: Path) -> None:
        # A non-group writer sits BETWEEN the two group members. The covered row
        # must stay in its original slot (last), not be hoisted next to the winner.
        w_codex = _make_whole_file_writer(AIToolID.CODEX, SyncConcern.RULES, "AGENTS.md")
        w_mid = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_agy = _make_whole_file_writer(AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES, "AGENTS.md")
        reg = _registry_with(w_codex, w_mid, w_agy)

        results = run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[AIToolID.CODEX, AIToolID.CLAUDE, AIToolID.ANTIGRAVITY_CLI],
            registry=reg,
        )
        order = [
            (r.tool_id, r.concern, r.action)
            for r in results
            if r.concern in {SyncConcern.RULES, SyncConcern.PERMISSIONS}
        ]
        assert order == [
            (AIToolID.CODEX, SyncConcern.RULES, "created"),
            (AIToolID.CLAUDE, SyncConcern.PERMISSIONS, "created"),
            (AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES, "skipped"),
        ]

    def test_grouping_applies_under_dry_run(self, tmp_path: Path) -> None:
        w_codex = _make_whole_file_writer(AIToolID.CODEX, SyncConcern.RULES, "AGENTS.md")
        w_agy = _make_whole_file_writer(AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES, "AGENTS.md")
        reg = _registry_with(w_codex, w_agy)

        results = run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI],
            dry_run=True,
            registry=reg,
        )
        assert w_codex.calls == [True]  # type: ignore[attr-defined]  # winner ran in dry-run
        assert w_agy.calls == []  # type: ignore[attr-defined]
        covered = [r for r in self._rules_rows(results) if r.action == "skipped"]
        assert len(covered) == 1
        assert covered[0].message == "covered by codex"

    def test_single_member_group_runs_when_codex_absent(self, tmp_path: Path) -> None:
        # agy-only install: the installed-tools filter removes Codex first, so agy
        # is the sole member of the AGENTS.md group and syncs normally.
        w_codex = _make_whole_file_writer(AIToolID.CODEX, SyncConcern.RULES, "AGENTS.md")
        w_agy = _make_whole_file_writer(AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES, "AGENTS.md")
        reg = _registry_with(w_codex, w_agy)

        results = run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[AIToolID.ANTIGRAVITY_CLI],
            registry=reg,
        )
        assert w_codex.calls == []  # type: ignore[attr-defined]  # filtered out
        assert w_agy.calls == [False]  # type: ignore[attr-defined]  # ran as sole member
        rows = self._rules_rows(results)
        assert len(rows) == 1
        assert rows[0].tool_id == AIToolID.ANTIGRAVITY_CLI
        assert rows[0].action != "skipped"

    def test_tool_filter_to_agy_runs_agy_alone(self, tmp_path: Path) -> None:
        # `--to antigravity-cli` → tool_id filter runs first; Codex is not in the
        # group, so agy syncs AGENTS.md.
        w_codex = _make_whole_file_writer(AIToolID.CODEX, SyncConcern.RULES, "AGENTS.md")
        w_agy = _make_whole_file_writer(AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES, "AGENTS.md")
        reg = _registry_with(w_codex, w_agy)

        run_sync(SyncData(), tmp_path, tool_id=AIToolID.ANTIGRAVITY_CLI, registry=reg)
        assert w_codex.calls == []  # type: ignore[attr-defined]
        assert w_agy.calls == [False]  # type: ignore[attr-defined]

    def test_winner_error_surfaces_and_still_covers_rest(self, tmp_path: Path) -> None:
        # No error-fallthrough: a winner whose sync returns error surfaces that
        # row, and the covered member is still emitted skipped (not promoted/run).
        w_codex = _make_whole_file_writer(
            AIToolID.CODEX, SyncConcern.RULES, "AGENTS.md", action="error"
        )
        w_agy = _make_whole_file_writer(AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES, "AGENTS.md")
        reg = _registry_with(w_codex, w_agy)

        results = run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI],
            registry=reg,
        )
        rows = self._rules_rows(results)
        assert any(r.tool_id == AIToolID.CODEX and r.action == "error" for r in rows)
        covered = [r for r in rows if r.tool_id == AIToolID.ANTIGRAVITY_CLI]
        assert len(covered) == 1
        assert covered[0].action == "skipped"
        assert w_agy.calls == []  # type: ignore[attr-defined]  # never promoted


class TestRunSyncContinueOnError:
    def test_error_recorded_other_writers_continue(self, tmp_path: Path) -> None:
        w_fail = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS, raises=RuntimeError("boom"))
        w_ok = _make_writer(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        reg = _registry_with(w_fail, w_ok)
        data = SyncData()

        results = run_sync(
            data,
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
        data = SyncData()

        results = run_sync(data, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert results[0].action == "error"
        assert "bad config" in (results[0].message or "")


class TestRunSyncDryRun:
    def test_dry_run_flag_passed_to_writer(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        data = SyncData()

        run_sync(data, tmp_path, tool_id=AIToolID.CLAUDE, dry_run=True, registry=reg)
        assert w.calls == [True]  # type: ignore[attr-defined]

    def test_no_dry_run_by_default(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        data = SyncData()

        run_sync(data, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert w.calls == [False]  # type: ignore[attr-defined]


class TestRunSyncForce:
    def test_force_flag_passed_to_writer(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        data = SyncData()

        run_sync(data, tmp_path, tool_id=AIToolID.CLAUDE, force=True, registry=reg)
        assert w.force_calls == [True]  # type: ignore[attr-defined]

    def test_no_force_by_default(self, tmp_path: Path) -> None:
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        data = SyncData()

        run_sync(data, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert w.force_calls == [False]  # type: ignore[attr-defined]


class TestRunSyncAutoDetect:
    def test_auto_detects_installed_tools_when_not_provided(self, tmp_path: Path) -> None:
        """When installed_tools is None, detect_installed() is called."""
        w = _make_writer(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg = _registry_with(w)
        data = SyncData()

        with patch(
            "crossby.ai_tools.base.AbstractAITool.detect_installed",
            return_value=[AIToolID.CLAUDE],
        ):
            results = run_sync(data, tmp_path, registry=reg)

        assert len(results) == 1
        assert results[0].tool_id == AIToolID.CLAUDE


class TestRunSyncPluginDiscovery:
    """Plugin discovery is injected into run_sync after writers."""

    def test_plugins_appear_in_results_when_present(self, tmp_path: Path) -> None:
        """A `.claude/plugins/<name>` dir produces one PLUGINS row per plugin."""
        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        (tmp_path / ".claude" / "plugins" / "team-macros").mkdir(parents=True)

        # Empty registry — only the plugin discovery should fire.
        results = run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[],
            registry=SyncRegistry(),
        )

        plugin_results = [r for r in results if r.concern == SyncConcern.PLUGINS]
        assert plugin_results
        assert all(r.action == "skipped" for r in plugin_results)
        assert any("team-macros" in (r.message or "") for r in plugin_results)

    def test_plugins_skipped_when_tool_id_filter_active(self, tmp_path: Path) -> None:
        """Per-tool runs don't reopen plugin discovery."""
        from crossby.models.ai import AIToolID
        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        (tmp_path / ".claude" / "plugins" / "team-macros").mkdir(parents=True)

        results = run_sync(
            SyncData(),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            installed_tools=[AIToolID.CLAUDE],
            registry=SyncRegistry(),
        )
        assert not [r for r in results if r.concern == SyncConcern.PLUGINS]

    def test_plugins_skipped_when_other_concern_filter_active(self, tmp_path: Path) -> None:
        """Asking for ``rules`` doesn't include plugin findings."""
        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        (tmp_path / ".claude" / "plugins" / "team-macros").mkdir(parents=True)

        results = run_sync(
            SyncData(),
            tmp_path,
            concern=SyncConcern.RULES,
            installed_tools=[],
            registry=SyncRegistry(),
        )
        assert not [r for r in results if r.concern == SyncConcern.PLUGINS]

    def test_plugins_concern_filter_keeps_only_plugin_rows(self, tmp_path: Path) -> None:
        """``concern=PLUGINS`` returns plugin rows even with no other writers."""
        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        (tmp_path / ".claude" / "plugins" / "team-macros").mkdir(parents=True)

        results = run_sync(
            SyncData(),
            tmp_path,
            concern=SyncConcern.PLUGINS,
            installed_tools=[],
            registry=SyncRegistry(),
        )
        assert results
        assert all(r.concern == SyncConcern.PLUGINS for r in results)

    def test_no_findings_when_no_plugin_dirs(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        results = run_sync(SyncData(), tmp_path, installed_tools=[], registry=SyncRegistry())
        assert not [r for r in results if r.concern == SyncConcern.PLUGINS]


class TestRunSyncOauthDiscovery:
    """MCP oauth-config discovery is injected into run_sync after writers,
    mirroring plugin discovery above."""

    def test_oauth_server_appears_in_results(self, tmp_path: Path) -> None:
        import json

        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        mcp_json = tmp_path / ".mcp.json"
        mcp_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "secure-srv": {
                            "url": "https://example.com/mcp",
                            "oauth": {"callbackPort": 3000},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        results = run_sync(
            SyncData(),
            tmp_path,
            installed_tools=[],
            registry=SyncRegistry(),
        )

        mcp_results = [r for r in results if r.concern == SyncConcern.MCP]
        assert mcp_results
        assert all(r.action == "skipped" and r.file_path is None for r in mcp_results)
        assert any("secure-srv" in (r.message or "") for r in mcp_results)

    def test_skipped_when_tool_id_filter_active(self, tmp_path: Path) -> None:
        import json

        from crossby.models.ai import AIToolID
        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"secure-srv": {"url": "x", "oauth": {}}}}),
            encoding="utf-8",
        )

        results = run_sync(
            SyncData(),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            installed_tools=[AIToolID.CLAUDE],
            registry=SyncRegistry(),
        )
        assert not [r for r in results if r.concern == SyncConcern.MCP]

    def test_no_findings_when_no_oauth_servers(self, tmp_path: Path) -> None:
        from crossby.sync import run_sync
        from crossby.sync.base import SyncConcern, SyncData, SyncRegistry

        results = run_sync(SyncData(), tmp_path, installed_tools=[], registry=SyncRegistry())
        assert not [r for r in results if r.concern == SyncConcern.MCP]


class TestRunSyncHooksRevocation:
    """The end-to-end revocable-sync behaviour: sync A then B → B only."""

    def _pre_tool_commands(self, tmp_path: Path) -> set[str]:
        import json

        data = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
        return {
            inner["command"]
            for entry in data["hooks"].get("PreToolUse", [])
            for inner in entry["hooks"]
        }

    def test_syncing_a_then_b_reflects_b_only(self, tmp_path: Path) -> None:
        from crossby.models.config import HookEntry
        from crossby.sync import run_sync
        from crossby.sync.base import SyncData, SyncRegistry
        from crossby.sync.hooks import ClaudeHooksWriter

        reg = SyncRegistry()
        reg.register(ClaudeHooksWriter())
        hook_a = HookEntry(event="pre_tool_use", command="guard-a", tools=["Edit"])
        hook_b = HookEntry(event="pre_tool_use", command="guard-b", tools=["Edit"])

        run_sync(SyncData(hooks=[hook_a]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert self._pre_tool_commands(tmp_path) == {"guard-a"}

        run_sync(SyncData(hooks=[hook_b]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        # A's hook is revoked (it was crossby-owned and absent from B); only B
        # remains — not the union.
        assert self._pre_tool_commands(tmp_path) == {"guard-b"}

    def test_second_identical_run_is_idempotent(self, tmp_path: Path) -> None:
        from crossby.models.config import HookEntry
        from crossby.sync import run_sync
        from crossby.sync.base import SyncData, SyncRegistry
        from crossby.sync.hooks import ClaudeHooksWriter

        reg = SyncRegistry()
        reg.register(ClaudeHooksWriter())
        data = SyncData(hooks=[HookEntry(event="pre_tool_use", command="guard", tools=["Edit"])])

        run_sync(data, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        results = run_sync(data, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert all(r.action == "skipped" for r in results if r.concern.value == "hooks")
