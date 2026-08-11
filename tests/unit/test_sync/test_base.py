"""Tests for SyncRegistry, SyncResult, and AbstractSyncWriter."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossby.models.ai import AIToolID
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncRegistry, SyncResult


class _FakeWriter(AbstractSyncWriter):
    """Test writer that records calls."""

    def __init__(self, tool_id: AIToolID, concern: SyncConcern, action: str = "skipped") -> None:
        self.tool_id = tool_id
        self.concern = concern
        self._action = action
        self.calls: list[tuple[SyncData, Path, bool]] = []

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        self.calls.append((data, project_root, dry_run))
        return SyncResult(tool_id=self.tool_id, concern=self.concern, action=self._action)  # type: ignore[arg-type]


class TestSyncRegistry:
    def test_register_and_get(self) -> None:
        reg = SyncRegistry()
        w = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg.register(w)
        assert reg.get_writers() == [w]

    def test_register_overwrites_same_key(self) -> None:
        reg = SyncRegistry()
        w1 = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w2 = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        reg.register(w1)
        reg.register(w2)
        writers = reg.get_writers()
        assert writers == [w2]

    def test_get_writers_filter_by_tool(self) -> None:
        reg = SyncRegistry()
        w_claude = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_cursor = _FakeWriter(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        reg.register(w_claude)
        reg.register(w_cursor)

        result = reg.get_writers(tool_id=AIToolID.CLAUDE)
        assert result == [w_claude]

    def test_get_writers_filter_by_concern(self) -> None:
        reg = SyncRegistry()
        w_perms = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w_rules = _FakeWriter(AIToolID.CLAUDE, SyncConcern.RULES)
        reg.register(w_perms)
        reg.register(w_rules)

        result = reg.get_writers(concern=SyncConcern.RULES)
        assert result == [w_rules]

    def test_get_writers_filter_both(self) -> None:
        reg = SyncRegistry()
        w1 = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        w2 = _FakeWriter(AIToolID.CURSOR, SyncConcern.PERMISSIONS)
        w3 = _FakeWriter(AIToolID.CLAUDE, SyncConcern.RULES)
        for w in [w1, w2, w3]:
            reg.register(w)

        result = reg.get_writers(tool_id=AIToolID.CLAUDE, concern=SyncConcern.PERMISSIONS)
        assert result == [w1]

    def test_get_writers_no_filters_returns_all(self) -> None:
        reg = SyncRegistry()
        writers = [
            _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS),
            _FakeWriter(AIToolID.CURSOR, SyncConcern.PERMISSIONS),
        ]
        for w in writers:
            reg.register(w)
        assert set(reg.get_writers()) == set(writers)

    def test_empty_registry_returns_empty(self) -> None:
        reg = SyncRegistry()
        assert reg.get_writers() == []


class TestSyncResult:
    def test_minimal_result(self) -> None:
        r = SyncResult(tool_id=AIToolID.CLAUDE, concern=SyncConcern.PERMISSIONS, action="skipped")
        assert r.tool_id == AIToolID.CLAUDE
        assert r.concern == SyncConcern.PERMISSIONS
        assert r.action == "skipped"
        assert r.file_path is None
        assert r.message is None

    def test_full_result(self, tmp_path: Path) -> None:
        fp = tmp_path / "settings.json"
        r = SyncResult(
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.PERMISSIONS,
            action="created",
            file_path=fp,
            message="test",
        )
        assert r.file_path == fp
        assert r.message == "test"


class TestTargetPath:
    """Whole-file ownership opt-in used by run_sync's target grouping."""

    def test_default_writer_returns_none(self, tmp_path: Path) -> None:
        # A writer that never opted in (default ``_owns_whole_file = False``).
        w = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        assert w.target_path(tmp_path) is None

    def test_rules_writer_returns_display_path(self, tmp_path: Path) -> None:
        from crossby.sync.rules import CodexRulesWriter

        assert CodexRulesWriter().target_path(tmp_path) == tmp_path / "AGENTS.md"

    def test_rules_writers_share_agents_md(self, tmp_path: Path) -> None:
        from crossby.sync.rules import AntigravityCLIRulesWriter, CodexRulesWriter

        assert (
            CodexRulesWriter().target_path(tmp_path)
            == AntigravityCLIRulesWriter().target_path(tmp_path)
            == tmp_path / "AGENTS.md"
        )

    def test_skills_writers_share_agents_skills(self, tmp_path: Path) -> None:
        from crossby.sync.skills import AntigravityCLISkillsWriter, CodexSkillsWriter

        assert (
            CodexSkillsWriter().target_path(tmp_path)
            == AntigravityCLISkillsWriter().target_path(tmp_path)
            == tmp_path / ".agents" / "skills"
        )

    def test_agents_writers_have_distinct_paths(self, tmp_path: Path) -> None:
        from crossby.sync.agents import (
            AntigravityCLIAgentsWriter,
            ClaudeAgentsWriter,
            CodexAgentsWriter,
            CopilotAgentsWriter,
            CursorAgentsWriter,
        )

        paths = [
            w().target_path(tmp_path)
            for w in (
                ClaudeAgentsWriter,
                CursorAgentsWriter,
                CopilotAgentsWriter,
                CodexAgentsWriter,
                AntigravityCLIAgentsWriter,
            )
        ]
        assert all(p is not None for p in paths)
        assert len(set(paths)) == len(paths)  # no accidental collisions

    def test_merge_writers_return_none(self, tmp_path: Path) -> None:
        # Merge writers co-write shared files (.claude/settings.json,
        # .codex/config.toml) by key — they must NEVER be grouped, so every one
        # returns None regardless of any _target_rel it may carry internally.
        from crossby.sync.hooks import (
            AntigravityCLIHooksWriter,
            ClaudeHooksWriter,
            CodexHooksWriter,
            CopilotHooksWriter,
            CursorHooksWriter,
        )
        from crossby.sync.mcp import (
            AntigravityCLIMCPWriter,
            ClaudeMCPWriter,
            CodexMCPWriter,
            CopilotMCPWriter,
            CursorMCPWriter,
        )
        from crossby.sync.permissions import ClaudePermissionWriter, CursorPermissionWriter

        merge_writers = [
            ClaudePermissionWriter(),
            CursorPermissionWriter(scope="project"),
            ClaudeMCPWriter(),
            CursorMCPWriter(),
            CopilotMCPWriter(),
            CodexMCPWriter(),
            AntigravityCLIMCPWriter(),
            ClaudeHooksWriter(),
            CursorHooksWriter(),
            CopilotHooksWriter(),
            CodexHooksWriter(),
            AntigravityCLIHooksWriter(),
        ]
        for w in merge_writers:
            assert w.target_path(tmp_path) is None, f"{type(w).__name__} must not be grouped"


class TestAbstractSyncWriter:
    def test_abstract_method_required(self) -> None:
        """Cannot instantiate AbstractSyncWriter without implementing sync()."""
        with pytest.raises(TypeError):
            AbstractSyncWriter()  # type: ignore[abstract]

    def test_concrete_writer_callable(self, tmp_path: Path) -> None:
        w = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        data = SyncData()
        result = w.sync(data, tmp_path)
        assert result.action == "skipped"
        assert w.calls == [(data, tmp_path, False)]

    def test_dry_run_passed_through(self, tmp_path: Path) -> None:
        w = _FakeWriter(AIToolID.CLAUDE, SyncConcern.PERMISSIONS)
        data = SyncData()
        w.sync(data, tmp_path, dry_run=True)
        assert w.calls[0][2] is True
