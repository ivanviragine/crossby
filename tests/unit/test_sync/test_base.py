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

    def sync(self, data: SyncData, project_root: Path, *, dry_run: bool = False, force: bool = False) -> SyncResult:
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
