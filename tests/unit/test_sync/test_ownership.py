"""Tests for the provenance ledger (``sync/ownership.py``) and run_sync's use of it.

The ledger is what makes crossby's sync *revocable* without ever deleting a
hand-authored entry: revocation is computed as ``ledger_owned - current`` in
run_sync and handed to writers explicitly. These tests cover the ledger
round-trip, its tolerance of a missing/corrupt file, the error-row gating that
keeps idempotency honest, and the never-delete-unowned invariant end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry
from crossby.sync import run_sync
from crossby.sync.base import (
    AbstractSyncWriter,
    SyncConcern,
    SyncData,
    SyncRegistry,
    SyncResult,
)
from crossby.sync.hooks import ClaudeHooksWriter
from crossby.sync.ownership import (
    LEDGER_PATH,
    LEDGER_VERSION,
    OwnershipLedger,
    load_ledger,
    save_ledger,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_ledger_json(project_root: Path) -> dict:
    return json.loads((project_root / LEDGER_PATH).read_text(encoding="utf-8"))


def _make_writer(
    concern: SyncConcern,
    action: Literal["created", "updated", "skipped", "error"] = "updated",
) -> AbstractSyncWriter:
    the_concern = concern
    the_action = action

    class _W(AbstractSyncWriter):
        tool_id = AIToolID.CLAUDE

        def sync(
            self,
            data: SyncData,
            project_root: Path,
            *,
            dry_run: bool = False,
            force: bool = False,
        ) -> SyncResult:
            return SyncResult(tool_id=self.tool_id, concern=the_concern, action=the_action)

    _W.concern = the_concern
    return _W()


def _registry(*writers: AbstractSyncWriter) -> SyncRegistry:
    reg = SyncRegistry()
    for w in writers:
        reg.register(w)
    return reg


# ---------------------------------------------------------------------------
# OwnershipLedger round-trip
# ---------------------------------------------------------------------------


class TestLedgerRoundTrip:
    def test_record_and_read_back(self) -> None:
        ledger = OwnershipLedger()
        ledger.record_hooks(AIToolID.CLAUDE, {("pre_tool_use", "guard")})
        ledger.record_permissions(AIToolID.CLAUDE, {"git diff:*"})
        ledger.record_mcp(AIToolID.CURSOR, {"github"})

        assert ledger.hooks(AIToolID.CLAUDE) == frozenset({("pre_tool_use", "guard")})
        assert ledger.permissions(AIToolID.CLAUDE) == frozenset({"git diff:*"})
        assert ledger.mcp(AIToolID.CURSOR) == frozenset({"github"})
        # A tool/concern with no record reads as empty, not an error.
        assert ledger.hooks(AIToolID.CURSOR) == frozenset()

    def test_save_then_load(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        ledger.record_hooks(AIToolID.CLAUDE, {("pre_tool_use", "a"), ("stop", "b")})
        assert save_ledger(tmp_path, ledger) is True

        loaded = load_ledger(tmp_path)
        assert loaded.hooks(AIToolID.CLAUDE) == frozenset({("pre_tool_use", "a"), ("stop", "b")})

    def test_persisted_file_carries_a_schema_version(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        ledger.record_permissions(AIToolID.CLAUDE, {"x:*"})
        save_ledger(tmp_path, ledger)
        assert _read_ledger_json(tmp_path)["version"] == LEDGER_VERSION

    def test_recording_empty_clears_ownership(self) -> None:
        ledger = OwnershipLedger()
        ledger.record_hooks(AIToolID.CLAUDE, {("pre_tool_use", "guard")})
        ledger.record_hooks(AIToolID.CLAUDE, set())
        assert ledger.hooks(AIToolID.CLAUDE) == frozenset()
        assert ledger.is_empty()

    def test_save_is_a_noop_when_unchanged(self, tmp_path: Path) -> None:
        ledger = OwnershipLedger()
        ledger.record_mcp(AIToolID.CLAUDE, {"srv"})
        assert save_ledger(tmp_path, ledger) is True
        # Re-saving identical content reports "no change" so the caller can skip
        # a redundant .gitignore touch.
        assert save_ledger(tmp_path, load_ledger(tmp_path)) is False

    def test_empty_ledger_is_not_materialised(self, tmp_path: Path) -> None:
        assert save_ledger(tmp_path, OwnershipLedger()) is False
        assert not (tmp_path / LEDGER_PATH).exists()


class TestLedgerDegradesGracefully:
    def test_missing_file_owns_nothing(self, tmp_path: Path) -> None:
        ledger = load_ledger(tmp_path)
        assert ledger.is_empty()
        assert ledger.hooks(AIToolID.CLAUDE) == frozenset()

    def test_malformed_json_owns_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / LEDGER_PATH
        path.parent.mkdir(parents=True)
        path.write_text("{not valid json!!", encoding="utf-8")
        # Degrades to empty rather than raising.
        assert load_ledger(tmp_path).is_empty()

    def test_wrong_shape_owns_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / LEDGER_PATH
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        assert load_ledger(tmp_path).is_empty()

    def test_partial_garbage_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / LEDGER_PATH
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "owned": {
                        "claude": {"hooks": [["pre_tool_use", "ok"], "garbage", [1, 2]]},
                        "cursor": "not-a-dict",
                    },
                }
            ),
            encoding="utf-8",
        )
        ledger = load_ledger(tmp_path)
        # Only the well-shaped pair survives; the malformed neighbours are dropped.
        assert ledger.hooks(AIToolID.CLAUDE) == frozenset({("pre_tool_use", "ok")})
        assert ledger.hooks(AIToolID.CURSOR) == frozenset()


# ---------------------------------------------------------------------------
# run_sync ledger integration
# ---------------------------------------------------------------------------

_HOOK = HookEntry(event="pre_tool_use", command="guard", tools=["Edit"])


class TestRunSyncLedgerGating:
    def test_success_records_ownership(self, tmp_path: Path) -> None:
        reg = _registry(_make_writer(SyncConcern.HOOKS, "updated"))
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        ledger = load_ledger(tmp_path)
        assert ledger.hooks(AIToolID.CLAUDE) == frozenset({("pre_tool_use", "guard")})

    def test_skipped_also_records_ownership(self, tmp_path: Path) -> None:
        reg = _registry(_make_writer(SyncConcern.HOOKS, "skipped"))
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset(
            {("pre_tool_use", "guard")}
        )

    def test_error_row_leaves_ledger_untouched(self, tmp_path: Path) -> None:
        reg = _registry(_make_writer(SyncConcern.HOOKS, "error"))
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        # An error must not record work that did not happen — the ledger (and
        # its file) stay empty so the next run retries cleanly.
        assert not (tmp_path / LEDGER_PATH).exists()
        assert load_ledger(tmp_path).is_empty()

    def test_error_row_preserves_prior_ownership(self, tmp_path: Path) -> None:
        # First a real write records ownership.
        good = _registry(_make_writer(SyncConcern.HOOKS, "updated"))
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=good)
        # A later erroring run must not clear what the earlier run recorded.
        bad = _registry(_make_writer(SyncConcern.HOOKS, "error"))
        run_sync(SyncData(hooks=[]), tmp_path, tool_id=AIToolID.CLAUDE, registry=bad)
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset(
            {("pre_tool_use", "guard")}
        )

    def test_dry_run_writes_no_ledger(self, tmp_path: Path) -> None:
        reg = _registry(_make_writer(SyncConcern.HOOKS, "updated"))
        run_sync(
            SyncData(hooks=[_HOOK]),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=reg,
            dry_run=True,
        )
        assert not (tmp_path / LEDGER_PATH).exists()

    def test_ledger_is_gitignored(self, tmp_path: Path) -> None:
        reg = _registry(_make_writer(SyncConcern.HOOKS, "updated"))
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert LEDGER_PATH.as_posix() in gitignore


class TestRunSyncNeverDeletesUnowned:
    def _settings(self, root: Path) -> Path:
        return root / ".claude" / "settings.json"

    def test_hand_written_hook_absent_from_ledger_survives(self, tmp_path: Path) -> None:
        # A hook crossby never wrote (empty ledger) must not be removed even
        # though it is absent from the current SyncData.
        settings = self._settings(tmp_path)
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Edit", "hooks": [{"type": "command", "command": "human"}]}
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        reg = _registry(ClaudeHooksWriter())
        run_sync(
            SyncData(hooks=[HookEntry(event="pre_tool_use", command="crossby", tools=["Write"])]),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=reg,
        )
        commands = {
            inner["command"]
            for entry in json.loads(settings.read_text())["hooks"]["PreToolUse"]
            for inner in entry["hooks"]
        }
        # The human hook is untouched; crossby's is added alongside it.
        assert "human" in commands
        assert "crossby" in commands
