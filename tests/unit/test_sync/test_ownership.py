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

import pytest

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry, MCPServerConfig
from crossby.sync import run_sync
from crossby.sync.base import (
    AbstractSyncWriter,
    SyncConcern,
    SyncData,
    SyncRegistry,
    SyncResult,
)
from crossby.sync.hooks import ClaudeHooksWriter
from crossby.sync.mcp import ClaudeMCPWriter
from crossby.sync.ownership import (
    LEDGER_PATH,
    LEDGER_VERSION,
    OwnershipLedger,
    load_ledger,
    save_ledger,
)
from crossby.sync.permissions import ClaudePermissionWriter

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
            # Simulate a real writer: report the identities it "wrote fresh" so
            # run_sync records ownership from them. A skip/error creates nothing.
            created: tuple[object, ...] = ()
            if the_action in {"created", "updated"}:
                if the_concern == SyncConcern.HOOKS:
                    created = tuple((h.event, h.command) for h in data.hooks)
                elif the_concern == SyncConcern.PERMISSIONS:
                    created = tuple(data.allowed_commands)
                elif the_concern == SyncConcern.MCP:
                    created = tuple(n for n, s in data.mcp_servers.items() if s.enabled)
            return SyncResult(
                tool_id=self.tool_id, concern=the_concern, action=the_action, created=created
            )

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

    def test_only_writer_created_ids_are_recorded(self, tmp_path: Path) -> None:
        # A writer that reports no `created` (e.g. the hook was already present,
        # authored by hand) must NOT have that identity recorded as owned — this
        # is what keeps crossby from later revoking a human entry.
        reg = _registry(_make_writer(SyncConcern.HOOKS, "skipped"))
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset()

    def test_skip_after_create_preserves_ownership(self, tmp_path: Path) -> None:
        # A real writer: first sync creates+owns the hook; an identical second
        # sync skips (created nothing) but ownership must persist.
        reg = _registry(ClaudeHooksWriter())
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset(
            {("pre_tool_use", "guard")}
        )
        results = run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert all(r.action == "skipped" for r in results)
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

    def test_ledger_save_oserror_does_not_discard_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The ledger is advisory: an OSError persisting it (read-only dir, full
        # disk) must not propagate out of run_sync and discard the SyncResults
        # for writes that already succeeded — matching run_sync's per-writer
        # isolation. save_ledger is imported inside run_sync, so patch the source.
        def _boom(*_args: object, **_kwargs: object) -> bool:
            raise OSError("read-only file system")

        monkeypatch.setattr("crossby.sync.ownership.save_ledger", _boom)
        results = run_sync(
            SyncData(hooks=[_HOOK]),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=_registry(ClaudeHooksWriter()),
        )
        # The write result survives the failed ledger save …
        assert any(r.concern == SyncConcern.HOOKS and r.action != "error" for r in results)
        # … and the hook actually landed on disk.
        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert "PreToolUse" in settings["hooks"]

    def test_gitignore_update_self_heals_after_transient_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A transient OSError updating .gitignore must not leave owned.json
        # unignored forever: because an identical re-sync makes save_ledger
        # return False, the retry is decoupled to run whenever the ledger file
        # exists. Fail the first .gitignore update, then let the second heal it.
        import crossby.sync.gitignore_utils as gi

        real = gi.update_managed_block
        calls = {"n": 0}

        def _flaky(*args: object, **kwargs: object) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient .gitignore failure")
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("crossby.sync.gitignore_utils.update_managed_block", _flaky)
        reg = _registry(ClaudeHooksWriter())
        gitignore = tmp_path / ".gitignore"

        # Sync 1: ledger is written, but the .gitignore update fails transiently.
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert (tmp_path / LEDGER_PATH).is_file()
        assert not gitignore.is_file() or LEDGER_PATH.as_posix() not in gitignore.read_text()

        # Sync 2: identical, so save_ledger no-ops — but the retry still runs
        # because the ledger file exists, and the block is now written.
        run_sync(SyncData(hooks=[_HOOK]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert LEDGER_PATH.as_posix() in gitignore.read_text()


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


class TestProvenanceNotIdentityCoincidence:
    """Ownership follows what crossby WROTE, not what merely matches the source.

    The subtle regression these guard against: recording the whole source set as
    owned would let crossby claim (and then narrow/revoke) a human entry that
    happens to share a ``(event, command)`` / pattern with the source.
    """

    def _settings(self, root: Path) -> Path:
        return root / ".claude" / "settings.json"

    def _pre_tool_entry(self, root: Path) -> dict:
        return json.loads(self._settings(root).read_text())["hooks"]["PreToolUse"][0]

    def test_human_hook_sharing_source_identity_is_never_narrowed_or_revoked(
        self, tmp_path: Path
    ) -> None:
        settings = self._settings(tmp_path)
        settings.parent.mkdir(parents=True)
        # Human wrote (pre_tool_use, guard) with a deliberately broad matcher.
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": ".*", "hooks": [{"type": "command", "command": "guard"}]}
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        reg = _registry(ClaudeHooksWriter())
        source = SyncData(hooks=[HookEntry(event="pre_tool_use", command="guard", tools=["Edit"])])

        # Sync 1: the command already exists and crossby did not create it — the
        # broad matcher is preserved (widen-only) and crossby claims nothing.
        run_sync(source, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert self._pre_tool_entry(tmp_path)["matcher"] == ".*"
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset()

        # Sync 2: STILL not owned, so the matcher is never narrowed to "Edit".
        run_sync(source, tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert self._pre_tool_entry(tmp_path)["matcher"] == ".*"

        # Sync 3: source drops the hook — the human's entry must NOT be revoked.
        run_sync(SyncData(hooks=[]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        commands = {
            inner["command"]
            for entry in json.loads(settings.read_text())["hooks"].get("PreToolUse", [])
            for inner in entry["hooks"]
        }
        assert "guard" in commands

    def test_crossby_written_hook_is_revoked_when_source_empties(self, tmp_path: Path) -> None:
        reg = _registry(ClaudeHooksWriter())
        run_sync(
            SyncData(hooks=[HookEntry(event="pre_tool_use", command="owned", tools=["Edit"])]),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=reg,
        )
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset(
            {("pre_tool_use", "owned")}
        )
        # Source now has no hooks at all — crossby's own entry is revoked.
        run_sync(SyncData(hooks=[]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        assert "PreToolUse" not in json.loads(self._settings(tmp_path).read_text()).get("hooks", {})
        assert load_ledger(tmp_path).hooks(AIToolID.CLAUDE) == frozenset()

    def test_human_permission_sharing_source_identity_survives(self, tmp_path: Path) -> None:
        settings = self._settings(tmp_path)
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"permissions": {"allow": ["Bash(git diff:*)"]}}))
        reg = _registry(ClaudePermissionWriter())

        # Source has git diff:* but it is already present → crossby creates
        # nothing and claims nothing.
        run_sync(
            SyncData(allowed_commands=["git diff:*"]),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=reg,
        )
        assert load_ledger(tmp_path).permissions(AIToolID.CLAUDE) == frozenset()

        # Source drops it → the human's pattern is not revoked.
        run_sync(SyncData(allowed_commands=[]), tmp_path, tool_id=AIToolID.CLAUDE, registry=reg)
        allow = json.loads(settings.read_text())["permissions"]["allow"]
        assert "Bash(git diff:*)" in allow

    def test_human_mcp_server_sharing_source_name_is_never_claimed_or_revoked(
        self, tmp_path: Path
    ) -> None:
        # MCP ownership is creation-only: overwriting a same-named hand-authored
        # server applies the merge but claims nothing, so the later disable path
        # (bounded by ``disabled ∩ owned``) can never delete it.
        mcp = tmp_path / ".mcp.json"
        mcp.write_text(
            json.dumps({"mcpServers": {"shared": {"command": "hand-written"}}}), encoding="utf-8"
        )
        reg = _registry(ClaudeMCPWriter())

        # Sync 1: source has `shared` enabled with a different config → crossby
        # overwrites the entry but, because it already existed, claims nothing.
        run_sync(
            SyncData(mcp_servers={"shared": MCPServerConfig(command="npx")}),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=reg,
        )
        assert load_ledger(tmp_path).mcp(AIToolID.CLAUDE) == frozenset()

        # Sync 2: source disables `shared`. Deletion is bounded by ownership and
        # crossby owns nothing, so the server survives.
        run_sync(
            SyncData(mcp_servers={"shared": MCPServerConfig(command="npx", enabled=False)}),
            tmp_path,
            tool_id=AIToolID.CLAUDE,
            registry=reg,
        )
        assert "shared" in json.loads(mcp.read_text())["mcpServers"]
