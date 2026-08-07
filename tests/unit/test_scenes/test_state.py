"""``content_hash`` / ``detect_drift`` — symlink-aware fingerprinting.

The regression these guard: a symlinked JSON/TOML config (a symlink-to-file) used
to be fingerprinted only by its link target, so editing the target's *contents*
left the hash unchanged and ``scene status`` never reported the drift. A symlinked
*directory* (a PROJECT re-point) must keep its dir signature.
"""

from __future__ import annotations

import json
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.scenes.state import (
    SceneState,
    SceneToolRecord,
    compute_hashes,
    content_hash,
    detect_drift,
)
from crossby.sync.base import SyncConcern, SyncResult


class TestContentHashSymlinks:
    def test_symlink_to_file_reflects_target_edit(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text(json.dumps({"a": 1}), encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(target)

        before = content_hash(link)
        # A symlink-to-file is hashed by its resolved contents, so an edit to the
        # link target now changes the hash (the old logic hashed the link target
        # string only and would have returned the same value).
        target.write_text(json.dumps({"a": 2}), encoding="utf-8")
        assert content_hash(link) != before
        # And it matches hashing the real file directly.
        assert content_hash(link) == content_hash(target)

    def test_symlink_to_file_ignores_neutral_reformat(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text('{"a": 1, "b": 2}', encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(target)

        before = content_hash(link)
        # Re-serialise the target with reordered keys / whitespace — normalised
        # JSON hashes identically, so this is not drift.
        target.write_text('{\n  "b": 2,\n  "a": 1\n}\n', encoding="utf-8")
        assert content_hash(link) == before

    def test_symlink_to_dir_keeps_dir_signature(self, tmp_path: Path) -> None:
        real = tmp_path / "src"
        real.mkdir()
        (real / "one").write_text("x", encoding="utf-8")
        link = tmp_path / "linkdir"
        link.symlink_to(real, target_is_directory=True)

        sig = content_hash(link)
        # A dir signature is entry names, not contents: editing a file inside is
        # not drift (matches a PROJECT re-point).
        (real / "one").write_text("changed", encoding="utf-8")
        assert content_hash(link) == sig
        # Adding an entry changes the signature.
        (real / "two").write_text("y", encoding="utf-8")
        assert content_hash(link) != sig

    def test_broken_symlink_is_missing(self, tmp_path: Path) -> None:
        link = tmp_path / "dangling.json"
        link.symlink_to(tmp_path / "does-not-exist.json")
        assert content_hash(link) == "missing"

    def test_real_file_and_dir_still_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "f.json"
        f.write_text(json.dumps({"a": 1}), encoding="utf-8")
        assert content_hash(f).startswith("sha256:")
        d = tmp_path / "d"
        d.mkdir()
        assert content_hash(d).startswith("sha256:")

    def test_missing_path_is_missing(self, tmp_path: Path) -> None:
        assert content_hash(tmp_path / "nope.json") == "missing"


class TestDetectDriftSymlinkedBaseline:
    def test_edit_of_symlinked_config_target_registers_as_drift(self, tmp_path: Path) -> None:
        target = tmp_path / ".claude" / "settings.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"skillOverrides": {"x": "off"}}), encoding="utf-8")
        link_rel = "linked-settings.json"
        (tmp_path / link_rel).symlink_to(target)

        state = SceneState(
            scene="s",
            applied_at="2026-01-01T00:00:00Z",
            status="applied",
            tools={"claude": SceneToolRecord(hashes={link_rel: content_hash(tmp_path / link_rel)})},
        )
        # Fresh baseline: no drift.
        assert detect_drift(tmp_path, state) == []

        # Editing the link *target*'s contents is now detected as drift for the
        # symlinked path (previously invisible under the target-string hash).
        target.write_text(json.dumps({"skillOverrides": {"x": "on"}}), encoding="utf-8")
        assert detect_drift(tmp_path, state) == [link_rel]


class TestComputeHashesPartialWrite:
    """An ``error`` row that still wrote something (a Codex mixed splice) must be
    baselined so ``status``/``clear`` can detect later drift on the partially
    written file; a pure failure (wrote nothing) must not.
    """

    def test_error_row_that_wrote_is_baselined(self, tmp_path: Path) -> None:
        cfg = tmp_path / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("[mcp_servers.github]\nenabled = false\n", encoding="utf-8")
        result = SyncResult(
            tool_id=AIToolID.CODEX,
            concern=SyncConcern.MCP,
            action="error",
            file_path=cfg,
            added=1,
        )
        hashes = compute_hashes(tmp_path, [result])
        assert hashes["codex"][".codex/config.toml"].startswith("sha256:")

    def test_error_row_that_wrote_nothing_is_skipped(self, tmp_path: Path) -> None:
        cfg = tmp_path / ".claude" / "settings.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("{}", encoding="utf-8")
        result = SyncResult(
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.MCP,
            action="error",
            file_path=cfg,
        )
        assert compute_hashes(tmp_path, [result]) == {}
