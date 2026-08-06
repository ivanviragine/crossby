"""The ownership ledger's scene DECLARE section (v2 schema extension)."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.sync.ownership import (
    LEDGER_PATH,
    LEDGER_VERSION,
    OwnershipLedger,
    SceneDeclareKey,
    load_ledger,
    save_ledger,
)


def _read(root: Path) -> dict:
    return json.loads((root / LEDGER_PATH).read_text(encoding="utf-8"))


def test_scene_section_round_trips(tmp_path: Path) -> None:
    ledger = OwnershipLedger()
    ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES, {"a", "b"})
    ledger.record_scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED, {"linear"})
    assert save_ledger(tmp_path, ledger) is True

    loaded = load_ledger(tmp_path)
    assert loaded.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES) == frozenset(
        {"a", "b"}
    )
    assert loaded.scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED) == frozenset(
        {"linear"}
    )
    assert _read(tmp_path)["version"] == LEDGER_VERSION


def test_recording_empty_clears_and_may_empty_the_ledger(tmp_path: Path) -> None:
    ledger = OwnershipLedger()
    ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS, {"Agent(x)"})
    assert not ledger.is_empty()
    ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS, set())
    assert ledger.is_empty()


def test_scene_section_absent_when_empty(tmp_path: Path) -> None:
    # A ledger with only revocable-sync ownership serialises exactly as under v1.
    ledger = OwnershipLedger()
    ledger.record_mcp(AIToolID.CLAUDE, {"srv"})
    save_ledger(tmp_path, ledger)
    assert "scene" not in _read(tmp_path)


def test_v1_file_without_scene_loads(tmp_path: Path) -> None:
    path = tmp_path / LEDGER_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"version": 1, "owned": {"claude": {"mcp": ["srv"]}}}), encoding="utf-8"
    )
    ledger = load_ledger(tmp_path)
    assert ledger.mcp(AIToolID.CLAUDE) == frozenset({"srv"})
    assert ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES) == frozenset()


def test_malformed_scene_section_degrades(tmp_path: Path) -> None:
    path = tmp_path / LEDGER_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "owned": {},
                "scene": {"claude": {"skill_overrides": ["ok", 5, None]}, "bad": "x"},
            }
        ),
        encoding="utf-8",
    )
    ledger = load_ledger(tmp_path)
    # Non-string members dropped; the malformed tool entry ignored.
    assert ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES) == frozenset(
        {"ok"}
    )
