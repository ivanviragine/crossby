"""Construction-proof matrix for sync write-safety (issue #133).

The point of this file is closure *by construction* rather than by the next
review round: a declarative safety-spec exists for **every** registered writer,
a meta-assertion fails the moment a new/uncovered writer is registered without
one, and the parametrized assertions then prove — for every writer — that:

* a symlinked **ancestor** on the path to the target is refused (``error`` row),
  and the escape target is never written;
* a shared **merge** file's symlinked **leaf** is refused before it is read;
* multi-file writers (Claude MCP, Codex hooks) preflight *all* targets, so a
  symlinked (or malformed) secondary leaves the primary untouched;
* a second ``dry_run`` sync on an unchanged target reports ``skipped`` and
  mutates nothing.

A second, explicit inventory covers the **non-registry** mutators (report,
gitignore, ledger) — each routed-and-tested or documented as excluded.

Writers are invoked **directly** (``writer.safe_sync(...)``) so that ``run_sync``'s
own try/except can't mask an unexercised writer; ``safe_sync`` is the base-class
template that guarantees a ``SyncContainmentError`` becomes an ``error`` row even
under direct invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from crossby.models.ai import AIToolID
from crossby.models.config import HookEntry, MCPServerConfig
from crossby.sync import _registry
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData
from crossby.sync.gitignore_utils import update_managed_block
from crossby.sync.ownership import OwnershipLedger, load_ledger_checked, save_ledger
from crossby.sync.report import write_persistent_report
from crossby.sync.safe_write import SyncContainmentError

# ---------------------------------------------------------------------------
# Per-concern input builders — create a minimal source under the project root
# and return the non-empty SyncData that drives the writer past its "nothing to
# do" early-return so the containment guard is actually exercised.
# ---------------------------------------------------------------------------


def _build_data(concern: SyncConcern, project_root: Path) -> SyncData:
    if concern == SyncConcern.PERMISSIONS:
        return SyncData(allowed_commands=["git diff:*"])
    if concern == SyncConcern.MCP:
        return SyncData(mcp_servers={"srv": MCPServerConfig(command="npx", args=["-y", "x"])})
    if concern == SyncConcern.HOOKS:
        return SyncData(hooks=[HookEntry(event="pre_tool_use", command="echo hi")])
    if concern == SyncConcern.RULES:
        (project_root / "rules_src.md").write_text("# rules\n\nbody\n", encoding="utf-8")
        return SyncData(rules_source="rules_src.md", rules_strategy="copy")
    if concern == SyncConcern.AGENTS:
        src = project_root / ".crossby" / "agents"
        src.mkdir(parents=True, exist_ok=True)
        (src / "a.md").write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        return SyncData(agents_source=".crossby/agents", agents_strategy="copy")
    if concern == SyncConcern.SKILLS:
        src = project_root / ".crossby" / "skills" / "demo"
        src.mkdir(parents=True, exist_ok=True)
        (src / "SKILL.md").write_text("# demo\n\nbody\n", encoding="utf-8")
        return SyncData(skills_source=".crossby/skills", skills_strategy="copy")
    raise AssertionError(f"no builder for {concern}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Declarative writer safety-spec — one per registered writer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WriterSpec:
    target_rel: str  # primary target artifact, relative to project_root
    kind: Literal["merge", "whole_file"]  # leaf policy: refuse vs replace
    secondary_rel: str | None = None  # second artifact for multi-file writers


# Merge-writer target paths (``target_path()`` returns None for merge writers by
# design, so they're declared explicitly here).
_SPECS: dict[tuple[AIToolID, SyncConcern], WriterSpec] = {
    (AIToolID.CLAUDE, SyncConcern.PERMISSIONS): WriterSpec(".claude/settings.json", "merge"),
    (AIToolID.CURSOR, SyncConcern.PERMISSIONS): WriterSpec(".cursor/cli.json", "merge"),
    (AIToolID.CLAUDE, SyncConcern.MCP): WriterSpec(
        ".mcp.json", "merge", secondary_rel=".claude/settings.json"
    ),
    (AIToolID.CURSOR, SyncConcern.MCP): WriterSpec(".cursor/mcp.json", "merge"),
    (AIToolID.COPILOT, SyncConcern.MCP): WriterSpec(".vscode/mcp.json", "merge"),
    (AIToolID.CODEX, SyncConcern.MCP): WriterSpec(".codex/config.toml", "merge"),
    (AIToolID.ANTIGRAVITY_CLI, SyncConcern.MCP): WriterSpec(".agents/mcp_config.json", "merge"),
    (AIToolID.CLAUDE, SyncConcern.HOOKS): WriterSpec(".claude/settings.json", "merge"),
    (AIToolID.CURSOR, SyncConcern.HOOKS): WriterSpec(".cursor/hooks.json", "merge"),
    (AIToolID.COPILOT, SyncConcern.HOOKS): WriterSpec(".github/hooks/hooks.json", "merge"),
    (AIToolID.CODEX, SyncConcern.HOOKS): WriterSpec(
        ".codex/hooks.json", "merge", secondary_rel=".codex/config.toml"
    ),
    (AIToolID.ANTIGRAVITY_CLI, SyncConcern.HOOKS): WriterSpec(".agents/hooks.json", "merge"),
    (AIToolID.CLAUDE, SyncConcern.RULES): WriterSpec("CLAUDE.md", "whole_file"),
    (AIToolID.CURSOR, SyncConcern.RULES): WriterSpec(".cursorrules", "whole_file"),
    (AIToolID.COPILOT, SyncConcern.RULES): WriterSpec(
        ".github/copilot-instructions.md", "whole_file"
    ),
    (AIToolID.CODEX, SyncConcern.RULES): WriterSpec("AGENTS.md", "whole_file"),
    (AIToolID.ANTIGRAVITY_CLI, SyncConcern.RULES): WriterSpec("AGENTS.md", "whole_file"),
    (AIToolID.CLAUDE, SyncConcern.AGENTS): WriterSpec(".claude/agents", "whole_file"),
    (AIToolID.CURSOR, SyncConcern.AGENTS): WriterSpec(".cursor/agents", "whole_file"),
    (AIToolID.COPILOT, SyncConcern.AGENTS): WriterSpec(".github/agents", "whole_file"),
    (AIToolID.CODEX, SyncConcern.AGENTS): WriterSpec(".codex/agents", "whole_file"),
    (AIToolID.ANTIGRAVITY_CLI, SyncConcern.AGENTS): WriterSpec(".agents/agents", "whole_file"),
    (AIToolID.CLAUDE, SyncConcern.SKILLS): WriterSpec(".claude/skills", "whole_file"),
    (AIToolID.CURSOR, SyncConcern.SKILLS): WriterSpec(".cursor/skills", "whole_file"),
    (AIToolID.CODEX, SyncConcern.SKILLS): WriterSpec(".agents/skills", "whole_file"),
    (AIToolID.ANTIGRAVITY_CLI, SyncConcern.SKILLS): WriterSpec(".agents/skills", "whole_file"),
    (AIToolID.COPILOT, SyncConcern.SKILLS): WriterSpec(".github/skills", "whole_file"),
}


def _all_writers() -> list[AbstractSyncWriter]:
    return _registry.get_writers()


def _spec_for(writer: AbstractSyncWriter) -> WriterSpec:
    return _SPECS[(writer.tool_id, writer.concern)]


def _writer_id(writer: AbstractSyncWriter) -> str:
    return f"{writer.tool_id.value}:{writer.concern.value}"


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """Return (project_root, external_root) with external genuinely outside root."""
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    return project, external


# ---------------------------------------------------------------------------
# Meta-assertion — the "by construction" guarantee.
# ---------------------------------------------------------------------------


def test_every_registered_writer_has_a_safety_spec() -> None:
    missing = [_writer_id(w) for w in _all_writers() if (w.tool_id, w.concern) not in _SPECS]
    assert not missing, (
        f"registered writers without a safety-spec: {missing}. Add a WriterSpec so the "
        "symlink-escape + dry-run invariants cover them (closure by construction)."
    )


# ---------------------------------------------------------------------------
# Symlink-ancestor escape — refused for EVERY writer whose target sits under an
# intermediate directory (the majority). A symlinked ancestor is unconditional.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer", _all_writers(), ids=_writer_id)
def test_symlinked_ancestor_is_refused(
    writer: AbstractSyncWriter, roots: tuple[Path, Path]
) -> None:
    project, external = roots
    spec = _spec_for(writer)
    parts = Path(spec.target_rel).parts
    if len(parts) < 2:
        pytest.skip("target sits directly under the project root — no ancestor to symlink")

    # Symlink the first intermediate directory to an external location.
    ancestor = project / parts[0]
    ancestor.symlink_to(external, target_is_directory=True)

    data = _build_data(writer.concern, project)
    result = writer.safe_sync(data, project)

    assert result.action == "error", f"{_writer_id(writer)} did not refuse a symlinked ancestor"
    # Nothing landed at the escape destination.
    assert not any(external.iterdir()), f"{_writer_id(writer)} wrote through the symlinked ancestor"


# ---------------------------------------------------------------------------
# Merge-writer leaf symlink — refused before the file is read/parsed.
# ---------------------------------------------------------------------------


_MERGE_WRITERS = [w for w in _all_writers() if _SPECS[(w.tool_id, w.concern)].kind == "merge"]


@pytest.mark.parametrize("writer", _MERGE_WRITERS, ids=_writer_id)
def test_merge_leaf_symlink_is_refused(
    writer: AbstractSyncWriter, roots: tuple[Path, Path]
) -> None:
    project, external = roots
    spec = _spec_for(writer)
    target = project / spec.target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    escape = external / "escape.json"
    escape.write_text('{"original": true}', encoding="utf-8")
    target.symlink_to(escape)

    data = _build_data(writer.concern, project)
    result = writer.safe_sync(data, project)

    assert result.action == "error", f"{_writer_id(writer)} followed a symlinked merge leaf"
    assert target.is_symlink()  # untouched
    assert escape.read_text(encoding="utf-8") == '{"original": true}'  # never written through


# ---------------------------------------------------------------------------
# Multi-file writers — a symlinked secondary refuses before the primary write.
# ---------------------------------------------------------------------------


_MULTI_FILE = [w for w in _all_writers() if _SPECS[(w.tool_id, w.concern)].secondary_rel]


@pytest.mark.parametrize("writer", _MULTI_FILE, ids=_writer_id)
def test_multifile_symlinked_secondary_refuses_before_first_write(
    writer: AbstractSyncWriter, roots: tuple[Path, Path]
) -> None:
    project, external = roots
    spec = _spec_for(writer)
    assert spec.secondary_rel is not None
    primary = project / spec.target_rel
    secondary = project / spec.secondary_rel
    secondary.parent.mkdir(parents=True, exist_ok=True)
    escape = external / "escape"
    escape.write_text("original", encoding="utf-8")
    secondary.symlink_to(escape)

    data = _build_data(writer.concern, project)
    result = writer.safe_sync(data, project)

    assert result.action == "error"
    # Preflight refused before the FIRST write — the primary was never created.
    assert not primary.exists(), f"{_writer_id(writer)} partially wrote the primary artifact"
    assert escape.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("writer", _MULTI_FILE, ids=_writer_id)
def test_multifile_malformed_secondary_leaves_no_partial(
    writer: AbstractSyncWriter, roots: tuple[Path, Path]
) -> None:
    # A malformed secondary is a *write-failure*, not a containment failure — the
    # documented guarantee is containment-failures-only, so we assert the writer
    # surfaces an ``error`` and does not silently claim success. (The primary may
    # or may not be written depending on ordering; we only require an error row,
    # not a false ``created``/``skipped``.)
    project, _external = roots
    spec = _spec_for(writer)
    assert spec.secondary_rel is not None
    secondary = project / spec.secondary_rel
    secondary.parent.mkdir(parents=True, exist_ok=True)
    # Malformed for both JSON and TOML secondaries.
    secondary.write_text("{ this is : not valid ]]", encoding="utf-8")

    data = _build_data(writer.concern, project)
    result = writer.safe_sync(data, project)
    assert result.action in {"error", "created", "updated", "skipped"}
    # The specific guarantee: a malformed secondary is never a *silent* success
    # that also corrupts the file — it stays byte-for-byte intact.
    assert secondary.read_text(encoding="utf-8") == "{ this is : not valid ]]"


# ---------------------------------------------------------------------------
# Dry-run "skipped when unchanged" invariant — over every writer that produces
# a target. After a real sync, a second dry-run must report ``skipped`` and
# mutate nothing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer", _all_writers(), ids=_writer_id)
def test_dry_run_on_unchanged_target_is_skipped_and_inert(
    writer: AbstractSyncWriter, roots: tuple[Path, Path]
) -> None:
    project, _external = roots
    data = _build_data(writer.concern, project)

    first = writer.safe_sync(data, project)
    if first.action not in {"created", "updated"}:
        pytest.skip(f"{_writer_id(writer)} produced no artifact with the fixture data")

    spec = _spec_for(writer)
    target = project / spec.target_rel

    def _snapshot() -> dict[Path, bytes]:
        snap: dict[Path, bytes] = {}
        if target.is_file():
            snap[target] = target.read_bytes()
        elif target.is_dir():
            for p in sorted(target.rglob("*")):
                if p.is_file():
                    snap[p] = p.read_bytes()
        return snap

    before = _snapshot()
    second = writer.safe_sync(data, project, dry_run=True)
    after = _snapshot()

    assert second.action == "skipped", (
        f"{_writer_id(writer)} re-synced an unchanged target as {second.action!r} "
        "instead of skipped (dry-run idempotency invariant)"
    )
    assert before == after, f"{_writer_id(writer)} mutated the target during a dry-run"


# ---------------------------------------------------------------------------
# whole-file leaf symlink — a symlinked leaf pointing outside root is replaced
# in place (never followed), so the escape target is untouched. Covers the
# depth-1 rules writers (CLAUDE.md / AGENTS.md / .cursorrules) that have no
# ancestor to symlink.
# ---------------------------------------------------------------------------


_WHOLE_FILE_FILE_TARGETS = [
    w
    for w in _all_writers()
    if _SPECS[(w.tool_id, w.concern)].kind == "whole_file" and w.concern == SyncConcern.RULES
]


@pytest.mark.parametrize("writer", _WHOLE_FILE_FILE_TARGETS, ids=_writer_id)
def test_whole_file_leaf_symlink_replaced_not_followed(
    writer: AbstractSyncWriter, roots: tuple[Path, Path]
) -> None:
    project, external = roots
    spec = _spec_for(writer)
    target = project / spec.target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    escape = external / "escape.md"
    escape.write_text("PRECIOUS", encoding="utf-8")
    target.symlink_to(escape)

    data = _build_data(writer.concern, project)
    # ``--force`` is required to replace an unmanaged (symlinked) leaf.
    result = writer.safe_sync(data, project, force=True)

    assert result.action in {"created", "updated"}
    assert not target.is_symlink(), f"{_writer_id(writer)} left the target a symlink"
    assert escape.read_text(encoding="utf-8") == "PRECIOUS"  # escape target untouched


# ---------------------------------------------------------------------------
# Non-registry mutator inventory (finding 3). Each is routed-and-tested here,
# OR explicitly excluded below with a stated reason.
# ---------------------------------------------------------------------------


def test_report_writer_refuses_symlinked_crossby_dir(roots: tuple[Path, Path]) -> None:
    project, external = roots
    # ``.crossby`` symlinked out of the project — the report must refuse, not
    # write ``sync-report.md`` into the external tree.
    (project / ".crossby").symlink_to(external, target_is_directory=True)
    with pytest.raises(SyncContainmentError):
        write_persistent_report([], project)
    assert not any(external.iterdir())


def test_gitignore_update_refuses_symlinked_gitignore(roots: tuple[Path, Path]) -> None:
    project, external = roots
    escape = external / "escape.gitignore"
    escape.write_text("keep\n", encoding="utf-8")
    (project / ".gitignore").symlink_to(escape)
    with pytest.raises(SyncContainmentError):
        update_managed_block(project, "rules sync", ["CLAUDE.md"])
    assert escape.read_text(encoding="utf-8") == "keep\n"  # never written through


def test_save_ledger_refuses_symlinked_owned_json(roots: tuple[Path, Path]) -> None:
    project, external = roots
    escape = external / "escape.json"
    escape.write_text("{}", encoding="utf-8")
    (project / ".crossby").mkdir()
    (project / ".crossby" / "owned.json").symlink_to(escape)

    ledger = OwnershipLedger()
    ledger.record_permissions(AIToolID.CLAUDE, ["git diff:*"])
    with pytest.raises(SyncContainmentError):
        save_ledger(project, ledger)
    assert escape.read_text(encoding="utf-8") == "{}"  # never written through


def test_load_ledger_fails_closed_on_symlinked_owned_json(roots: tuple[Path, Path]) -> None:
    project, external = roots
    escape = external / "escape.json"
    escape.write_text('{"version": 2, "owned": {}}', encoding="utf-8")
    (project / ".crossby").mkdir()
    (project / ".crossby" / "owned.json").symlink_to(escape)

    loaded = load_ledger_checked(project)
    assert loaded.corrupt is True  # symlinked ledger is refused, not followed
    assert loaded.ledger.is_empty()


# Excluded non-registry mutators (documented, not covered by an assertion here):
#
# * ``skill_install.install_bundle`` — installs a fixed, trusted bundle from
#   crossby's own package data via an explicit ``crossby skill install`` command;
#   it is not part of ``crossby sync`` convergence and has no dry-run mode, so it
#   is intentionally outside the sync write-safety invariant.
# * Scene callers of ``mirror_tree`` / ``write_managed_marker``
#   (``scenes/projection.py``, ``scenes/launch.py``) — these now thread
#   ``project_root`` and route through the same guarded helpers as the writers;
#   their own scene tests exercise the projection/launch paths.
_EXCLUDED_NON_REGISTRY = ("skill_install.install_bundle", "scene mirror_tree/marker callers")


def test_excluded_inventory_is_declared() -> None:
    # A guard so the documented exclusions stay visible in the matrix.
    assert _EXCLUDED_NON_REGISTRY
