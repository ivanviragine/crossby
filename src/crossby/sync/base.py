"""Sync framework base — SyncConcern, SyncData, SyncResult, AbstractSyncWriter, SyncRegistry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from crossby.models.ai import AIToolID
from crossby.sync.file_utils import first_symlinked_ancestor
from crossby.sync.safe_write import SyncContainmentError

if TYPE_CHECKING:
    from crossby.models.config import HookEntry, MCPServerConfig


class SyncConcern(StrEnum):
    """Top-level sync categories — each maps to a set of writers."""

    PERMISSIONS = "permissions"
    RULES = "rules"
    MCP = "mcp"
    AGENTS = "agents"
    SKILLS = "skills"
    HOOKS = "hooks"
    PLUGINS = "plugins"


@dataclass
class SyncData:
    """Sync input data — populated by readers, consumed by writers.

    Replaces ``CrossbyConfig`` in the sync layer.  Each field group
    corresponds to one :class:`SyncConcern`.  A ``None`` source means
    "nothing to sync for this concern" and the writer will skip.
    """

    # Rules concern
    rules_source: str | None = None  # relative path to canonical instruction file
    rules_strategy: Literal["symlink", "copy"] = "symlink"
    rules_gitignore: bool = True

    # Agents concern
    agents_source: str | None = None  # relative path to canonical agents directory
    agents_strategy: Literal["symlink", "copy", "translate"] = "symlink"
    agents_gitignore: bool = True

    # Skills concern
    skills_source: str | None = None  # relative path to canonical skills directory
    skills_strategy: Literal["symlink", "copy", "translate"] = "symlink"
    skills_gitignore: bool = True

    # MCP servers concern
    mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict)

    # Permissions concern
    allowed_commands: list[str] = field(default_factory=list)

    # Hooks concern
    hooks: list[HookEntry] = field(default_factory=list)

    # --- Revocation channels ------------------------------------------------
    # Computed *exclusively* by ``run_sync()`` from the ownership ledger
    # (``sync/ownership.py``) and handed to writers explicitly. These are the
    # ONLY inputs a writer may act on to remove an item — a writer must never
    # infer "absent from the additive fields above ⇒ remove", or the direct
    # single-hook callers (``config/claude_allowlist.configure_plan_hooks`` and
    # friends) would wipe every other entry in the target file on every session
    # setup. All default to empty, so those direct callers revoke nothing.

    # Canonical ``(event, command)`` pairs to remove from each hooks target.
    hooks_remove: list[tuple[str, str]] = field(default_factory=list)
    # Canonical ``(event, command)`` pairs crossby already owns for the target
    # tool. Consulted only to decide whether a matcher may *narrow* (crossby's
    # own over-broad matcher) versus must *widen* (protect a human's broader
    # scope) — never used to remove anything.
    hooks_owned: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    # Canonical permission patterns to remove from each permissions target.
    permissions_remove: list[str] = field(default_factory=list)
    # MCP server names crossby may delete (ledger-bounded disabled set).
    mcp_remove: frozenset[str] = field(default_factory=frozenset)


@dataclass
class SyncResult:
    """Result from a single sync writer run."""

    tool_id: AIToolID | None
    concern: SyncConcern
    action: Literal["created", "updated", "skipped", "error"]
    file_path: Path | None = None
    message: str | None = None
    # Counts used by the report/plan to distinguish a revocation-only row from
    # an additive one. ``added`` counts entries written/widened this run;
    # ``revoked`` counts entries removed. A row with ``revoked > 0`` and
    # ``added == 0`` classifies as ``Removed`` rather than ``Added``.
    added: int = 0
    revoked: int = 0
    # Identities the writer wrote **fresh** this run (a new hook entry, a newly
    # added permission pattern, a newly written MCP server) — NOT ones that were
    # already present. ``run_sync`` records ownership from this, never from the
    # whole source set, so crossby never claims (and later narrows/revokes) a
    # human entry that merely shares an identity with a source entry. Hooks use
    # ``(event, command)`` tuples; permissions/MCP use strings.
    created: tuple[object, ...] = ()
    # True when this row reports a concern the tool has *no* lever for — a
    # narrowing the scene asked for that simply cannot be applied (e.g. a scene
    # dropping an MCP server on a tool with no per-server disable key). Distinct
    # from a benign ``skipped`` (already applied / already linked): the launch
    # fallback surfaces only ``unsupported`` rows as warnings, never benign
    # skips. Set by the ``UNSUPPORTED`` branch of ``scenes/engine._declare_mcp``.
    unsupported: bool = False


class AbstractSyncWriter(ABC):
    """Base for all sync writer adapters.

    Concrete subclasses must set ``tool_id`` and ``concern`` as class variables
    and implement ``sync()``.  Using ABC with @abstractmethod catches missing
    implementations at class definition time, consistent with AbstractAITool.
    """

    tool_id: AIToolID
    concern: SyncConcern

    # Whole-file ownership opt-in. Only writers that *overwrite an entire
    # physical artifact* (rules/agents/skills) set this to True; ``run_sync``
    # then groups every writer sharing one target path and lets a single winner
    # write it (see ``run_sync``). Merge-style writers (permissions, MCP, hooks)
    # leave it False: they co-write shared files (``.claude/settings.json``,
    # ``.codex/config.toml``) by key and must never be collapsed to one writer.
    # The flag is an *explicit* opt-in — the grouping never keys off a stray
    # ``_target_rel`` attribute, so a future merge writer that happens to define
    # ``_target_rel`` is not grouped by accident.
    _owns_whole_file: bool = False

    def target_path(self, project_root: Path) -> Path | None:
        """Physical whole-file artifact this writer overwrites, or ``None``.

        Returns ``project_root / self._target_rel`` for whole-file overwrite
        writers (those with ``_owns_whole_file = True`` and a ``_target_rel``);
        ``None`` for merge writers and anything without a declared target.

        This is the *display* path used for :attr:`SyncResult.file_path`.
        ``run_sync`` derives the grouping key separately (canonicalising the
        parent) so a symlinked project root never leaks into the reported path.
        """
        if not self._owns_whole_file:
            return None
        rel = getattr(self, "_target_rel", None)
        if not isinstance(rel, str):
            return None
        return project_root / rel

    def contained_or_error(self, project_root: Path, target: Path) -> SyncResult | None:
        """Return an ``error`` result when a *parent* of *target* is a symlink.

        ``mkdir(parents=True)`` and ``create_symlink`` follow a symlinked ancestor
        (e.g. ``.agents -> /outside``), landing writes outside the project even
        when the final target component is guarded. Every writer must call this
        before creating or writing its target so no sync can escape the project
        root through a symlinked tool directory.
        """
        bad = first_symlinked_ancestor(project_root, target)
        if bad is None:
            return None
        try:
            shown_bad = bad.relative_to(project_root)
            shown_target: Path | str = target.relative_to(project_root)
        except ValueError:
            shown_bad, shown_target = bad, target
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="error",
            message=(
                f"{shown_bad} is a symlinked directory on the path to "
                f"{shown_target}; refusing to write through it (it may point "
                "outside the project). Remove the symlink and re-run."
            ),
        )

    def merge_target_or_error(self, project_root: Path, target: Path) -> SyncResult | None:
        """Ancestor **and leaf** containment for a shared *merge* config file.

        ``contained_or_error`` guards only the *ancestor* chain (a symlinked
        parent), because whole-file writers replace a symlink *leaf* in place.
        Merge writers (permissions / MCP / hooks) instead read-modify-write a
        shared file the user may legitimately keep as a symlink into a dotfiles
        repo — following it would write through to an arbitrary destination. So
        for those files a symlink **leaf** is refused too, and — critically —
        this runs **before** the file is read/parsed, so a refused symlink never
        reads or rewrites the target. Returns an ``error`` result or ``None``.
        """
        err = self.contained_or_error(project_root, target)
        if err is not None:
            return err
        if target.is_symlink():
            try:
                shown: Path | str = target.relative_to(project_root)
            except ValueError:
                shown = target
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=(
                    f"{shown} is a symlink; refusing to write through it (it may point "
                    "outside the project). Remove the symlink and re-run."
                ),
            )
        return None

    def preflight(self, project_root: Path, *targets: Path) -> SyncResult | None:
        """Ancestor+leaf-contain **all** *targets* before the first write.

        Multi-file writers (``ClaudeMCPWriter``, ``CodexHooksWriter``) write two
        artifacts. Calling this on every target up front means a refused symlink
        on the *second* target can't leave the *first* partially applied — the
        writer bails with the returned ``error`` result before touching disk.
        Uses :meth:`merge_target_or_error` (leaf-aware) since both multi-file
        writers are merge writers. Returns the first containment error, or
        ``None`` when every target is clear.
        """
        for target in targets:
            err = self.merge_target_or_error(project_root, target)
            if err is not None:
                return err
        return None

    def _containment_error(self, exc: SyncContainmentError) -> SyncResult:
        """Wrap a :class:`SyncContainmentError` into an ``error`` result row."""
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="error",
            message=str(exc),
        )

    def safe_sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        """Run :meth:`sync` and convert any escaped :class:`SyncContainmentError`
        into an ``action="error"`` result.

        ``run_sync`` wraps each writer in its own try/except, but a writer may be
        invoked directly (the safety-spec matrix does exactly this). Routing all
        direct invocations through here guarantees a refused write surfaces as an
        ``error`` row rather than an escaped exception, regardless of caller — so
        every writer satisfies the "refusal ⇒ error row" contract by construction.
        """
        try:
            return self.sync(data, project_root, dry_run=dry_run, force=force)
        except SyncContainmentError as exc:
            return self._containment_error(exc)

    @abstractmethod
    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        """Sync data to tool-specific files.

        Args:
            data: Sync input data (from readers or wizard).
            project_root: Project root directory.
            dry_run: If True, compute the result without writing any files.
            force: If True, overwrite existing target files/directories (with
                backup).  Merge-style writers (permissions, MCP) perform
                non-destructive appends, so ``force`` is a no-op for them.

        Returns:
            SyncResult describing what happened.
        """
        ...


class SyncRegistry:
    """Registry of sync writers keyed by (tool_id, concern).

    Each (tool_id, concern) pair maps to exactly one writer instance.
    Registering a writer for an existing key overwrites the previous one.
    """

    def __init__(self) -> None:
        self._writers: dict[tuple[AIToolID, SyncConcern], AbstractSyncWriter] = {}

    def register(self, writer: AbstractSyncWriter) -> None:
        """Register a writer. Overwrites any existing for the same key."""
        self._writers[(writer.tool_id, writer.concern)] = writer

    def get_writers(
        self,
        *,
        tool_id: AIToolID | None = None,
        concern: SyncConcern | None = None,
    ) -> list[AbstractSyncWriter]:
        """Return writers optionally filtered by tool_id and/or concern."""
        writers = list(self._writers.values())
        if tool_id is not None:
            writers = [w for w in writers if w.tool_id == tool_id]
        if concern is not None:
            writers = [w for w in writers if w.concern == concern]
        return writers
