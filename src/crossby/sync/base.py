"""Sync framework base — SyncConcern, SyncData, SyncResult, AbstractSyncWriter, SyncRegistry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from crossby.models.ai import AIToolID

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


class AbstractSyncWriter(ABC):
    """Base for all sync writer adapters.

    Concrete subclasses must set ``tool_id`` and ``concern`` as class variables
    and implement ``sync()``.  Using ABC with @abstractmethod catches missing
    implementations at class definition time, consistent with AbstractAITool.
    """

    tool_id: AIToolID
    concern: SyncConcern

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
