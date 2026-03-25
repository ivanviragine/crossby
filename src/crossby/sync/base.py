"""Sync framework base — SyncConcern, SyncResult, AbstractSyncWriter, SyncRegistry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig


class SyncConcern(StrEnum):
    """Top-level sync categories — each maps to a set of writers."""

    PERMISSIONS = "permissions"
    RULES = "rules"
    MCP = "mcp"
    AGENTS = "agents"


@dataclass
class SyncResult:
    """Result from a single sync writer run."""

    tool_id: AIToolID | None
    concern: SyncConcern
    action: Literal["created", "updated", "skipped", "error"]
    file_path: Path | None = None
    message: str | None = None


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
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        """Sync config to tool-specific files.

        Args:
            config: Loaded CrossbyConfig.
            project_root: Project root directory.
            dry_run: If True, compute the result without writing any files.
            force: If True, overwrite existing target directories (with backup).

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
