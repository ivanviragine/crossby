"""Sync framework base classes — AbstractSyncWriter, SyncResult, SyncAction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from crossby.models.config import MCPServerConfig

SyncAction = Literal["created", "updated", "skipped", "error"]


@dataclass
class SyncResult:
    """Result of a single sync writer operation."""

    tool: str
    path: Path
    action: SyncAction
    message: str = field(default="")
    dry_run: bool = field(default=False)


class AbstractSyncWriter(ABC):
    """Base class for all sync writers."""

    @property
    @abstractmethod
    def tool_id(self) -> str:
        """Tool identifier (e.g. 'claude', 'cursor')."""

    @abstractmethod
    def write(
        self,
        servers: dict[str, MCPServerConfig],
        project_root: Path,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        """Write MCP server config for this tool.

        Args:
            servers: All servers from .crossby.yml (enabled and disabled).
            project_root: Project root directory.
            dry_run: If True, compute what would change but don't write.

        Returns:
            List of SyncResult (one per affected file/operation).
        """
