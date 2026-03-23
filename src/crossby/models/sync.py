"""Sync domain models — strategy, action, result."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class SyncStrategy(StrEnum):
    """How a config item should be ported to a target tool."""

    LINK = "link"
    CONVERT = "convert"
    WARN = "warn"
    UNSUPPORTED = "unsupported"


@dataclass
class SyncAction:
    """A single sync operation planned or performed."""

    config_type: str  # "instructions", "skills", "allowlist"
    strategy: SyncStrategy
    source_path: Path | None = None
    target_path: Path | None = None
    message: str = ""


@dataclass
class SyncResult:
    """Outcome of a sync operation."""

    actions: list[SyncAction] = field(default_factory=list)
    linked: int = 0
    converted: int = 0
    warnings: list[str] = field(default_factory=list)
