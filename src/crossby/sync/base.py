"""Sync framework base types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class SyncAction(StrEnum):
    """Outcome of a single sync target."""

    CREATED = "created"
    UPDATED = "updated"
    SKIPPED = "skipped"
    UP_TO_DATE = "up_to_date"
    ERROR = "error"


class SyncResult(BaseModel):
    """Result of syncing one target file."""

    target: str
    action: SyncAction
    message: str = ""
    dry_run: bool = False
