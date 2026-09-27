"""Shared value types."""

from __future__ import annotations

import json
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Use(StrEnum):
    """Things we may want to do with a source's data. Each needs its own permission."""

    COLLECT = "collect"
    RETAIN_HISTORY = "retain_history"
    EMBED = "embed"
    AI_PROCESSING = "ai_processing"
    DISPLAY_EXCERPTS = "display_excerpts"
    PUBLISH_DERIVED = "publish_derived"


class RightStatus(StrEnum):
    ALLOWED = "allowed"
    CONDITIONAL = "conditional"  # allowed under the conditions written in the registry
    PENDING = "pending"  # not confirmed yet: treated as not allowed
    DENIED = "denied"


class PriceType(StrEnum):
    """Price types answer different questions and must never be mixed silently."""

    LIST = "list"
    FIPE_REFERENCE = "fipe_reference"
    ASKING = "asking"
    TRANSACTION = "transaction"


class Observation(BaseModel):
    """One fact from one source about one subject for one period."""

    source_id: str
    metric: str
    subject_type: str
    subject_id: str
    dims: dict[str, str | int] = Field(default_factory=dict)
    as_of_start: date
    as_of_end: date
    value: float | None
    unit: str
    price_type: PriceType | None = None
    published_at: datetime | None = None

    @model_validator(mode="after")
    def _check_period(self) -> Observation:
        if self.as_of_end < self.as_of_start:
            raise ValueError("as_of_end is before as_of_start")
        return self

    def dims_key(self) -> str:
        """Canonical JSON for `dims`, so equal dims always compare equal in SQL."""
        return canonical_json(self.dims)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
