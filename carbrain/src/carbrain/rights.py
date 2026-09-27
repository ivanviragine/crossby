"""Source registry and usage-rights enforcement.

Permission to collect data is separate from permission to keep its history, embed it,
send it to an AI provider, show excerpts or publish derived work. Every one of those
uses goes through `Registry.require()`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from carbrain.models import RightStatus, Use


class RightsError(PermissionError):
    """Raised when a use of a source's data is not allowed."""


class SourceRights(BaseModel):
    collect: RightStatus
    retain_history: RightStatus
    embed: RightStatus
    ai_processing: RightStatus
    display_excerpts: RightStatus
    publish_derived: RightStatus

    def status(self, use: Use) -> RightStatus:
        status: RightStatus = getattr(self, use.value)
        return status


SourceType = Literal[
    "official_statistics",
    "vehicle_registry",
    "industry_association",
    "lab_testing",
    "price_reference",
    "specialist_media",
    "specialist_database",
    "social_platform",
    "consumer_complaints",
    "marketplace",
    "commercial_data",
]


class SourceSpec(BaseModel):
    id: str
    name: str
    type: SourceType
    url: str
    cadence: Literal["daily", "weekly", "monthly", "event"]
    freshness_days: int
    license: str
    attribution: str
    evidence: str
    rights: SourceRights
    conditions: dict[Use, str] = Field(default_factory=dict)
    retention_days: dict[str, int] = Field(default_factory=dict)
    dev_sample_limit: int | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Decision:
    source_id: str
    use: Use
    status: RightStatus
    condition: str | None
    dev_sample: bool = False


class Registry:
    def __init__(self, sources: dict[str, SourceSpec]) -> None:
        self._sources = sources

    @classmethod
    def load(cls, path: Path) -> Registry:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        sources = {
            source_id: SourceSpec(id=source_id, **spec)
            for source_id, spec in raw["sources"].items()
        }
        return cls(sources)

    def __iter__(self) -> Iterator[SourceSpec]:
        return iter(self._sources.values())

    def get(self, source_id: str) -> SourceSpec:
        try:
            return self._sources[source_id]
        except KeyError:
            raise RightsError(
                f"Unknown source '{source_id}'. Register it in sources.yaml."
            ) from None

    def check(self, source_id: str, use: Use) -> Decision:
        spec = self.get(source_id)
        status = spec.rights.status(use)
        return Decision(source_id, use, status, spec.conditions.get(use))

    def allows(self, source_id: str, use: Use) -> bool:
        return self.check(source_id, use).status in (RightStatus.ALLOWED, RightStatus.CONDITIONAL)

    def require(
        self,
        source_id: str,
        use: Use,
        *,
        dev_sample: bool = False,
        conn: sqlite3.Connection | None = None,
    ) -> Decision:
        """Return the decision if `use` is allowed, else raise `RightsError`.

        `dev_sample` lets a developer collect a few records from a source whose collection
        rights are still pending, capped by `dev_sample_limit` requests per UTC day.
        """
        decision = self.check(source_id, use)
        if decision.status in (RightStatus.ALLOWED, RightStatus.CONDITIONAL):
            return decision
        spec = self.get(source_id)
        if (
            dev_sample
            and use is Use.COLLECT
            and decision.status is RightStatus.PENDING
            and spec.dev_sample_limit
        ):
            used = requests_today(conn, source_id) if conn is not None else 0
            if used >= spec.dev_sample_limit:
                raise RightsError(
                    f"{spec.name}: dev-sample limit of {spec.dev_sample_limit} requests "
                    f"per day reached ({used} used)."
                )
            return Decision(source_id, use, decision.status, spec.notes, dev_sample=True)
        raise RightsError(
            f"{spec.name}: '{use.value}' is {decision.status.value}. {spec.evidence}. "
            "Update sources.yaml only with written evidence of permission."
        )


def requests_today(conn: sqlite3.Connection, source_id: str) -> int:
    today = datetime.now(UTC).date().isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(requests), 0) FROM run_log WHERE source_id = ? AND started_at >= ?",
        (source_id, today),
    ).fetchone()
    return int(row[0])
