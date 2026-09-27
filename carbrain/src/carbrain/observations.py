"""Writing and reading facts.

A fact is keyed by (source, metric, subject, dims, period). Writing the same key again
with the same value is a no-op; a different value becomes a new revision, so every
correction a source makes stays visible.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from carbrain.models import Observation, canonical_json


@dataclass
class WriteStats:
    inserted: int = 0
    revised: int = 0
    unchanged: int = 0
    revised_keys: list[str] = field(default_factory=list)


def write_observations(
    conn: sqlite3.Connection,
    observations: Iterable[Observation],
    *,
    fetched_at: str,
    snapshot_id: int | None,
) -> WriteStats:
    stats = WriteStats()
    for obs in observations:
        dims = obs.dims_key()
        key = (
            obs.source_id,
            obs.metric,
            obs.subject_type,
            obs.subject_id,
            dims,
            obs.as_of_start.isoformat(),
            obs.as_of_end.isoformat(),
        )
        current = conn.execute(
            "SELECT revision, value, unit, price_type FROM observation "
            "WHERE source_id = ? AND metric = ? AND subject_type = ? AND subject_id = ? "
            "AND dims = ? AND as_of_start = ? AND as_of_end = ? "
            "ORDER BY revision DESC LIMIT 1",
            key,
        ).fetchone()
        price_type = obs.price_type.value if obs.price_type else None
        if current is not None and (
            _same(current["value"], obs.value)
            and current["unit"] == obs.unit
            and current["price_type"] == price_type
        ):
            stats.unchanged += 1
            continue
        revision = 1 if current is None else current["revision"] + 1
        conn.execute(
            "INSERT INTO observation (source_id, metric, subject_type, subject_id, dims, "
            "as_of_start, as_of_end, value, unit, price_type, published_at, fetched_at, "
            "snapshot_id, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *key,
                obs.value,
                obs.unit,
                price_type,
                obs.published_at.isoformat() if obs.published_at else None,
                fetched_at,
                snapshot_id,
                revision,
            ),
        )
        if revision == 1:
            stats.inserted += 1
        else:
            stats.revised += 1
            stats.revised_keys.append("|".join(key))
    conn.commit()
    return stats


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


@dataclass(frozen=True)
class Fact:
    """A stored fact as returned to callers, with everything needed to cite it."""

    source_id: str
    metric: str
    subject_type: str
    subject_id: str
    dims: dict[str, Any]
    as_of_start: date
    as_of_end: date
    value: float | None
    unit: str
    price_type: str | None
    fetched_at: str
    revision: int


def _to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        source_id=row["source_id"],
        metric=row["metric"],
        subject_type=row["subject_type"],
        subject_id=row["subject_id"],
        dims=json.loads(row["dims"]),
        as_of_start=date.fromisoformat(row["as_of_start"]),
        as_of_end=date.fromisoformat(row["as_of_end"]),
        value=row["value"],
        unit=row["unit"],
        price_type=row["price_type"],
        fetched_at=row["fetched_at"],
        revision=row["revision"],
    )


def query_facts(
    conn: sqlite3.Connection,
    metric: str,
    *,
    subject_type: str | None = None,
    subject_id: str | None = None,
    dims: dict[str, str | int] | None = None,
    source_id: str | None = None,
    since: date | None = None,
    latest_only: bool = False,
) -> list[Fact]:
    """Current revision of matching facts, oldest period first.

    `dims` filters on exact values of the given keys; other keys may have any value.
    With `latest_only`, only facts for the most recent period are returned.
    """
    sql = ["SELECT * FROM observation_current WHERE metric = ?"]
    params: list[Any] = [metric]
    for column, value in (
        ("subject_type", subject_type),
        ("subject_id", subject_id),
        ("source_id", source_id),
    ):
        if value is not None:
            sql.append(f"AND {column} = ?")
            params.append(value)
    if since is not None:
        sql.append("AND as_of_start >= ?")
        params.append(since.isoformat())
    for dim_key, dim_value in (dims or {}).items():
        sql.append("AND json_extract(dims, ?) = ?")
        params.extend([f"$.{dim_key}", dim_value])
    sql.append("ORDER BY as_of_start, subject_id, dims")
    facts = [_to_fact(r) for r in conn.execute(" ".join(sql), params)]
    if latest_only and facts:
        last = max(f.as_of_start for f in facts)
        facts = [f for f in facts if f.as_of_start == last]
    return facts


def revisions(conn: sqlite3.Connection, fact: Fact) -> list[Fact]:
    """Every stored revision of the fact's key, oldest first."""
    rows = conn.execute(
        "SELECT * FROM observation WHERE source_id = ? AND metric = ? AND subject_type = ? "
        "AND subject_id = ? AND dims = ? AND as_of_start = ? AND as_of_end = ? ORDER BY revision",
        (
            fact.source_id,
            fact.metric,
            fact.subject_type,
            fact.subject_id,
            canonical_json(fact.dims),
            fact.as_of_start.isoformat(),
            fact.as_of_end.isoformat(),
        ),
    )
    return [_to_fact(r) for r in rows]
