"""Vehicle catalog: loading the seed, recording source mappings, and the review queue."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from carbrain.db import utcnow
from carbrain.resolve import Match, Resolver


def load_catalog(conn: sqlite3.Connection, resolver: Resolver) -> int:
    """Upsert brands and families from the resolver's seed. Returns the family count."""
    for brand in resolver.brands.values():
        conn.execute(
            "INSERT INTO brand (id, name) VALUES (?, ?) "
            "ON CONFLICT (id) DO UPDATE SET name = excluded.name",
            (brand.id, brand.name),
        )
    for fam in resolver.families.values():
        conn.execute(
            "INSERT INTO family (id, brand_id, name, segment, body, status, powertrains, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET "
            "brand_id = excluded.brand_id, name = excluded.name, segment = excluded.segment, "
            "body = excluded.body, status = excluded.status, "
            "powertrains = excluded.powertrains, notes = excluded.notes",
            (
                fam.id,
                fam.brand,
                fam.name,
                fam.segment,
                fam.body,
                fam.status,
                json.dumps(fam.powertrains),
                fam.notes,
            ),
        )
    conn.commit()
    return len(resolver.families)


def load_events(conn: sqlite3.Connection, path: Path) -> int:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for ev in raw["events"]:
        conn.execute(
            "INSERT INTO event (id, date, kind, title, scope, source_url) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET date = excluded.date, kind = excluded.kind, "
            "title = excluded.title, scope = excluded.scope, source_url = excluded.source_url",
            (ev["id"], str(ev["date"]), ev["kind"], ev["title"], ev.get("scope"), ev.get("url")),
        )
    conn.commit()
    return len(raw["events"])


def record_match(
    conn: sqlite3.Connection, source_id: str, external_key: str, match: Match
) -> str | None:
    """Store how a source names a vehicle. Returns the family ID when the mapping is usable.

    A person's decision (confirmed/rejected) is never overwritten by automatic matching.
    Ambiguous matches go to the review queue.
    """
    existing = conn.execute(
        "SELECT target_id, status FROM external_mapping WHERE source_id = ? AND external_key = ?",
        (source_id, external_key),
    ).fetchone()
    if existing is not None and existing["status"] in ("confirmed", "rejected"):
        return str(existing["target_id"]) if existing["status"] == "confirmed" else None
    if match.accepted and match.family_id is not None:
        conn.execute(
            "INSERT INTO external_mapping (source_id, external_key, level, target_id, confidence, "
            "method, status, created_at) VALUES (?, ?, 'family', ?, ?, ?, 'auto', ?) "
            "ON CONFLICT (source_id, external_key) DO UPDATE SET target_id = excluded.target_id, "
            "confidence = excluded.confidence, method = excluded.method",
            (source_id, external_key, match.family_id, match.confidence, match.method, utcnow()),
        )
        return match.family_id
    if match.method == "ambiguous":
        enqueue_review(
            conn,
            source_id,
            external_key,
            candidate_id=",".join(match.candidates),
            confidence=match.confidence,
            reason="Several families match equally well",
        )
    return None


def enqueue_review(
    conn: sqlite3.Connection,
    source_id: str,
    external_key: str,
    *,
    candidate_id: str | None,
    confidence: float | None,
    reason: str,
) -> None:
    conn.execute(
        "INSERT INTO review_queue (source_id, external_key, candidate_level, candidate_id, "
        "confidence, reason, created_at) VALUES (?, ?, 'family', ?, ?, ?, ?) "
        "ON CONFLICT (source_id, external_key) DO NOTHING",
        (source_id, external_key, candidate_id, confidence, reason, utcnow()),
    )


@dataclass(frozen=True)
class ReviewItem:
    id: int
    source_id: str
    external_key: str
    candidate_id: str | None
    confidence: float | None
    reason: str


def open_reviews(conn: sqlite3.Connection, limit: int = 50) -> list[ReviewItem]:
    rows = conn.execute(
        "SELECT id, source_id, external_key, candidate_id, confidence, reason FROM review_queue "
        "WHERE resolved_at IS NULL ORDER BY confidence DESC, id LIMIT ?",
        (limit,),
    )
    return [ReviewItem(**dict(r)) for r in rows]


def resolve_review(
    conn: sqlite3.Connection, review_id: int, *, family_id: str | None, reviewer_note: str = ""
) -> None:
    """Confirm a mapping to `family_id`, or reject the label when `family_id` is None."""
    item = conn.execute("SELECT * FROM review_queue WHERE id = ?", (review_id,)).fetchone()
    if item is None:
        raise KeyError(f"No review item {review_id}")
    if (
        family_id is not None
        and conn.execute("SELECT 1 FROM family WHERE id = ?", (family_id,)).fetchone() is None
    ):
        raise KeyError(f"Unknown family {family_id}")
    status = "confirmed" if family_id else "rejected"
    conn.execute(
        "INSERT INTO external_mapping (source_id, external_key, level, target_id, confidence, "
        "method, status, created_at) VALUES (?, ?, 'family', ?, 1.0, 'human', ?, ?) "
        "ON CONFLICT (source_id, external_key) DO UPDATE SET target_id = excluded.target_id, "
        "confidence = 1.0, method = 'human', status = excluded.status",
        (item["source_id"], item["external_key"], family_id or "", status, utcnow()),
    )
    conn.execute(
        "UPDATE review_queue SET resolved_at = ?, resolution = ? WHERE id = ?",
        (utcnow(), f"{status}:{family_id or ''} {reviewer_note}".strip(), review_id),
    )
    conn.commit()
