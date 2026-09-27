"""Retention: text evidence expires when the source's terms say so."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from carbrain.models import Use
from carbrain.rights import Registry, RightsError


def store_text(
    conn: sqlite3.Connection,
    registry: Registry,
    *,
    source_id: str,
    external_id: str,
    text: str,
    published_at: str | None = None,
    vehicle: tuple[str, str, float] | None = None,
    now: datetime | None = None,
) -> None:
    """Store a text item with an expiry taken from the source's `retention_days.text`.

    Collection must be allowed. Without a text retention limit, keeping the text also
    needs `retain_history`.
    """
    registry.require(source_id, Use.COLLECT)
    spec = registry.get(source_id)
    now = now or datetime.now(UTC)
    days = spec.retention_days.get("text")
    if days is None:
        registry.require(source_id, Use.RETAIN_HISTORY)
        expires_at = None
    else:
        expires_at = (now + timedelta(days=days)).replace(microsecond=0).isoformat()
    level, vehicle_id, confidence = vehicle if vehicle else (None, None, None)
    conn.execute(
        "INSERT INTO text_item (source_id, external_id, vehicle_level, vehicle_id, "
        "match_confidence, text, published_at, fetched_at, expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (source_id, external_id) DO UPDATE SET "
        "text = excluded.text, fetched_at = excluded.fetched_at, expires_at = excluded.expires_at",
        (
            source_id,
            external_id,
            level,
            vehicle_id,
            confidence,
            text,
            published_at,
            now.replace(microsecond=0).isoformat(),
            expires_at,
        ),
    )
    conn.commit()


def purge_expired(
    conn: sqlite3.Connection, *, now: datetime | None = None, registry: Registry | None = None
) -> int:
    """Delete text and content metadata whose retention period has passed.

    With a registry, also clears platform statistics (e.g. YouTube subscriber counts)
    older than the platform's text retention. Returns the number of rows affected.
    """
    moment = now or datetime.now(UTC)
    cutoff = moment.replace(microsecond=0).isoformat()
    affected = 0
    for table in ("text_item", "content_item"):
        cur = conn.execute(
            f"DELETE FROM {table} WHERE expires_at IS NOT NULL AND expires_at <= ?", (cutoff,)
        )
        affected += cur.rowcount
    if registry is not None:
        from carbrain.content.catalog import PLATFORM_SOURCE

        for platform, source_id in PLATFORM_SOURCE.items():
            days = registry.get(source_id).retention_days.get("text")
            if days is None:
                continue
            stale = (moment - timedelta(days=days)).replace(microsecond=0).isoformat()
            cur = conn.execute(
                "UPDATE channel SET stats = NULL, stats_fetched_at = NULL "
                "WHERE platform = ? AND stats_fetched_at IS NOT NULL AND stats_fetched_at <= ?",
                (platform, stale),
            )
            affected += cur.rowcount
    conn.commit()
    return affected


def assert_can_send_to_ai(registry: Registry, source_ids: set[str]) -> None:
    """Raise if any source's text may not be sent to an AI provider."""
    blocked = sorted(s for s in source_ids if not registry.allows(s, Use.AI_PROCESSING))
    if blocked:
        raise RightsError(f"AI processing not allowed for: {', '.join(blocked)}")
