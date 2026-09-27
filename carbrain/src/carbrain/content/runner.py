"""Run every platform adapter through the same steps.

For each platform: check the source's collection rights once, skip the platform (with a
reason) if rights or credentials are missing, then fetch each channel's latest items,
store their metadata with the source's retention period, and link them to the vehicle
families they mention.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from carbrain.content.base import ChannelRef, ContentItem, Fetcher, NotConfiguredError
from carbrain.content.feeds import FeedFetcher
from carbrain.content.social import InstagramFetcher, YouTubeFetcher
from carbrain.db import get_meta, set_meta, utcnow
from carbrain.http import Http
from carbrain.models import Use
from carbrain.resolve import Resolver
from carbrain.rights import Registry, RightsError

log = logging.getLogger(__name__)

#: Minimum confidence for linking an item to a vehicle family (pattern matches only).
MENTION_MIN_CONFIDENCE = 0.92
MENTIONS_CATALOG_KEY = "mentions_catalog"


def default_fetchers() -> dict[str, Fetcher]:
    fetchers: list[Fetcher] = [FeedFetcher(), YouTubeFetcher(), InstagramFetcher()]
    return {f.platform: f for f in fetchers}


@dataclass
class PlatformResult:
    platform: str
    status: str = "ok"
    reason: str | None = None
    channels: int = 0
    items_new: int = 0
    items_seen: int = 0
    errors: list[str] = field(default_factory=list)


def channels_for(
    conn: sqlite3.Connection, platform: str, publisher_ids: list[str] | None = None
) -> list[ChannelRef]:
    sql = "SELECT * FROM channel WHERE platform = ?"
    params: list[str] = [platform]
    if publisher_ids:
        sql += f" AND publisher_id IN ({','.join('?' * len(publisher_ids))})"
        params.extend(publisher_ids)
    return [ChannelRef.from_row(r) for r in conn.execute(sql + " ORDER BY id", params)]


def sync_content(
    conn: sqlite3.Connection,
    registry: Registry,
    resolver: Resolver,
    http: Http,
    *,
    fetchers: dict[str, Fetcher] | None = None,
    platforms: list[str] | None = None,
    publisher_ids: list[str] | None = None,
    limit: int = 50,
    now: datetime | None = None,
) -> list[PlatformResult]:
    fetchers = fetchers or default_fetchers()
    if get_meta(conn, MENTIONS_CATALOG_KEY) != resolver.fingerprint:
        # The catalog changed since mentions were linked (new family, better pattern).
        relink_mentions(conn, resolver)
    results = []
    for platform in platforms or list(fetchers):
        fetcher = fetchers[platform]
        result = PlatformResult(platform)
        results.append(result)
        try:
            registry.require(fetcher.source_id, Use.COLLECT)
            fetcher.check_configured()
        except (RightsError, NotConfiguredError) as exc:
            result.status, result.reason = "skipped", str(exc)
            continue
        expires = _expiry(registry, fetcher.source_id, now or datetime.now(UTC))
        run_id = _start_run(conn, fetcher.source_id)
        requests_before = http.requests
        for channel in channels_for(conn, platform, publisher_ids):
            result.channels += 1
            try:
                items = fetcher.fetch(channel, http, limit=limit)
            except Exception as exc:  # one broken channel must not stop the others
                log.warning("%s: %s", channel.id, exc)
                result.errors.append(f"{channel.id}: {type(exc).__name__}: {exc}")
                continue
            for item in items:
                result.items_seen += 1
                if _store(conn, fetcher.source_id, channel, item, expires):
                    result.items_new += 1
                _link_mentions(conn, resolver, channel, item)
            conn.commit()
        if result.errors:
            result.status = "partial" if result.items_seen else "error"
        _finish_run(conn, run_id, result, http.requests - requests_before)
    return results


def verify_channels(
    conn: sqlite3.Connection,
    registry: Registry,
    http: Http,
    platform: str,
    *,
    fetchers: dict[str, Fetcher] | None = None,
) -> list[tuple[str, str]]:
    """Confirm channels through the platform's official interface; returns (id, outcome)."""
    fetcher = (fetchers or default_fetchers())[platform]
    registry.require(fetcher.source_id, Use.COLLECT)
    fetcher.check_configured()
    outcomes = []
    for channel in channels_for(conn, platform):
        try:
            v = fetcher.verify(channel, http)
        except Exception as exc:
            outcomes.append((channel.id, f"failed: {type(exc).__name__}: {exc}"))
            continue
        conn.execute(
            "UPDATE channel SET status = ?, external_id = ?, checked_on = ?, stats = ?, "
            "stats_fetched_at = ? WHERE id = ?",
            (
                "feed_verified" if platform == "rss" else "api_verified",
                v.external_id,
                utcnow()[:10],
                json.dumps(v.stats),
                utcnow(),
                channel.id,
            ),
        )
        outcomes.append((channel.id, f"ok: {v.title}"))
    conn.commit()
    return outcomes


def _expiry(registry: Registry, source_id: str, now: datetime) -> str | None:
    days = registry.get(source_id).retention_days.get("text")
    if days is None:
        return None
    return (now + timedelta(days=days)).replace(microsecond=0).isoformat()


def _store(
    conn: sqlite3.Connection,
    source_id: str,
    channel: ChannelRef,
    item: ContentItem,
    expires_at: str | None,
) -> bool:
    """Upsert the item's metadata. Returns True if the item was not stored before."""
    existed = conn.execute(
        "SELECT 1 FROM content_item WHERE channel_id = ? AND external_id = ?",
        (channel.id, item.external_id),
    ).fetchone()
    conn.execute(
        "INSERT INTO content_item (source_id, channel_id, external_id, kind, url, title, "
        "published_at, fetched_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (channel_id, external_id) DO UPDATE SET title = excluded.title, "
        "url = excluded.url, fetched_at = excluded.fetched_at, expires_at = excluded.expires_at",
        (
            source_id,
            channel.id,
            item.external_id,
            item.kind,
            item.url,
            item.title,
            item.published_at.isoformat() if item.published_at else None,
            utcnow(),
            expires_at,
        ),
    )
    return existed is None


def _link_mentions(
    conn: sqlite3.Connection, resolver: Resolver, channel: ChannelRef, item: ContentItem
) -> None:
    content_id = conn.execute(
        "SELECT id FROM content_item WHERE channel_id = ? AND external_id = ?",
        (channel.id, item.external_id),
    ).fetchone()["id"]
    # Recompute from scratch, so improved matching rules also fix earlier mistakes.
    conn.execute("DELETE FROM content_mention WHERE content_id = ?", (content_id,))
    for match in resolver.find(item.title, editorial=True):
        if match.method != "text_pattern" or match.confidence < MENTION_MIN_CONFIDENCE:
            continue
        conn.execute(
            "INSERT INTO content_mention (content_id, family_id, confidence, matched_text) "
            "VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (content_id, match.family_id, match.confidence, match.matched_text),
        )


def relink_mentions(conn: sqlite3.Connection, resolver: Resolver) -> int:
    """Recompute vehicle mentions for every stored item (after matching rules change)."""
    rows = conn.execute(
        "SELECT c.*, ch.publisher_id, ch.platform, ch.handle, ch.url AS ch_url, "
        "ch.external_id AS ch_external_id FROM content_item c "
        "JOIN channel ch ON ch.id = c.channel_id"
    ).fetchall()
    for r in rows:
        channel = ChannelRef(
            r["channel_id"],
            r["publisher_id"],
            r["platform"],
            r["handle"],
            r["ch_url"],
            r["ch_external_id"],
        )
        item = ContentItem(r["external_id"], r["kind"], r["title"], r["url"], None)
        _link_mentions(conn, resolver, channel, item)
    conn.commit()
    set_meta(conn, MENTIONS_CATALOG_KEY, resolver.fingerprint)
    return len(rows)


def _start_run(conn: sqlite3.Connection, source_id: str) -> int:
    cur = conn.execute(
        "INSERT INTO run_log (source_id, started_at, status) VALUES (?, ?, 'running')",
        (source_id, utcnow()),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def _finish_run(
    conn: sqlite3.Connection, run_id: int, result: PlatformResult, requests: int
) -> None:
    conn.execute(
        "UPDATE run_log SET finished_at = ?, status = ?, files_fetched = ?, files_new = ?, "
        "requests = ?, error = ? WHERE id = ?",
        (
            utcnow(),
            result.status,
            result.channels,
            result.items_new,
            requests,
            "; ".join(result.errors)[:2000] or None,
            run_id,
        ),
    )
    conn.commit()
