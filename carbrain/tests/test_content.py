from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from carbrain.config import PACKAGE_DATA
from carbrain.content.catalog import (
    ChannelSpec,
    PublisherSpec,
    load_publishers,
    sync_catalog,
)
from carbrain.content.feeds import FeedError, parse_feed
from carbrain.content.runner import sync_content, verify_channels
from carbrain.content.social import YouTubeFetcher
from carbrain.http import Http
from carbrain.resolve import Resolver
from carbrain.retention import purge_expired
from carbrain.rights import Registry
from carbrain.tools import ToolContext, call_tool

from .conftest import fixture_bytes

FEEDS = {
    "quatrorodas.abril.com.br/feed": "rss_quatrorodas.xml",
    "autoesporte.globo.com/rss": "rss_autoesporte.xml",
    "motor1.uol.com.br/rss": "rss_motor1.xml",
    "autopapo.com.br/feed": "rss_autopapo.xml",
    "garagem360.com.br/feed": "rss_motor1.xml",  # any valid feed will do
}

CRAFTED_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<title>Quatro Rodas</title>
<item><title>Teste: Polo Track 2027 melhora o consumo</title>
<link>https://quatrorodas.abril.com.br/testes/polo-track-2027/</link>
<guid>https://quatrorodas.abril.com.br/?p=1</guid>
<pubDate>Sat, 26 Sep 2026 10:00:00 +0000</pubDate>
<description>SECRET BODY TEXT</description>
<content:encoded><![CDATA[<p>SECRET BODY TEXT</p>]]></content:encoded></item>
</channel></rss>"""

# Shapes from the YouTube Data API v3 reference (channels.list, playlistItems.list).
# No API key in this environment, so these are doc-shaped, not captured responses.
YT_CHANNEL = {
    "kind": "youtube#channelListResponse",
    "items": [
        {
            "kind": "youtube#channel",
            "id": "UCGBIIPnw0AYM3BFsmTsjeAw",
            "snippet": {"title": "Acelerados", "customUrl": "@acelerados"},
            "contentDetails": {"relatedPlaylists": {"uploads": "UUGBIIPnw0AYM3BFsmTsjeAw"}},
            "statistics": {"subscriberCount": "2030000", "videoCount": "1500"},
        }
    ],
}
YT_UPLOADS = {
    "items": [
        {
            "snippet": {
                "title": "BYD Dolphin Mini: vale a pena?",
                "publishedAt": "2026-09-20T12:00:00Z",
            },
            "contentDetails": {"videoId": "vid123", "videoPublishedAt": "2026-09-20T12:00:00Z"},
        }
    ]
}


def mock_http(routes: dict[str, Any]) -> Http:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for key, body in routes.items():
            if key in url:
                if isinstance(body, dict):
                    return httpx.Response(200, json=body)
                return httpx.Response(200, content=body)
        return httpx.Response(404)

    return Http(transport=httpx.MockTransport(handler), sleep=lambda _s: None, min_interval=0)


@pytest.fixture
def catalog(seeded: sqlite3.Connection) -> sqlite3.Connection:
    sync_catalog(seeded, load_publishers(PACKAGE_DATA / "publishers.yaml"))
    return seeded


# --- catalog ---------------------------------------------------------------------------


def test_catalog_is_valid_and_organized() -> None:
    pubs = load_publishers(PACKAGE_DATA / "publishers.yaml")
    kinds = {p.kind for p in pubs}
    assert kinds == {"specialist_media", "creator"} and len(pubs) >= 30
    for pub in pubs:
        for ch in pub.channels:
            if ch.status == "feed_verified":
                assert ch.platform == "rss" and ch.url and ch.checked_on
    quatro = next(p for p in pubs if p.id == "quatro-rodas")
    assert {c.platform for c in quatro.channels} >= {"website", "rss", "youtube"}


@pytest.mark.parametrize(
    ("channel", "error"),
    [
        ({"platform": "youtube", "handle": "@x", "status": "web_evidence"}, "evidence"),
        ({"platform": "youtube", "handle": "x"}, "start with '@'"),
        ({"platform": "instagram", "handle": "@x"}, "without '@'"),
        ({"platform": "rss"}, "handle, url or external_id"),
    ],
)
def test_channel_rules(channel: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        ChannelSpec(**channel)


def test_focus_vocabulary_is_controlled() -> None:
    with pytest.raises(ValueError, match="unknown"):
        PublisherSpec(
            id="x", name="X", kind="creator", focus=["gossip"], evidence_kind=[], channels=[]
        )


def test_resync_keeps_api_verification(catalog: sqlite3.Connection) -> None:
    catalog.execute("UPDATE channel SET status = 'api_verified' WHERE id LIKE 'acelerados:%'")
    sync_catalog(catalog, load_publishers(PACKAGE_DATA / "publishers.yaml"))
    (status,) = catalog.execute(
        "SELECT status FROM channel WHERE id LIKE 'acelerados:%'"
    ).fetchone()
    assert status == "api_verified"


# --- feeds -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "title", "count"),
    [
        ("rss_quatrorodas.xml", "Quatro Rodas", 4),
        ("rss_autoesporte.xml", "autoesporte", 4),
        ("rss_motor1.xml", "Motor1.com Brasil - Notícias", 4),
        ("rss_autopapo.xml", "AutoPapo", 3),
    ],
)
def test_real_rss_feeds(name: str, title: str, count: int) -> None:
    feed_title, items = parse_feed(fixture_bytes(name))
    assert feed_title == title and len(items) == count
    for item in items:
        assert item.title and item.url and item.url.startswith("https://")
        assert item.published_at is not None and item.published_at.tzinfo is not None
        assert item.published_at.year == 2026


def test_real_atom_feed() -> None:
    title, items = parse_feed(fixture_bytes("atom_sample.xml"))
    assert title == "The Go Blog" and len(items) == 2
    assert all(i.published_at and i.url for i in items)


def test_not_a_feed() -> None:
    with pytest.raises(FeedError):
        parse_feed(b"<html><body>nope</body></html>")


def test_entity_expansion_is_refused() -> None:
    bomb = b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><rss>&a;</rss>'
    with pytest.raises(Exception, match=r"(?i)entit"):
        parse_feed(bomb)


# --- runner ----------------------------------------------------------------------------


def _feed_routes(crafted: bool = False) -> dict[str, Any]:
    routes: dict[str, Any] = {k: fixture_bytes(v) for k, v in FEEDS.items()}
    if crafted:
        routes["quatrorodas.abril.com.br/feed"] = CRAFTED_FEED
    return routes


def test_sync_feeds_is_incremental(
    catalog: sqlite3.Connection, registry: Registry, resolver: Resolver
) -> None:
    http = mock_http(_feed_routes())
    (first,) = sync_content(catalog, registry, resolver, http, platforms=["rss"])
    (second,) = sync_content(catalog, registry, resolver, http, platforms=["rss"])
    assert first.status == "ok" and first.channels == 5 and first.items_new == 19
    assert second.items_new == 0 and second.items_seen == 19


def test_article_text_is_never_stored(
    catalog: sqlite3.Connection, registry: Registry, resolver: Resolver
) -> None:
    sync_content(catalog, registry, resolver, mock_http(_feed_routes(True)), platforms=["rss"])
    dump = "\n".join(catalog.iterdump())
    assert "Polo Track 2027" in dump and "SECRET BODY" not in dump


def test_mentions_link_items_to_families(
    catalog: sqlite3.Connection, registry: Registry, resolver: Resolver
) -> None:
    sync_content(catalog, registry, resolver, mock_http(_feed_routes(True)), platforms=["rss"])
    row = catalog.execute(
        "SELECT m.family_id, c.title FROM content_mention m "
        "JOIN content_item c ON c.id = m.content_id"
    ).fetchone()
    assert row["family_id"] == "volkswagen-polo-track"


def test_catalog_change_relinks_stored_items(
    catalog: sqlite3.Connection, registry: Registry, resolver: Resolver
) -> None:
    families = [f for f in resolver.families.values() if f.id != "volkswagen-polo-track"]
    older = Resolver(list(resolver.brands.values()), families)
    sync_content(catalog, registry, older, mock_http(_feed_routes(True)), platforms=["rss"])
    linked = "SELECT COUNT(*) FROM content_mention WHERE family_id = 'volkswagen-polo-track'"
    assert catalog.execute(linked).fetchone()[0] == 0
    # Nothing is fetched (Instagram is skipped): the catalog change alone relinks.
    sync_content(catalog, registry, resolver, mock_http({}), platforms=["instagram"])
    assert catalog.execute(linked).fetchone()[0] == 1


def test_platforms_without_rights_or_keys_are_skipped(
    catalog: sqlite3.Connection,
    registry: Registry,
    resolver: Resolver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    results = {
        r.platform: r
        for r in sync_content(
            catalog, registry, resolver, mock_http({}), platforms=["youtube", "instagram"]
        )
    }
    assert results["instagram"].status == "skipped" and "pending" in (
        results["instagram"].reason or ""
    )
    assert results["youtube"].status == "skipped" and "YOUTUBE_API_KEY" in (
        results["youtube"].reason or ""
    )


def test_a_broken_channel_does_not_stop_the_others(
    catalog: sqlite3.Connection, registry: Registry, resolver: Resolver
) -> None:
    routes = _feed_routes()
    routes["motor1.uol.com.br/rss"] = b"<html>maintenance</html>"
    (result,) = sync_content(catalog, registry, resolver, mock_http(routes), platforms=["rss"])
    assert result.status == "partial" and len(result.errors) == 1 and result.items_new == 15


# --- YouTube ---------------------------------------------------------------------------


def test_youtube_videos_expire_after_30_days(
    catalog: sqlite3.Connection, registry: Registry, resolver: Resolver
) -> None:
    http = mock_http({"/channels": YT_CHANNEL, "/playlistItems": YT_UPLOADS})
    now = datetime(2026, 9, 27, tzinfo=UTC)
    fetchers = {"youtube": YouTubeFetcher(api_key="test")}
    (result,) = sync_content(
        catalog,
        registry,
        resolver,
        http,
        fetchers=fetchers,
        platforms=["youtube"],
        publisher_ids=["acelerados"],
        now=now,
    )
    assert result.status == "ok" and result.items_new == 1
    mention = catalog.execute("SELECT family_id FROM content_mention").fetchone()
    assert mention["family_id"] == "byd-dolphin-mini"
    assert purge_expired(catalog, now=now + timedelta(days=29)) == 0
    assert purge_expired(catalog, now=now + timedelta(days=30)) == 1
    assert catalog.execute("SELECT COUNT(*) FROM content_mention").fetchone()[0] == 0


def test_youtube_verification_and_stats_retention(
    catalog: sqlite3.Connection, registry: Registry
) -> None:
    http = mock_http({"/channels": YT_CHANNEL})
    outcomes = verify_channels(
        catalog, registry, http, "youtube", fetchers={"youtube": YouTubeFetcher(api_key="k")}
    )
    assert any(o.startswith("ok: Acelerados") for _id, o in outcomes)
    row = catalog.execute(
        "SELECT status, external_id, stats FROM channel WHERE id LIKE 'acelerados:youtube:%'"
    ).fetchone()
    assert (
        row["status"] == "api_verified" and json.loads(row["stats"])["subscriberCount"] == 2030000
    )
    later = datetime.now(UTC) + timedelta(days=31)
    purge_expired(catalog, now=later, registry=registry)
    assert (
        catalog.execute(
            "SELECT stats FROM channel WHERE id LIKE 'acelerados:youtube:%'"
        ).fetchone()["stats"]
        is None
    )


def test_youtube_request_uses_handle_when_no_id(catalog: sqlite3.Connection) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=YT_CHANNEL)

    http = Http(transport=httpx.MockTransport(handler), sleep=lambda _s: None, min_interval=0)
    from carbrain.content.runner import channels_for

    (chan,) = channels_for(catalog, "youtube", ["carro-chefe"])
    YouTubeFetcher(api_key="k").verify(chan, http)
    assert "forHandle=%40CarroChefe" in seen[0]


# --- tools -----------------------------------------------------------------------------


@pytest.fixture
def ctx(catalog: sqlite3.Connection, registry: Registry, resolver: Resolver) -> ToolContext:
    return ToolContext(catalog, registry, resolver, today=datetime(2026, 9, 28).date())


def test_information_sources_by_focus(ctx: ToolContext) -> None:
    result = call_tool(ctx, "information_sources", {"focus": "ev"})
    assert [p["id"] for p in result.data] == ["eletricar-br"]
    media = call_tool(ctx, "information_sources", {"kind": "specialist_media", "platform": "rss"})
    assert {p["id"] for p in media.data} >= {"quatro-rodas", "autoesporte", "motor1-brasil"}
    assert all(ch["platform"] == "rss" for p in media.data for ch in p["channels"])


def test_expert_content_respects_display_rights(ctx: ToolContext) -> None:
    sync_content(
        ctx.conn, ctx.registry, ctx.resolver, mock_http(_feed_routes(True)), platforms=["rss"]
    )
    yt = mock_http(
        {
            "/channels": YT_CHANNEL,
            "/playlistItems": {
                "items": [
                    {
                        "snippet": {
                            "title": "Polo Track: teste completo",
                            "publishedAt": "2026-09-25T12:00:00Z",
                        },
                        "contentDetails": {
                            "videoId": "v2",
                            "videoPublishedAt": "2026-09-25T12:00:00Z",
                        },
                    }
                ]
            },
        }
    )
    sync_content(
        ctx.conn,
        ctx.registry,
        ctx.resolver,
        yt,
        platforms=["youtube"],
        fetchers={"youtube": YouTubeFetcher(api_key="k")},
        publisher_ids=["acelerados"],
    )
    result = call_tool(ctx, "expert_content", {"family_id": "volkswagen-polo-track"})
    assert result.status == "ok"
    assert [i["publisher"] for i in result.data] == ["Quatro Rodas"]
    assert result.citations[0].attribution == "Fonte: Quatro Rodas"
    assert any("without display rights" in n for n in result.notes)  # the YouTube video
    assert any("Headline, publisher and a link" in n for n in result.notes)
