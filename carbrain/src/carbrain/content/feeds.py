"""RSS 2.0 and Atom feeds (publisher websites).

Only metadata is read: title, link, date, id and categories. Article text
(`description`, `content:encoded`, Atom `content`/`summary`) is never kept.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as SafeET

from carbrain.content.base import ChannelRef, ContentItem, Fetcher, Verification
from carbrain.http import Http

ATOM = "{http://www.w3.org/2005/Atom}"


class FeedError(ValueError):
    pass


def parse_feed(data: bytes) -> tuple[str, list[ContentItem]]:
    """(feed title, items) for an RSS 2.0 or Atom document. Uses a hardened XML parser."""
    root = SafeET.fromstring(data)
    if root.tag == "rss":
        channel = root.find("channel")
        if channel is None:
            raise FeedError("RSS document without <channel>")
        return _text(channel, "title"), [_rss_item(i) for i in channel.findall("item")]
    if root.tag == f"{ATOM}feed":
        return _text(root, f"{ATOM}title"), [_atom_entry(e) for e in root.findall(f"{ATOM}entry")]
    raise FeedError(f"Not an RSS or Atom feed: <{root.tag}>")


def _text(node: Element, tag: str) -> str:
    child = node.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _rss_item(item: Element) -> ContentItem:
    link = _text(item, "link") or None
    guid = _text(item, "guid") or link or _text(item, "title")
    published = None
    raw_date = _text(item, "pubDate")
    if raw_date:
        try:
            published = _utc(parsedate_to_datetime(raw_date))
        except (TypeError, ValueError):
            published = None
    return ContentItem(
        external_id=guid,
        kind="article",
        title=_text(item, "title"),
        url=link,
        published_at=published,
        categories=tuple((c.text or "").strip() for c in item.findall("category") if c.text),
    )


def _atom_entry(entry: Element) -> ContentItem:
    link = None
    for el in entry.findall(f"{ATOM}link"):
        if el.get("rel", "alternate") == "alternate":
            link = el.get("href")
            break
    raw_date = _text(entry, f"{ATOM}published") or _text(entry, f"{ATOM}updated")
    return ContentItem(
        external_id=_text(entry, f"{ATOM}id") or link or _text(entry, f"{ATOM}title"),
        kind="article",
        title=_text(entry, f"{ATOM}title"),
        url=link,
        published_at=_utc(datetime.fromisoformat(raw_date)) if raw_date else None,
        categories=tuple(
            c.get("term", "") for c in entry.findall(f"{ATOM}category") if c.get("term")
        ),
    )


def _utc(moment: datetime) -> datetime:
    """UTC-aware datetime. RFC 5322 "-0000" (e.g. Autoesporte) means "zone unknown" and
    parses as naive; it is treated as UTC so stored timestamps always compare correctly."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


class FeedFetcher(Fetcher):
    platform = "rss"
    source_id = "publisher_feeds"

    def verify(self, channel: ChannelRef, http: Http) -> Verification:
        if not channel.url:
            raise FeedError(f"{channel.id} has no feed URL")
        title, items = parse_feed(http.get(channel.url).content)
        return Verification(external_id=channel.url, title=title, stats={"items": len(items)})

    def fetch(self, channel: ChannelRef, http: Http, *, limit: int = 50) -> list[ContentItem]:
        if not channel.url:
            raise FeedError(f"{channel.id} has no feed URL")
        _title, items = parse_feed(http.get(channel.url).content)
        return items[:limit]
