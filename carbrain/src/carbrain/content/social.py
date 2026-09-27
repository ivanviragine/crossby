"""Social platforms through their official APIs only.

YouTube Data API v3 (needs `YOUTUBE_API_KEY`) and Instagram Graph API business discovery
(needs `INSTAGRAM_ACCESS_TOKEN`, `INSTAGRAM_BUSINESS_ACCOUNT_ID` and
`INSTAGRAM_GRAPH_VERSION`). YouTube's terms forbid automated access to its web pages, so
there is no page-scraping fallback. Response shapes follow the official API docs; this
environment had no keys, so they are tested with doc-shaped fixtures (see LEARNINGS.md).
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

from carbrain.content.base import (
    ChannelRef,
    ContentItem,
    Fetcher,
    NotConfiguredError,
    Verification,
)
from carbrain.http import Http

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"


class YouTubeFetcher(Fetcher):
    """Channel lookup (1 quota unit) and latest uploads (1 unit per page of 50)."""

    platform = "youtube"
    source_id = "youtube"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")

    def check_configured(self) -> None:
        if not self.api_key:
            raise NotConfiguredError("Set YOUTUBE_API_KEY to use the YouTube Data API.")

    def _channel(self, channel: ChannelRef, http: Http) -> dict[str, Any]:
        self.check_configured()
        params: dict[str, Any] = {"part": "snippet,statistics,contentDetails", "key": self.api_key}
        if channel.external_id:
            params["id"] = channel.external_id
        elif channel.handle:
            params["forHandle"] = channel.handle
        elif channel.url and (legacy := re.search(r"youtube\.com/user/([^/?#]+)", channel.url)):
            params["forUsername"] = legacy.group(1)
        else:
            raise NotConfiguredError(f"{channel.id}: no channel ID, handle or /user/ URL")
        items = http.get(f"{YOUTUBE_API}/channels", params=params).json().get("items") or []
        if not items:
            raise LookupError(f"{channel.id}: YouTube has no channel for {params}")
        found: dict[str, Any] = items[0]
        return found

    def verify(self, channel: ChannelRef, http: Http) -> Verification:
        data = self._channel(channel, http)
        stats = data.get("statistics", {})
        return Verification(
            external_id=data["id"],
            title=data["snippet"]["title"],
            stats={
                k: int(stats[k])
                for k in ("subscriberCount", "videoCount", "viewCount")
                if str(stats.get(k, "")).isdigit()
            },
            extra={
                "handle": data["snippet"].get("customUrl"),
                "uploads": data["contentDetails"]["relatedPlaylists"]["uploads"],
            },
        )

    def fetch(self, channel: ChannelRef, http: Http, *, limit: int = 50) -> list[ContentItem]:
        uploads = self._channel(channel, http)["contentDetails"]["relatedPlaylists"]["uploads"]
        response = http.get(
            f"{YOUTUBE_API}/playlistItems",
            params={
                "part": "snippet,contentDetails",
                "playlistId": uploads,
                "maxResults": min(limit, 50),
                "key": self.api_key,
            },
        ).json()
        items = []
        for it in response.get("items", []):
            video_id = it["contentDetails"]["videoId"]
            published = it["contentDetails"].get("videoPublishedAt") or it["snippet"].get(
                "publishedAt"
            )
            items.append(
                ContentItem(
                    external_id=video_id,
                    kind="video",
                    title=it["snippet"]["title"],
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    published_at=(
                        datetime.fromisoformat(published).astimezone(UTC) if published else None
                    ),
                )
            )
        return items


class InstagramFetcher(Fetcher):
    """Business discovery: public profile data and recent posts of business/creator accounts."""

    platform = "instagram"
    source_id = "instagram"

    def __init__(self) -> None:
        self.token = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
        self.account = os.environ.get("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        self.version = os.environ.get("INSTAGRAM_GRAPH_VERSION")

    def check_configured(self) -> None:
        if not (self.token and self.account and self.version):
            raise NotConfiguredError(
                "Set INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID and "
                "INSTAGRAM_GRAPH_VERSION (a current Graph API version, e.g. from Meta's docs)."
            )

    def _discover(self, channel: ChannelRef, http: Http, fields: str) -> dict[str, Any]:
        self.check_configured()
        if not channel.handle:
            raise NotConfiguredError(f"{channel.id}: no Instagram handle")
        response = http.get(
            f"https://graph.facebook.com/{self.version}/{self.account}",
            params={
                "fields": f"business_discovery.username({channel.handle}){{{fields}}}",
                "access_token": self.token,
            },
        ).json()
        found: dict[str, Any] = response["business_discovery"]
        return found

    def verify(self, channel: ChannelRef, http: Http) -> Verification:
        data = self._discover(channel, http, "id,username,name,followers_count,media_count")
        return Verification(
            external_id=data["id"],
            title=data.get("name") or data["username"],
            stats={k: data[k] for k in ("followers_count", "media_count") if k in data},
        )

    def fetch(self, channel: ChannelRef, http: Http, *, limit: int = 50) -> list[ContentItem]:
        data = self._discover(
            channel, http, f"media.limit({limit}){{id,caption,permalink,timestamp}}"
        )
        items = []
        for post in data.get("media", {}).get("data", []):
            caption = (post.get("caption") or "").strip()
            items.append(
                ContentItem(
                    external_id=post["id"],
                    kind="post",
                    # First line of the caption only, as a title; the rest is not kept.
                    title=caption.splitlines()[0][:200] if caption else "(no caption)",
                    url=post.get("permalink"),
                    published_at=datetime.strptime(post["timestamp"], "%Y-%m-%dT%H:%M:%S%z")
                    if post.get("timestamp")
                    else None,
                )
            )
        return items
