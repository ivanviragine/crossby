"""The publisher catalog: who publishes car information, where, and how sure we are."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Platform = Literal["website", "rss", "youtube", "instagram", "tiktok"]
ChannelStatus = Literal["unverified", "web_evidence", "feed_verified", "api_verified"]
PublisherKind = Literal["specialist_media", "creator"]

FOCUS = {
    "reviews", "comparisons", "buying_advice", "news", "instrumented_tests",
    "long_term_tests", "used_cars", "ev", "mechanics", "motorsport", "entertainment",
    "customization", "auctions", "classics", "market",
}  # fmt: skip
EVIDENCE_KINDS = {"measured_tests", "expert_opinion", "news", "owner_experience", "entertainment"}
#: Which rights-registry source governs each platform's data.
PLATFORM_SOURCE = {"rss": "publisher_feeds", "youtube": "youtube", "instagram": "instagram"}


class Evidence(BaseModel):
    url: str
    dated: date | None = None


class ChannelSpec(BaseModel):
    platform: Platform
    handle: str | None = None
    url: str | None = None
    external_id: str | None = None
    status: ChannelStatus = "unverified"
    evidence: list[Evidence] = Field(default_factory=list)
    checked_on: date | None = None

    @model_validator(mode="after")
    def _check(self) -> ChannelSpec:
        if not (self.handle or self.url or self.external_id):
            raise ValueError("a channel needs a handle, url or external_id")
        if self.status == "web_evidence" and not self.evidence:
            raise ValueError("web_evidence channels must cite their evidence")
        if self.platform == "youtube" and self.handle and not self.handle.startswith("@"):
            raise ValueError(f"YouTube handles start with '@': {self.handle}")
        if self.platform == "instagram" and self.handle and self.handle.startswith("@"):
            raise ValueError(f"store Instagram handles without '@': {self.handle}")
        return self

    def key(self) -> str:
        return self.external_id or self.handle or self.url or ""


class PublisherSpec(BaseModel):
    id: str
    name: str
    kind: PublisherKind
    owner: str | None = None
    people: list[str] = Field(default_factory=list)
    focus: list[str]
    evidence_kind: list[str]
    notes: str | None = None
    channels: list[ChannelSpec]

    @model_validator(mode="after")
    def _vocabulary(self) -> PublisherSpec:
        unknown = (set(self.focus) - FOCUS) | (set(self.evidence_kind) - EVIDENCE_KINDS)
        if unknown:
            raise ValueError(f"{self.id}: unknown focus/evidence terms {sorted(unknown)}")
        return self


def channel_id(publisher_id: str, channel: ChannelSpec) -> str:
    slug = re.sub(r"[^a-z0-9@._-]+", "-", channel.key().lower()).strip("-")
    return f"{publisher_id}:{channel.platform}:{slug}"


def load_publishers(path: Path) -> list[PublisherSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    publishers = [PublisherSpec(**p) for p in raw["publishers"]]
    ids = [p.id for p in publishers]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate publisher ids")
    return publishers


def sync_catalog(conn: sqlite3.Connection, publishers: list[PublisherSpec]) -> int:
    """Upsert publishers and channels. A channel verified through an API keeps that status."""
    for pub in publishers:
        conn.execute(
            "INSERT INTO publisher (id, name, kind, focus, evidence_kind, notes) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO UPDATE SET name = excluded.name, "
            "kind = excluded.kind, focus = excluded.focus, "
            "evidence_kind = excluded.evidence_kind, notes = excluded.notes",
            (
                pub.id,
                pub.name,
                pub.kind,
                json.dumps(pub.focus),
                json.dumps(pub.evidence_kind),
                pub.notes,
            ),
        )
        for ch in pub.channels:
            conn.execute(
                "INSERT INTO channel (id, publisher_id, platform, handle, url, external_id, "
                "status, evidence, checked_on) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET handle = excluded.handle, url = excluded.url, "
                "external_id = COALESCE(channel.external_id, excluded.external_id), "
                "status = CASE WHEN channel.status = 'api_verified' THEN channel.status "
                "ELSE excluded.status END, evidence = excluded.evidence, "
                "checked_on = COALESCE(channel.checked_on, excluded.checked_on)",
                (
                    channel_id(pub.id, ch),
                    pub.id,
                    ch.platform,
                    ch.handle,
                    ch.url,
                    ch.external_id,
                    ch.status,
                    json.dumps([e.model_dump(mode="json") for e in ch.evidence]),
                    ch.checked_on.isoformat() if ch.checked_on else None,
                ),
            )
    conn.commit()
    return len(publishers)
