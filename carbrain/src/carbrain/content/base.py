"""The common interface every platform adapter implements."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from carbrain.http import Http

ItemKind = Literal["article", "video", "post"]


class NotConfiguredError(RuntimeError):
    """The adapter needs credentials or settings that are not present."""


@dataclass(frozen=True)
class ChannelRef:
    """A channel as stored in the `channel` table."""

    id: str
    publisher_id: str
    platform: str
    handle: str | None
    url: str | None
    external_id: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ChannelRef:
        return cls(
            row["id"],
            row["publisher_id"],
            row["platform"],
            row["handle"],
            row["url"],
            row["external_id"],
        )


@dataclass(frozen=True)
class ContentItem:
    """Metadata of one article, video or post. Never the body text."""

    external_id: str
    kind: ItemKind
    title: str
    url: str | None
    published_at: datetime | None
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verification:
    """What the platform says about a channel, straight from its official interface."""

    external_id: str
    title: str
    stats: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class Fetcher(ABC):
    """One adapter per platform.

    `source_id` names the rights-registry entry that governs the platform's data, so the
    runner can check collection rights and retention before any request is made.
    """

    platform: ClassVar[str]
    source_id: ClassVar[str]

    def check_configured(self) -> None:  # noqa: B027 - optional hook, default is a no-op
        """Raise `NotConfiguredError` when credentials are missing. Default: nothing needed."""

    @abstractmethod
    def verify(self, channel: ChannelRef, http: Http) -> Verification:
        """Confirm that the channel exists, and return its platform ID and public stats."""

    @abstractmethod
    def fetch(self, channel: ChannelRef, http: Http, *, limit: int = 50) -> list[ContentItem]:
        """Latest items of the channel, newest first."""
