"""Connector contract and the run loop shared by every source.

A run: check collection rights → discover URLs → fetch → archive (hash) → parse only
new or re-versioned files → write observations with revisions → log the run.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from carbrain.archive import RawArchive, Snapshot
from carbrain.db import utcnow
from carbrain.http import FetchError, Http
from carbrain.models import Observation, Use
from carbrain.observations import write_observations
from carbrain.resolve import Resolver
from carbrain.rights import Registry, requests_today

log = logging.getLogger(__name__)


@dataclass
class ParseContext:
    conn: sqlite3.Connection
    resolver: Resolver


class Connector(ABC):
    source_id: ClassVar[str]
    parser_version: ClassVar[str]
    #: Large files are streamed to disk instead of held in memory.
    large_files: ClassVar[bool] = False
    #: A 404 for a discovered URL means "not published yet", not a failure.
    missing_ok: ClassVar[bool] = False
    #: Each URL always serves the same content (e.g. one file per month), so a URL
    #: already parsed by this parser version is not downloaded again.
    immutable_urls: ClassVar[bool] = False
    #: Parsing maps source names to catalog families, so a catalog change (new family,
    #: better pattern) re-parses the latest files, from the archive when they are there.
    uses_catalog: ClassVar[bool] = False

    @abstractmethod
    def discover(self, http: Http) -> list[str]:
        """URLs to fetch in this run, newest last."""

    @abstractmethod
    def parse(self, snapshot: Snapshot, ctx: ParseContext) -> Iterable[Observation]:
        """Observations contained in one archived file."""


@dataclass
class RunResult:
    source_id: str
    status: str
    files_fetched: int = 0
    files_new: int = 0
    inserted: int = 0
    revised: int = 0
    unchanged: int = 0
    requests: int = 0
    error: str | None = None


def run_connector(
    connector: Connector,
    *,
    conn: sqlite3.Connection,
    http: Http,
    archive: RawArchive,
    registry: Registry,
    resolver: Resolver,
    dev_sample: bool = False,
    reparse: bool = False,
) -> RunResult:
    source_id = connector.source_id
    decision = registry.require(source_id, Use.COLLECT, dev_sample=dev_sample, conn=conn)
    if decision.dev_sample:
        limit = registry.get(source_id).dev_sample_limit or 0
        http.budget = http.requests + max(0, limit - requests_today(conn, source_id))
    started = utcnow()
    run_id = conn.execute(
        "INSERT INTO run_log (source_id, started_at, status) VALUES (?, ?, 'running')",
        (source_id, started),
    ).lastrowid
    conn.commit()
    result = RunResult(source_id, "ok")
    requests_before = http.requests
    ctx = ParseContext(conn, resolver)
    version = parser_version(connector, resolver)
    try:
        for url in connector.discover(http):
            if (
                connector.immutable_urls
                and not reparse
                and _parsed_before(conn, source_id, url, version)
            ):
                result.files_fetched += 1
                continue
            try:
                snapshot = _fetch(connector, http, archive, url, version)
            except FetchError as exc:
                if connector.missing_ok and exc.status_code == 404:
                    log.info("%s: %s not published yet", source_id, url)
                    continue
                raise
            result.files_fetched += 1
            if not (snapshot.needs_parse or reparse):
                continue
            result.files_new += 1
            stats = write_observations(
                conn,
                connector.parse(snapshot, ctx),
                fetched_at=snapshot.fetched_at,
                snapshot_id=snapshot.id,
            )
            archive.mark_parsed(snapshot, version)
            result.inserted += stats.inserted
            result.revised += stats.revised
            result.unchanged += stats.unchanged
        if result.files_fetched and not result.files_new:
            result.status = "unchanged"
    except Exception as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        log.exception("%s run failed", source_id)
    result.requests = http.requests - requests_before
    conn.execute(
        "UPDATE run_log SET finished_at = ?, status = ?, files_fetched = ?, files_new = ?, "
        "observations_inserted = ?, observations_revised = ?, requests = ?, error = ? WHERE id = ?",
        (
            utcnow(),
            result.status,
            result.files_fetched,
            result.files_new,
            result.inserted,
            result.revised,
            result.requests,
            result.error,
            run_id,
        ),
    )
    conn.commit()
    return result


def parser_version(connector: Connector, resolver: Resolver) -> str:
    """The parser version recorded on parsed files, including the catalog when it matters."""
    if connector.uses_catalog:
        return f"{connector.parser_version}+catalog.{resolver.fingerprint}"
    return connector.parser_version


def _parsed_before(conn: sqlite3.Connection, source_id: str, url: str, version: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM snapshot WHERE source_id = ? AND url = ? AND parser_version = ? "
        "AND parsed_at IS NOT NULL LIMIT 1",
        (source_id, url, version),
    ).fetchone()
    return row is not None


def _fetch(
    connector: Connector, http: Http, archive: RawArchive, url: str, version: str
) -> Snapshot:
    if connector.immutable_urls:
        # The content can't have changed, so a file already in the archive is reused.
        archived = archive.find(connector.source_id, url, version)
        if archived is not None:
            return archived
    if connector.large_files:
        archive.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=archive.root) as tmp:
            path = http.download(url, Path(tmp) / "download")
            return archive.store_file(connector.source_id, url, path, None, version)
    response = http.get(url)
    return archive.store_bytes(
        connector.source_id,
        url,
        response.content,
        response.headers.get("Content-Type"),
        version,
    )
