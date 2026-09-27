"""Raw archive: every fetched file is kept unchanged and addressed by its SHA-256.

A file whose hash was already parsed with the current parser version is not parsed
again, which is what makes "check daily, store only on change" cheap.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from carbrain.db import utcnow


@dataclass(frozen=True)
class Snapshot:
    id: int
    source_id: str
    url: str
    sha256: str
    path: Path
    fetched_at: str
    content_type: str | None
    needs_parse: bool


class RawArchive:
    def __init__(self, conn: sqlite3.Connection, root: Path) -> None:
        self.conn = conn
        self.root = root

    def store_bytes(
        self,
        source_id: str,
        url: str,
        content: bytes,
        content_type: str | None,
        parser_version: str,
    ) -> Snapshot:
        digest = hashlib.sha256(content).hexdigest()
        path = self._path_for(source_id, digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return self._register(
            source_id, url, digest, len(content), content_type, path, parser_version
        )

    def store_file(
        self, source_id: str, url: str, file: Path, content_type: str | None, parser_version: str
    ) -> Snapshot:
        """Move a downloaded file into the archive (for files too large to hold in memory)."""
        digest = _sha256_file(file)
        path = self._path_for(source_id, digest)
        size = file.stat().st_size
        if path.exists():
            file.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), path)
        return self._register(source_id, url, digest, size, content_type, path, parser_version)

    def mark_parsed(self, snapshot: Snapshot, parser_version: str) -> None:
        self.conn.execute(
            "UPDATE snapshot SET parsed_at = ?, parser_version = ? WHERE id = ?",
            (utcnow(), parser_version, snapshot.id),
        )
        self.conn.commit()

    def _path_for(self, source_id: str, digest: str) -> Path:
        return self.root / source_id / digest[:2] / digest

    def _register(
        self,
        source_id: str,
        url: str,
        digest: str,
        size: int,
        content_type: str | None,
        path: Path,
        parser_version: str,
    ) -> Snapshot:
        row = self.conn.execute(
            "SELECT id, fetched_at, parser_version, parsed_at FROM snapshot "
            "WHERE source_id = ? AND url = ? AND sha256 = ?",
            (source_id, url, digest),
        ).fetchone()
        if row is not None:
            return Snapshot(
                row["id"],
                source_id,
                url,
                digest,
                path,
                row["fetched_at"],
                content_type,
                not self._already_parsed(source_id, url, digest, parser_version),
            )
        fetched_at = utcnow()
        cur = self.conn.execute(
            "INSERT INTO snapshot (source_id, url, fetched_at, sha256, size, content_type, path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source_id, url, fetched_at, digest, size, content_type, str(path)),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        needs_parse = not self._already_parsed(source_id, url, digest, parser_version)
        return Snapshot(
            cur.lastrowid, source_id, url, digest, path, fetched_at, content_type, needs_parse
        )

    def _already_parsed(self, source_id: str, url: str, digest: str, parser_version: str) -> bool:
        """These bytes, from this URL, were already parsed by this parser version.

        The URL is part of the key because parsers read meaning from it (the SGS series
        code, the SENATRAN month), so identical bytes from two URLs are not the same fact.
        """
        row = self.conn.execute(
            "SELECT 1 FROM snapshot WHERE source_id = ? AND url = ? AND sha256 = ? "
            "AND parser_version = ? AND parsed_at IS NOT NULL LIMIT 1",
            (source_id, url, digest, parser_version),
        ).fetchone()
        return row is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
