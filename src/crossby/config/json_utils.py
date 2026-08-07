"""File read/write utilities shared across config and sync layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Read a JSON file, returning (data, error_message, was_new).

    Returns:
        (dict, None, False) on success for an existing file.
        (None, error_message, False) if file is malformed.
        ({}, None, True) if file does not exist.
    """
    if not path.exists():
        return {}, None, True
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, f"contains invalid JSON: {e}", False
    except (OSError, UnicodeDecodeError) as e:
        return None, f"could not be read: {e}", False
    if not isinstance(raw, dict):
        return None, "root value is not a JSON object", False
    return raw, None, False


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via a **unique** temp file plus atomic rename.

    A crash or interrupt part-way through leaves the original file intact
    rather than truncated — which matters most for the config files crossby
    merges into rather than owns (``.codex/config.toml``, tool settings).

    The temp file is created with :func:`tempfile.mkstemp` in the target's own
    directory (so the final ``os.replace`` stays on one filesystem and is
    atomic), and its name is unique per call — two concurrent writers to the
    same path therefore never clobber or unlink each other's temp file. The
    content race is still last-writer-wins, but neither writer ever observes a
    torn file.
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic write of a JSON dict with consistent formatting.

    Uses 2-space indent, sorted keys, and a tmp+replace pattern to avoid
    partial writes on crash.
    """
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
