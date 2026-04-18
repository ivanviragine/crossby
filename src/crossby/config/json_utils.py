"""JSON read/write utilities shared across config and sync layers."""

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


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic write of a JSON dict with consistent formatting.

    Uses 2-space indent, sorted keys, and a tmp+replace pattern to avoid
    partial writes on crash.
    """
    json_text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_text, encoding="utf-8")
    tmp.replace(path)
