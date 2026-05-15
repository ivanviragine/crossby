"""Shared JSON allowlist read/merge/write helper.

Internal utility — not exported from ``crossby.config``.  Used by sync writers
that store permissions in a ``{"permissions": {"allow": [...]}}`` JSON file.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import structlog

from crossby.config.json_utils import read_json_file, write_json_file

logger = structlog.get_logger()


AllowlistAction = Literal["created", "updated", "skipped", "error"]


def configure_json_allowlist(
    config_path: Path,
    patterns: list[str],
    *,
    pattern_converter: Callable[[str], str],
    log_event: str = "allowlist.configured",
) -> tuple[AllowlistAction, str | None]:
    """Read JSON, ensure permissions.allow contains required patterns, write back.

    Returns ``(action, error_message)`` where ``action`` is one of
    ``"created"``, ``"updated"``, ``"skipped"``, or ``"error"``.

    No-op if *patterns* is empty (returns ``("skipped", None)``).
    Idempotent — patterns already present are not duplicated.
    Repairs a missing or malformed ``permissions`` dict or ``allow`` list
    rather than failing.

    Refuses to overwrite a malformed JSON file: parse failure returns
    ``("error", msg)`` with no write, matching the safer policy used by
    hooks/MCP writers (instead of silently replacing the user's file with
    a fresh ``{}``-derived document).
    """
    if not patterns:
        return "skipped", None

    data, error, was_new = read_json_file(config_path)
    if error is not None:
        msg = (
            f"{config_path} {error} — skipping permissions sync. "
            "Fix the file manually or delete it."
        )
        warnings.warn(msg, stacklevel=2)
        logger.warning("allowlist_util.read_error", path=str(config_path), error=error)
        return "error", msg
    existing: dict[str, object] = data if data is not None else {}

    permissions = existing.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
        existing["permissions"] = permissions

    allow_list = permissions.setdefault("allow", [])
    if not isinstance(allow_list, list):
        allow_list = []
        permissions["allow"] = allow_list

    changed = False
    for pat in (pattern_converter(p) for p in patterns):
        if pat not in allow_list:
            allow_list.append(pat)
            changed = True

    if not changed:
        return "skipped", None

    write_json_file(config_path, existing)
    logger.info(log_event, path=str(config_path))
    return ("created" if was_new else "updated"), None
