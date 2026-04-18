"""Shared JSON allowlist read/merge/write helper.

Internal utility — not exported from ``crossby.config``.  Used by sync writers
that store permissions in a ``{"permissions": {"allow": [...]}}`` JSON file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import structlog

logger = structlog.get_logger()


def configure_json_allowlist(
    config_path: Path,
    patterns: list[str],
    *,
    pattern_converter: Callable[[str], str],
    log_event: str = "allowlist.configured",
) -> None:
    """Read JSON, ensure permissions.allow contains required patterns, write back.

    No-op if *patterns* is empty.  Idempotent — patterns already present are
    not duplicated.  Repairs a missing or malformed ``permissions`` dict or
    ``allow`` list rather than failing.
    """
    if not patterns:
        return

    # Lazy import to avoid a circular dependency:
    # allowlist_util → crossby.sync (package __init__) → permissions → allowlist_util
    from crossby.sync.json_utils import read_json_file, write_json_file

    data, error, _was_new = read_json_file(config_path)
    if error is not None:
        logger.warning("allowlist_util.read_error", path=str(config_path), error=error)
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
    for pat in [pattern_converter(p) for p in patterns]:
        if pat not in allow_list:
            allow_list.append(pat)
            changed = True

    if not changed:
        return

    write_json_file(config_path, existing)
    logger.info(log_event, path=str(config_path))
