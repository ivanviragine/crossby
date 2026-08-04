"""Shared JSON allowlist read/merge/write helper.

Internal utility — not exported from ``crossby.config``.  Used by sync writers
that store permissions in a ``{"permissions": {"allow": [...]}}`` JSON file.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
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
    revoke: Iterable[str] = (),
    log_event: str = "allowlist.configured",
    dry_run: bool = False,
) -> tuple[AllowlistAction, str | None, list[str], int]:
    """Read JSON, add required patterns, remove revoked ones, write back.

    Returns ``(action, error_message, created, revoked)`` where ``action`` is one
    of ``"created"``, ``"updated"``, ``"skipped"``, or ``"error"``, ``created``
    is the list of **canonical** patterns written *fresh* this call (those not
    already present — never a pattern that was there by hand), and ``revoked`` is
    the count removed.

    No-op if both *patterns* and *revoke* are empty (returns
    ``("skipped", None, [], 0)``). Idempotent — patterns already present are not
    duplicated, and revoking a pattern that is absent removes nothing. Repairs a
    missing or malformed ``permissions`` dict or ``allow`` list rather than
    failing.

    *revoke* is a set of canonical patterns crossby is permitted to remove
    (computed by ``run_sync`` from the ownership ledger) — it is converted
    through the same *pattern_converter* before matching, so only entries crossby
    itself wrote are dropped and a hand-authored pattern that merely resembles
    one is never touched.

    Refuses to overwrite a malformed JSON file: parse failure returns
    ``("error", msg, [], 0)`` with no write, matching the safer policy used by
    hooks/MCP writers (instead of silently replacing the user's file with
    a fresh ``{}``-derived document).
    """
    revoke_patterns = {pattern_converter(p) for p in revoke}
    if not patterns and not revoke_patterns:
        return "skipped", None, [], 0

    data, error, was_new = read_json_file(config_path)
    if error is not None:
        msg = (
            f"{config_path} {error} — skipping permissions sync. "
            "Fix the file manually or delete it."
        )
        warnings.warn(msg, stacklevel=2)
        logger.warning("allowlist_util.read_error", path=str(config_path), error=error)
        return "error", msg, [], 0
    existing: dict[str, object] = data if data is not None else {}

    permissions = existing.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
        existing["permissions"] = permissions

    allow_list = permissions.setdefault("allow", [])
    if not isinstance(allow_list, list):
        allow_list = []
        permissions["allow"] = allow_list

    # Track the *canonical* patterns written fresh, so run_sync records ownership
    # only of what crossby actually added — never a hand-authored pattern that
    # coincidentally matches the source.
    created: list[str] = []
    for canonical in patterns:
        pat = pattern_converter(canonical)
        if pat in revoke_patterns:
            # A pattern requested for both add and revoke is dropped by the loop
            # below — never append it or claim it in ``created``, or run_sync
            # would record ownership of a pattern absent from the file on disk.
            continue
        if pat not in allow_list:
            allow_list.append(pat)
            created.append(canonical)

    revoked = 0
    if revoke_patterns:
        kept = [p for p in allow_list if p not in revoke_patterns]
        revoked = len(allow_list) - len(kept)
        if revoked:
            # Mutate in place so ``permissions["allow"]`` keeps referencing the
            # same list object we validated/repaired above.
            allow_list[:] = kept

    if not created and not revoked:
        return "skipped", None, [], 0

    if not dry_run:
        write_json_file(config_path, existing)
        logger.info(log_event, path=str(config_path))
    return ("created" if was_new else "updated"), None, created, revoked
