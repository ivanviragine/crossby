"""Provenance ledger — a record of what crossby wrote, so it can revoke it later.

``.crossby/owned.json`` is a per-machine, gitignored record of the items crossby
has written to each tool, keyed by ``(tool_id, concern)``. :func:`run_sync
<crossby.sync.run_sync>` diffs this ledger against the current sync data to
compute what to revoke, so a writer never removes an entry a human authored.

Item identities per concern:

- **hooks** — ``(event, command)`` pairs, stored as 2-element JSON lists.
- **permissions** — canonical command patterns (strings, e.g. ``"git diff:*"``).
- **mcp** — server names (strings).

A missing or malformed ledger degrades to "own nothing" — purely additive
behaviour, never a crash. Because the file is gitignored it is per-machine: a
fresh clone starts with an empty ledger and can only *add* until it catches up
with what is already on disk (it never revokes an entry it has no record of
writing).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from crossby.config.json_utils import atomic_write_text
from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern

logger = structlog.get_logger()

# Sits beside ``.crossby/sync-report.md`` (see ``sync/report.py``).
LEDGER_PATH = Path(".crossby") / "owned.json"
LEDGER_VERSION = 1

# Only these three concerns carry revocation semantics.
_HOOKS = SyncConcern.HOOKS.value
_PERMISSIONS = SyncConcern.PERMISSIONS.value
_MCP = SyncConcern.MCP.value


@dataclass
class OwnershipLedger:
    """In-memory view of ``.crossby/owned.json``: tool → concern → item list.

    Construct via :func:`load_ledger`; persist via :func:`save_ledger`. The
    accessor methods (:meth:`hooks`, :meth:`permissions`, :meth:`mcp`) return a
    ``frozenset`` of the identities crossby currently owns for a tool, and the
    ``record_*`` methods overwrite ownership for one ``(tool, concern)`` pair.
    """

    # tool_id str → concern str → list of item identities (JSON-native shapes).
    _data: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    # -- read ----------------------------------------------------------------

    def hooks(self, tool_id: AIToolID) -> frozenset[tuple[str, str]]:
        """Owned ``(event, command)`` pairs for *tool_id*."""
        out: set[tuple[str, str]] = set()
        for item in self._concern_items(tool_id, _HOOKS):
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                out.add((item[0], item[1]))
        return frozenset(out)

    def permissions(self, tool_id: AIToolID) -> frozenset[str]:
        """Owned canonical permission patterns for *tool_id*."""
        return frozenset(
            item for item in self._concern_items(tool_id, _PERMISSIONS) if isinstance(item, str)
        )

    def mcp(self, tool_id: AIToolID) -> frozenset[str]:
        """Owned MCP server names for *tool_id*."""
        return frozenset(
            item for item in self._concern_items(tool_id, _MCP) if isinstance(item, str)
        )

    def _concern_items(self, tool_id: AIToolID, concern: str) -> list[Any]:
        tool = self._data.get(str(tool_id))
        if not isinstance(tool, dict):
            return []
        items = tool.get(concern)
        return items if isinstance(items, list) else []

    # -- write ---------------------------------------------------------------

    def record_hooks(self, tool_id: AIToolID, pairs: Iterable[tuple[str, str]]) -> None:
        """Replace the owned hooks for *tool_id* with *pairs* (sorted, deduped)."""
        self._set(tool_id, _HOOKS, [[event, command] for event, command in sorted(set(pairs))])

    def record_permissions(self, tool_id: AIToolID, patterns: Iterable[str]) -> None:
        """Replace the owned permission patterns for *tool_id*."""
        self._set(tool_id, _PERMISSIONS, sorted(set(patterns)))

    def record_mcp(self, tool_id: AIToolID, names: Iterable[str]) -> None:
        """Replace the owned MCP server names for *tool_id*."""
        self._set(tool_id, _MCP, sorted(set(names)))

    def _set(self, tool_id: AIToolID, concern: str, items: list[Any]) -> None:
        key = str(tool_id)
        tool = self._data.setdefault(key, {})
        if items:
            tool[concern] = items
        else:
            # Owning nothing for a concern drops the key rather than storing an
            # empty list — keeps the on-disk file minimal and comparisons clean.
            tool.pop(concern, None)
            if not tool:
                self._data.pop(key, None)

    def is_empty(self) -> bool:
        """True when crossby owns nothing anywhere."""
        return not self._data

    def to_json(self) -> dict[str, Any]:
        """Serialisable form written to ``owned.json``."""
        return {"version": LEDGER_VERSION, "owned": self._data}


def load_ledger(project_root: Path) -> OwnershipLedger:
    """Read ``.crossby/owned.json``; a missing or malformed file yields an empty ledger."""
    path = project_root / LEDGER_PATH
    if not path.is_file():
        return OwnershipLedger()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        # Degrade to "own nothing" rather than crash — the ledger is advisory,
        # and a corrupt one must never block a sync or delete anything.
        logger.warning("ownership.malformed_ledger", path=str(path))
        return OwnershipLedger()
    if not isinstance(raw, dict):
        return OwnershipLedger()
    owned = raw.get("owned")
    if not isinstance(owned, dict):
        return OwnershipLedger()

    data: dict[str, dict[str, list[Any]]] = {}
    for tool, concerns in owned.items():
        if not isinstance(tool, str) or not isinstance(concerns, dict):
            continue
        clean: dict[str, list[Any]] = {}
        for concern, items in concerns.items():
            if isinstance(concern, str) and isinstance(items, list):
                clean[concern] = items
        if clean:
            data[tool] = clean
    return OwnershipLedger(data)


def save_ledger(project_root: Path, ledger: OwnershipLedger) -> bool:
    """Write the ledger to ``.crossby/owned.json``.

    Returns ``True`` when the file was created or its contents changed, ``False``
    when the on-disk file already matched (so callers can skip a redundant
    ``.gitignore`` touch on an idempotent re-run).
    """
    path = project_root / LEDGER_PATH
    exists = path.is_file()
    # Don't materialise an empty ledger just because a ledger-bearing concern
    # ran with nothing to record — but a file that already exists must still be
    # updated (e.g. cleared) when ownership shrinks to nothing.
    if not exists and ledger.is_empty():
        return False
    body = json.dumps(ledger.to_json(), indent=2, sort_keys=True) + "\n"
    if exists:
        try:
            if path.read_text(encoding="utf-8") == body:
                return False
        except OSError:
            pass
    # Atomic tmp+rename — a crash mid-write must not corrupt the ledger (which
    # would silently wipe all accumulated provenance), matching the other writers.
    atomic_write_text(path, body)
    return True


__all__ = [
    "LEDGER_PATH",
    "LEDGER_VERSION",
    "OwnershipLedger",
    "load_ledger",
    "save_ledger",
]
