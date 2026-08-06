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
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

from crossby.config.json_utils import atomic_write_text
from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern

logger = structlog.get_logger()

# Sits beside ``.crossby/sync-report.md`` (see ``sync/report.py``).
LEDGER_PATH = Path(".crossby") / "owned.json"
# v2 adds the ``scene`` section (DECLARE-key provenance). Bumped from 1; older
# v1 files load unchanged because the version is advisory (never gates a read)
# and a missing ``scene`` section degrades to "own no DECLARE keys".
LEDGER_VERSION = 2

# Only these three concerns carry revocation semantics.
_HOOKS = SyncConcern.HOOKS.value
_PERMISSIONS = SyncConcern.PERMISSIONS.value
_MCP = SyncConcern.MCP.value


class SceneDeclareKey(StrEnum):
    """The scene DECLARE surfaces whose crossby-written entries need provenance.

    Every one is *new* — none is covered by the revocable-sync ledger above,
    which tracks hooks / ``permissions.allow`` / MCP presence. Reverting a scene
    (``clear_scene``) must revert only the entries crossby wrote for these keys,
    so each is recorded per ``(tool, key)`` exactly like the additive concerns.
    """

    #: Claude ``skillOverrides: {"<name>": "off"}`` in ``.claude/settings.json``.
    SKILL_OVERRIDES = "skill_overrides"
    #: Claude ``permissions.deny: ["Agent(<name>)"]`` entries.
    DENY_AGENTS = "deny_agents"
    #: Claude ``disabledMcpjsonServers`` server names.
    DISABLED_MCP = "disabled_mcp"
    #: Codex ``mcp_servers.<id>.enabled = false`` server ids.
    CODEX_MCP_DISABLED = "codex_mcp_disabled"
    #: Antigravity CLI ``mcpServers.<name>.disabled = true`` server names.
    ANTIGRAVITY_MCP_DISABLED = "antigravity_mcp_disabled"


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
    # tool_id str → SceneDeclareKey value → list of names crossby wrote to that
    # DECLARE surface. Kept in a separate section so scene provenance never
    # collides with the revocable-sync concerns above (a concern name and a
    # DECLARE-key name could otherwise clash in one namespace).
    _scene: dict[str, dict[str, list[str]]] = field(default_factory=dict)

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

    # -- scene DECLARE provenance --------------------------------------------

    def scene_declare(self, tool_id: AIToolID, key: SceneDeclareKey) -> frozenset[str]:
        """Owned entry names crossby wrote to *key* on *tool_id*."""
        tool = self._scene.get(str(tool_id))
        if not isinstance(tool, dict):
            return frozenset()
        items = tool.get(key.value)
        if not isinstance(items, list):
            return frozenset()
        return frozenset(item for item in items if isinstance(item, str))

    def record_scene_declare(
        self, tool_id: AIToolID, key: SceneDeclareKey, names: Iterable[str]
    ) -> None:
        """Replace the owned entries for *key* on *tool_id* (sorted, deduped).

        Recording an empty set drops the key — and the tool, if it then owns no
        DECLARE surfaces — mirroring :meth:`_set` so the on-disk file stays
        minimal and ``is_empty`` comparisons stay clean.
        """
        tool_key = str(tool_id)
        items = sorted(set(names))
        tool = self._scene.setdefault(tool_key, {})
        if items:
            tool[key.value] = items
        else:
            tool.pop(key.value, None)
            if not tool:
                self._scene.pop(tool_key, None)

    def is_empty(self) -> bool:
        """True when crossby owns nothing anywhere (revocable or scene DECLARE)."""
        return not self._data and not self._scene

    def to_json(self) -> dict[str, Any]:
        """Serialisable form written to ``owned.json``.

        The ``scene`` section is emitted only when non-empty, so a project that
        never used scenes serialises exactly as it did under the v1 schema.
        """
        out: dict[str, Any] = {"version": LEDGER_VERSION, "owned": self._data}
        if self._scene:
            out["scene"] = self._scene
        return out


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

    scene = _load_scene_section(raw.get("scene"))
    return OwnershipLedger(data, scene)


def _load_scene_section(raw_scene: object) -> dict[str, dict[str, list[str]]]:
    """Parse the ``scene`` section, keeping only well-formed string entries.

    A missing/malformed section (older v1 ledgers, corruption) yields an empty
    map — the same graceful degradation the ``owned`` section uses, so scene
    provenance never blocks a load or invents ownership.
    """
    scene: dict[str, dict[str, list[str]]] = {}
    if not isinstance(raw_scene, dict):
        return scene
    for tool, keys in raw_scene.items():
        if not isinstance(tool, str) or not isinstance(keys, dict):
            continue
        clean: dict[str, list[str]] = {}
        for key, items in keys.items():
            if isinstance(key, str) and isinstance(items, list):
                clean[key] = [item for item in items if isinstance(item, str)]
        if clean:
            scene[tool] = clean
    return scene


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
    "SceneDeclareKey",
    "load_ledger",
    "save_ledger",
]
