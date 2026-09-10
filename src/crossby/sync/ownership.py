"""Provenance ledger — a record of what crossby wrote, so it can revoke it later.

``.crossby/owned.json`` is a per-machine, gitignored record of the items crossby
has written to each tool, keyed by ``(tool_id, concern)``, plus exact PROJECT
path baselines used by scenes. :func:`run_sync
<crossby.sync.run_sync>` diffs this ledger against the current sync data to
compute what to revoke, so a writer never removes an entry a human authored.

Item identities per concern:

- **hooks** — ``(event, command)`` pairs, stored as 2-element JSON lists.
- **permissions** — canonical command patterns (strings, e.g. ``"git diff:*"``).
- **mcp** — server names (strings).

A missing or malformed ledger degrades to "own nothing" for ordinary sync —
purely additive behaviour, never a crash. Scene operations use the checked load
and fail closed on malformed restore authority. Because the file is gitignored
it is per-machine: a fresh clone starts with an empty ledger and can only *add*
until it catches up with what is already on disk (it never revokes an entry it
has no record of writing).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

import structlog

from crossby.config.json_utils import atomic_write_text
from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern

logger = structlog.get_logger()

# Sits beside ``.crossby/sync-report.md`` (see ``sync/report.py``).
LEDGER_PATH = Path(".crossby") / "owned.json"
# v2 added the ``scene`` section (DECLARE-key provenance); v3 adds
# ``scene_paths`` (exact PROJECT-path restore provenance). v4 records the
# device/inode of a directory immediately before its restore rename, allowing a
# retry to authenticate a completed rename after cleanup persistence failed.
# v5 records canonical PROJECT sources that a scene deliberately left untouched,
# so their managed markers are never mistaken for legacy scene output on clear.
# v6 records whether a directory baseline has actually been displaced and its
# identity at capture time. This prevents a missing backup from making generated
# scene output look like the original directory captured before displacement.
# The version remains advisory, and older files without either section remain
# readable.
LEDGER_VERSION = 6

# Only these three concerns carry revocation semantics.
_HOOKS = SyncConcern.HOOKS.value
_PERMISSIONS = SyncConcern.PERMISSIONS.value
_MCP = SyncConcern.MCP.value

# Sentinel distinguishing an *absent* ``scene`` key (fine) from an explicit
# ``null`` (corrupt) in corruption classification — ``dict.get`` conflates them.
_MISSING_SCENE: object = object()
_MISSING_SCENE_PATHS: object = object()


class ScenePathRestoreKind(StrEnum):
    """The supported pre-scene states of a PROJECT target path."""

    ABSENT = "absent"
    SYMLINK = "symlink"
    DIRECTORY = "directory"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class ScenePathRestore:
    """Exact baseline needed to restore one physical PROJECT target."""

    kind: ScenePathRestoreKind
    link_target: str | None = None
    backup_path: str | None = None
    directory_displaced: bool | None = None
    directory_device: int | None = None
    directory_inode: int | None = None

    @classmethod
    def absent(cls) -> ScenePathRestore:
        return cls(ScenePathRestoreKind.ABSENT)

    @classmethod
    def symlink(cls, literal_target: str) -> ScenePathRestore:
        return cls(ScenePathRestoreKind.SYMLINK, link_target=literal_target)

    @classmethod
    def directory(
        cls,
        backup_path: str,
        *,
        displaced: bool | None = None,
        device: int | None = None,
        inode: int | None = None,
    ) -> ScenePathRestore:
        return cls(
            ScenePathRestoreKind.DIRECTORY,
            backup_path=backup_path,
            directory_displaced=displaced,
            directory_device=device,
            directory_inode=inode,
        )

    @classmethod
    def unchanged(cls) -> ScenePathRestore:
        """Record a canonical source a scene intentionally did not mutate."""
        return cls(ScenePathRestoreKind.UNCHANGED)

    def with_directory_identity(self, *, device: int, inode: int) -> ScenePathRestore:
        """Return a directory descriptor authenticated for an attempted restore."""
        if self.kind != ScenePathRestoreKind.DIRECTORY or self.backup_path is None:
            raise ValueError("only a directory restore descriptor has a directory identity")
        return self.directory(
            self.backup_path,
            displaced=self.directory_displaced,
            device=device,
            inode=inode,
        )

    def with_directory_displaced(self) -> ScenePathRestore:
        """Return a directory descriptor whose displacement is durable."""
        if self.kind != ScenePathRestoreKind.DIRECTORY or self.backup_path is None:
            raise ValueError("only a directory restore descriptor has displacement state")
        return self.directory(
            self.backup_path,
            displaced=True,
            device=self.directory_device,
            inode=self.directory_inode,
        )

    def to_json(self) -> dict[str, str | int]:
        out: dict[str, str | int] = {"kind": self.kind.value}
        if self.link_target is not None:
            out["target"] = self.link_target
        if self.backup_path is not None:
            out["backup"] = self.backup_path
        if self.directory_displaced is not None:
            out["displaced"] = self.directory_displaced
        if self.directory_device is not None:
            out["device"] = self.directory_device
            # Validation requires device and inode to appear as a pair.
            assert self.directory_inode is not None
            out["inode"] = self.directory_inode
        return out


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
    # Normalised project-relative physical target path → exact pre-scene state.
    _scene_paths: dict[str, ScenePathRestore] = field(default_factory=dict)

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

    # -- scene PROJECT path provenance --------------------------------------

    def scene_restore(self, target_path: str | Path) -> ScenePathRestore | None:
        """Return the recorded baseline for a registered physical target."""
        return self._scene_paths.get(_normalise_scene_target(target_path))

    def scene_restores(self) -> dict[str, ScenePathRestore]:
        """Return a copy of every path-keyed restore descriptor."""
        return dict(self._scene_paths)

    def record_scene_restore(self, target_path: str | Path, descriptor: ScenePathRestore) -> bool:
        """Record the first baseline for *target_path*; never overwrite it.

        Returns ``True`` when a descriptor was added and ``False`` when that
        physical path already had recovery authority.
        """
        target = _normalise_scene_target(target_path)
        _validate_scene_restore(target, descriptor)
        if target in self._scene_paths:
            return False
        self._scene_paths[target] = descriptor
        return True

    def record_scene_absent(self, target_path: str | Path) -> bool:
        return self.record_scene_restore(target_path, ScenePathRestore.absent())

    def record_scene_symlink(self, target_path: str | Path, literal_target: str) -> bool:
        return self.record_scene_restore(target_path, ScenePathRestore.symlink(literal_target))

    def record_scene_directory(self, target_path: str | Path, backup_path: str | Path) -> bool:
        return self.record_scene_restore(
            target_path, ScenePathRestore.directory(PurePosixPath(backup_path).as_posix())
        )

    def record_scene_restore_directory_identity(
        self, target_path: str | Path, *, device: int, inode: int
    ) -> ScenePathRestore:
        """Persist the identity of a directory before moving or restoring it.

        A matching path identity on a later retry proves that the exact baseline
        was moved. The baseline itself is never replaceable.
        """
        target = _normalise_scene_target(target_path)
        descriptor = self._scene_paths.get(target)
        if descriptor is None or descriptor.kind != ScenePathRestoreKind.DIRECTORY:
            raise ValueError("directory restore identity requires an existing directory descriptor")
        if descriptor.directory_device is not None:
            if (descriptor.directory_device, descriptor.directory_inode) != (device, inode):
                raise ValueError("directory restore identity is already recorded")
            return descriptor
        updated = descriptor.with_directory_identity(device=device, inode=inode)
        _validate_scene_restore(target, updated)
        self._scene_paths[target] = updated
        return updated

    def record_scene_directory_displaced(self, target_path: str | Path) -> ScenePathRestore:
        """Mark a recorded directory baseline as having reached its backup."""
        target = _normalise_scene_target(target_path)
        descriptor = self._scene_paths.get(target)
        if descriptor is None or descriptor.kind != ScenePathRestoreKind.DIRECTORY:
            raise ValueError("directory displacement requires an existing directory descriptor")
        if descriptor.directory_displaced is True:
            return descriptor
        updated = descriptor.with_directory_displaced()
        _validate_scene_restore(target, updated)
        self._scene_paths[target] = updated
        return updated

    def clear_scene_restore(self, target_path: str | Path) -> None:
        """Drop a descriptor only after its baseline is confirmed restored."""
        self._scene_paths.pop(_normalise_scene_target(target_path), None)

    def is_empty(self) -> bool:
        """True when crossby owns nothing anywhere (revocable or scene DECLARE)."""
        return not self._data and not self._scene and not self._scene_paths

    def to_json(self) -> dict[str, Any]:
        """Serialisable form written to ``owned.json``.

        The ``scene`` section is emitted only when non-empty, so a project that
        never used scenes serialises exactly as it did under the v1 schema.
        """
        out: dict[str, Any] = {"version": LEDGER_VERSION, "owned": self._data}
        if self._scene:
            out["scene"] = self._scene
        if self._scene_paths:
            out["scene_paths"] = {
                path: descriptor.to_json() for path, descriptor in sorted(self._scene_paths.items())
            }
        return out


@dataclass(frozen=True)
class LoadedLedger:
    """Outcome of reading ``owned.json`` with corruption detection.

    ``ledger`` is always usable (an empty ledger on any failure). ``corrupt`` is
    ``True`` only when the path *exists* but could not be read as a valid ledger —
    the signal a fail-closed caller (``scene use`` / ``clear``) uses to refuse
    rather than revert from provenance it cannot parse.
    """

    ledger: OwnershipLedger
    corrupt: bool = False  # path exists but could not be read as a valid ledger


def load_ledger(project_root: Path) -> OwnershipLedger:
    """Read ``.crossby/owned.json``; a missing or malformed file yields an empty ledger."""
    return load_ledger_checked(project_root).ledger


def load_ledger_checked(project_root: Path) -> LoadedLedger:
    """Read ``.crossby/owned.json``, flagging a corrupt/unreadable file.

    ``corrupt`` fails closed on anything that is neither a truly-absent path nor a
    valid ledger:

    - **Not corrupt** when the path is *truly absent* (``os.path.lexists`` is
      ``False`` — a fresh per-machine clone legitimately owns nothing) or when it
      holds a valid, possibly-empty ledger.
    - **Corrupt** when the path exists but is a directory or broken symlink
      (``not path.is_file()``), ``json.loads`` raises, or the parsed root is not a
      ``dict``. ``os.path.lexists`` (not ``Path.is_file``) is deliberate so a
      broken symlink at ``owned.json`` is caught rather than mistaken for missing.

    Lenient sub-parsing of malformed additive ``owned`` entries is preserved.
    Malformed ``scene`` or ``scene_paths`` entries fail closed because they are
    restore authority.
    """
    path = project_root / LEDGER_PATH
    if not os.path.lexists(path):
        # Truly absent — never corrupt; a fresh clone starts from an empty ledger.
        return LoadedLedger(OwnershipLedger())
    if path.is_symlink() or not path.is_file():
        # A symlink (even one pointing at a valid file), directory, or broken
        # link at the ledger path: present but not a plain ledger. Fail closed —
        # ``run_sync`` reads the ledger *before* the post-writer save, so a
        # symlinked ``owned.json`` must not be followed (it could redirect the
        # read/write outside ``.crossby/``). ``load_ledger`` degrades this to an
        # empty ledger (own-nothing / additive-only); the fail-closed callers
        # (scene use/clear) refuse on ``corrupt``.
        logger.warning("ownership.malformed_ledger", path=str(path))
        return LoadedLedger(OwnershipLedger(), corrupt=True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        # Degrade to "own nothing" rather than crash — the ledger is advisory —
        # but flag corruption so a fail-closed caller can refuse to proceed.
        logger.warning("ownership.malformed_ledger", path=str(path))
        return LoadedLedger(OwnershipLedger(), corrupt=True)
    if not isinstance(raw, dict):
        logger.warning("ownership.malformed_ledger", path=str(path))
        return LoadedLedger(OwnershipLedger(), corrupt=True)

    # The scene section is the *authority* for a scene revert, so — unlike the
    # additive ``owned`` section — a malformed entry here fails closed rather than
    # degrading silently: dropping a real (tool, key) would let ``clear`` revert
    # nothing yet delete ``scene-state.json``, leaving the setting applied with no
    # recovery record (the exact #119 bug). ``owned`` stays lenient (a fresh clone
    # can only add, never revoke what it has no record of), so its malformed
    # sub-entries are dropped, not treated as corruption. A sentinel distinguishes
    # an *absent* scene key (older v1 ledgers — fine) from an explicit ``null``.
    scene_corrupt = _scene_section_corrupt(raw.get("scene", _MISSING_SCENE))
    scene_paths_raw = raw.get("scene_paths", _MISSING_SCENE_PATHS)
    scene_paths_corrupt = _scene_paths_section_corrupt(scene_paths_raw)

    owned = raw.get("owned")
    if not isinstance(owned, dict):
        # Root is a dict but ``owned`` is missing/malformed — leniently empty
        # (mirrors the historical degradation), while valid scene provenance
        # remains available for exact recovery.
        scene = _load_scene_section(raw.get("scene"))
        scene_paths = _load_scene_paths_section(scene_paths_raw)
        return LoadedLedger(
            OwnershipLedger({}, scene, scene_paths),
            corrupt=scene_corrupt or scene_paths_corrupt,
        )

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
    scene_paths = _load_scene_paths_section(scene_paths_raw)
    return LoadedLedger(
        OwnershipLedger(data, scene, scene_paths),
        corrupt=scene_corrupt or scene_paths_corrupt,
    )


def _scene_section_corrupt(raw_scene: object) -> bool:
    """True when a *present* scene section can't be read as clean provenance.

    crossby only ever writes ``{<AIToolID>: {<SceneDeclareKey>: [str, ...]}}`` (and
    omits the key entirely when it owns nothing), so anything else is external
    tampering/corruption of the revert authority and fails closed. Fail-closed
    triggers, all of which would silently orphan real provenance:

    - the section is *present but not a dict* — including an explicit ``null``
      (distinguished from an absent key, which is ``_MISSING_SCENE`` → not corrupt,
      so older v1 ledgers stay readable);
    - a **tool name is not a known** :class:`~crossby.models.ai.AIToolID`, or a
      **key is not a known** :class:`SceneDeclareKey` — an entry no revert handler
      will ever consult (a typo, or a newer schema this build can't revert);
    - a tool entry is not a dict, a key value is not a list, or a list holds a
      non-string.

    A valid-but-empty ``{}`` is not corrupt.
    """
    if raw_scene is _MISSING_SCENE:
        return False
    if not isinstance(raw_scene, dict):
        return True
    known_tools = {tool.value for tool in AIToolID}
    known_keys = {key.value for key in SceneDeclareKey}
    for tool, keys in raw_scene.items():
        if not isinstance(tool, str) or tool not in known_tools or not isinstance(keys, dict):
            return True
        for key, items in keys.items():
            if not isinstance(key, str) or key not in known_keys or not isinstance(items, list):
                return True
            if any(not isinstance(item, str) for item in items):
                return True
    return False


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


def _registered_scene_targets() -> frozenset[str]:
    """Every physical path a built-in PROJECT mechanism may mutate."""
    # Local imports avoid making the low-level ownership module part of the
    # sync writer import cycle during module initialisation.
    from crossby.config.skills import SKILLS_DIR
    from crossby.sync.agents import _AGENT_TARGET_PATHS

    return frozenset({*SKILLS_DIR.values(), *_AGENT_TARGET_PATHS.values()})


def _normalise_relative_path(path: str | Path) -> str:
    raw = str(path)
    pure = PurePosixPath(raw)
    if not raw or raw == "." or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe project-relative path: {raw!r}")
    normalised = pure.as_posix()
    if normalised.startswith("./") or "\\" in normalised:
        raise ValueError(f"path is not normalized POSIX project-relative form: {raw!r}")
    return normalised


def _normalise_scene_target(path: str | Path) -> str:
    target = _normalise_relative_path(path)
    if target not in _registered_scene_targets():
        raise ValueError(f"unregistered scene target path: {target!r}")
    return target


def _normalise_backup_path(target: str, backup: str | Path) -> str:
    value = _normalise_relative_path(backup)
    target_path = PurePosixPath(target)
    backup_path = PurePosixPath(value)
    # A scene displacement is always allocated beside the exact target using
    # backup_path's ``<name>.bak[2...]`` convention. This rejects a plausible
    # but unsafe reference elsewhere in the project.
    prefix = target_path.name + ".bak"
    suffix = backup_path.name[len(prefix) :] if backup_path.name.startswith(prefix) else None
    if (
        backup_path.parent != target_path.parent
        or suffix is None
        or (suffix and (not suffix.isdigit() or int(suffix) < 2))
    ):
        raise ValueError(f"unsafe scene backup path {value!r} for target {target!r}")
    return value


def _validate_scene_restore(target: str, descriptor: ScenePathRestore) -> None:
    if not isinstance(descriptor, ScenePathRestore):
        raise ValueError("scene restore descriptor has an invalid type")
    if descriptor.kind == ScenePathRestoreKind.ABSENT:
        if (
            descriptor.link_target is not None
            or descriptor.backup_path is not None
            or descriptor.directory_displaced is not None
            or descriptor.directory_device is not None
            or descriptor.directory_inode is not None
        ):
            raise ValueError("absent scene restore descriptor has unexpected fields")
        return
    if descriptor.kind == ScenePathRestoreKind.UNCHANGED:
        if (
            descriptor.link_target is not None
            or descriptor.backup_path is not None
            or descriptor.directory_displaced is not None
            or descriptor.directory_device is not None
            or descriptor.directory_inode is not None
        ):
            raise ValueError("unchanged scene restore descriptor has unexpected fields")
        return
    if descriptor.kind == ScenePathRestoreKind.SYMLINK:
        if (
            not isinstance(descriptor.link_target, str)
            or not descriptor.link_target
            or "\x00" in descriptor.link_target
            or descriptor.backup_path is not None
            or descriptor.directory_displaced is not None
            or descriptor.directory_device is not None
            or descriptor.directory_inode is not None
        ):
            raise ValueError("symlink scene restore descriptor requires only a literal target")
        return
    if descriptor.kind == ScenePathRestoreKind.DIRECTORY:
        if descriptor.link_target is not None or not isinstance(descriptor.backup_path, str):
            raise ValueError("directory scene restore descriptor requires only a backup path")
        if descriptor.directory_displaced is not None and not isinstance(
            descriptor.directory_displaced, bool
        ):
            raise ValueError("directory displacement state must be a boolean")
        if _normalise_backup_path(target, descriptor.backup_path) != descriptor.backup_path:
            raise ValueError("scene backup path is not normalized")
        identity = (descriptor.directory_device, descriptor.directory_inode)
        if descriptor.directory_displaced is False and identity == (None, None):
            raise ValueError("pending directory displacement requires a directory identity")
        if identity != (None, None) and (
            any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in identity
            )
        ):
            raise ValueError(
                "directory restore identity must be non-negative integer device and inode"
            )
        return
    raise ValueError(f"unknown scene restore descriptor kind: {descriptor.kind!r}")


def _parse_scene_restore(target: str, raw: object) -> ScenePathRestore:
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("scene restore descriptor must be an object")
    kind = raw.get("kind")
    if kind == ScenePathRestoreKind.ABSENT.value and set(raw) == {"kind"}:
        descriptor = ScenePathRestore.absent()
    elif kind == ScenePathRestoreKind.UNCHANGED.value and set(raw) == {"kind"}:
        descriptor = ScenePathRestore.unchanged()
    elif kind == ScenePathRestoreKind.SYMLINK.value and set(raw) == {"kind", "target"}:
        literal = raw.get("target")
        if not isinstance(literal, str):
            raise ValueError("symlink target must be a string")
        descriptor = ScenePathRestore.symlink(literal)
    elif kind == ScenePathRestoreKind.DIRECTORY.value and set(raw) in (
        {"kind", "backup"},
        {"kind", "backup", "displaced"},
        {"kind", "backup", "device", "inode"},
        {"kind", "backup", "displaced", "device", "inode"},
    ):
        backup = raw.get("backup")
        if not isinstance(backup, str):
            raise ValueError("directory backup must be a string")
        device = raw.get("device")
        inode = raw.get("inode")
        displaced = raw.get("displaced")
        if "displaced" in raw and not isinstance(displaced, bool):
            raise ValueError("directory displacement state must be a boolean")
        if "device" in raw and (
            not isinstance(device, int)
            or isinstance(device, bool)
            or not isinstance(inode, int)
            or isinstance(inode, bool)
        ):
            raise ValueError("directory restore identity must contain integer device and inode")
        descriptor = ScenePathRestore.directory(
            backup, displaced=displaced, device=device, inode=inode
        )
    else:
        raise ValueError("unknown or malformed scene restore descriptor")
    _validate_scene_restore(target, descriptor)
    return descriptor


def _scene_paths_section_corrupt(raw_scene_paths: object) -> bool:
    """True when present PROJECT restore authority is malformed or unsafe."""
    if raw_scene_paths is _MISSING_SCENE_PATHS:
        return False
    if not isinstance(raw_scene_paths, dict):
        return True
    try:
        for raw_target, raw_descriptor in raw_scene_paths.items():
            if not isinstance(raw_target, str):
                return True
            target = _normalise_scene_target(raw_target)
            if target != raw_target:
                return True
            _parse_scene_restore(target, raw_descriptor)
    except ValueError:
        return True
    return False


def _load_scene_paths_section(raw_scene_paths: object) -> dict[str, ScenePathRestore]:
    if not isinstance(raw_scene_paths, dict):
        return {}
    out: dict[str, ScenePathRestore] = {}
    try:
        for raw_target, raw_descriptor in raw_scene_paths.items():
            if not isinstance(raw_target, str):
                return {}
            target = _normalise_scene_target(raw_target)
            out[target] = _parse_scene_restore(target, raw_descriptor)
    except ValueError:
        return {}
    return out


def save_ledger(project_root: Path, ledger: OwnershipLedger) -> bool:
    """Write the ledger to ``.crossby/owned.json``.

    Returns ``True`` when the file was created or its contents changed, ``False``
    when the on-disk file already matched (so callers can skip a redundant
    ``.gitignore`` touch on an idempotent re-run).

    Fails closed on a symlinked ``owned.json`` (or a symlinked ``.crossby``
    ancestor) *before* the read below — the ledger is provenance state, never
    ordinary generated output, so it is refused rather than followed (mirroring
    the ``load_ledger`` fail-closed path). Raises :class:`SyncContainmentError`.
    """
    from crossby.sync.safe_write import ProjectScope, SyncContainmentError, assert_ancestors

    path = project_root / LEDGER_PATH
    assert_ancestors(ProjectScope(project_root), path)
    if path.is_symlink():
        raise SyncContainmentError(
            f"{path} is a symlink; refusing to write the ownership ledger through it. "
            "Remove the symlink and re-run."
        )
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
    "LoadedLedger",
    "OwnershipLedger",
    "SceneDeclareKey",
    "ScenePathRestore",
    "ScenePathRestoreKind",
    "load_ledger",
    "load_ledger_checked",
    "save_ledger",
]
