"""Scene activation state — the CLI-layer record of what scene is active.

``.crossby/scene-state.json`` is a per-machine, gitignored bookkeeping file the
``crossby scene`` command writes on ``use`` and reads on ``status`` / ``clear``
/ a switch. It records the active scene name, when it was applied, the per-tool
mechanism, an applied/partial flag, and a normalised content hash per
scene-managed file so ``status`` can detect drift.

It is deliberately **not** the revert authority: :func:`clear_scene
<crossby.scenes.engine.clear_scene>` reverts from the ownership ledger
(``owned.json``), which is the record of what crossby actually wrote. This file
answers "which scene is active, and has it drifted"; the ledger answers "what
did crossby write, and how do I take it back". They can desync (a plain
``crossby sync`` between two scene commands touches the ledger only), so the two
roles are kept separate on purpose.

A missing file means "no active scene". A malformed file or an unrecognised
schema version is treated as *unreadable*: the caller warns and proceeds as if no
scene were active, never attempting a revert from state it cannot interpret.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from crossby.config.json_utils import atomic_write_text
from crossby.sync.base import SyncResult

logger = structlog.get_logger()

SCENE_STATE_PATH = Path(".crossby") / "scene-state.json"
# Bumped when the on-disk shape changes incompatibly. A file whose version does
# not match is treated as unreadable (see :func:`load_scene_state`).
SCENE_STATE_VERSION = 1
_GITIGNORE_BLOCK_ID = "scene state"


@dataclass
class SceneToolRecord:
    """What one tool carried under the active scene.

    ``mechanisms`` maps each participating concern to the mechanism used
    (``declare`` / ``project`` / ``unsupported``); ``status`` is ``applied`` or
    ``failed`` (the tool produced an ``error`` row during apply).
    """

    mechanisms: dict[str, str] = field(default_factory=dict)
    status: str = "applied"


@dataclass
class SceneState:
    """The parsed contents of ``scene-state.json``."""

    scene: str
    applied_at: str
    status: str  # "applied" | "partial"
    tools: dict[str, SceneToolRecord] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)

    @property
    def tool_ids(self) -> list[str]:
        """The tool-id strings this scene was applied to, sorted."""
        return sorted(self.tools)


@dataclass(frozen=True)
class LoadedSceneState:
    """Outcome of reading the state file.

    ``state`` is ``None`` when there is no active scene. ``warning`` is set only
    when a file *existed* but could not be interpreted (corruption or a schema
    version mismatch) — the caller surfaces it and proceeds as "no active scene".
    """

    state: SceneState | None
    warning: str | None = None


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_scene_state(project_root: Path) -> LoadedSceneState:
    """Read ``.crossby/scene-state.json``.

    A missing file yields ``LoadedSceneState(None)``. A malformed file or an
    unrecognised schema version yields ``LoadedSceneState(None, warning=...)`` —
    never an exception, and never a partially-trusted state.
    """
    path = project_root / SCENE_STATE_PATH
    if not path.is_file():
        return LoadedSceneState(None)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        logger.warning("scene_state.malformed", path=str(path))
        return LoadedSceneState(
            None,
            warning=f"{SCENE_STATE_PATH.as_posix()} is unreadable; treating as no active scene",
        )
    if not isinstance(raw, dict):
        return LoadedSceneState(
            None, warning=f"{SCENE_STATE_PATH.as_posix()} is malformed; treating as no active scene"
        )
    version = raw.get("version")
    if version != SCENE_STATE_VERSION:
        return LoadedSceneState(
            None,
            warning=(
                f"{SCENE_STATE_PATH.as_posix()} has unrecognised schema version {version!r} "
                f"(expected {SCENE_STATE_VERSION}); treating as no active scene"
            ),
        )
    scene = raw.get("scene")
    applied_at = raw.get("applied_at")
    if not isinstance(scene, str) or not isinstance(applied_at, str):
        return LoadedSceneState(
            None,
            warning=(
                f"{SCENE_STATE_PATH.as_posix()} is missing required fields; "
                "treating as no active scene"
            ),
        )
    status = raw.get("status")
    return LoadedSceneState(
        SceneState(
            scene=scene,
            applied_at=applied_at,
            status=status if isinstance(status, str) else "applied",
            tools=_parse_tools(raw.get("tools")),
            hashes=_parse_hashes(raw.get("hashes")),
        )
    )


def save_scene_state(project_root: Path, state: SceneState) -> None:
    """Write *state* atomically (temp file + rename) and gitignore it.

    The state file is the sole record of what scene is active; a crash mid-write
    must never leave a half-written file, so the write goes through
    :func:`~crossby.config.json_utils.atomic_write_text`.
    """
    path = project_root / SCENE_STATE_PATH
    body = json.dumps(_to_json(state), indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, body)
    _ensure_gitignored(project_root)


def clear_scene_state(project_root: Path) -> None:
    """Remove ``scene-state.json`` — the "no scene active" state."""
    path = project_root / SCENE_STATE_PATH
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _to_json(state: SceneState) -> dict[str, Any]:
    return {
        "version": SCENE_STATE_VERSION,
        "scene": state.scene,
        "applied_at": state.applied_at,
        "status": state.status,
        "tools": {
            tool: {"mechanisms": rec.mechanisms, "status": rec.status}
            for tool, rec in state.tools.items()
        },
        "hashes": state.hashes,
    }


def _parse_tools(raw: object) -> dict[str, SceneToolRecord]:
    out: dict[str, SceneToolRecord] = {}
    if not isinstance(raw, dict):
        return out
    for tool, rec in raw.items():
        if not isinstance(tool, str) or not isinstance(rec, dict):
            continue
        mechanisms = {
            k: v
            for k, v in (rec.get("mechanisms") or {}).items()
            if isinstance(k, str) and isinstance(v, str)
        }
        status = rec.get("status")
        out[tool] = SceneToolRecord(
            mechanisms=mechanisms, status=status if isinstance(status, str) else "applied"
        )
    return out


def _parse_hashes(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def _ensure_gitignored(project_root: Path) -> None:
    from crossby.sync.gitignore_utils import update_managed_block

    update_managed_block(project_root, _GITIGNORE_BLOCK_ID, [SCENE_STATE_PATH.as_posix()])


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """A UTC ISO-8601 timestamp (seconds precision) for ``applied_at``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def scene_managed_paths(project_root: Path, results: Sequence[SyncResult]) -> list[str]:
    """The relative paths of the files a scene apply touched, order-preserved.

    Derived from the ``file_path`` each :class:`SyncResult` carries, so no
    per-tool path knowledge is duplicated here.
    """
    seen: dict[str, None] = {}
    for result in results:
        if result.file_path is None:
            continue
        rel = _rel(project_root, result.file_path)
        seen.setdefault(rel, None)
    return list(seen)


def compute_hashes(project_root: Path, results: Sequence[SyncResult]) -> dict[str, str]:
    """Capture a normalised content hash for every file the apply touched."""
    return {
        rel: content_hash(project_root / rel) for rel in scene_managed_paths(project_root, results)
    }


def detect_drift(project_root: Path, state: SceneState) -> list[str]:
    """Return the scene-managed paths whose current content differs from apply.

    A semantically-neutral reformat (key reordering / whitespace) leaves the
    normalised hash unchanged, so it is *not* reported; a content change or a
    deleted file is.
    """
    drifted: list[str] = []
    for rel, expected in sorted(state.hashes.items()):
        if content_hash(project_root / rel) != expected:
            drifted.append(rel)
    return drifted


def content_hash(path: Path) -> str:
    """A normalised content hash for *path*.

    JSON and TOML files are parsed and re-serialised canonically so a reformat or
    key reordering hashes identically; a directory (or symlinked directory — a
    PROJECT re-point) is reduced to its link target and sorted entry names;
    anything else is hashed raw. A missing path hashes to the sentinel
    ``"missing"`` so its disappearance reads as drift.
    """
    if path.is_symlink() or path.is_dir():
        return _dir_signature(path)
    if not path.is_file():
        return "missing"
    try:
        data = path.read_bytes()
    except OSError:
        return "unreadable"
    suffix = path.suffix.lower()
    if suffix == ".json":
        normalized = _normalize_json(data)
        if normalized is not None:
            return _sha(normalized)
    elif suffix == ".toml":
        normalized = _normalize_toml(data)
        if normalized is not None:
            return _sha(normalized)
    return _sha(data)


def _dir_signature(path: Path) -> str:
    parts: list[str] = []
    if path.is_symlink():
        with contextlib.suppress(OSError):
            parts.append("->" + os.readlink(path))
    if path.is_dir():
        with contextlib.suppress(OSError):
            for child in sorted(path.iterdir(), key=lambda c: c.name):
                parts.append(child.name + ("/" if child.is_dir() else ""))
    elif not parts:
        # A broken symlink whose target could not be read.
        return "missing"
    return _sha("\n".join(parts).encode("utf-8"))


def _normalize_json(data: bytes) -> bytes | None:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _normalize_toml(data: bytes) -> bytes | None:
    try:
        obj = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _rel(project_root: Path, path: Path) -> str:
    rel = path
    if rel.is_absolute():
        with contextlib.suppress(ValueError):
            rel = rel.relative_to(project_root)
    return rel.as_posix()


__all__ = [
    "SCENE_STATE_PATH",
    "SCENE_STATE_VERSION",
    "LoadedSceneState",
    "SceneState",
    "SceneToolRecord",
    "clear_scene_state",
    "compute_hashes",
    "content_hash",
    "detect_drift",
    "load_scene_state",
    "now_iso",
    "save_scene_state",
    "scene_managed_paths",
]
