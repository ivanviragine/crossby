"""Session-scoped scene launch — render artefacts and carry per-launch context.

``crossby scene use`` persists a scene into each tool's tracked config files.
This module is the *session-scoped* counterpart driving ``crossby launch
--scene``: it renders a scene into throwaway artefacts under the gitignored
``.crossby/scene/<name>/launch/`` tree and hands each adapter the flags/env that
point its CLI at those artefacts for one session — touching nothing tracked.

Two types cross the adapter boundary:

- :class:`SceneLaunchArgs` — what one adapter's ``scene_launch_args`` returns:
  extra argv plus extra environment for the child process.
- :class:`SceneLaunchContext` — everything an adapter needs to render: the
  scene name (for the artefact path), the tool-agnostic :class:`ResolvedScene`
  selection, the project root, and the :class:`SyncData` inventory (so an
  adapter can read the *full* MCP server definitions, not just their names).
  It is a superset of "the resolved scene" because :class:`ResolvedScene`
  records neither the scene's name nor the concrete MCP configs an artefact
  file needs.

**Codex is the deliberate artefact-location exception.** ``codex --profile
<name>`` only reads ``$CODEX_HOME/<name>.config.toml`` (typically ``~/.codex``,
shared across projects), so its profile can't live under ``.crossby/scene/``.
Those files are namespaced by a short project-root hash to avoid two repos'
same-named scenes colliding, and carry a generated-by header so pruning can tell
crossby's own files from a hand-written profile that happens to match the naming
pattern — the header is the ownership test, never the filename alone.

Rendered artefacts are written temp-file-then-atomic-rename (via
:func:`crossby.config.json_utils.atomic_write_text`), so a crash mid-render
leaves no half-written file. Concurrent launches of the same scene race to the
same paths; the atomic rename removes the torn-read failure mode and
last-writer-wins on content is accepted rather than locked.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from crossby.config.json_utils import PathContainmentError, assert_within, atomic_write_text
from crossby.sync.file_utils import MANAGED_MARKER_NAME, write_managed_marker

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crossby.models.config import MCPServerConfig
    from crossby.services.scene_resolution import ResolvedScene
    from crossby.sync.base import SyncData

# Kept in lockstep with ``crossby.scenes.projection.SCENE_PROJECTION_ROOT`` so
# the persistent projection tree and the launch artefacts share one root
# (``.crossby/scene/``).
SCENE_ROOT = Path(".crossby") / "scene"

# Scene names are interpolated into filesystem paths (``.crossby/scene/<name>/``
# and ``$CODEX_HOME/crossby-<slug>-<name>.config.toml``), so they are validated
# against this pattern first: it forbids separators and leading punctuation, and
# ``..``/``active`` are rejected separately.
_SAFE_SCENE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# First line of every crossby-generated Codex profile. Pruning deletes a profile
# only when its text starts with this marker — the ownership test that keeps a
# hand-written profile matching the same filename pattern safe.
CODEX_PROFILE_MARKER = "# crossby:scene-launch"

# Minimum Codex CLI on which ``--profile <name>`` layers
# ``$CODEX_HOME/<name>.config.toml`` over the base config. The legacy
# ``[profiles.<name>]`` in-config tables were removed in 0.134.0, so on an older
# build the profile file is never read — the launch path falls back to
# persistent activation instead.
CODEX_PROFILE_MIN = (0, 134, 0)


@dataclass(frozen=True)
class SceneLaunchArgs:
    """Extra argv and environment one adapter contributes for a scene launch.

    ``args`` are appended to the tool's normal launch command; ``env`` are
    additions the launcher merges over ``os.environ`` for the child. Both empty
    is the "no session-scoped lever for this scene" default.
    """

    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneLaunchContext:
    """Everything an adapter needs to render a scene for one launch.

    ``allow_tools`` carries the resolved profile's approval-layer allow entries
    (Copilot ``--allow-tool``); an adapter filters out any that name a
    scene-excluded tool before re-emitting them, so a profile can't re-allow
    what the scene excluded.
    """

    name: str
    resolved: ResolvedScene
    project_root: Path
    sync_data: SyncData
    allow_tools: tuple[str, ...] = ()

    # -- selections -------------------------------------------------------
    def selected(self, concern: str) -> set[str]:
        """The scene-selected item names for *concern* (union across groups)."""
        return set(self.resolved.names(concern))

    def mcp_universe(self) -> dict[str, MCPServerConfig]:
        """Every MCP server discovered for the project, keyed by name."""
        return dict(self.sync_data.mcp_servers)

    def selected_mcp(self) -> dict[str, MCPServerConfig]:
        """The discovered MCP servers the scene keeps enabled, keyed by name."""
        keep = self.selected("mcp")
        return {name: cfg for name, cfg in self.sync_data.mcp_servers.items() if name in keep}

    def deselected_mcp(self) -> set[str]:
        """Discovered MCP servers the scene drops (universe minus selected)."""
        return set(self.sync_data.mcp_servers) - self.selected("mcp")

    def narrows_mcp(self) -> bool:
        """True when the scene drops at least one discovered MCP server.

        When it narrows nothing there is nothing to scope for the session, so an
        adapter emits no MCP flags (mirroring the persistent engine, whose empty
        disable set is a no-op) rather than restricting the tool to exactly the
        project servers and hiding the user's global ones.
        """
        return bool(self.deselected_mcp())

    # -- artefact paths ---------------------------------------------------
    @property
    def launch_dir(self) -> Path:
        """The gitignored ``.crossby/scene/<name>/launch/`` artefact directory."""
        return self.project_root / SCENE_ROOT / self.name / "launch"

    def artifact(self, filename: str) -> Path:
        """Absolute path of a named artefact inside :attr:`launch_dir`."""
        return self.launch_dir / filename

    def write_artifact(self, filename: str, content: str) -> Path:
        """Render *content* to a named artefact, temp-file-then-atomic-rename.

        Also stamps the launch dir with the crossby-managed marker (so pruning
        can tell its own trees from a hand-made directory) and adds
        ``.crossby/scene/`` to ``.git/info/exclude`` — never ``.gitignore`` — so
        a session-scoped launch keeps its "writes nothing tracked" promise.

        Both write targets — the artefact leaf *and* the managed marker — are
        containment-checked up front (via :func:`assert_within`), so a symlinked
        component anywhere in ``.crossby/scene/<name>/launch`` or a pre-existing
        symlinked marker leaf refuses the whole write rather than leaving a
        partial artefact behind. Because an adapter may write several artefacts
        across successive calls (Claude renders ``mcp.json`` then
        ``settings.json``), any pre-existing symlink already sitting in the
        launch dir — a planted sibling leaf a *later* call would target — is also
        rejected before the *first* artefact lands, so an aborted launch never
        leaves an earlier clean artefact behind. crossby never creates symlinks
        here, so any symlink present is foreign. Raises
        :class:`PathContainmentError` on a violation; ``within=`` is retained on
        the ``atomic_write_text`` call as defense in depth for any direct caller.
        """
        path = self.artifact(filename)
        marker = self.launch_dir / MANAGED_MARKER_NAME
        assert_within(self.project_root, path)
        assert_within(self.project_root, marker)
        if self.launch_dir.is_dir() and not self.launch_dir.is_symlink():
            for entry in self.launch_dir.iterdir():
                if entry.is_symlink():
                    raise PathContainmentError(
                        f"refusing to write scene artefacts: {entry} is a symlink "
                        "(crossby never creates symlinks under the launch dir)"
                    )
        atomic_write_text(path, content, within=self.project_root)
        write_managed_marker(self.launch_dir)
        ensure_launch_excluded(self.project_root)
        return path


# ---------------------------------------------------------------------------
# Scene-name validation and shared rendering helpers
# ---------------------------------------------------------------------------


def validate_scene_name(name: str) -> None:
    """Reject a scene name that could escape the artefact tree or reserved dirs.

    Scene names are interpolated into filesystem paths, so a name with a path
    separator, ``..``, a leading dot, or the reserved ``active`` (the persistent
    projection's own subdir) is refused before any path is built — raising
    :class:`ValueError`. Names are project-local config keys, but a typo like
    ``../../etc`` must never reach the filesystem.
    """
    if (
        not name
        or name in (".", "..", "active")
        or "/" in name
        or "\\" in name
        or ".." in name
        or _SAFE_SCENE_NAME.match(name) is None
    ):
        raise ValueError(
            f"unsafe scene name {name!r}: use letters, digits, '.', '_' or '-' "
            "(no path separators, no '..', not the reserved 'active')"
        )


def ensure_launch_excluded(project_root: Path) -> None:
    """Add ``.crossby/scene/`` to ``.git/info/exclude`` — never ``.gitignore``.

    ``.gitignore`` is a tracked file; a session-scoped launch must not mutate it.
    ``.git/info/exclude`` is the per-clone, untracked equivalent, so the artefacts
    stay out of ``git status`` without touching anything tracked. Best-effort and
    idempotent: resolved via ``git rev-parse --git-path`` (so it works from a
    worktree), and a no-op outside a git repo or on any I/O error.
    """
    import subprocess

    entry = SCENE_ROOT.as_posix() + "/"
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--git-path", "info/exclude"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if proc.returncode != 0 or not proc.stdout.strip():
        return
    exclude_path = Path(proc.stdout.strip())
    if not exclude_path.is_absolute():
        exclude_path = project_root / exclude_path
    try:
        existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
        if entry in existing.splitlines():
            return
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        with exclude_path.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(entry + "\n")
    except OSError:
        return


def project_slug(project_root: Path) -> str:
    """A short, stable hash of the project root, for global artefact namespacing.

    Used to keep two repositories that define a scene of the same name from
    colliding in one global ``$CODEX_HOME`` file. Deterministic — no clock or
    randomness (both are unavailable in workflow scripts and would break
    idempotence anyway).
    """
    digest = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()
    return digest[:10]


def mcp_json_config(servers: dict[str, MCPServerConfig]) -> str:
    """Render *servers* as a Claude/Cursor ``{"mcpServers": {...}}`` JSON file.

    Reuses the sync layer's per-transport entry shaping so a launch-time config
    is byte-compatible with what ``crossby sync`` would write.
    """
    import json

    from crossby.sync.mcp import _to_json_entry

    body = {"mcpServers": {name: _to_json_entry(cfg) for name, cfg in sorted(servers.items())}}
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# Codex profile — the $CODEX_HOME exception
# ---------------------------------------------------------------------------


def codex_home() -> Path:
    """Resolve ``$CODEX_HOME`` (default ``~/.codex``), expanded."""
    import os

    raw = os.environ.get("CODEX_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".codex"


def codex_profile_name(project_root: Path, scene_name: str) -> str:
    """The ``--profile`` name for *scene_name*, namespaced by project.

    ``crossby-<slug>-<scene>`` — the ``crossby-`` prefix plus the header inside
    the file are what let pruning recognise crossby's own profiles.
    """
    return f"crossby-{project_slug(project_root)}-{scene_name}"


def codex_profile_path(project_root: Path, scene_name: str) -> Path:
    """Absolute ``$CODEX_HOME/<profile-name>.config.toml`` path."""
    return codex_home() / f"{codex_profile_name(project_root, scene_name)}.config.toml"


def render_codex_profile(scene_name: str, disabled_servers: set[str]) -> str:
    """Render a Codex profile config that disables *disabled_servers*.

    The profile layers over the base ``config.toml``: setting
    ``[mcp_servers.<id>] enabled = false`` per deselected server narrows the MCP
    set for the session without touching the base file. The leading marker line
    is the ownership test pruning relies on.
    """
    import tomli_w

    lines = [
        CODEX_PROFILE_MARKER,
        f"# Generated for scene {scene_name!r} — regenerated on each launch; do not edit.",
        "",
    ]
    if disabled_servers:
        table = {"mcp_servers": {name: {"enabled": False} for name in sorted(disabled_servers)}}
        # tomli_w.dumps already ends in a newline, so joining keeps a single one.
        lines.append(tomli_w.dumps(table))
    return "\n".join(lines)


def is_crossby_codex_profile(path: Path) -> bool:
    """True when *path* is a crossby-generated Codex profile (header present).

    The ownership test for pruning: a file whose first line is not
    :data:`CODEX_PROFILE_MARKER` — including a hand-written profile that happens
    to match the ``crossby-*.config.toml`` naming pattern — is never deleted.
    An unreadable or non-UTF-8 file returns ``False`` (fail closed, never
    deleted) so a bad profile can never abort :func:`prune_stale_artifacts`;
    ``UnicodeDecodeError`` is a :class:`ValueError`, not an :class:`OSError`.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
    except (OSError, UnicodeDecodeError):
        return False
    return first.strip() == CODEX_PROFILE_MARKER


def write_codex_profile(project_root: Path, scene_name: str, disabled_servers: set[str]) -> Path:
    """Render the Codex profile for *scene_name* and return its path.

    Refuses to overwrite a file at the target path that lacks crossby's header —
    that would be a hand-written profile colliding on the namespaced name, and
    silently clobbering it is exactly what the ownership test exists to prevent.
    """
    path = codex_profile_path(project_root, scene_name)
    if path.exists() and not is_crossby_codex_profile(path):
        raise FileExistsError(
            f"{path} exists but is not a crossby-generated profile (no header); "
            "refusing to overwrite a hand-written Codex profile."
        )
    atomic_write_text(path, render_codex_profile(scene_name, disabled_servers))
    return path


# ---------------------------------------------------------------------------
# Pruning — stale launch trees and stale Codex profiles
# ---------------------------------------------------------------------------


def prune_stale_artifacts(project_root: Path, defined_scenes: set[str]) -> list[str]:
    """Remove launch artefacts for scenes no longer defined; return what was pruned.

    Two accumulation sites are cleaned:

    - Local ``.crossby/scene/<name>/launch/`` trees for a ``<name>`` not in
      *defined_scenes* (a renamed or deleted scene). The persistent projection's
      ``active/`` tree and any ``<name>/`` dir that is still a defined scene are
      left alone.
    - Global ``$CODEX_HOME/crossby-<slug>-*.config.toml`` files for this project
      whose scene is no longer defined — and **only** files that pass the
      ownership header test. A file matching the naming pattern but lacking the
      header is never deleted.

    Best-effort: any single unreadable/undeletable path is skipped, never raised.
    Both top-level enumerations are independently guarded, so an I/O error
    listing one accumulation site never aborts the launch nor blocks the other
    cleanup path. A ``launch`` reached through a symlinked component is skipped
    (never followed into a deletion outside the project) via
    :func:`assert_within`, which covers pre-existing symlinks only (a concurrent
    TOCTOU symlink insertion is out of scope).
    """
    import shutil

    pruned: list[str] = []

    scene_root = project_root / SCENE_ROOT
    try:
        children = list(scene_root.iterdir()) if scene_root.is_dir() else []
    except OSError:
        children = []
    for child in children:
        if not child.is_dir() or child.name == "active" or child.name in defined_scenes:
            continue
        launch = child / "launch"
        # Only remove a tree crossby stamped — never a hand-made directory
        # that happens to sit under .crossby/scene/ (the ownership test,
        # mirroring the Codex-profile header check).
        if not launch.is_dir() or not (launch / MANAGED_MARKER_NAME).is_file():
            continue
        try:
            # Refuse to rmtree through a symlinked .crossby/scene/<name>/launch
            # component — that would delete content outside the project.
            assert_within(project_root, launch)
            shutil.rmtree(launch)
            if not any(child.iterdir()):
                child.rmdir()
            pruned.append((child.relative_to(project_root) / "launch").as_posix())
        except (OSError, PathContainmentError):
            continue

    prefix = f"crossby-{project_slug(project_root)}-"
    home = codex_home()
    try:
        profiles = sorted(home.glob(f"{prefix}*.config.toml")) if home.is_dir() else []
    except OSError:
        profiles = []
    for profile in profiles:
        scene = profile.name[len(prefix) : -len(".config.toml")]
        if scene in defined_scenes or not is_crossby_codex_profile(profile):
            continue
        try:
            profile.unlink()
            pruned.append(str(profile))
        except OSError:
            continue

    return pruned
