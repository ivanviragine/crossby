"""Codex project-trust detection.

Codex only loads a project's ``.codex/config.toml`` when the project is
*trusted*; in an untrusted directory a DECLARE toggle written there silently
no-ops. Codex records trust in the user-global ``~/.codex/config.toml`` as::

    [projects."/abs/path/to/project"]
    trust_level = "trusted"

No trust introspection existed in crossby before scenes (``supports_trusted_dirs``
on :class:`AIToolCapabilities` is a launch-flag marker, not a way to read state),
so this reads that file directly. Best-effort: a missing/malformed file or an
absent entry reads as "not trusted", and the caller reports the toggle as
possibly-ineffective rather than failing.
"""

from __future__ import annotations

import contextlib
import tomllib
from pathlib import Path

import structlog

logger = structlog.get_logger()

_CODEX_CONFIG_REL = Path(".codex") / "config.toml"
_TRUSTED = "trusted"


def _codex_config_path(home: Path | None) -> Path:
    return (home or Path.home()) / _CODEX_CONFIG_REL


def codex_trusts_project(project_root: Path, *, home: Path | None = None) -> bool:
    """Return True when Codex records *project_root* as ``trust_level = "trusted"``.

    *home* overrides ``~`` (for tests). The project path is resolved to an
    absolute, symlink-free form before matching, because Codex stores the
    canonical absolute path as the table key. Any read/parse error, or the
    absence of a matching ``[projects."<path>"]`` entry, returns ``False``.
    """
    config_path = _codex_config_path(home)
    if not config_path.is_file():
        return False
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.debug("scene.codex_trust_unreadable", path=str(config_path), error=str(exc))
        return False

    projects = data.get("projects")
    if not isinstance(projects, dict):
        return False

    wanted = _candidate_keys(project_root)
    for key, entry in projects.items():
        if key in wanted and isinstance(entry, dict) and entry.get("trust_level") == _TRUSTED:
            return True
    return False


def _candidate_keys(project_root: Path) -> set[str]:
    """The path spellings Codex might have stored for *project_root*.

    Matches both the resolved (symlink-free) path and the plain absolute path,
    since a worktree may be reached through either.
    """
    keys = {str(Path(project_root).absolute())}
    with contextlib.suppress(OSError):
        keys.add(str(project_root.resolve()))
    return keys
