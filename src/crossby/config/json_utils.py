"""File read/write utilities shared across config and sync layers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class PathContainmentError(Exception):
    """A write/delete target escapes the project root it must stay within.

    Raised by :func:`assert_within` when a path either passes through a
    symlinked component below the containment root, or resolves outside it.
    """


def assert_within(within: Path, path: Path) -> None:
    """Assert *path* stays inside *within*, refusing symlinked escape routes.

    Two independent checks, either of which trips :class:`PathContainmentError`:

    1. **Reject symlinked components below** *within*. Each path component
       strictly beneath *within* (e.g. ``.crossby``, ``scene``, ``<name>``,
       ``launch``, the leaf) is inspected; if an existing one ``is_symlink()``
       the write is refused, naming the offending component. Components *at or
       above* *within* are never inspected, so a project living under a
       symlinked ancestor (macOS ``/tmp`` → ``/private/tmp``, a pytest
       ``tmp_path`` under a symlinked root) is not a false positive.
    2. **Resolved-containment assertion** (defense in depth): ``path.resolve()``
       must be relative to ``within.resolve()``. Resolving *both* sides cancels
       a symlinked ancestor of *within* itself, so the legitimate project above
       still passes while a genuine escape does not.

    **Guarantee scope:** this defends against symlinked components that exist at
    check time. A concurrent local attacker racing a symlink insertion between
    this check and the subsequent ``mkdir``/``mkstemp``/``rmtree`` (a TOCTOU
    window) is explicitly out of scope — crossby does not adopt
    ``openat``/``O_NOFOLLOW``/``mkdirat`` machinery here.
    """
    within_abs = Path(os.path.abspath(within))
    path_abs = Path(os.path.abspath(path))

    # 1. Reject symlinked components strictly below `within`.
    try:
        relative = path_abs.relative_to(within_abs)
    except ValueError:
        relative = None
    if relative is not None:
        current = within_abs
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise PathContainmentError(
                    f"refusing to write through symlinked path component {current} "
                    f"(would escape {within_abs})"
                )

    # 2. Resolved-containment assertion (defense in depth). Resolving both sides
    #    cancels a symlinked ancestor of `within` itself.
    resolved_within = within_abs.resolve()
    resolved_path = path_abs.resolve()
    if not resolved_path.is_relative_to(resolved_within):
        raise PathContainmentError(
            f"{path} resolves outside {within} (resolved to {resolved_path})"
        )


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


def atomic_write_text(path: Path, text: str, *, within: Path | None = None) -> None:
    """Write *text* to *path* via a **unique** temp file plus atomic rename.

    A crash or interrupt part-way through leaves the original file intact
    rather than truncated — which matters most for the config files crossby
    merges into rather than owns (``.codex/config.toml``, tool settings).

    The temp file is created with :func:`tempfile.mkstemp` in the target's own
    directory (so the final ``os.replace`` stays on one filesystem and is
    atomic), and its name is unique per call — two concurrent writers to the
    same path therefore never clobber or unlink each other's temp file. The
    content race is still last-writer-wins, but neither writer ever observes a
    torn file.

    When *within* is set, :func:`assert_within` runs **before** the
    ``mkdir`` — which would otherwise follow a symlinked parent component — so a
    scene artefact can never escape the project root. Callers that write
    outside the project by design (``write_codex_profile`` under ``$CODEX_HOME``,
    the merged sync-config writers) pass no *within* and stay unaffected.
    """
    import tempfile

    if within is not None:
        assert_within(within, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Atomic write of a JSON dict with consistent formatting.

    Uses 2-space indent, sorted keys, and a tmp+replace pattern to avoid
    partial writes on crash.
    """
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
