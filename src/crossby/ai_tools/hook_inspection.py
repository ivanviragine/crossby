"""Which of a folder's own hooks a tool would run during a session.

A read-only profile blocks a tool's file, edit and shell tools, but on
Cursor, Copilot, OpenCode and Antigravity CLI it leaves hooks enabled, and a
hook is a command that can change any file. Verified 2026-11-10 with sentinel
hooks written by ``crossby sync hooks``: Cursor's ``.cursor/hooks.json``
``preToolUse`` hook and Antigravity CLI's ``.agents/hooks.json`` hook both ran
during read-only turns (see
``docs/unattended-headless-verification.md#project-hooks``).

This module reads only those project sources, the ones crossby itself writes
(plus OpenCode's plugins, which load code). It never claims a folder is free
of hooks that run: a file it cannot read is reported as unreadable, and
user-level configuration is not inspected.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crossby.models.ai import AIToolID

# Project hook sources per tool, relative to the folder. A directory entry
# means every ``*.json`` file in it.
_PROJECT_HOOK_FILES: dict[AIToolID, tuple[str, ...]] = {
    AIToolID.CURSOR: (".cursor/hooks.json",),
    AIToolID.COPILOT: (".github/hooks/",),
    AIToolID.ANTIGRAVITY_CLI: (".agents/hooks.json",),
}
_OPENCODE_PLUGIN_DIRS = (".opencode/plugin", ".opencode/plugins")
_OPENCODE_CONFIGS = ("opencode.json", "opencode.jsonc")
_MAX_SOURCE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ProjectHooks:
    """A folder's own hook sources for one tool, as relative paths."""

    found: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()

    @property
    def clear(self) -> bool:
        """Nothing found and nothing unreadable."""
        return not self.found and not self.unreadable

    def describe(self) -> str:
        parts = [*self.found, *(f"{name} (unreadable)" for name in self.unreadable)]
        return ", ".join(parts)


def _has_entries(value: Any) -> bool:
    """Whether a hook document holds any entry (a non-empty list anywhere)."""
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, dict):
        return any(_has_entries(item) for item in value.values())
    return False


def _read_json(path: Path) -> Any:
    """Parse a hook source; anything but a readable regular file raises ``OSError``.

    The source is opened non-blocking with its type checked on the open
    descriptor, so a FIFO (or a link to one) can never hang the inspection,
    which runs before the session's own deadline is watched.
    """
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"{path} is not a regular file")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            data = handle.read(_MAX_SOURCE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) > _MAX_SOURCE_BYTES:
        raise OSError(f"{path} is larger than {_MAX_SOURCE_BYTES} bytes")
    return json.loads(data.decode("utf-8"))


def _check(folder: Path, relative: str, found: list[str], unreadable: list[str]) -> None:
    path = folder / relative
    if not path.exists() and not path.is_symlink():
        return
    try:
        if _has_entries(_read_json(path)):
            found.append(relative)
    except (OSError, ValueError):
        unreadable.append(relative)


def project_hook_sources(tool: AIToolID, folder: Path) -> ProjectHooks:
    """The hook sources in ``folder`` that ``tool`` would load, if any."""
    found: list[str] = []
    unreadable: list[str] = []
    if tool is AIToolID.OPENCODE:
        for relative in _OPENCODE_PLUGIN_DIRS:
            directory = folder / relative
            if directory.is_dir() and any(directory.iterdir()):
                found.append(relative)
        for relative in _OPENCODE_CONFIGS:
            path = folder / relative
            if not path.exists():
                continue
            try:
                config = _read_json(path)
            except (OSError, ValueError):
                # JSONC comments are not parsed: never treat such a file as clear.
                unreadable.append(relative)
                continue
            plugins = config.get("plugin") if isinstance(config, dict) else None
            if plugins:
                found.append(relative)
        return ProjectHooks(tuple(found), tuple(unreadable))
    for relative in _PROJECT_HOOK_FILES.get(tool, ()):
        if relative.endswith("/"):
            directory = folder / relative
            if directory.is_dir():
                for path in sorted(directory.glob("*.json")):
                    _check(folder, path.relative_to(folder).as_posix(), found, unreadable)
            continue
        _check(folder, relative, found, unreadable)
    return ProjectHooks(tuple(found), tuple(unreadable))


__all__ = ["ProjectHooks", "project_hook_sources"]
