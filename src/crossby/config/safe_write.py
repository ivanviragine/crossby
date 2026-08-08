"""Safe, verified writes for ``.crossby.yml``.

The single write path both ``crossby init`` and ``crossby scene`` funnel
through: back up any existing file, write atomically, re-parse through the real
loader, and — if that parse fails — restore the backup byte-for-byte (or remove
the just-written file when there was none) before raising.

On a *failure* the backup is always cleaned up (the original is back in place).
On *success* the caller decides via ``keep_backup``: ``crossby init --force`` is
a destructive full-file overwrite, so it keeps the ``.bak`` as a recovery net;
the scene commands are surgical single-entry splices and leave nothing behind.

This is the sequence originally inlined in :mod:`crossby.cli.init`
(``init.py:82-106``); lifting it into one helper keeps ``init`` and the scene
authoring commands from drifting, since both must never leave the user with a
broken — or vanished — config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from crossby.models.config import CrossbyConfig


class ConfigWriteError(Exception):
    """Raised when a checked config write produced an unparseable file.

    ``original`` is the parse (or write) exception that triggered the rollback;
    ``restored`` is ``True`` when a previous file was put back byte-for-byte and
    ``False`` when the freshly-written file was simply removed (there was no
    prior file). Callers use ``restored`` to word their error message.
    """

    def __init__(self, original: Exception, *, restored: bool) -> None:
        self.original: Exception = original
        self.restored: bool = restored
        super().__init__(str(original))


def write_config_checked(
    target: Path,
    rendered: str,
    *,
    validate: Callable[[CrossbyConfig], object] | None = None,
    keep_backup: bool = False,
) -> Path | None:
    """Write *rendered* to *target*, verifying it round-trips through the loader.

    Backs up an existing *target* first. Writes atomically, then re-parses via
    :func:`crossby.config.loader.parse_config_file` and — if *validate* is given
    — runs it on the parsed config. On any failure (parse error or a *validate*
    that raises) the original file is restored byte-for-byte (or the new file
    removed when *target* did not previously exist) and :class:`ConfigWriteError`
    is raised — the backup is always removed on this path.

    *validate* is the hook for checks the structural parse cannot make on its
    own — e.g. resolving a scene's ``extends`` chain, whose cycle and
    undefined-parent errors surface only in
    :meth:`crossby.models.config.CrossbyConfig.get_scene`.

    When *target* is a symlink, the write goes through to the resolved real
    file so the link itself survives — ``os.replace`` inside
    :func:`~crossby.config.json_utils.atomic_write_text` would otherwise
    replace the symlink with a regular file. Backup, atomic write, and
    rollback all operate on the resolved path; only the re-parse reads through
    the original *target* (the real read path). A broken symlink (points
    nowhere yet) is resolved to its intended, not-yet-existing target: no
    backup is taken, and on failure the newly created file is removed rather
    than restored, leaving the link exactly as broken as it started.

    A symlink resolving outside the project root is followed intentionally,
    not refused — a config split out into a dotfiles repo is a legitimate,
    supported layout, and this write path has never containment-checked
    (``atomic_write_text`` is called here without ``within=``); scene
    artefact writes are the ones that stay containment-checked.

    Returns:
        On success, the retained backup path when *keep_backup* is set and a
        prior file existed, else ``None``. When *keep_backup* is false the
        backup is removed on success. Note the backup sits beside the
        *resolved* target, so for a symlinked config it can land outside the
        project root.

    Raises:
        ConfigWriteError: the rendered text did not parse or failed *validate*;
            the filesystem is left exactly as it was before the call.
    """
    from crossby.config.json_utils import atomic_write_text
    from crossby.config.loader import parse_config_file
    from crossby.sync.file_utils import backup_path

    write_target = target.resolve() if target.is_symlink() else target

    backup: Path | None = None
    if write_target.exists():
        backup = backup_path(write_target)
        backup.write_bytes(write_target.read_bytes())

    try:
        atomic_write_text(write_target, rendered)
        config = parse_config_file(target)
        if validate is not None:
            validate(config)
    except Exception as exc:
        if backup is not None:
            write_target.write_bytes(backup.read_bytes())
            backup.unlink(missing_ok=True)
        else:
            write_target.unlink(missing_ok=True)
        raise ConfigWriteError(exc, restored=backup is not None) from exc

    if backup is not None and keep_backup:
        return backup
    if backup is not None:
        backup.unlink(missing_ok=True)
    return None
