"""Symlink management for config file linking."""

from __future__ import annotations

import os
from pathlib import Path

import structlog

logger = structlog.get_logger()


def create_symlink(
    source: Path,
    link: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Create a relative symlink from *link* pointing to *source*.

    Args:
        source: The real file/directory to point at.
        link: The symlink path to create.
        force: If True, remove an existing file/symlink at *link*.
        dry_run: If True, only report what would happen.

    Returns:
        True if a symlink was created (or would be in dry-run mode).
    """
    rel_target = os.path.relpath(source, link.parent)

    # Already a correct symlink — idempotent no-op
    if link.is_symlink():
        existing = os.readlink(link)
        if existing == rel_target:
            return False
        # Resolve both paths to catch equivalent but differently-expressed targets
        try:
            existing_target = (link.parent / existing).resolve(strict=False)
            resolved_source = source.resolve(strict=False)
            if existing_target == resolved_source:
                return False
        except (OSError, RuntimeError):
            pass
        if not force:
            return False
        if not dry_run:
            os.unlink(link)

    elif link.exists():
        if not force:
            return False
        if link.is_dir():
            # Refuse to delete real directories
            return False
        if not dry_run:
            os.unlink(link)

    # Guard against circular symlinks
    try:
        resolved_source = source.resolve(strict=False)
        resolved_link = link.resolve(strict=False)
        if resolved_source == resolved_link:
            logger.warning("symlink.circular", link=str(link), source=str(source))
            return False
    except (OSError, RuntimeError):
        return False

    if dry_run:
        return True

    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(rel_target, link)
    logger.info("symlink.created", link=str(link), target=rel_target)
    return True
