"""Source ownership for scoped translation cleanup, separate from tree management."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from crossby.sync.safe_write import ProjectScope, safe_write_text

PROVENANCE_NAME = ".crossby-translated-skills.json"


def skill_fingerprint(directory: Path) -> str:
    """Hash the whole output tree without following symlinks.

    A human edit or added support file prevents deletion of the owned directory.
    """
    digest = hashlib.sha256()

    def visit(path: Path) -> None:
        relative = path.relative_to(directory).as_posix()
        value: list[str | int]
        if path.is_symlink():
            value = ["link", relative, os.readlink(path)]
        elif path.is_dir():
            value = ["dir", relative, path.stat().st_mode & 0o7777]
        else:
            value = [
                "file",
                relative,
                path.stat().st_mode & 0o7777,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
        digest.update(json.dumps(value).encode())
        if path.is_dir() and not path.is_symlink():
            for child in sorted(path.iterdir()):
                visit(child)

    visit(directory)
    return digest.hexdigest()


def read_provenance(target_dir: Path) -> dict[str, dict[str, str]]:
    """Never trust a symlink or malformed provenance to authorize a mutation."""
    path = target_dir / PROVENANCE_NAME
    if path.is_symlink():
        raise ValueError(f"{PROVENANCE_NAME} is a symlink")
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"invalid {PROVENANCE_NAME}")
    entries = raw.get("skills")
    if not isinstance(entries, dict):
        raise ValueError(f"invalid {PROVENANCE_NAME} entries")
    result: dict[str, dict[str, str]] = {}
    for name, entry in entries.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(entry, dict)
            or not isinstance(entry.get("source"), str)
            or not isinstance(entry.get("fingerprint"), str)
        ):
            raise ValueError(f"invalid {PROVENANCE_NAME} entry")
        result[name] = {"source": entry["source"], "fingerprint": entry["fingerprint"]}
    return result


def write_provenance(
    target_dir: Path,
    entries: dict[str, dict[str, str]],
    *,
    project_root: Path,
    dry_run: bool = False,
) -> bool:
    return safe_write_text(
        ProjectScope(project_root),
        target_dir / PROVENANCE_NAME,
        json.dumps({"version": 1, "skills": entries}, indent=2, sort_keys=True) + "\n",
        dry_run=dry_run,
        leaf_policy="replace",
    )
