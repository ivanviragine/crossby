"""Install the bundled crossby-sync skill into a project.

Crossby ships a Skill-format runbook (``src/crossby/data/skill/``) so an
LLM agent inside Claude / Codex / Cursor / etc. can drive ``crossby
sync`` end-to-end. This module copies the bundle into a project so the
agent finds it under the tool's expected skills path.

Two install layouts are supported:

- ``per-tool``: write ``<tool-skills-dir>/crossby-sync/`` for every
  installed tool. Use this when there is no canonical Crossby skills
  source yet (i.e. the user hasn't run sync, or hasn't picked a
  source). It's the safest first-time install.
- ``canonical``: write ``<source-skills-dir>/crossby-sync/``. Use this
  when the project already has a skills source — the regular skills
  sync will then propagate the bundle to every other tool on the next
  ``crossby sync`` run.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from crossby.ai_tools.base import AbstractAITool
from crossby.config.skills import SKILLS_DIR
from crossby.models.ai import AIToolID

SKILL_NAME = "crossby-sync"


@dataclass(frozen=True)
class SkillInstallResult:
    """Outcome of writing the bundle into one target skill directory."""

    target_dir: Path
    action: str  # "created" | "updated" | "skipped"


def bundled_skill_root() -> Path:
    """Return the absolute path of the bundled SKILL.md root.

    ``importlib.resources`` is used so the lookup works whether crossby
    is installed from source, a wheel, or a PyInstaller bundle.
    """
    files_root = resources.files("crossby.data.skill")
    # ``files`` returns a Traversable; for our use we always have a real
    # filesystem path under ``src/`` because the bundle ships as plain
    # files. ``with as_file`` would handle the zip case too but we don't
    # ship a zipped wheel of these files today.
    return Path(str(files_root))


_PYTHON_NOISE = (".pyc", ".pyo")


def _walk_bundle(root: Path) -> Iterable[Path]:
    """Yield every payload file inside the bundle, sorted.

    Skips Python sub-package boilerplate (``__init__.py``, ``__pycache__``)
    that exists only so :mod:`importlib.resources` can find the bundle.
    """
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "__init__.py":
            continue
        if path.suffix in _PYTHON_NOISE:
            continue
        if "__pycache__" in path.parts:
            continue
        out.append(path)
    return out


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def install_bundle(target_dir: Path) -> SkillInstallResult:
    """Copy the bundled SKILL.md tree into ``target_dir/crossby-sync/``.

    Idempotent: when the target already matches the bundle byte-for-byte
    the result is ``"skipped"``. When the target exists but differs,
    it's overwritten and the result is ``"updated"``. Otherwise
    ``"created"``.
    """
    bundle_root = bundled_skill_root()
    skill_dir = target_dir / SKILL_NAME
    bundle_files = list(_walk_bundle(bundle_root))

    existing_state = "missing"
    if skill_dir.is_dir():
        existing_state = "matches"
        for source_file in bundle_files:
            relative = source_file.relative_to(bundle_root)
            target_file = skill_dir / relative
            if not target_file.is_file():
                existing_state = "differs"
                break
            if _hash(target_file.read_bytes()) != _hash(source_file.read_bytes()):
                existing_state = "differs"
                break

    if existing_state == "matches":
        return SkillInstallResult(target_dir=skill_dir, action="skipped")

    skill_dir.mkdir(parents=True, exist_ok=True)
    for source_file in bundle_files:
        relative = source_file.relative_to(bundle_root)
        target_file = skill_dir / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

    action = "created" if existing_state == "missing" else "updated"
    return SkillInstallResult(target_dir=skill_dir, action=action)


def install_for_tools(project_root: Path, tools: Iterable[AIToolID]) -> list[SkillInstallResult]:
    """Install the bundle under every tool's skills directory."""
    results: list[SkillInstallResult] = []
    for tool in tools:
        rel = SKILLS_DIR.get(tool)
        if rel is None:
            continue
        results.append(install_bundle(project_root / rel))
    return results


def install_canonical(project_root: Path, canonical_skills_source: Path) -> SkillInstallResult:
    """Install the bundle under a single canonical skills directory."""
    return install_bundle(project_root / canonical_skills_source)


def install_to_installed_tools(project_root: Path) -> list[SkillInstallResult]:
    """Convenience: install under every detected-installed tool's skills dir."""
    installed = AbstractAITool.detect_installed()
    return install_for_tools(project_root, installed)


__all__ = [
    "SKILL_NAME",
    "SkillInstallResult",
    "bundled_skill_root",
    "install_bundle",
    "install_canonical",
    "install_for_tools",
    "install_to_installed_tools",
]
