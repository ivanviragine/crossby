"""Owned disposable checkouts. Ownership is a deletion precondition, never a name guess."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from crossby.config.json_utils import PathContainmentError, assert_within
from crossby.config.skills import _SCAN_ORDER, SKILLS_DIR
from crossby.models.ai import AIToolID
from crossby.models.config import SceneConfig

_MARKER = "crossby-scene-owner.json"
_GIT_ENV = {
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
}


class WorktreeError(RuntimeError):
    """An isolation request or owned cleanup cannot safely complete."""


def _git(root: Path, *args: str) -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _GIT_ENV and not k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorktreeError(f"Git isolation operation failed: {exc}") from exc
    if result.returncode:
        raise WorktreeError(result.stderr.strip() or "Git isolation operation failed")
    return result.stdout.strip()


def isolation_requested(scene: SceneConfig | None, flag: str | None, keep: bool) -> bool:
    """Reject contradictory modes before allocating anything."""
    if flag not in (None, "worktree", "none"):
        raise WorktreeError("--isolate must be 'worktree' or 'none'.")
    configured = scene.isolation if scene is not None else None
    if flag is not None and configured is not None and flag != configured:
        raise WorktreeError("--isolate contradicts the scene's isolation setting.")
    isolated = (flag or configured) == "worktree"
    if (isolated or keep) and scene is None:
        raise WorktreeError("Worktree isolation requires --scene.")
    if keep and not isolated:
        raise WorktreeError("--keep-worktree requires worktree isolation.")
    if scene is not None and scene.inject and scene.inject.skills and not isolated:
        raise WorktreeError("Scene skill injection requires worktree isolation.")
    return isolated


@dataclass(frozen=True)
class WorktreeRequest:
    repository: Path
    common: Path
    commit: str
    cwd_relative: Path
    scene_relative: Path
    skills_relative: Path | None
    bundles: dict[str, dict[Path, bytes]]
    canonical_skills_relative: Path | None = None

    @classmethod
    def prepare(
        cls, cwd: Path, scene_root: Path, scene: SceneConfig, tool: AIToolID
    ) -> WorktreeRequest:
        repository = Path(_git(cwd, "rev-parse", "--show-toplevel")).resolve()
        common = Path(_git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        commit = _git(cwd, "rev-parse", "--verify", "HEAD^{commit}")
        hook_path = Path(_git(repository, "rev-parse", "--git-path", "hooks/post-checkout"))
        if not hook_path.is_absolute():
            hook_path = repository / hook_path
        if hook_path.is_file() and os.access(hook_path, os.X_OK):
            raise WorktreeError("Isolated v1 launches require no executable post-checkout hook.")
        try:
            cwd_relative = cwd.resolve().relative_to(repository)
            scene_relative = scene_root.resolve().relative_to(repository)
        except ValueError as exc:
            raise WorktreeError("Scene configuration must be inside the Git checkout.") from exc
        for relative in (cwd_relative, scene_relative):
            if (
                relative != Path(".")
                and _git(repository, "cat-file", "-t", f"{commit}:{relative.as_posix()}") != "tree"
            ):
                raise WorktreeError("The launch directory must exist in committed HEAD.")
        # Refuse external committed links and submodules before checkout allocation.
        entries = _git(repository, "ls-tree", "-rz", commit).split("\0")
        for entry in filter(None, entries):
            header, name = entry.split("\t", 1)
            mode = header.split()[0]
            if mode == "160000":
                raise WorktreeError("Isolated v1 launches do not initialize Git submodules.")
            if mode == "120000":
                link = _git(repository, "show", f"{commit}:{name}")
                destination = Path(os.path.normpath(str(Path(name).parent / link)))
                if Path(link).is_absolute() or destination.parts[:1] == ("..",):
                    raise WorktreeError(f"Committed symlink escapes the isolated checkout: {name}")
        sources = scene.inject.skills if scene.inject is not None else {}
        target = SKILLS_DIR.get(tool)
        if sources and target is None:
            raise WorktreeError(f"{tool} has no supported skills directory.")
        bundles: dict[str, dict[Path, bytes]] = {}
        for name, source in sources.items():
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
                raise WorktreeError(f"Unsafe injected skill name: {name!r}")
            selector = scene.skills
            if selector is not None and (
                (
                    selector.include is not None
                    and not any(fnmatch.fnmatchcase(name, p) for p in selector.include)
                )
                or any(fnmatch.fnmatchcase(name, p) for p in selector.exclude)
            ):
                raise WorktreeError(f"Injected skill {name!r} is excluded by the scene.")
            if source.startswith("catalog:"):
                if source != "catalog:crossby-sync":
                    raise WorktreeError(f"Unknown skill catalog source: {source!r}")
                from crossby.sync.skill_install import bundled_skill_root

                bundle = bundled_skill_root()
            else:
                bundle = Path(source).expanduser()
                if not bundle.is_absolute():
                    bundle = scene_root / bundle
            if not source.strip() or bundle.is_symlink() or not bundle.is_dir():
                raise WorktreeError(f"Skill source must be a real directory: {source!r}")
            payload: dict[Path, bytes] = {}
            for path in sorted(bundle.rglob("*")):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise WorktreeError(f"Skill source contains a symlink or special file: {path}")
                relative = path.relative_to(bundle)
                if ".git" in relative.parts:
                    raise WorktreeError("Skill bundles must not contain Git metadata.")
                if path.is_file() and "__pycache__" not in relative.parts:
                    payload[relative] = path.read_bytes()
            if Path("SKILL.md") not in payload:
                raise WorktreeError(f"Skill source has no regular SKILL.md: {source!r}")
            bundles[name] = payload
        skills_relative = scene_relative / target if target is not None else None
        canonical = None
        if bundles:
            for skill_directory in dict.fromkeys(SKILLS_DIR[tool] for tool in _SCAN_ORDER):
                candidate = scene_relative / skill_directory
                found = _git(repository, "ls-tree", commit, "--", candidate.as_posix())
                if found.startswith("040000 "):
                    canonical = candidate
                    break
        if bundles and skills_relative is not None:
            # Check all destination components against HEAD, not dirty source state.
            for directory in (skills_relative, canonical):
                if directory is None:
                    continue
                for name in bundles:
                    destination = directory / name
                    parts = destination.parts
                    for index in range(1, len(parts) + 1):
                        found = _git(
                            repository, "ls-tree", commit, "--", Path(*parts[:index]).as_posix()
                        )
                        if found and (index == len(parts) or not found.startswith("040000 ")):
                            raise WorktreeError(
                                f"Injected skill destination collides with HEAD: {destination}"
                            )
        return cls(
            repository,
            common.resolve(),
            commit,
            cwd_relative,
            scene_relative,
            skills_relative,
            bundles,
            canonical,
        )

    @contextmanager
    def create(self, *, keep: bool = False) -> Iterator[OwnedWorktree]:
        token = uuid.uuid4().hex
        container = Path(tempfile.mkdtemp(prefix=f"crossby-scene-{token}-")).resolve()
        workspace = container / "checkout"
        owned: OwnedWorktree | None = None
        yielded = False

        def bind() -> OwnedWorktree:
            # Only the unique container allocated by this call can be bound. An
            # interrupted Git add is recoverable once reciprocal Git links exist.
            gitdir = Path(_git(workspace, "rev-parse", "--absolute-git-dir")).resolve()
            if gitdir.parent != self.common / "worktrees":
                raise WorktreeError("New worktree has an unexpected private Git directory.")
            if Path((gitdir / "gitdir").read_text().strip()) != workspace / ".git":
                raise WorktreeError("New worktree lacks its reciprocal Git binding.")
            if _git(workspace, "rev-parse", "HEAD") != self.commit:
                raise WorktreeError("New worktree does not match the requested HEAD.")
            record = {
                "version": 1,
                "owner": "crossby-scene",
                "token": token,
                "container": str(container),
                "workspace": str(workspace),
                "common": str(self.common),
                "gitdir": str(gitdir),
                "commit": self.commit,
            }
            marker = json.dumps(record, sort_keys=True) + "\n"
            (container / _MARKER).write_text(marker)
            (gitdir / _MARKER).write_text(marker)
            return OwnedWorktree(workspace)

        try:
            _git(
                self.repository,
                "worktree",
                "add",
                "--detach",
                str(workspace),
                self.commit,
            )
            owned = bind()
            directories = list(
                dict.fromkeys((self.canonical_skills_relative, self.skills_relative))
            )
            if (
                self.bundles
                and self.skills_relative is not None
                and self.canonical_skills_relative is not None
                and self.skills_relative != self.canonical_skills_relative
            ):
                native = workspace / self.skills_relative
                canonical = workspace / self.canonical_skills_relative
                assert_within(workspace, native)
                assert_within(workspace, canonical)
                if not native.exists():
                    native.parent.mkdir(parents=True, exist_ok=True)
                    native.symlink_to(
                        os.path.relpath(canonical, native.parent), target_is_directory=True
                    )
                    directories.remove(self.skills_relative)
            for directory in directories:
                if directory is None:
                    continue
                for name, payload in self.bundles.items():
                    destination = workspace / directory / name
                    assert_within(workspace, destination)
                    if destination.exists():
                        raise WorktreeError(
                            f"Injected skill destination already exists: {destination}"
                        )
                    for relative, content in payload.items():
                        path = destination / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
            try:
                yielded = True
                yield owned
            except BaseException as exc:
                from crossby.utils.owned_launch import ProcessTreeCleanupError

                if isinstance(exc, ProcessTreeCleanupError):
                    keep = True
                raise
        except BaseException as exc:
            if keep and owned is not None and not yielded:
                raise WorktreeError(f"Setup failed; retained worktree: {workspace}: {exc}") from exc
            raise
        finally:
            if owned is None and workspace.exists():
                try:
                    owned = bind()
                except (OSError, WorktreeError) as exc:
                    raise WorktreeError(
                        f"Incomplete worktree retained for inspection: {workspace}: {exc}"
                    ) from exc
            if owned is not None:
                if not keep:
                    owned.cleanup(discard_changes=True)
            else:
                container.rmdir()


@dataclass(frozen=True)
class OwnedWorktree:
    path: Path

    def cleanup(self, *, discard_changes: bool = False) -> None:
        """Remove only a checkout with matching container and reciprocal Git ownership."""
        workspace = Path(os.path.abspath(self.path))
        container = workspace.parent
        try:
            assert_within(container.parent, container / _MARKER)
            assert_within(container, workspace / ".git")
            record = json.loads((container / _MARKER).read_text())
            if not isinstance(record, dict) or set(record) != {
                "version",
                "owner",
                "token",
                "container",
                "workspace",
                "common",
                "gitdir",
                "commit",
            }:
                raise ValueError("invalid ownership fields")
            token = record["token"]
            if (
                type(record["version"]) is not int
                or record["version"] != 1
                or record["owner"] != "crossby-scene"
                or not isinstance(token, str)
                or re.fullmatch(r"[0-9a-f]{32}", token) is None
                or not container.name.startswith(f"crossby-scene-{token}-")
                or workspace.name != "checkout"
                or record["container"] != str(container)
                or record["workspace"] != str(workspace)
                or container.stat().st_mode & 0o077
                or (hasattr(os, "getuid") and container.stat().st_uid != os.getuid())
            ):
                raise ValueError("ownership identity mismatch")
            common, gitdir = Path(record["common"]), Path(record["gitdir"])
            assert_within(common, gitdir / _MARKER)
            if gitdir.parent != common / "worktrees":
                raise ValueError("private Git directory mismatch")
            if json.loads((gitdir / _MARKER).read_text()) != record:
                raise ValueError("reciprocal ownership mismatch")
            if (workspace / ".git").read_text().strip() != f"gitdir: {gitdir}":
                raise ValueError("checkout Git binding mismatch")
            if Path((gitdir / "gitdir").read_text().strip()) != workspace / ".git":
                raise ValueError("reciprocal Git binding mismatch")
            if (
                Path(
                    _git(workspace, "rev-parse", "--path-format=absolute", "--git-common-dir")
                ).resolve()
                != common
            ):
                raise ValueError("repository mismatch")
            registrations = _git(workspace, "worktree", "list", "--porcelain", "-z").split("\0")
            if f"worktree {workspace}" not in registrations:
                raise ValueError("checkout is not registered")
            for path in workspace.rglob(".git"):
                if path != workspace / ".git":
                    raise ValueError(f"nested repository preserved: {path}")
            if not discard_changes and (
                _git(workspace, "status", "--porcelain", "--untracked-files=all")
                or _git(workspace, "rev-parse", "HEAD") != record["commit"]
            ):
                raise WorktreeError(
                    "Worktree has changes; inspect it, then use --discard-changes to remove it."
                )
        except WorktreeError:
            raise
        except (OSError, ValueError, TypeError, KeyError, PathContainmentError) as exc:
            raise WorktreeError(
                f"Refusing ambiguous worktree cleanup for {workspace}: {exc}"
            ) from exc
        _git(workspace, "worktree", "remove", "--force", str(workspace))
        (container / _MARKER).unlink()
        # Extra user files in the container are preserved; never rmtree it.
        with suppress(OSError):
            container.rmdir()


@contextmanager
def cancellation_cleanup() -> Iterator[None]:
    """Let SIGTERM unwind the owned launch, restoring the caller's signal policy."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
