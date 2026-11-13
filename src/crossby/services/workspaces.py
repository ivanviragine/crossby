"""Persona folders: a plain folder holding one tool's own instructions and skills.

:func:`provision` writes a caller's instructions, skills and material files
verbatim into the native layout of one tool (its
``AbstractAITool.OWNED_WORKSPACE_LAYOUT``), so the agent reads them the way it
reads any project of its own: no envelope, no crossby header, no extra file.
The returned :class:`OwnedWorkspace` carries a manifest (relative path to
SHA-256) of exactly what was written; it is the only ownership record, so
nothing of crossby's ever sits in the folder for the agent to read.

An ask in an owned folder refuses one that drifted from its manifest, and
checks again afterwards. :func:`cleanup` removes only manifest files whose
content still matches, then the directories left empty; it never deletes
anything it did not write.

One folder serves one tool: tools read each other's files (Copilot loads
``CLAUDE.md`` and ``AGENTS.md``, Cursor ``.claude/skills``), so a folder for
several would hand some of them the instructions twice.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from crossby.models.ai import AIToolID, OwnedWorkspaceLayout
from crossby.models.config import MCPServerConfig

_SKILL_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
# Also a Claude tool prefix, a TOML bare key and an Antigravity allow rule.
_MCP_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
# Whether writes and removals can be anchored to directory descriptors (every
# POSIX platform; not Windows). Fixed at import, so it cannot change mid-run.
_ANCHORED = {os.open, os.mkdir, os.unlink, os.rmdir, os.rename, os.link} <= os.supports_dir_fd
_MAX_SKILL_FILES = 1000
_MAX_BYTES = 20 * 1024 * 1024
# File names a supported tool reads as instructions or configuration, at the
# top level or in subfolders (Claude Code loads a subfolder's CLAUDE.md, and
# Codex prefers AGENTS.override.md over AGENTS.md). Material files and skill
# files may not use them at any depth, so they can never add or replace
# instructions, MCP servers, hooks or permissions.
_RESERVED_NAMES = frozenset(
    {
        "agents.md",
        "agents.override.md",
        "claude.md",
        "claude.local.md",
        "gemini.md",
        "opencode.json",
        "opencode.jsonc",
        "copilot-instructions.md",
    }
)


class WorkspaceError(Exception):
    """A persona folder cannot be provisioned, used or removed as asked."""


class WorkspaceConflictError(WorkspaceError):
    """The folder already holds something other than this persona."""


class WorkspaceDriftError(WorkspaceError):
    """The folder no longer matches what was provisioned into it."""

    def __init__(self, message: str, *, changed: tuple[str, ...]) -> None:
        super().__init__(message)
        self.changed = changed


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OwnedWorkspaceSpec(_Frozen):
    """What to put in a persona folder for one tool.

    ``instructions`` is literal text for the tool's own instruction file.
    ``skills`` are skill directories (each holding a ``SKILL.md``), copied
    whole under the tool's skills directory by their directory name.
    ``files`` are material files (relative path to text) the agent can read,
    such as interview notes; they may not be hidden or named like an
    instruction or configuration file.
    """

    path: Path
    tool: AIToolID
    instructions: str | None = None
    skills: tuple[Path, ...] = ()
    files: dict[str, str] = Field(default_factory=dict)
    # MCP servers this persona's agent gets for each ask, by name, on top of
    # the ones the tool already has. Their tools are pre-approved.
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @field_validator("mcp_servers")
    @classmethod
    def _mcp_servers_are_named_plainly(
        cls, value: dict[str, MCPServerConfig]
    ) -> dict[str, MCPServerConfig]:
        for name in value:
            if not _MCP_NAME.fullmatch(name):
                raise ValueError(
                    f"MCP server name {name!r} must be lowercase letters, digits, hyphens "
                    "and underscores"
                )
        return value

    @field_validator("instructions")
    @classmethod
    def _instructions_are_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("instructions must be non-blank when given")
        return value

    @field_validator("files")
    @classmethod
    def _files_are_plain_material(cls, value: dict[str, str]) -> dict[str, str]:
        for name in value:
            _material_path(name)
        return value


class OwnedWorkspace(_Frozen):
    """A provisioned persona folder: where it is, for which tool, and what it holds."""

    kind: Literal["owned"] = "owned"
    path: Path
    tool: AIToolID
    # Relative POSIX path of every file written, to its SHA-256.
    manifest: dict[str, str]
    layout: OwnedWorkspaceLayout
    # The persona's MCP servers, passed to the tool on every ask in this folder.
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


def layout_for(tool: AIToolID) -> OwnedWorkspaceLayout:
    """The verified persona-folder layout for ``tool``, or :class:`WorkspaceError`."""
    import crossby.ai_tools  # noqa: F401  (registers every adapter)
    from crossby.ai_tools.base import AbstractAITool

    layout = AbstractAITool.get(tool).OWNED_WORKSPACE_LAYOUT
    if layout is None:
        raise WorkspaceError(f"{tool.value} has no verified persona-folder layout.")
    return layout


def _material_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        # A colon makes a Windows drive path ("C:/x", "C:x") that escapes the
        # folder when joined, or names an NTFS alternate stream.
        or ":" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError(f"{name!r} is not a relative file path")
    if any(part.startswith(".") for part in path.parts):
        raise ValueError(f"{name!r} is hidden; tools read hidden folders as configuration")
    if path.name.lower() in _RESERVED_NAMES:
        raise ValueError(f"{name!r} is an instruction or configuration file name")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path) -> bytes | None:
    """A regular file's bytes, or ``None`` for anything else.

    Opened non-blocking and without following links, with the type checked
    on the open descriptor, so a FIFO or link swapped in can neither hang the
    read nor redirect it.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _skill_files(source: Path) -> tuple[str, dict[str, bytes]]:
    """A skill directory's name and its files, read without following links."""
    if source.is_symlink() or not source.is_dir():
        raise WorkspaceError(f"Skill {source} is not a directory.")
    name = source.name
    if not _SKILL_NAME.fullmatch(name):
        raise WorkspaceError(
            f"Skill directory name {name!r} must be lowercase letters, digits and hyphens."
        )
    if not (source / "SKILL.md").is_file() or (source / "SKILL.md").is_symlink():
        raise WorkspaceError(f"Skill {source} has no SKILL.md.")
    files: dict[str, bytes] = {}

    def unreadable(error: OSError) -> None:
        # A part of the skill that cannot be listed must never be dropped silently.
        raise WorkspaceError(f"Skill {source} cannot be read completely: {error}")

    for root, dirs, names in os.walk(source, followlinks=False, onerror=unreadable):
        for entry in (*dirs, *names):
            if Path(root, entry).is_symlink():
                raise WorkspaceError(f"Skill {source} contains a symbolic link: {entry}")
        for entry in names:
            full = Path(root, entry)
            if entry.lower() in _RESERVED_NAMES:
                raise WorkspaceError(
                    f"Skill {source} contains {entry}, an instruction or configuration file name."
                )
            try:
                data = _read_regular(full)
            except OSError as exc:
                raise WorkspaceError(f"Skill {source} cannot be read completely: {exc}") from None
            if data is None:
                raise WorkspaceError(f"Skill {source} contains a special file: {entry}")
            files[full.relative_to(source).as_posix()] = data
            if len(files) > _MAX_SKILL_FILES:
                raise WorkspaceError(f"Skill {source} has more than {_MAX_SKILL_FILES} files.")
    return name, files


_REFERENCE = re.compile(r"\$\{[^}]*\}")
_WHOLE_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_BEARER_REFERENCE = re.compile(r"Bearer\s+\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _codex_forwards(header: str, value: str) -> bool:
    """Whether Codex CLI can take this header's reference from the environment.

    As ``rewrite_headers_for_codex`` does: any header that is one reference, and
    ``Authorization: Bearer ${VAR}``. A default (``${VAR:-x}``) is not forwarded.
    """
    if _WHOLE_REFERENCE.fullmatch(value):
        return True
    return header.lower() == "authorization" and _BEARER_REFERENCE.fullmatch(value) is not None


def _session_server(tool: AIToolID, name: str, server: MCPServerConfig) -> MCPServerConfig:
    """``server`` as ``tool`` can take it for one session, ``${VAR}`` references included.

    Claude Code expands any reference itself. Codex CLI can forward a variable
    under its own name (``KEY = "${KEY}"``) and a header that is one reference;
    Antigravity CLI expands nothing, but its servers inherit the environment,
    so a same-name entry is simply left out. Any other reference would reach
    the server as the literal text, so it is refused.
    """
    if tool is AIToolID.CLAUDE:
        return server
    same_name = {key for key, value in server.env.items() if value == f"${{{key}}}"}
    literal = [
        *(value for key, value in server.env.items() if key not in same_name),
        *server.args,
        server.command or "",
        server.url or "",
        *(
            value
            for header, value in server.headers.items()
            if tool is not AIToolID.CODEX or not _codex_forwards(header, value)
        ),
    ]
    if any(_REFERENCE.search(value) for value in literal):
        raise WorkspaceError(
            f"MCP server {name!r} uses a ${{VAR}} reference that {tool.value} would pass on "
            'as text. An environment entry under its own name (KEY: "${KEY}") is forwarded '
            "there; Codex CLI also forwards a header that is one reference, and "
            "`Authorization: Bearer ${VAR}`."
        )
    if tool is AIToolID.ANTIGRAVITY_CLI and same_name:
        kept = {key: value for key, value in server.env.items() if key not in same_name}
        return server.model_copy(update={"env": kept})
    return server


def _enabled_servers(spec: OwnedWorkspaceSpec) -> dict[str, MCPServerConfig]:
    return {
        name: _session_server(spec.tool, name, server)
        for name, server in sorted(spec.mcp_servers.items())
        if server.enabled
    }


def planned_files(spec: OwnedWorkspaceSpec) -> dict[str, bytes]:
    """Every file :func:`provision` writes for ``spec``, by relative path."""
    layout = layout_for(spec.tool)
    planned: dict[str, bytes] = {}
    if spec.instructions is not None:
        planned[layout.instructions] = spec.instructions.encode("utf-8")
    seen: set[str] = set()
    for source in spec.skills:
        name, files = _skill_files(source)
        if name in seen:
            raise WorkspaceError(f"Two skills are named {name!r}.")
        seen.add(name)
        for relative, data in files.items():
            planned[f"{layout.skills}/{name}/{relative}"] = data
    servers = _enabled_servers(spec)
    if servers:
        if layout.mcp is None:
            raise WorkspaceError(
                f"{spec.tool.value} has no verified way to take MCP servers for one session."
            )
        if layout.mcp == "file" and layout.mcp_config is not None:
            from crossby.sync.mcp import _to_antigravity_cli_entry

            entries = {name: _to_antigravity_cli_entry(server) for name, server in servers.items()}
            text = json.dumps({"mcpServers": entries}, indent=2, sort_keys=True) + "\n"
            planned[layout.mcp_config] = text.encode("utf-8")
    for relative, text in spec.files.items():
        key = _material_path(relative).as_posix()
        if key in planned:
            raise WorkspaceError(f"Material file {relative!r} collides with a persona file.")
        planned[key] = text.encode("utf-8")
    if sum(len(data) for data in planned.values()) > _MAX_BYTES:
        raise WorkspaceError("A persona folder may hold at most 20 MiB.")
    # Refuse a layout the filesystem cannot hold before anything is written:
    # a path that is both a file and a folder, or two that differ only in
    # case (one file on case-insensitive filesystems).
    folders = _expected_directories(planned)
    both = sorted(name for name in planned if name in folders)
    if both:
        raise WorkspaceError(f"{both[0]!r} would be both a file and a folder.")
    folded: dict[str, str] = {}
    for name in sorted({*planned, *folders}):
        other = folded.setdefault(name.casefold(), name)
        if other != name:
            raise WorkspaceError(f"{other!r} and {name!r} differ only in case.")
    return planned


def _expected_directories(manifest: Mapping[str, object]) -> set[str]:
    return {
        parent.as_posix()
        for relative in manifest
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }


def _inventory(
    root: Path, expected_directories: set[str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Every regular file under ``root`` with its hash, and anything else found.

    "Anything else" is a link, a special file, or a directory that is not the
    parent of an expected file (when ``expected_directories`` is given).
    """
    found: dict[str, str] = {}
    other: list[str] = []
    unlisted: list[OSError] = []
    for current, dirs, names in os.walk(root, followlinks=False, onerror=unlisted.append):
        for entry in dirs:
            relative = Path(current, entry).relative_to(root).as_posix()
            if Path(current, entry).is_symlink() or (
                expected_directories is not None and relative not in expected_directories
            ):
                other.append(relative)
        for entry in names:
            full = Path(current, entry)
            relative = full.relative_to(root).as_posix()
            try:
                data = _read_regular(full)
            except OSError:
                # Unreadable, a link, or gone mid-scan: never the provisioned file.
                data = None
            if data is None:
                other.append(relative)
            else:
                found[relative] = _sha256(data)
    for error in unlisted:
        # A folder that cannot be listed cannot be shown to match.
        name = error.filename if isinstance(error.filename, str) else str(root)
        try:
            other.append(Path(name).relative_to(root).as_posix())
        except ValueError:
            other.append(".")
    return found, other


def _folder(path: Path) -> Path:
    if path.is_symlink():
        raise WorkspaceConflictError(f"{path} is a symbolic link.")
    return path.resolve()


def _write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise WorkspaceConflictError(f"{path} appeared while provisioning.") from None
        raise
    # The file is ours once created: a failed write or close removes it, so a
    # rollback never leaves a partial file without a manifest entry.
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def provision(spec: OwnedWorkspaceSpec) -> OwnedWorkspace:
    """Write ``spec`` into a new or empty folder, or confirm it is already there.

    The folder is created when missing. An existing folder must be empty, or
    hold exactly what ``spec`` would write (provisioning is idempotent);
    anything else raises :class:`WorkspaceConflictError` and nothing is written.
    """
    layout = layout_for(spec.tool)
    planned = planned_files(spec)
    manifest = {relative: _sha256(data) for relative, data in sorted(planned.items())}
    root = _folder(spec.path)
    workspace = OwnedWorkspace(
        path=root,
        tool=spec.tool,
        manifest=manifest,
        layout=layout,
        mcp_servers=_enabled_servers(spec),
    )
    if root.exists():
        if not root.is_dir():
            raise WorkspaceConflictError(f"{root} is not a folder.")
        found, other = _inventory(root, _expected_directories(manifest))
        if found == manifest and not other:
            return workspace
        if found or other or any(root.iterdir()):
            raise WorkspaceConflictError(
                f"{root} is not empty and does not hold this persona; provision into a new "
                "or empty folder."
            )
        created_root = False
    else:
        root.mkdir(parents=True)
        created_root = True
    written: list[str] = []
    # One descriptor for the folder through writing, verification and any
    # rollback, so a folder renamed or replaced mid-way is never mistaken for
    # this one.
    root_fd = os.open(root, _DIRECTORY_FLAGS) if _ANCHORED else None
    try:
        try:
            if root_fd is not None:
                _write_anchored(root_fd, root, planned, written)
            else:  # pragma: no cover - platforms without dir_fd support (Windows)
                for relative, data in planned.items():
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _write_new(target, data)
                    written.append(relative)
            # The folder must hold exactly what was written, nothing redirected.
            verify(workspace)
        except BaseException:
            # Leave nothing behind that the caller has no manifest for.
            _remove_written(root, root_fd, written, planned, manifest, created_root=created_root)
            raise
    finally:
        if root_fd is not None:
            os.close(root_fd)
    return workspace


def _write_anchored(
    root_fd: int, root: Path, planned: Mapping[str, bytes], written: list[str]
) -> None:
    """Write every planned file relative to directory descriptors, never following links.

    Each directory is created and opened from its verified parent with
    ``O_NOFOLLOW``, so a directory replaced by a link mid-way fails the write
    instead of redirecting it outside the folder.
    """
    for relative, data in planned.items():
        parts = relative.split("/")
        opened: list[int] = []
        current = root_fd
        try:
            for part in parts[:-1]:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, 0o755, dir_fd=current)
                current = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
                opened.append(current)
            name = parts[-1]
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(name, flags, 0o644, dir_fd=current)
            except FileExistsError:
                raise WorkspaceConflictError(
                    f"{root / relative} appeared while provisioning."
                ) from None
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(name, dir_fd=current)
                raise
            written.append(relative)
        finally:
            for item in reversed(opened):
                os.close(item)


def _remove_written(
    root: Path,
    root_fd: int | None,
    written: list[str],
    planned: Mapping[str, object],
    manifest: Mapping[str, str],
    *,
    created_root: bool,
) -> None:
    """Undo a provisioning that failed part-way: only the files it wrote.

    Removal goes through the descriptor the files were written through, and a
    file is removed only while it still holds what was written, so a folder
    renamed or replaced mid-way never loses anything of its own. Every planned
    directory is removed if it ended up empty; the folder itself only if its
    path still names that same directory.
    """
    for relative in reversed(written):
        _remove_entry(root, root_fd, relative, digest=manifest[relative])
    for directory in sorted(
        _expected_directories(planned),
        key=lambda item: item.count("/"),
        reverse=True,
    ):
        _remove_entry(root, root_fd, directory, digest=None)
    if created_root and root_fd is not None:
        with contextlib.suppress(OSError):
            here, there = os.fstat(root_fd), os.lstat(root)
            if (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino):
                root.rmdir()
    elif created_root:  # pragma: no cover - platforms without dir_fd support (Windows)
        with contextlib.suppress(OSError):
            root.rmdir()


def _remove_entry(root: Path, root_fd: int | None, relative: str, *, digest: str | None) -> None:
    """Remove one written file (``digest`` given, if it still matches) or empty directory."""
    if root_fd is None:  # pragma: no cover - platforms without dir_fd support (Windows)
        target = root / relative
        with contextlib.suppress(OSError):
            if digest is None:
                target.rmdir()
            elif not target.is_symlink() and _sha256(target.read_bytes()) == digest:
                target.unlink()
        return
    try:
        parent, name, opened = _open_parent(root_fd, relative)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            if digest is None:
                os.rmdir(name, dir_fd=parent)
            else:
                _unlink_if_unchanged(parent, name, digest)
    finally:
        for item in reversed(opened):
            os.close(item)


def verify(workspace: OwnedWorkspace) -> None:
    """Raise :class:`WorkspaceDriftError` unless the folder holds exactly its manifest."""
    root = workspace.path
    if root.is_symlink() or not root.is_dir():
        raise WorkspaceDriftError(f"{root} is no longer a folder.", changed=(".",))
    found, other = _inventory(root, _expected_directories(workspace.manifest))
    changed = sorted(
        {
            *other,
            *(name for name in found if workspace.manifest.get(name) != found[name]),
            *(name for name in workspace.manifest if name not in found),
        }
    )
    if changed:
        more = f" and {len(changed) - 5} more" if len(changed) > 5 else ""
        shown = ", ".join(changed[:5]) + more
        raise WorkspaceDriftError(
            f"The persona folder {root} changed since it was provisioned: {shown}.",
            changed=tuple(changed),
        )


def cleanup(workspace: OwnedWorkspace) -> None:
    """Remove a persona folder that still holds exactly what was provisioned.

    A drifted folder raises :class:`WorkspaceDriftError` and is left alone:
    only files crossby wrote, and the directories they leave empty, are ever
    removed. Each file is re-checked against its manifest hash and removed
    through directory descriptors opened without following links, so a
    directory replaced by a link after the check can never redirect a removal
    outside the folder.
    """
    verify(workspace)
    root = workspace.path
    if _ANCHORED:
        _cleanup_anchored(workspace)
    else:  # pragma: no cover - platforms without dir_fd support (Windows)
        _cleanup_by_path(workspace)
    try:
        root.rmdir()
    except OSError as exc:
        raise WorkspaceDriftError(
            f"The persona folder {root} gained content during cleanup.", changed=(".",)
        ) from exc


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_parent(root_fd: int, relative: str) -> tuple[int, str, list[int]]:
    """The descriptor of ``relative``'s parent, walked from ``root_fd`` without links."""
    parts = relative.split("/")
    opened: list[int] = []
    current = root_fd
    try:
        for part in parts[:-1]:
            current = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            opened.append(current)
    except OSError:
        for descriptor in opened:
            os.close(descriptor)
        raise
    return current, parts[-1], opened


def _drifted(workspace: OwnedWorkspace, relative: str) -> WorkspaceDriftError:
    return WorkspaceDriftError(
        f"The persona folder {workspace.path} changed during cleanup: {relative}.",
        changed=(relative,),
    )


def _unlink_if_unchanged(parent: int, name: str, digest: str) -> bool:
    """Remove ``name`` from ``parent`` only if it still holds ``digest``.

    The entry is first renamed to a private name in the same directory, which
    is atomic, so the content checked is exactly the content removed: a file
    substituted at ``name`` meanwhile is never deleted unchecked. On a
    mismatch the entry is linked back under ``name`` (never over a newer one).
    """
    try:
        entry = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError:
        return False
    if not stat.S_ISREG(entry.st_mode):
        # Only a regular file is ever moved: a directory or link put in its
        # place is left where it is.
        return False
    private = f".crossby-removing-{secrets.token_hex(8)}"
    try:
        os.rename(name, private, src_dir_fd=parent, dst_dir_fd=parent)
    except OSError:
        return False
    try:
        # Non-blocking, so a FIFO swapped in before the rename cannot hang the
        # open; its type is checked before anything is read.
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(private, flags, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                matches = False
            else:
                with os.fdopen(os.dup(descriptor), "rb") as handle:
                    matches = _sha256(handle.read()) == digest
        finally:
            os.close(descriptor)
        if matches:
            os.unlink(private, dir_fd=parent)
            return True
    except OSError:
        pass
    # Every unsuccessful removal puts the entry back.
    _restore(parent, private, name)
    return False


def _restore(parent: int, private: str, name: str) -> None:
    """Put a rejected entry back under ``name``, never over a newer entry."""
    try:
        os.link(private, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
    except FileExistsError:
        return  # something newer took the name; the entry keeps its private name
    except OSError:
        # Not hard-linkable (a directory swapped in after the check): rename it
        # back, only while nothing holds the name.
        with contextlib.suppress(OSError):
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                os.rename(private, name, src_dir_fd=parent, dst_dir_fd=parent)
        return
    with contextlib.suppress(OSError):
        os.unlink(private, dir_fd=parent)


def _cleanup_anchored(workspace: OwnedWorkspace) -> None:
    root_fd = os.open(workspace.path, _DIRECTORY_FLAGS)
    try:
        for relative, digest in workspace.manifest.items():
            try:
                parent, name, opened = _open_parent(root_fd, relative)
            except OSError:
                raise _drifted(workspace, relative) from None
            try:
                if not _unlink_if_unchanged(parent, name, digest):
                    raise _drifted(workspace, relative)
            finally:
                for descriptor in reversed(opened):
                    os.close(descriptor)
        for directory in sorted(
            _expected_directories(workspace.manifest),
            key=lambda item: item.count("/"),
            reverse=True,
        ):
            try:
                parent, name, opened = _open_parent(root_fd, directory)
            except OSError:
                continue
            try:
                # A directory that still holds something not written here stays.
                with contextlib.suppress(OSError):
                    os.rmdir(name, dir_fd=parent)
            finally:
                for descriptor in reversed(opened):
                    os.close(descriptor)
    finally:
        os.close(root_fd)


def _cleanup_by_path(workspace: OwnedWorkspace) -> None:  # pragma: no cover - Windows
    root = workspace.path
    for relative, digest in workspace.manifest.items():
        target = root / relative
        if any(parent.is_symlink() for parent in target.parents if root in parent.parents):
            raise _drifted(workspace, relative)
        if target.is_symlink() or _sha256(target.read_bytes()) != digest:
            raise _drifted(workspace, relative)
        target.unlink()
    for directory in sorted(
        _expected_directories(workspace.manifest),
        key=lambda item: item.count("/"),
        reverse=True,
    ):
        with contextlib.suppress(OSError):
            (root / directory).rmdir()


def workspace_unchanged(workspace: OwnedWorkspace) -> bool:
    """Whether the folder still holds exactly its manifest; never raises."""
    try:
        verify(workspace)
    except (WorkspaceDriftError, OSError):
        return False
    return True


__all__ = [
    "OwnedWorkspace",
    "OwnedWorkspaceSpec",
    "WorkspaceConflictError",
    "WorkspaceDriftError",
    "WorkspaceError",
    "cleanup",
    "layout_for",
    "planned_files",
    "provision",
    "verify",
    "workspace_unchanged",
]
