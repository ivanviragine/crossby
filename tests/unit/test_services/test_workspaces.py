"""Persona folders: one tool's own instructions and skills, written verbatim.

The layouts pinned here were verified live in plain folders, with no Git
repository (see ``docs/unattended-headless-verification.md#persona-folders``).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.headless import HeadlessUnsupportedError
from crossby.api import (
    AskRequest,
    CliBackend,
    OwnedWorkspaceSpec,
    WorkspaceChangedError,
    WorkspaceConflictError,
    WorkspaceDriftError,
    WorkspaceError,
    ask,
    cleanup,
    provision,
    verify_workspace,
)
from crossby.models.ai import AIToolID, HeadlessSessionRequest
from crossby.services.converse import DEPTH_ENV
from crossby.utils.versioning import BinaryVersion, parse_semver
from tests.unit.test_ai_tools.test_headless_adapters import CODEX_EVENTS, _jsonl

LAYOUTS = {
    AIToolID.CLAUDE: ("CLAUDE.md", ".claude/skills", "2.1.317"),
    AIToolID.CODEX: ("AGENTS.md", ".agents/skills", "0.166.0"),
    AIToolID.CURSOR: ("AGENTS.md", ".cursor/skills", "2026.11.05-86047df"),
    AIToolID.COPILOT: (".github/copilot-instructions.md", ".github/skills", "1.0.105"),
    AIToolID.OPENCODE: ("AGENTS.md", ".agents/skills", "1.19.5"),
    AIToolID.ANTIGRAVITY_CLI: ("AGENTS.md", ".agents/skills", "1.5.5"),
}
PERSONA = "# Dana\n\nYou are Dana, a skeptical procurement lead.\n"


@pytest.fixture(autouse=True)
def _top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


def _skill(root: Path, name: str = "voice", body: str = "Speak plainly.") -> Path:
    folder = root / "skills-src" / name
    (folder / "references").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: How Dana talks.\n---\n\n{body}\n"
    )
    (folder / "references" / "examples.md").write_text("Example answers.\n")
    return folder


def _spec(root: Path, tool: AIToolID = AIToolID.CODEX, **updates: Any) -> OwnedWorkspaceSpec:
    values: dict[str, Any] = {
        "path": root / "personas" / "dana",
        "tool": tool,
        "instructions": PERSONA,
        "skills": (_skill(root),),
        "files": {"notes/interview-1.md": "Budget is fixed for the year.\n"},
    }
    values.update(updates)
    return OwnedWorkspaceSpec(**values)


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# Layouts and provisioning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", list(LAYOUTS))
def test_each_tool_gets_its_own_verified_layout(tool: AIToolID, tmp_path: Path) -> None:
    instructions, skills, verified = LAYOUTS[tool]
    layout = AbstractAITool.get(tool).OWNED_WORKSPACE_LAYOUT
    assert layout is not None
    assert (layout.instructions, layout.skills, layout.verified_version) == (
        instructions,
        skills,
        verified,
    )

    workspace = provision(_spec(tmp_path, tool))

    written = _files(workspace.path)
    assert written == {
        instructions: PERSONA.encode(),
        f"{skills}/voice/SKILL.md": (tmp_path / "skills-src/voice/SKILL.md").read_bytes(),
        f"{skills}/voice/references/examples.md": b"Example answers.\n",
        "notes/interview-1.md": b"Budget is fixed for the year.\n",
    }
    # Exactly one instruction file and one skills directory: nothing of
    # crossby's sits in the folder for the agent to read.
    assert workspace.manifest == {
        name: hashlib.sha256(data).hexdigest() for name, data in written.items()
    }
    assert workspace.tool is tool
    assert workspace.layout == layout


def test_tools_without_a_verified_layout_are_refused(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="no verified persona-folder layout"):
        provision(_spec(tmp_path, AIToolID.VSCODE))
    assert not (tmp_path / "personas").exists()


def test_provisioning_is_idempotent_and_deterministic(tmp_path: Path) -> None:
    first = provision(_spec(tmp_path))
    again = provision(_spec(tmp_path))
    assert again == first


def test_a_folder_with_other_content_is_never_written_to(tmp_path: Path) -> None:
    folder = tmp_path / "personas" / "dana"
    folder.mkdir(parents=True)
    (folder / "mine.txt").write_text("keep me")
    with pytest.raises(WorkspaceConflictError, match="not empty"):
        provision(_spec(tmp_path))
    assert _files(folder) == {"mine.txt": b"keep me"}
    # Different persona content is a conflict too.
    other = tmp_path / "other"
    provision(_spec(tmp_path, path=other))
    with pytest.raises(WorkspaceConflictError):
        provision(_spec(tmp_path, path=other, instructions="Someone else."))


def test_an_empty_folder_is_used(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    assert provision(_spec(tmp_path, path=folder)).path == folder.resolve()


def test_a_link_or_file_is_never_a_persona_folder(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(WorkspaceConflictError, match="symbolic link"):
        provision(_spec(tmp_path, path=link))
    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(WorkspaceConflictError, match="not a folder"):
        provision(_spec(tmp_path, path=file))


@pytest.mark.parametrize(
    "name",
    [
        ".claude/settings.json",
        ".cursor/hooks.json",
        "notes/.hidden.md",
        "AGENTS.md",
        "claude.md",
        "GEMINI.md",
        "opencode.json",
        "/etc/passwd",
        "../escape.md",
        "a/../b.md",
        "a\\b.md",
        "C:/outside.md",
        "C:outside.md",
        "notes/x.md:stream",
        "",
    ],
)
def test_material_files_cannot_configure_a_tool_or_escape(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _spec(tmp_path, files={name: "x"})


def test_skills_are_checked_before_anything_is_written(tmp_path: Path) -> None:
    no_skill_md = tmp_path / "skills-src" / "empty"
    no_skill_md.mkdir(parents=True)
    bad_name = _skill(tmp_path, "Bad_Name")
    linked = _skill(tmp_path, "linked")
    (linked / "secret").symlink_to("/etc/hosts")
    duplicate = tmp_path / "elsewhere" / "voice"
    duplicate.mkdir(parents=True)
    (duplicate / "SKILL.md").write_text("---\nname: voice\n---\n")
    for skills, message in (
        ((no_skill_md,), "no SKILL.md"),
        ((bad_name,), "lowercase"),
        ((linked,), "symbolic link"),
        ((_skill(tmp_path, "voice2"), _skill(tmp_path, "voice3"), tmp_path / "missing"), "not a"),
    ):
        with pytest.raises(WorkspaceError, match=message):
            provision(_spec(tmp_path, skills=skills, path=tmp_path / "p"))
        assert not (tmp_path / "p").exists()
    with pytest.raises(WorkspaceError, match="Two skills"):
        provision(_spec(tmp_path, skills=(_skill(tmp_path), duplicate), path=tmp_path / "p"))
    assert not (tmp_path / "p").exists()


# ---------------------------------------------------------------------------
# Drift and cleanup
# ---------------------------------------------------------------------------


def _drifted(tmp_path: Path, change: str) -> Any:
    workspace = provision(_spec(tmp_path))
    root = workspace.path
    if change == "edited":
        (root / "AGENTS.md").write_text("Something else.")
    elif change == "added":
        (root / "new.md").write_text("x")
    elif change == "removed":
        (root / "notes" / "interview-1.md").unlink()
    elif change == "empty-dir":
        (root / "extra").mkdir()
    elif change == "link":
        (root / "link.md").symlink_to(root / "AGENTS.md")
    return workspace


@pytest.mark.parametrize("change", ["edited", "added", "removed", "empty-dir", "link"])
def test_any_change_is_drift(tmp_path: Path, change: str) -> None:
    workspace = _drifted(tmp_path, change)
    with pytest.raises(WorkspaceDriftError) as raised:
        verify_workspace(workspace)
    assert raised.value.changed


@pytest.mark.parametrize("change", ["edited", "added", "empty-dir"])
def test_cleanup_never_removes_a_drifted_folder(tmp_path: Path, change: str) -> None:
    workspace = _drifted(tmp_path, change)
    before = _files(workspace.path)
    with pytest.raises(WorkspaceDriftError):
        cleanup(workspace)
    assert _files(workspace.path) == before


def test_cleanup_removes_only_what_it_wrote(tmp_path: Path) -> None:
    workspace = provision(_spec(tmp_path))
    (tmp_path / "personas" / "sibling.md").write_text("not ours")
    cleanup(workspace)
    assert not workspace.path.exists()
    assert (tmp_path / "personas" / "sibling.md").read_text() == "not ours"


# ---------------------------------------------------------------------------
# Asking in a persona folder
# ---------------------------------------------------------------------------


def _codex_cli(
    monkeypatch: pytest.MonkeyPatch, record: Path, *, version: str = "0.166.0", write: bool = False
) -> None:
    program = "\n".join(
        (
            "import json,os,sys",
            "data=sys.stdin.read()",
            f"open({str(record)!r},'w').write(json.dumps({{'stdin': data, 'cwd': os.getcwd()}}))",
            "open('scribble.txt','w').write('x')" if write else "pass",
            f"sys.stdout.write({_jsonl(CODEX_EVENTS)!r})",
        )
    )
    adapter = AbstractAITool.get(AIToolID.CODEX)
    normalized = parse_semver(version)
    assert normalized is not None
    monkeypatch.setattr(
        type(adapter),
        "_headless_command",
        lambda self, request, **_k: [sys.executable, "-c", program],
        raising=False,
    )
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion(normalized, version),
        raising=True,
    )


def test_an_ask_in_a_persona_folder_sends_the_question_as_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "record.json"
    _codex_cli(monkeypatch, record)
    workspace = provision(_spec(tmp_path))

    result = ask(
        AskRequest(
            backend=CliBackend(tool="codex"), prompt="Would you buy it?", workspace=workspace
        )
    )

    assert result.succeeded
    seen = json.loads(record.read_text())
    # No envelope: the instructions are the tool's own AGENTS.md.
    assert seen["stdin"] == "Would you buy it?"
    assert seen["cwd"] == os.path.realpath(workspace.path)
    assert result.delivery == "native_files"
    assert result.envelope_version is None
    assert result.workspace_unchanged is True


def test_a_drifted_persona_folder_is_refused_before_the_agent_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "record.json"
    _codex_cli(monkeypatch, record)
    workspace = provision(_spec(tmp_path))
    (workspace.path / "AGENTS.md").write_text("tampered")

    with pytest.raises(WorkspaceChangedError) as raised:
        ask(AskRequest(backend=CliBackend(tool="codex"), prompt="q", workspace=workspace))

    assert raised.value.changed == ("AGENTS.md",)
    assert not record.exists()


def test_a_change_during_the_ask_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _codex_cli(monkeypatch, tmp_path / "record.json", write=True)
    workspace = provision(_spec(tmp_path))
    result = ask(AskRequest(backend=CliBackend(tool="codex"), prompt="q", workspace=workspace))
    assert result.workspace_unchanged is False


def test_a_persona_folder_runs_only_on_its_own_tool(tmp_path: Path) -> None:
    workspace = provision(_spec(tmp_path))
    with pytest.raises(ValidationError, match="provisioned for codex, not claude"):
        AskRequest(backend=CliBackend(tool="claude"), prompt="q", workspace=workspace)


def test_a_build_older_than_the_layout_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # agy's layout was verified on 1.5.5, after its read-only floor (1.5.4).
    adapter = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI)
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion((1, 5, 4), "1.5.4"),
        raising=True,
    )
    workspace = provision(_spec(tmp_path, AIToolID.ANTIGRAVITY_CLI))
    from crossby.api import AskUnsupportedError

    with pytest.raises(AskUnsupportedError, match=r"needs Antigravity CLI 1\.5\.5 or newer"):
        ask(AskRequest(backend=CliBackend(tool="antigravity-cli"), prompt="q", workspace=workspace))


def test_minimum_version_is_checked_against_the_detected_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="not a version"):
        HeadlessSessionRequest(prompt="p", working_dir=tmp_path, minimum_version="soon")
    adapter = AbstractAITool.get(AIToolID.CODEX)
    monkeypatch.setattr(
        type(adapter),
        "_detect_headless_version",
        lambda self, **_kwargs: BinaryVersion((0, 166, 0), "codex-cli 0.166.0"),
        raising=True,
    )
    request = HeadlessSessionRequest(prompt="p", working_dir=tmp_path, minimum_version="0.170.0")
    with pytest.raises(HeadlessUnsupportedError, match=r"needs Codex CLI 0\.170\.0 or newer"):
        adapter.preflight_headless_session(request)


# ---------------------------------------------------------------------------
# Review hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["AGENTS.override.md", "notes/CLAUDE.md", "deep/a/Agents.md", "x/gemini.md"]
)
def test_instruction_names_are_reserved_at_every_depth(name: str, tmp_path: Path) -> None:
    # Codex prefers AGENTS.override.md over AGENTS.md, and Claude Code loads a
    # subfolder's CLAUDE.md, so either would replace or add instructions.
    with pytest.raises(ValidationError, match="instruction or configuration"):
        _spec(tmp_path, files={name: "different instructions"})


def test_a_skill_cannot_carry_instruction_files(tmp_path: Path) -> None:
    skill = _skill(tmp_path, "sneaky")
    (skill / "references" / "AGENTS.md").write_text("different instructions")
    with pytest.raises(WorkspaceError, match="instruction or configuration file name"):
        provision(_spec(tmp_path, skills=(skill,)))
    assert not (tmp_path / "personas").exists()


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"notes": "one", "notes/interview.md": "two"}, "both a file and a folder"),
        ({"Notes/a.md": "one", "notes/b.md": "two"}, "differ only in case"),
    ],
)
def test_layouts_the_filesystem_cannot_hold_are_refused_before_writing(
    tmp_path: Path, files: dict[str, str], message: str
) -> None:
    with pytest.raises(WorkspaceError, match=message):
        provision(_spec(tmp_path, files=files))
    assert not (tmp_path / "personas").exists()


def test_a_failed_write_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.services import workspaces

    real_fdopen = os.fdopen
    calls = {"count": 0}

    def flaky(fd: int, mode: str) -> Any:
        calls["count"] += 1
        if calls["count"] == 3:
            os.close(fd)
            raise OSError("disk full")
        return real_fdopen(fd, mode)

    monkeypatch.setattr(workspaces.os, "fdopen", flaky)
    with pytest.raises(OSError, match="disk full"):
        provision(_spec(tmp_path))
    assert not (tmp_path / "personas" / "dana").exists()
    # A pre-existing empty folder is kept, emptied again.
    folder = tmp_path / "empty"
    folder.mkdir()
    calls["count"] = 0
    with pytest.raises(OSError):
        provision(_spec(tmp_path, path=folder))
    assert folder.is_dir() and not any(folder.iterdir())


@pytest.mark.skipif(os.mkdir not in os.supports_dir_fd, reason="needs dir_fd support")
def test_provisioning_never_follows_a_folder_replaced_mid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.services import workspaces

    outside = tmp_path / "outside"
    outside.mkdir()
    folder = tmp_path / "p"
    real_mkdir = os.mkdir

    def mkdir_then_swap(name: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        real_mkdir(name, mode, dir_fd=dir_fd)
        if name == "notes":
            os.rmdir(name, dir_fd=dir_fd)
            os.symlink(outside, name, dir_fd=dir_fd)

    monkeypatch.setattr(workspaces.os, "mkdir", mkdir_then_swap)
    with pytest.raises(OSError):
        provision(_spec(tmp_path, path=folder))
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.unlink not in os.supports_dir_fd, reason="needs dir_fd support")
def test_cleanup_never_follows_a_folder_replaced_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.services import workspaces

    workspace = provision(_spec(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    # Same name and content as the manifest entry, so only the link check can stop it.
    (outside / "interview-1.md").write_text("Budget is fixed for the year.\n")
    real_verify = workspaces.verify

    def verify_then_swap(target: Any) -> None:
        real_verify(target)
        notes = workspace.path / "notes"
        (notes / "interview-1.md").unlink()
        notes.rmdir()
        notes.symlink_to(outside)

    monkeypatch.setattr(workspaces, "verify", verify_then_swap)
    with pytest.raises(WorkspaceDriftError, match="changed during cleanup"):
        cleanup(workspace)
    assert (outside / "interview-1.md").read_text() == "Budget is fixed for the year.\n"


def test_a_write_that_fails_part_way_removes_the_file_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.services import workspaces

    real_fdopen = os.fdopen

    class Failing:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Failing:
            return self

        def __exit__(self, *exc: object) -> None:
            self.handle.close()

        def write(self, data: bytes) -> int:
            self.handle.write(data[:10])
            raise OSError("file too large")

    monkeypatch.setattr(workspaces.os, "fdopen", lambda fd, mode: Failing(real_fdopen(fd, mode)))
    folder = tmp_path / "p"
    with pytest.raises(OSError, match="file too large"):
        provision(_spec(tmp_path, path=folder))
    assert not folder.exists()
