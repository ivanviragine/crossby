"""Real Git checkouts and local stub binaries; no native agent turns."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig, SceneConfig
from crossby.scenes.worktree import OwnedWorktree, WorktreeError, WorktreeRequest

runner = CliRunner()


def git(root: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.check_output(["git", "-C", str(root), *args], env=env, text=True).strip()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "fixture@example.test")
    git(repo, "config", "user.name", "Fixture")
    (repo / "package").mkdir()
    (repo / "package" / "app.py").write_text("committed\n")
    (repo / ".claude" / "skills" / "ordinary").mkdir(parents=True)
    (repo / ".claude" / "skills" / "ordinary" / "SKILL.md").write_text("ordinary\n")
    (repo / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"keep": {"command": "keep"}, "drop": {"command": "drop"}}})
    )
    source = tmp_path / "review-source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review a change\n---\nReview\n"
    )
    (source / "resources").mkdir()
    (source / "resources" / "guide.md").write_text("explicit payload\n")
    config: dict[str, Any] = {
        "ai": {"default_tool": "claude"},
        "scenes": {
            "review": {
                "isolation": "worktree",
                "inject": {"skills": {"review": str(source)}},
                "mcp": {"include": ["keep"]},
            }
        },
    }
    (repo / ".crossby.yml").write_text(yaml.safe_dump(config))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stub = binaries / "claude"
    stub.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, pathlib, sys
if '--version' in sys.argv:
    print('2.1.300')
    sys.exit(0)
root = pathlib.Path.cwd()
while not (root / '.git').exists():
    root = root.parent
original = pathlib.Path(os.environ['CROSSBY_TEST_ORIGINAL'])
evidence = {'cwd': str(pathlib.Path.cwd()), 'root': str(root), 'argv': sys.argv,
    'committed': (root / 'package/app.py').read_text(),
    'secret': (root / 'secret.env').exists(),
    'skill': (root / '.claude/skills/review/SKILL.md').read_text(),
    'original_skill_during': (original / '.claude/skills/review').exists()}
pathlib.Path(os.environ['CROSSBY_TEST_EVIDENCE']).write_text(json.dumps(evidence))
sys.exit(int(os.environ.get('CROSSBY_TEST_EXIT', '0')))
"""
    )
    stub.chmod(0o755)
    evidence = tmp_path / "evidence.json"
    owners = tmp_path / "owned"
    owners.mkdir()
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(
        "crossby.scenes.worktree.tempfile.mkdtemp", lambda **kw: real_mkdtemp(dir=owners, **kw)
    )
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CROSSBY_TEST_ORIGINAL", str(repo))
    monkeypatch.setenv("CROSSBY_TEST_EVIDENCE", str(evidence))
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        lambda: [AIToolID.CLAUDE, AIToolID.CODEX],
    )
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)
    monkeypatch.setattr("crossby.scenes.versioning.detect_tool_version", lambda *_: (0, 200, 0))
    return {
        "repo": repo,
        "source": source,
        "owners": owners,
        "evidence": evidence,
        "stub": stub,
        "tmp": tmp_path,
    }


def invoke(project: dict[str, Path], *args: str) -> Any:
    return runner.invoke(
        app,
        [
            "launch",
            str(project["repo"] / "package"),
            "--tool",
            "claude",
            "--scene",
            "review",
            *args,
        ],
    )


def request(project: dict[str, Path]) -> WorktreeRequest:
    from crossby.config.loader import load_config

    scene = load_config(project["repo"]).get_scene("review")
    assert scene is not None
    return WorktreeRequest.prepare(
        project["repo"] / "package", project["repo"], scene, AIToolID.CLAUDE
    )


def test_head_injection_compilation_and_original_preservation(project: dict[str, Path]) -> None:
    repo = project["repo"]
    (repo / "package" / "app.py").write_text("local changes\n")
    (repo / "secret.env").write_text("untracked secret\n")
    original = {
        p.relative_to(repo): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    excluded = (repo / ".git/info/exclude").read_bytes()
    result = invoke(project)
    assert result.exit_code == 0, result.output
    evidence = json.loads(project["evidence"].read_text())
    assert evidence["committed"] == "committed\n"
    assert evidence["secret"] is False
    assert evidence["original_skill_during"] is False
    assert evidence["cwd"] == str(Path(evidence["root"]) / "package")
    assert "Review" in evidence["skill"]
    assert "--mcp-config" in evidence["argv"]
    assert "--strict-mcp-config" in evidence["argv"]
    assert not Path(evidence["root"]).exists()
    assert not list(project["owners"].iterdir())
    assert (repo / ".git/info/exclude").read_bytes() == excluded
    assert original == {
        p.relative_to(repo): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    assert len(git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2


@pytest.mark.parametrize("exit_code", [7, 130])
def test_child_failure_cleans_owned_checkout(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    monkeypatch.setenv("CROSSBY_TEST_EXIT", str(exit_code))
    result = invoke(project)
    assert result.exit_code == exit_code, result.output
    assert not list(project["owners"].iterdir())


def test_keep_and_explicit_cleanup_preserve_dirty_original(project: dict[str, Path]) -> None:
    result = invoke(project, "--keep-worktree")
    assert result.exit_code == 0, result.output
    workspace = Path(json.loads(project["evidence"].read_text())["root"])
    assert workspace.exists()
    assert (
        workspace / ".claude/skills/review/resources/guide.md"
    ).read_text() == "explicit payload\n"
    (workspace / "child-work.txt").write_text("child changes")
    refused = runner.invoke(app, ["scene", "cleanup-worktree", str(workspace)])
    assert refused.exit_code == 1 and workspace.exists()
    assert "--discard-changes" in refused.output
    removed = runner.invoke(app, ["scene", "cleanup-worktree", str(workspace), "--discard-changes"])
    assert removed.exit_code == 0, removed.output
    assert not workspace.exists()
    assert not (project["repo"] / "child-work.txt").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--isolate", "none"],
        ["--isolate", "wrong"],
        ["--resume", "session"],
    ],
)
def test_invalid_cli_before_allocation(project: dict[str, Path], args: list[str]) -> None:
    result = invoke(project, *args)
    assert result.exit_code == 1, result.output
    assert not list(project["owners"].iterdir())
    assert not project["evidence"].exists()


@pytest.mark.parametrize("mutation", ["missing", "symlink", "collision", "excluded", "unsafe"])
def test_invalid_injection_before_allocation(project: dict[str, Path], mutation: str) -> None:
    from crossby.models.config import SceneInjection, SceneSelector

    scene = SceneConfig(
        isolation="worktree", inject=SceneInjection(skills={"review": str(project["source"])})
    )
    if mutation == "missing":
        (project["source"] / "SKILL.md").unlink()
    elif mutation == "symlink":
        (project["source"] / "secret-link").symlink_to(project["repo"] / ".crossby.yml")
    elif mutation == "collision":
        scene.inject = SceneInjection(skills={"ordinary": str(project["source"])})
    elif mutation == "excluded":
        scene.skills = SceneSelector(exclude=["review"])
    else:
        scene.inject = SceneInjection(skills={"../escape": str(project["source"])})
    with pytest.raises(WorktreeError):
        WorktreeRequest.prepare(project["repo"], project["repo"], scene, AIToolID.CLAUDE)
    assert not list(project["owners"].iterdir())


@pytest.mark.parametrize(
    "mutation", ["marker", "token", "gitlink", "reciprocal", "nested", "symlink"]
)
def test_ambiguous_ownership_is_never_deleted(project: dict[str, Path], mutation: str) -> None:
    with request(project).create(keep=True) as owned:
        workspace = owned.path
    marker = workspace.parent / "crossby-scene-owner.json"
    original = marker.read_text()
    record = json.loads(original)
    gitdir = Path(record["gitdir"])
    restore: tuple[Path, str] | None = None
    if mutation == "marker":
        marker.write_text("broken")
        restore = marker, original
    elif mutation == "token":
        record["token"] = "a" * 32
        marker.write_text(json.dumps(record))
        restore = marker, original
    elif mutation == "gitlink":
        restore = workspace / ".git", (workspace / ".git").read_text()
        (workspace / ".git").write_text(f"gitdir: {project['repo'] / '.git'}\n")
    elif mutation == "reciprocal":
        restore = gitdir / "crossby-scene-owner.json", original
        restore[0].write_text("{}")
    elif mutation == "nested":
        (workspace / "nested").mkdir()
        git(workspace / "nested", "init", "-q")
    else:
        marker.unlink()
        marker.symlink_to(project["repo"] / ".crossby.yml")
    with pytest.raises(WorktreeError):
        owned.cleanup(discard_changes=True)
    assert workspace.exists()
    if restore is not None:
        restore[0].write_text(restore[1])
    elif mutation == "nested":
        import shutil

        shutil.rmtree(workspace / "nested")
    else:
        marker.unlink()
        marker.write_text(original)
    owned.cleanup(discard_changes=True)


def test_foreign_worktree_is_preserved(project: dict[str, Path]) -> None:
    foreign = project["tmp"] / "foreign"
    git(project["repo"], "worktree", "add", "--detach", str(foreign), "HEAD")
    with pytest.raises(WorktreeError):
        OwnedWorktree(foreign).cleanup(discard_changes=True)
    assert foreign.exists()
    git(project["repo"], "worktree", "remove", str(foreign))


def test_inherited_isolation_fields_roundtrip() -> None:
    from crossby.scenes.authoring import render_scene_entry

    config = CrossbyConfig.model_validate(
        {
            "scenes": {
                "base": {
                    "isolation": "worktree",
                    "inject": {"skills": {"review": "catalog:crossby-sync"}},
                },
                "child": {"extends": "base"},
            }
        }
    )
    scene = config.get_scene("child")
    assert scene is not None
    assert scene.isolation == "worktree"
    assert scene.inject is not None
    rendered = yaml.safe_load(render_scene_entry("child", scene))["child"]
    assert rendered["isolation"] == "worktree"
    assert rendered["inject"]["skills"] == {"review": "catalog:crossby-sync"}


@pytest.mark.parametrize("transcript", [False, True])
@pytest.mark.parametrize("cancel_signal", [signal.SIGTERM, signal.SIGINT])
def test_cancel_stops_descendants_before_checkout_removal(
    project: dict[str, Path],
    transcript: bool,
    cancel_signal: int,
) -> None:
    # Real separate launcher, so actual signal unwinding and subprocess teardown run.
    script = project["tmp"] / "launcher.py"
    ready = project["tmp"] / "ready.json"
    heartbeat = project["tmp"] / "heartbeat"
    tool_script = project["tmp"] / "spawn.py"
    heartbeat_script = project["tmp"] / "heartbeat.py"
    heartbeat_script.write_text(
        "import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\n"
        "while True:\n p.write_text(str(time.time()))\n time.sleep(.02)\n"
    )
    tool_script.write_text(
        "import json,pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable," + repr(str(heartbeat_script)) + ",sys.argv[2]])\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'child':child.pid,"
        "'cwd':str(pathlib.Path.cwd())}))\nwhile True: time.sleep(1)\n"
    )
    script.write_text("""import pathlib, sys
from crossby.models.ai import AIToolID
from crossby.models.config import SceneConfig
from crossby.scenes.worktree import WorktreeRequest, cancellation_cleanup
from crossby.utils.process import run_with_transcript
repo, ready, heartbeat, tool, transcript = map(pathlib.Path, sys.argv[1:])
request = WorktreeRequest.prepare(repo, repo, SceneConfig(), AIToolID.CLAUDE)
with cancellation_cleanup(), request.create() as owned:
    run_with_transcript([sys.executable, str(tool), str(ready), str(heartbeat)],
        transcript if str(transcript) != 'none' else None, cwd=owned.path, owned=True)
""")
    transcript_path = project["tmp"] / "transcript.txt" if transcript else Path("none")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(project["repo"]),
            str(ready),
            str(heartbeat),
            str(tool_script),
            str(transcript_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() or not heartbeat.exists():
            if proc.poll() is not None or time.monotonic() >= deadline:
                proc.send_signal(signal.SIGTERM)
                pytest.fail(f"Stub launch failed: {proc.communicate(timeout=3)[1]!r}")
            time.sleep(0.02)
        workspace = Path(json.loads(ready.read_text())["cwd"])
        assert workspace.exists()
        proc.send_signal(cancel_signal)
        _, errors = proc.communicate(timeout=10)
        assert not workspace.exists(), errors.decode()
        last = heartbeat.read_text()
        time.sleep(0.15)
        assert heartbeat.read_text() == last, "Descendant kept running after worktree cleanup"
        if transcript:
            assert transcript_path.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


def test_cli_flag_enables_isolation_and_catalog(project: dict[str, Path]) -> None:
    config = yaml.safe_load((project["repo"] / ".crossby.yml").read_text())
    config["scenes"]["review"].pop("isolation")
    config["scenes"]["review"]["inject"]["skills"]["review"] = "catalog:crossby-sync"
    (project["repo"] / ".crossby.yml").write_text(yaml.safe_dump(config))
    result = invoke(project, "--isolate", "worktree")
    assert result.exit_code == 0, result.output
    assert "crossby" in json.loads(project["evidence"].read_text())["skill"].lower()
    assert not list(project["owners"].iterdir())


def test_codex_isolated_projection_and_inline_mcp(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = yaml.safe_load((project["repo"] / ".crossby.yml").read_text())
    config["scenes"]["review"]["skills"] = {"include": ["review"]}
    (project["repo"] / ".crossby.yml").write_text(yaml.safe_dump(config))
    codex = project["stub"].with_name("codex")
    codex.write_text(
        project["stub"]
        .read_text()
        .replace(".claude/skills/review/SKILL.md", ".agents/skills/review/SKILL.md")
    )
    codex.chmod(0o755)
    home = project["tmp"] / "codex-home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    result = runner.invoke(
        app,
        [
            "launch",
            str(project["repo"] / "package"),
            "--tool",
            "codex",
            "--scene",
            "review",
            "--keep-worktree",
        ],
    )
    assert result.exit_code == 0, result.output
    evidence = json.loads(project["evidence"].read_text())
    workspace = Path(evidence["root"])
    assert "Review" in evidence["skill"]
    assert not (workspace / ".agents/skills/ordinary").exists()
    assert 'mcp_servers."drop".enabled=false' in evidence["argv"]
    assert "--profile" not in evidence["argv"]
    assert not list(home.iterdir())
    assert not (project["repo"] / ".agents").exists()
    OwnedWorktree(workspace).cleanup(discard_changes=True)


def test_spawn_error_cleans_checkout(project: dict[str, Path]) -> None:
    project["stub"].unlink()
    result = invoke(project)
    assert result.exit_code == 1, result.output
    assert not list(project["owners"].iterdir())


def test_malformed_native_input_never_falls_back_to_original(project: dict[str, Path]) -> None:
    (project["repo"] / ".mcp.json").write_text('{"mcpServers": []}')
    git(project["repo"], "add", ".mcp.json")
    git(project["repo"], "commit", "-qm", "invalid fixture MCP")
    original = (project["repo"] / ".mcp.json").read_bytes()
    result = invoke(project)
    assert result.exit_code == 1, result.output
    assert not project["evidence"].exists()
    assert not list(project["owners"].iterdir())
    assert (project["repo"] / ".mcp.json").read_bytes() == original
    assert not (project["repo"] / ".crossby/scene-state.json").exists()


def test_keep_on_keyboard_cancellation(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def cancel(*_args: Any, **_kw: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("crossby.utils.process.run_with_transcript", cancel)
    result = invoke(project, "--keep-worktree")
    assert result.exit_code == 130, result.output
    workspace = next(project["owners"].iterdir()) / "checkout"
    assert workspace.exists()
    assert "Retained worktree" in result.output
    OwnedWorktree(workspace).cleanup(discard_changes=True)


def test_setup_error_cleanup(project: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    real_write = Path.write_bytes

    def fail_skill(path: Path, data: bytes) -> int:
        if "crossby-scene-" in str(path) and path.name == "SKILL.md":
            raise OSError("fixture copy failure")
        return real_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_skill)
    result = invoke(project)
    assert result.exit_code == 1, result.output
    assert not list(project["owners"].iterdir())


@pytest.mark.parametrize("kind", ["uncommitted-cwd", "external-link", "hook"])
def test_invalid_repository_state_before_allocation(project: dict[str, Path], kind: str) -> None:
    repo = project["repo"]
    cwd = repo
    if kind == "uncommitted-cwd":
        cwd = repo / "local-only"
        cwd.mkdir()
    elif kind == "external-link":
        (repo / "external").symlink_to(project["source"], target_is_directory=True)
        git(repo, "add", "external")
        git(repo, "commit", "-qm", "link fixture")
    else:
        hook = repo / ".git/hooks/post-checkout"
        hook.write_text("#!/bin/sh\nexit 0\n")
        hook.chmod(0o755)
    with pytest.raises(WorktreeError):
        WorktreeRequest.prepare(cwd, repo, SceneConfig(), AIToolID.CLAUDE)
    assert not list(project["owners"].iterdir())
    assert not (repo / "unexpected").exists()


def test_teardown_failure_retains_checkout(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.utils.owned_launch import ProcessTreeCleanupError

    def failed_teardown(*_args: Any, **_kw: Any) -> int:
        raise ProcessTreeCleanupError("fixture tool teardown timed out")

    monkeypatch.setattr("crossby.utils.process.run_with_transcript", failed_teardown)
    result = invoke(project)
    assert result.exit_code == 1, result.output
    assert "teardown timed out" in result.output
    workspace = next(project["owners"].iterdir()) / "checkout"
    assert workspace.exists()
    assert "Retained worktree" in result.output
    OwnedWorktree(workspace).cleanup(discard_changes=True)


def test_interrupted_add_recovers_and_cleans(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import crossby.scenes.worktree as module

    real_git = module._git

    def interrupted(root: Path, *args: str) -> str:
        result = real_git(root, *args)
        if args[:2] == ("worktree", "add"):
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(module, "_git", interrupted)
    result = invoke(project)
    assert result.exit_code == 130, result.output
    assert not list(project["owners"].iterdir())
    assert len(git(project["repo"], "worktree", "list", "--porcelain").split("worktree ")) == 2


@pytest.mark.parametrize("transcript", [False, True])
def test_normal_exit_stops_remaining_descendants(
    project: dict[str, Path], transcript: bool
) -> None:
    from crossby.utils.process import run_with_transcript

    heartbeat = project["tmp"] / "normal-heartbeat"
    child_script = project["tmp"] / "normal-child.py"
    child_script.write_text(
        "import pathlib,sys,time\np=pathlib.Path(sys.argv[1])\n"
        "while True:\n p.write_text(str(time.time()))\n time.sleep(.02)\n"
    )
    parent_script = project["tmp"] / "normal-parent.py"
    parent_script.write_text(
        "import pathlib,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable," + repr(str(child_script)) + ",sys.argv[1]])\n"
        "deadline=time.monotonic()+5\n"
        "while not pathlib.Path(sys.argv[1]).exists():\n"
        " if time.monotonic()>deadline: sys.exit(10)\n time.sleep(.01)\n"
        "print('finished', flush=True)\nsys.exit(7)\n"
    )
    transcript_path = project["tmp"] / "normal-transcript" if transcript else None
    with request(project).create() as owned:
        exit_code = run_with_transcript(
            [sys.executable, str(parent_script), str(heartbeat)],
            transcript_path,
            cwd=owned.path,
            owned=True,
        )
        assert exit_code == 7
        last = heartbeat.read_text()
        time.sleep(0.15)
        assert heartbeat.read_text() == last
        if transcript_path is not None:
            assert b"finished" in transcript_path.read_bytes()
    assert not list(project["owners"].iterdir())


def test_clean_child_commit_requires_explicit_discard(project: dict[str, Path]) -> None:
    original_head = git(project["repo"], "rev-parse", "HEAD")
    with request(project).create(keep=True) as owned:
        (owned.path / "child.txt").write_text("child work\n")
        git(owned.path, "add", ".")
        git(owned.path, "commit", "-qm", "child fixture commit")
    assert git(owned.path, "status", "--porcelain") == ""
    with pytest.raises(WorktreeError, match="--discard-changes"):
        owned.cleanup()
    assert owned.path.exists()
    assert git(project["repo"], "rev-parse", "HEAD") == original_head
    owned.cleanup(discard_changes=True)


def test_injection_preserves_existing_canonical_source(project: dict[str, Path]) -> None:
    repo = project["repo"]
    (repo / ".agents").mkdir()
    (repo / ".claude/skills").rename(repo / ".agents/skills")
    config = yaml.safe_load((repo / ".crossby.yml").read_text())
    config["scenes"]["review"]["skills"] = {"include": ["ordinary", "review"]}
    (repo / ".crossby.yml").write_text(yaml.safe_dump(config))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "canonical fixture")
    result = invoke(project, "--keep-worktree")
    assert result.exit_code == 0, result.output
    workspace = Path(json.loads(project["evidence"].read_text())["root"])
    assert (workspace / ".claude/skills/ordinary/SKILL.md").read_text() == "ordinary\n"
    assert (workspace / ".agents/skills/ordinary/SKILL.md").read_text() == "ordinary\n"
    assert not (repo / ".claude/skills").exists()
    assert not (repo / ".agents/skills/review").exists()
    OwnedWorktree(workspace).cleanup(discard_changes=True)


def test_transcript_drains_final_output(project: dict[str, Path]) -> None:
    from crossby.utils.process import run_with_transcript

    output = project["tmp"] / "large-transcript"
    with request(project).create() as owned:
        code = run_with_transcript(
            [sys.executable, "-c", "import os; os.write(1, b'x' * 200000 + b'END')"],
            output,
            cwd=owned.path,
            owned=True,
        )
    assert code == 0
    assert output.read_bytes() == b"x" * 200000 + b"END"


def test_chained_symlink_escape_refused_before_allocation(project: dict[str, Path]) -> None:
    repo = project["repo"]
    (repo / "alias").symlink_to(".", target_is_directory=True)
    (repo / "up").symlink_to("alias/..", target_is_directory=True)
    (repo / ".codex").symlink_to("up", target_is_directory=True)
    git(repo, "add", "alias", "up", ".codex")
    git(repo, "commit", "-qm", "chained escape fixture")
    original = (repo / ".crossby.yml").read_bytes()
    with pytest.raises(WorktreeError, match="escapes"):
        request(project)
    assert not list(project["owners"].iterdir())
    assert (repo / ".crossby.yml").read_bytes() == original


def test_internal_symlink_chain_is_supported(project: dict[str, Path]) -> None:
    repo = project["repo"]
    (repo / "alias").symlink_to("package", target_is_directory=True)
    (repo / "same").symlink_to("alias/../package", target_is_directory=True)
    git(repo, "add", "alias", "same")
    git(repo, "commit", "-qm", "internal links fixture")
    with request(project).create() as owned:
        assert (owned.path / "same/app.py").read_text() == "committed\n"


def test_injected_resource_retains_executable_mode(project: dict[str, Path]) -> None:
    script = project["source"] / "resources" / "check"
    script.write_text("#!/bin/sh\nprintf 'resource works'\n")
    script.chmod(0o755)
    with request(project).create() as owned:
        target = owned.path / ".claude/skills/review/resources/check"
        assert target.stat().st_mode & 0o777 == 0o755
        assert subprocess.check_output([str(target)], cwd=owned.path) == b"resource works"


def test_transcript_preserves_piped_input_and_eof(project: dict[str, Path]) -> None:
    script = project["tmp"] / "pipe-launcher.py"
    output = project["tmp"] / "pipe-transcript"
    script.write_text("""import pathlib,sys
from crossby.utils.process import run_with_transcript
command=[sys.executable, '-c', "import sys; data=sys.stdin.read(); print(repr(data)); sys.exit(7)"]
sys.exit(run_with_transcript(command, pathlib.Path(sys.argv[1]), cwd=sys.argv[2], owned=True))
""")
    result = subprocess.run(
        [sys.executable, str(script), str(output), str(project["repo"])],
        input="hello\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 7, result.stderr
    assert "'hello\\n'" in result.stdout
    assert b"'hello\\n'" in output.read_bytes()


def test_committed_relative_hook_refused_even_when_locally_deleted(
    project: dict[str, Path],
) -> None:
    repo = project["repo"]
    hooks = repo / ".hooks"
    hooks.mkdir()
    hook = hooks / "post-checkout"
    hook.write_text("#!/bin/sh\nexit 0\n")
    hook.chmod(0o755)
    git(repo, "add", ".hooks")
    git(repo, "commit", "-qm", "relative hook fixture")
    git(repo, "config", "core.hooksPath", ".hooks")
    hook.unlink()
    with pytest.raises(WorktreeError, match="post-checkout"):
        request(project)
    assert not list(project["owners"].iterdir())
