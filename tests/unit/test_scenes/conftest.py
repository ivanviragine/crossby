"""Shared fixtures/helpers for scene-engine tests."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from crossby.models.ai import AIToolID
from crossby.models.config import SceneConfig
from crossby.services.scene_resolution import ResolvedScene, resolve_scene
from crossby.sync.readers import scan_project

DEFAULT_TOOLS = [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI]


def make_skill(root: Path, rel_dir: str, name: str) -> None:
    skill_dir = root / rel_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody\n", encoding="utf-8")


def make_agent(root: Path, rel_dir: str, filename: str) -> None:
    agent_dir = root / rel_dir
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / filename).write_text("---\nname: x\n---\nagent body\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def symlink_dir(link: Path, target: Path) -> None:
    """Point *link* at *target* with a relative symlink (mirrors a prior sync)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, link.parent)
    link.symlink_to(rel, target_is_directory=True)


def populate_project(root: Path) -> None:
    """A multi-tool project: 3 skills, 2 agents, 2 MCP servers.

    Skills live in ``.claude/skills`` with ``.agents/skills`` symlinked to it (the
    normal post-sync state), so codex + antigravity-cli share that resolved path.
    """
    for name in ("review-skill", "knowledge", "deploy-prod"):
        make_skill(root, ".claude/skills", name)
    symlink_dir(root / ".agents" / "skills", root / ".claude" / "skills")

    make_agent(root, ".claude/agents", "code-reviewer.md")
    make_agent(root, ".claude/agents", "deployer.md")

    write_json(
        root / ".mcp.json",
        {"mcpServers": {"github": {"command": "gh-mcp"}, "linear": {"command": "lin-mcp"}}},
    )


def resolve(root: Path, scene: SceneConfig, tools: list[AIToolID] | None = None) -> ResolvedScene:
    scan = scan_project(root, tools or DEFAULT_TOOLS)
    return resolve_scene(scene, scan, root)


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """Keep scene tests from caching a stdlib-bound logger into shared state.

    ``test_logging`` configures structlog globally with ``cache_logger_on_first_use``
    at import time; running ``apply_scene`` (which touches the readers logger)
    would then cache that logger in stdlib mode and break a later ``capture_logs``
    test elsewhere in the suite. Reset to cache-free defaults for the duration of
    each scene test and restore the prior config afterwards, so these tests are
    transparent to the rest of the suite.
    """
    saved = structlog.get_config()
    structlog.reset_defaults()
    try:
        yield
    finally:
        structlog.configure(**saved)


@pytest.fixture(autouse=True)
def _fixed_claude_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the detected Claude version so skillOverrides isn't version-gated.

    Individual tests override this to exercise the gate. Also stub Codex trust so
    tests never read the developer's real ``~/.codex/config.toml``.
    """
    monkeypatch.setattr("crossby.scenes.versioning.detect_tool_version", lambda _tool: (2, 1, 218))
    monkeypatch.setattr("crossby.scenes.trust.codex_trusts_project", lambda *a, **k: True)
    # Pin the installed-tool set so projection targets are deterministic and don't
    # depend on which CLIs the developer happens to have on PATH.
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        classmethod(lambda _cls: list(DEFAULT_TOOLS)),
    )
