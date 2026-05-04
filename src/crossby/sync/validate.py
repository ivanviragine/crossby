"""Post-sync target validators.

Re-parse each tool's output files and surface structural issues that
``run_sync`` itself can't catch. The OpenAI ``migrate-to-codex`` skill
runs an equivalent ``--validate-target`` pass after every migration; the
checks here mirror that:

- ``.codex/config.toml`` parses as TOML; every ``[mcp_servers.<name>]``
  whose ``command`` is set has the binary on ``PATH``.
- ``.codex/agents/*.toml`` parses and carries ``name`` /
  ``description`` / ``developer_instructions``.
- ``<tool>/skills/<name>/SKILL.md`` carries ``name`` and ``description``
  frontmatter.
- ``AGENTS.md`` / ``CLAUDE.md`` / ``GEMINI.md`` / ``.cursorrules`` /
  ``.github/copilot-instructions.md`` are under a 32KB review threshold.
- Tool-specific JSON config files (``.claude/settings.json``,
  ``.cursor/cli.json``, ``.cursor/mcp.json``, ``.vscode/mcp.json``,
  ``.gemini/settings.json``) parse cleanly.

Findings are tool-neutral; the CLI renders them as a Rich table.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern


ValidationLevel = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class ValidationFinding:
    """One outcome from a validator: a fact about a path."""

    tool_id: AIToolID | None
    concern: SyncConcern | None
    level: ValidationLevel
    path: Path
    detail: str


# Files that should stay under this size threshold for an instructions file.
# Mirrors migrate-to-codex's MAX_AGENTS_MD_BYTES.
INSTRUCTION_SIZE_LIMIT_BYTES = 32 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok(
    project_root: Path,
    path: Path,
    detail: str,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        tool_id=tool_id,
        concern=concern,
        level="ok",
        path=_relative(path, project_root),
        detail=detail,
    )


def _warn(
    project_root: Path,
    path: Path,
    detail: str,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        tool_id=tool_id,
        concern=concern,
        level="warning",
        path=_relative(path, project_root),
        detail=detail,
    )


def _error(
    project_root: Path,
    path: Path,
    detail: str,
    *,
    tool_id: AIToolID | None = None,
    concern: SyncConcern | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        tool_id=tool_id,
        concern=concern,
        level="error",
        path=_relative(path, project_root),
        detail=detail,
    )


def _relative(path: Path, project_root: Path) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path


def _parse_skill_frontmatter(skill_md: Path) -> dict[str, object] | None:
    text = skill_md.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    try:
        data = yaml.safe_load(text[4:end])
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    return data


# ---------------------------------------------------------------------------
# Codex validators
# ---------------------------------------------------------------------------


def validate_codex_config(project_root: Path) -> list[ValidationFinding]:
    config_path = project_root / ".codex" / "config.toml"
    if not config_path.is_file():
        return []
    findings: list[ValidationFinding] = []
    try:
        parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        findings.append(
            _error(
                project_root,
                config_path,
                f"invalid TOML: {exc}",
                tool_id=AIToolID.CODEX,
                concern=SyncConcern.MCP,
            )
        )
        return findings

    findings.append(
        _ok(
            project_root,
            config_path,
            "valid TOML",
            tool_id=AIToolID.CODEX,
            concern=SyncConcern.MCP,
        )
    )
    mcp_servers = parsed.get("mcp_servers")
    if isinstance(mcp_servers, dict):
        for name, server in sorted(mcp_servers.items()):
            if not isinstance(server, dict):
                continue
            command = server.get("command")
            if not command:
                continue
            command_str = str(command)
            if shutil.which(command_str):
                findings.append(
                    _ok(
                        project_root,
                        config_path,
                        f"MCP `{name}` command `{command_str}` on PATH",
                        tool_id=AIToolID.CODEX,
                        concern=SyncConcern.MCP,
                    )
                )
            else:
                findings.append(
                    _warn(
                        project_root,
                        config_path,
                        f"MCP `{name}` command `{command_str}` not on PATH",
                        tool_id=AIToolID.CODEX,
                        concern=SyncConcern.MCP,
                    )
                )
    return findings


def validate_codex_agents(project_root: Path) -> list[ValidationFinding]:
    agents_dir = project_root / ".codex" / "agents"
    if not agents_dir.is_dir():
        return []
    required = ("name", "description", "developer_instructions")
    findings: list[ValidationFinding] = []
    for agent_file in sorted(agents_dir.glob("*.toml")):
        try:
            parsed = tomllib.loads(agent_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            findings.append(
                _error(
                    project_root,
                    agent_file,
                    f"invalid TOML: {exc}",
                    tool_id=AIToolID.CODEX,
                    concern=SyncConcern.AGENTS,
                )
            )
            continue
        missing = [key for key in required if not parsed.get(key)]
        if missing:
            findings.append(
                _error(
                    project_root,
                    agent_file,
                    "missing required field(s): " + ", ".join(missing),
                    tool_id=AIToolID.CODEX,
                    concern=SyncConcern.AGENTS,
                )
            )
            continue
        findings.append(
            _ok(
                project_root,
                agent_file,
                "agent TOML has required fields",
                tool_id=AIToolID.CODEX,
                concern=SyncConcern.AGENTS,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Skills validators (markdown frontmatter, applies to every tool)
# ---------------------------------------------------------------------------


_SKILLS_LOCATIONS: dict[AIToolID, Path] = {
    AIToolID.CLAUDE: Path(".claude") / "skills",
    AIToolID.CURSOR: Path(".cursor") / "skills",
    AIToolID.CODEX: Path(".agents") / "skills",
    AIToolID.GEMINI: Path(".gemini") / "skills",
    AIToolID.COPILOT: Path(".github") / "skills",
}


def validate_skill_frontmatter(project_root: Path) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for tool, rel in _SKILLS_LOCATIONS.items():
        root = project_root / rel
        if not root.is_dir():
            continue
        # Resolve symlinks so we walk the underlying tree once even if multiple
        # tool dirs symlink to the same canonical source.
        try:
            real_root = root.resolve()
        except OSError:
            real_root = root
        for skill_md in sorted(real_root.glob("*/SKILL.md")):
            data = _parse_skill_frontmatter(skill_md)
            relative = _relative(skill_md, project_root)
            if data is None:
                findings.append(
                    _error(
                        project_root,
                        skill_md,
                        "skill frontmatter missing or unparseable",
                        tool_id=tool,
                        concern=SyncConcern.SKILLS,
                    )
                )
                continue
            missing = [key for key in ("name", "description") if not data.get(key)]
            if missing:
                findings.append(
                    _error(
                        project_root,
                        skill_md,
                        "skill frontmatter missing " + ", ".join(missing),
                        tool_id=tool,
                        concern=SyncConcern.SKILLS,
                    )
                )
                continue
            findings.append(
                ValidationFinding(
                    tool_id=tool,
                    concern=SyncConcern.SKILLS,
                    level="ok",
                    path=relative,
                    detail="skill has name and description",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Instructions validator (size threshold)
# ---------------------------------------------------------------------------


_INSTRUCTION_FILES: dict[AIToolID, Path] = {
    AIToolID.CLAUDE: Path("CLAUDE.md"),
    AIToolID.CODEX: Path("AGENTS.md"),
    AIToolID.GEMINI: Path("GEMINI.md"),
    AIToolID.CURSOR: Path(".cursorrules"),
    AIToolID.COPILOT: Path(".github") / "copilot-instructions.md",
}


def validate_instruction_sizes(project_root: Path) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    seen: set[Path] = set()
    for tool, rel in _INSTRUCTION_FILES.items():
        path = project_root / rel
        if not path.is_file():
            continue
        # Resolve so symlinked targets aren't double-counted.
        try:
            real = path.resolve()
        except OSError:
            real = path
        if real in seen:
            continue
        seen.add(real)
        size = path.stat().st_size
        kb = size / 1024
        if size > INSTRUCTION_SIZE_LIMIT_BYTES:
            findings.append(
                _warn(
                    project_root,
                    path,
                    f"{kb:.1f}KB exceeds the 32KB review threshold",
                    tool_id=tool,
                    concern=SyncConcern.RULES,
                )
            )
        else:
            findings.append(
                _ok(
                    project_root,
                    path,
                    f"{kb:.1f}KB",
                    tool_id=tool,
                    concern=SyncConcern.RULES,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# JSON config validators
# ---------------------------------------------------------------------------


_JSON_CONFIGS: dict[AIToolID, list[Path]] = {
    AIToolID.CLAUDE: [Path(".claude") / "settings.json"],
    AIToolID.CURSOR: [Path(".cursor") / "cli.json", Path(".cursor") / "mcp.json"],
    AIToolID.GEMINI: [Path(".gemini") / "settings.json"],
    AIToolID.COPILOT: [Path(".vscode") / "mcp.json"],
}


def validate_json_configs(project_root: Path) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for tool, rels in _JSON_CONFIGS.items():
        for rel in rels:
            path = project_root / rel
            if not path.is_file():
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                findings.append(
                    _error(
                        project_root,
                        path,
                        f"invalid JSON: {exc}",
                        tool_id=tool,
                        concern=SyncConcern.MCP,
                    )
                )
                continue
            findings.append(
                _ok(
                    project_root,
                    path,
                    "valid JSON",
                    tool_id=tool,
                    concern=SyncConcern.MCP,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def validate_target(project_root: Path) -> list[ValidationFinding]:
    """Run every validator against ``project_root`` and return findings."""
    findings: list[ValidationFinding] = []
    findings.extend(validate_codex_config(project_root))
    findings.extend(validate_codex_agents(project_root))
    findings.extend(validate_skill_frontmatter(project_root))
    findings.extend(validate_instruction_sizes(project_root))
    findings.extend(validate_json_configs(project_root))
    return findings


def has_errors(findings: list[ValidationFinding]) -> bool:
    return any(f.level == "error" for f in findings)


__all__ = [
    "INSTRUCTION_SIZE_LIMIT_BYTES",
    "ValidationFinding",
    "ValidationLevel",
    "has_errors",
    "validate_codex_agents",
    "validate_codex_config",
    "validate_instruction_sizes",
    "validate_json_configs",
    "validate_skill_frontmatter",
    "validate_target",
]
