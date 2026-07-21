"""Agent sync writers — distribute agents from a canonical source to each tool's directory."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml

from crossby.config.linker import create_symlink
from crossby.models.ai import AIToolID
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult
from crossby.sync.file_utils import (
    MANAGED_MARKER_NAME,
    backup_path,
    has_managed_marker,
    write_managed_marker,
)
from crossby.sync.gitignore_utils import update_managed_block

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Gitignore managed-block
# ---------------------------------------------------------------------------

_GITIGNORE_BLOCK_ID = "agents sync"

# Per-tool agent directory paths (relative to project root).
# Codex uses ``.codex/agents`` for custom-agent TOML files (per Codex docs);
# ``.agents/skills/`` is a *skills* directory, handled by sync/skills.py.
_AGENT_TARGET_PATHS: dict[str, str] = {
    "claude": ".claude/agents",
    "copilot": ".github/agents",
    "cursor": ".cursor/agents",
    "codex": ".codex/agents",
    "antigravity-cli": ".agents/agents",
}


def update_agents_gitignore(
    data: SyncData,
    project_root: Path,
    *,
    dry_run: bool = False,
    installed_tools: list[AIToolID] | None = None,
) -> SyncResult | None:
    """Write/update the crossby-managed block in .gitignore.

    Returns a SyncResult if a change was made (or would be in dry-run), else None.
    The source directory itself is never gitignored.
    """
    if data.agents_source is None or not data.agents_gitignore:
        return None

    # Determine which tool target paths to include in the block
    if installed_tools is not None:
        entries = [
            _AGENT_TARGET_PATHS[str(t)] for t in installed_tools if str(t) in _AGENT_TARGET_PATHS
        ]
    else:
        entries = list(_AGENT_TARGET_PATHS.values())

    if not entries:
        return None

    gitignore_path = project_root / ".gitignore"
    action: Literal["created", "updated"] = "updated" if gitignore_path.is_file() else "created"

    changed = update_managed_block(project_root, _GITIGNORE_BLOCK_ID, entries, dry_run=dry_run)
    if not changed:
        return None

    return SyncResult(
        tool_id=None,
        concern=SyncConcern.AGENTS,
        action=action,
        file_path=gitignore_path,
        message="gitignore",
    )


# ---------------------------------------------------------------------------
# Tool name translation (copy strategy)
# ---------------------------------------------------------------------------

_TOOL_NAME_MAP: dict[str, dict[str, str]] = {
    "copilot": {
        "Read": "read",
        "Edit": "edit",
        "Grep": "search",
        "Glob": "glob",
        "Bash": "shell",
        "WebSearch": "web_search",
        "WebFetch": "web_fetch",
    },
    "cursor": {
        "Bash": "Shell",
    },
}


def _translate_tools(tools: list[str], tool_id: str) -> list[str]:
    """Map canonical tool names to tool-specific names."""
    mapping = _TOOL_NAME_MAP.get(tool_id, {})
    return [mapping.get(t, t) for t in tools]


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> tuple[dict[str, object] | None, str]:
    """Split YAML frontmatter from markdown body.

    Returns (fm_dict, body) where fm_dict is None when there is no frontmatter
    or when it could not be parsed (missing closing delimiter or invalid YAML).
    Callers must treat None as "no parseable frontmatter" and copy verbatim.
    """
    if not content.startswith("---\n"):
        return None, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return None, content
    try:
        raw = yaml.safe_load(content[4:end])
        if not isinstance(raw, dict):
            # Non-dict YAML (list, scalar, etc.) — copy verbatim to avoid data loss
            return None, content
        fm: dict[str, object] = raw
    except yaml.YAMLError:
        return None, content
    return fm, content[end + 5 :]


def _render_frontmatter(fm: dict[str, object], body: str) -> str:
    """Reassemble frontmatter + body into a markdown string."""
    return f"---\n{yaml.dump(fm, default_flow_style=False, sort_keys=False)}---\n{body}"


def detect_legacy_codex_agents(project_root: Path) -> Path | None:
    """Return the legacy `.agents/` path when it carries old codex-agent content.

    Crossby ≤ 0.2.x synced codex agents to ``<project>/.agents/`` (either as a
    directory symlink to the source or as a markdown copy). The current path
    per upstream Codex docs is ``<project>/.codex/agents/<name>.toml``;
    `.agents/skills/` is the Codex *skills* root and is left alone.

    Detection: the legacy path is reported when ``.agents`` is a symlink
    (the old default), OR when it's a real directory containing one or more
    top-level ``*.md`` files (old copy fallback). A directory containing
    only ``skills/`` (current Codex skills layout) returns ``None``.
    """
    legacy = project_root / ".agents"
    if legacy.is_symlink():
        return legacy
    if not legacy.is_dir():
        return None
    try:
        for child in legacy.iterdir():
            if child.is_file() and child.suffix == ".md":
                return legacy
    except OSError:
        return None
    return None


def _warn_legacy_codex_agents_path(project_root: Path) -> None:
    """Log a one-shot warning when the legacy `.agents/` path is present.

    Non-destructive: never auto-deletes, never blocks the sync. Users can
    remove the directory or symlink at their convenience now that codex
    agents live at ``.codex/agents/``.
    """
    legacy = detect_legacy_codex_agents(project_root)
    if legacy is None:
        return
    logger.warning(
        "agents.legacy_codex_path",
        path=str(legacy),
        hint=(
            f"`.agents/` is no longer the codex agents target — codex agents "
            f"now sync to `.codex/agents/`. The legacy path was left untouched; "
            f"remove `{legacy}` once you've confirmed nothing else relies on it."
        ),
    )


# Sync writers delegate cross-tool translation to :mod:`crossby.subagents`,
# the canonical agent IR + parser + emitter layer. The helpers below add
# the directory-level concerns that one-shot translator doesn't care about:
# source-tool inference from the source-dir path, in-file manual-fix block
# emission from `ConversionWarning`s, and Codex's two-output emit
# (agent TOML + optional ``[agents.<name>]`` config fragment).

# Map tool-default agent paths back to their AIToolID, used by
# _infer_source_tool to pick the right parser without an explicit source-tool
# flag. Tool-neutral source dirs (e.g. ``.crossby/agents``) fall through to
# claude — the most common markdown frontmatter shape.
_SOURCE_TOOL_BY_PATH: dict[str, str] = {
    ".claude/agents": "claude",
    ".cursor/agents": "cursor",
    ".github/agents": "copilot",
    ".codex/agents": "codex",
}


def _infer_source_tool(source_dir: Path) -> str:
    """Return the subagents tool name (`claude`, `cursor`, …) for a source dir.

    Falls back to ``claude`` for tool-neutral paths (markdown + YAML
    frontmatter is the prevailing shape across Claude / Cursor / Copilot,
    and the Claude parser is the most lenient). Also the fallback for
    Antigravity CLI's ``.agents/agents`` — that tool has no dedicated
    subagents parser/emitter yet.
    """
    s = source_dir.as_posix()
    for needle, tool in _SOURCE_TOOL_BY_PATH.items():
        if s.endswith(needle):
            return tool
    return "claude"


def _significant_warnings(warnings: list[Any]) -> list[Any]:
    """Filter `ConversionWarning`s to lossy/dropped — info notes are silent."""
    from crossby.subagents.ir import WarningSeverity

    return [w for w in warnings if w.severity != WarningSeverity.INFO]


def _ir_body_with_manual_fix(ir: Any, warnings: list[Any]) -> Any:
    """Return a copy of ``ir`` with a `crossby:manual-fix` block in its body.

    Strips any pre-existing manual-fix block from the IR body before
    appending the fresh one. Without that strip, a source file that
    already contains a block (because the user round-tripped it, or
    edited a previously-translated artifact and fed it back) would
    accumulate a new block on every sync. Returns ``ir`` unchanged
    when there are no significant warnings *and* the body has no
    leftover block to strip.
    """
    from crossby.sync.manual_fix import (
        ManualFixNote,
        append_manual_fix_block,
        strip_manual_fix_blocks,
    )

    body = ir.body or ""
    cleaned_body = strip_manual_fix_blocks(body)
    if not warnings:
        # Nothing fresh to attach, but if we cleaned a stale block, return
        # the cleaned body so re-translates remove the old artifact.
        if cleaned_body == body:
            return ir
        return ir.model_copy(update={"body": cleaned_body})
    notes = [
        ManualFixNote(
            category=w.field,
            message=f"[{w.severity.value}] {w.field}: {w.message}",
        )
        for w in warnings
    ]
    new_body = append_manual_fix_block(cleaned_body, notes)
    return ir.model_copy(update={"body": new_body})


def _translate_markdown_agent(
    *,
    content: str,
    from_tool: str,
    to_tool: str,
    source_path: Path | None = None,
) -> str:
    """Translate a markdown-shape agent file and embed manual-fix notes.

    Returns the rendered markdown for the target tool. Used by
    :class:`_BaseAgentsWriter._sync_translate` for the markdown-shape
    targets (Claude / Cursor / Copilot). For Codex output see
    :func:`_translate_codex_agent`.

    ``source_path`` is forwarded to the parser so name inference can
    recover the canonical agent name from the filename when frontmatter
    omits it — without it, parsers fall back to ``"agent"`` for every
    file, collapsing distinct agents into one rendering.
    """
    from crossby.subagents.api import emit as _emit
    from crossby.subagents.api import parse as _parse

    ir, parse_warnings = _parse(from_tool, content, source_path)
    # First pass: discover what the target emitter would warn about.
    _, emit_warnings = _emit(to_tool, ir)
    notes = _significant_warnings(parse_warnings + emit_warnings)
    rich_ir = _ir_body_with_manual_fix(ir, notes)
    payload, _ = _emit(to_tool, rich_ir)
    # Markdown emitters always return ``str``; only emit_codex returns a
    # CodexEmission, and this helper is for the markdown-shape targets.
    assert isinstance(payload, str)
    return payload


def _translate_codex_agent(
    *,
    content: str,
    from_tool: str,
    source_path: Path | None = None,
) -> tuple[str, str]:
    """Translate to Codex; returns (agent.toml, config.toml fragment).

    Two-pass pattern: emit once to collect warnings, mutate IR body to
    carry a `crossby:manual-fix` block, re-emit so the block lands inside
    ``developer_instructions``. Also pre-translates Claude-family model
    ids and effort tiers to Codex equivalents — subagents.emitters.emit_codex
    passes ``model`` through verbatim, which would hand Codex an id it
    rejects (``claude-sonnet-4.6``); the family mapping in sync.translation
    is the right place to land that conversion.
    """
    from crossby.subagents.api import emit as _emit
    from crossby.subagents.api import parse as _parse
    from crossby.sync.translation import (
        find_claude_family,
        map_effort_claude_to_codex,
        map_model_claude_to_codex,
    )

    ir, parse_warnings = _parse(from_tool, content, source_path)

    # Cross-provider translation for the model + effort pair.  Only fires
    # when the model belongs to a known Claude family — anything else
    # (gpt-*, custom ids, etc.) passes through to the emitter unchanged.
    if ir.model and find_claude_family(ir.model) is not None:
        translated_model = map_model_claude_to_codex(ir.model)
        translated_effort = ir.effort
        if ir.effort:
            mapped = map_effort_claude_to_codex(ir.model, ir.effort)
            if mapped is not None:
                translated_effort = mapped.value
        ir = ir.model_copy(update={"model": translated_model, "effort": translated_effort})

    _, emit_warnings = _emit("codex", ir)
    notes = _significant_warnings(parse_warnings + emit_warnings)
    rich_ir = _ir_body_with_manual_fix(ir, notes)
    emission, _ = _emit("codex", rich_ir)
    return emission.agent_toml, emission.config_fragment


def _copy_agent_file(source: Path, target: Path, tool_id: str) -> bool:
    """Copy one agent file to target, translating tool names.

    Returns True when the target was written or rewritten, False when the
    on-disk content was already byte-identical to the rendered output
    (idempotent re-run).
    """
    content = source.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(content)
    if isinstance(fm, dict):
        raw_tools = fm.get("tools")
        if isinstance(raw_tools, list):
            fm["tools"] = _translate_tools([str(t) for t in raw_tools], tool_id)
        out = _render_frontmatter(fm, body)
    else:
        # Frontmatter could not be parsed — copy verbatim to avoid data loss
        out = content
    if target.is_file():
        try:
            if target.read_text(encoding="utf-8") == out:
                return False
        except OSError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(out, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Base writer
# ---------------------------------------------------------------------------


class _BaseAgentsWriter(AbstractSyncWriter):
    """Common sync logic for non-Copilot agent writers (directory-level symlinks)."""

    concern = SyncConcern.AGENTS
    _target_rel: str  # e.g. ".claude/agents"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.agents_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents source detected",
            )

        source_dir = project_root / data.agents_source
        if not source_dir.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {data.agents_source}",
            )
        if not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source path is not a directory: {data.agents_source}",
            )

        target_dir = project_root / self._target_rel

        # For copy strategy, explicitly guard against following a symlinked target
        # directory — copies would land in the symlink's destination, potentially
        # outside the project root.
        if data.agents_strategy == "copy" and target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} is a symlinked directory. "
                        "Refusing to copy agents into a symlink target. "
                        "Remove the symlink or re-run with --force to replace it."
                    ),
                )
            if not dry_run:
                target_dir.unlink()
                logger.info("agents.symlink_replaced", target=str(target_dir))

        # Existing real directory — may need to error, proceed, or back up.
        # Ownership is determined by the ``.crossby-managed`` marker file
        # crossby drops on every write-bearing sync. The previous heuristic
        # ("looks like .md files only") couldn't distinguish a crossby-managed
        # directory from a hand-curated one, so a user's native ``.claude/agents``
        # could be wiped on a copy/translate run. The marker fixes that: unmarked
        # directories are user-owned and require ``--force`` (which writes the
        # marker as a side effect of the backup-and-replace path).
        dir_was_cleared = False
        if target_dir.is_dir() and not target_dir.is_symlink():
            if not force:
                contents = [f for f in target_dir.iterdir() if f.name != MANAGED_MARKER_NAME]
                if contents and not has_managed_marker(target_dir):
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="error",
                        message=(
                            f"{self._target_rel} exists as a directory but is not "
                            "crossby-managed (no .crossby-managed marker). "
                            f"Migrate its contents to {data.agents_source} first, "
                            "or use --force to back it up and replace it."
                        ),
                    )
                # Managed (or empty) directory: re-sync via the configured
                # strategy so subsequent runs preserve translate/copy semantics.
                # Symlink strategy can't symlink-over an existing real dir, so
                # it falls back to copy (matching pre-marker behavior).
                if data.agents_strategy == "translate":
                    return self._sync_translate(source_dir, target_dir, dry_run=dry_run)
                return self._sync_copy(source_dir, target_dir, dry_run=dry_run)
            else:
                dir_was_cleared = True
                if not dry_run:
                    bak = backup_path(target_dir)
                    shutil.copytree(str(target_dir), str(bak))
                    shutil.rmtree(str(target_dir))
                    logger.info("agents.dir_backed_up", original=str(target_dir), backup=str(bak))

        if data.agents_strategy == "translate":
            return self._sync_translate(source_dir, target_dir, dry_run=dry_run)

        if data.agents_strategy == "copy":
            return self._sync_copy(source_dir, target_dir, dry_run=dry_run)

        return self._sync_symlink(
            source_dir, target_dir, dry_run=dry_run, force=force, dir_was_cleared=dir_was_cleared
        )

    def _sync_symlink(
        self,
        source_dir: Path,
        target_dir: Path,
        *,
        dry_run: bool,
        force: bool,
        dir_was_cleared: bool = False,
    ) -> SyncResult:
        # When force cleared a real directory (dry_run skips the removal), still
        # report "created" — the symlink would succeed once the directory is gone.
        if dir_was_cleared and dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="created",
                file_path=target_dir,
                message="(dry-run: would replace existing directory)",
            )
        try:
            created = create_symlink(source_dir, target_dir, force=force, dry_run=dry_run)
        except OSError as exc:
            logger.warning("agents.symlink_failed", tool=str(self.tool_id), error=str(exc))
            # Fallback: copy. Mark the directory so the next sync recognizes
            # its own output and doesn't refuse it as "not crossby-managed".
            try:
                if not dry_run:
                    _copy_all_agents(source_dir, target_dir, str(self.tool_id))
                    write_managed_marker(target_dir)
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="created",
                    file_path=target_dir,
                    message="copy (symlink failed)",
                )
            except Exception as copy_exc:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=str(copy_exc),
                )

        if not created:
            if target_dir.is_symlink() and target_dir.resolve() != source_dir.resolve():
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    file_path=target_dir,
                    message="symlink points to a different location; use --force to replace",
                )
            if target_dir.exists() and not target_dir.is_symlink():
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    file_path=target_dir,
                    message=(
                        f"{self._target_rel} already exists as a regular file or directory; "
                        "use --force to replace with a symlink"
                    ),
                )
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already linked",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="created",
            file_path=target_dir,
        )

    def _sync_copy(self, source_dir: Path, target_dir: Path, *, dry_run: bool) -> SyncResult:
        target_existed = target_dir.is_dir()
        action: Literal["created", "updated"] = "updated" if target_existed else "created"
        if dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=target_dir,
                message="copy (dry-run)",
            )
        wrote = _copy_all_agents(source_dir, target_dir, str(self.tool_id))
        write_managed_marker(target_dir)
        if not wrote and target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already copied",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=target_dir,
        )

    def _sync_translate(self, source_dir: Path, target_dir: Path, *, dry_run: bool) -> SyncResult:
        """Per-file translation with lossy-field annotation.

        Delegates parse/emit to :mod:`crossby.subagents.api.convert` — that's
        the canonical cross-tool subagent translator (rich `SubagentIR`,
        per-tool parsers/emitters, structured `ConversionWarning`s). The
        sync writer adds the directory-level concerns the one-shot CLI
        doesn't care about: file iteration, hash-based idempotency, stale
        cleanup, and emitting any lossy/dropped warnings as an in-file
        ``<!-- crossby:manual-fix -->`` block so the user sees the lossy
        edge inside the artifact, not just on the terminal.

        Source tool is inferred from the source-dir path against the known
        per-tool paths (``.claude/agents`` → claude, etc.); falls back to
        ``claude`` when the source is a tool-neutral directory like
        ``.crossby/agents``.
        """
        from_tool = _infer_source_tool(source_dir)

        source_files = sorted(source_dir.glob("*.md"))
        target_existed = target_dir.is_dir()
        action: Literal["created", "updated"] = "updated" if target_existed else "created"

        if not source_files and not target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="no agents to translate",
            )

        target_tool = str(self.tool_id)

        def _target_name(src: Path) -> str:
            """Drop a Copilot ``.agent.md`` suffix when crossing to a markdown
            target that uses plain ``.md`` filenames. Without this, a Copilot
            source ``foo.agent.md`` would land as ``foo.agent.md`` under e.g.
            ``.claude/agents/`` where Claude expects ``foo.md``."""
            if from_tool == "copilot" and src.name.endswith(".agent.md"):
                return src.name.removesuffix(".agent.md") + ".md"
            return src.name

        if dry_run:
            # Render in memory so manual-fix blocks are visible to plan
            # summarization. Without this pass, --plan would under-report the
            # manual review surface for translate dry-runs.
            from crossby.sync.manual_fix import has_manual_fix_block

            manual_fix_count = 0
            for src in source_files:
                rendered = _translate_markdown_agent(
                    content=src.read_text(encoding="utf-8"),
                    from_tool=from_tool,
                    to_tool=target_tool,
                    source_path=src,
                )
                if has_manual_fix_block(rendered):
                    manual_fix_count += 1
            message = (
                f"translated (dry-run, {manual_fix_count} manual-fix)"
                if manual_fix_count
                else "translated (dry-run)"
            )
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=target_dir,
                message=message,
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        write_managed_marker(target_dir)
        wanted = {_target_name(src) for src in source_files}
        removed_any = False
        for existing in target_dir.glob("*.md"):
            if existing.name not in wanted:
                existing.unlink()
                logger.info("agents.stale_removed", path=str(existing))
                removed_any = True

        wrote_any = False
        for src in source_files:
            rendered = _translate_markdown_agent(
                content=src.read_text(encoding="utf-8"),
                from_tool=from_tool,
                to_tool=target_tool,
                source_path=src,
            )
            target_file = target_dir / _target_name(src)
            if target_file.is_file() and target_file.read_text(encoding="utf-8") == rendered:
                continue
            target_file.write_text(rendered, encoding="utf-8")
            wrote_any = True

        if not wrote_any and not removed_any and target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already translated",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=target_dir,
            message="translated",
        )


def _copy_all_agents(source_dir: Path, target_dir: Path, tool_id: str) -> bool:
    """Copy all .md agent files from source to target, translating tool names.

    Returns True when at least one file was written or rewritten, False
    when every file was already up to date (idempotent re-run).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    for src in source_dir.glob("*.md"):
        if _copy_agent_file(src, target_dir / src.name, tool_id):
            wrote_any = True
    return wrote_any


# ---------------------------------------------------------------------------
# Concrete writers
# ---------------------------------------------------------------------------


class ClaudeAgentsWriter(_BaseAgentsWriter):
    """Sync agents → .claude/agents/"""

    tool_id = AIToolID.CLAUDE
    _target_rel = ".claude/agents"


class CursorAgentsWriter(_BaseAgentsWriter):
    """Sync agents → .cursor/agents/"""

    tool_id = AIToolID.CURSOR
    _target_rel = ".cursor/agents"


class AntigravityCLIAgentsWriter(_BaseAgentsWriter):
    """Sync agents → .agents/agents/"""

    tool_id = AIToolID.ANTIGRAVITY_CLI
    _target_rel = ".agents/agents"


class CodexAgentsWriter(AbstractSyncWriter):
    """Sync agents → .codex/agents/<name>.toml.

    Codex agents use a TOML schema (``name``, ``description``,
    ``developer_instructions`` plus optional ``model``,
    ``model_reasoning_effort``, ``sandbox_mode``). When the source is a
    different tool (Claude/Cursor/Copilot all use markdown +
    YAML frontmatter), we translate per file via
    :mod:`crossby.sync.agent_models`. Lossy fields (``permissionMode:
    plan``, ``allowed-tools``, etc.) become a ``crossby:manual-fix``
    block at the end of the rendered ``developer_instructions``.

    Idempotent: identical rendered TOML produces ``action="skipped"``.
    Stale cleanup: ``.toml`` files whose source ``.md`` is gone are
    removed, matching the behaviour of CopilotAgentsWriter for
    ``.agent.md`` files.
    """

    tool_id = AIToolID.CODEX
    concern = SyncConcern.AGENTS
    _target_rel = ".codex/agents"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.agents_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents source detected",
            )

        source_dir = project_root / data.agents_source
        if not source_dir.exists() or not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {data.agents_source}",
            )

        _warn_legacy_codex_agents_path(project_root)

        target_dir = project_root / self._target_rel

        if target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} exists as a symlink. "
                        "Remove it or rerun with --force to replace it."
                    ),
                )
            if not dry_run:
                target_dir.unlink()
                logger.info("agents.symlink_replaced", path=str(target_dir))

        # When a real directory exists, ownership is established by the
        # ``.crossby-managed`` marker. ``.toml`` filenames alone aren't a
        # reliable signal since a user could maintain ``.codex/agents/*.toml``
        # by hand; only the marker is authoritative.
        if target_dir.is_dir():
            contents = [f for f in target_dir.iterdir() if f.name != MANAGED_MARKER_NAME]
            unmarked = contents and not has_managed_marker(target_dir)
            if unmarked and not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} exists as a directory but is not "
                        "crossby-managed (no .crossby-managed marker). "
                        "Migrate it first or rerun with --force."
                    ),
                )
            if unmarked and force and not dry_run:
                bak = backup_path(target_dir)
                shutil.copytree(str(target_dir), str(bak))
                shutil.rmtree(str(target_dir))
                logger.info(
                    "agents.dir_backed_up",
                    original=str(target_dir),
                    backup=str(bak),
                )

        return self._translate_all(source_dir, target_dir, dry_run=dry_run)

    def _source_files(self, source_dir: Path) -> list[Path]:
        # Source can be either Codex TOML (round-trip) or markdown agents.
        return sorted(
            [p for p in source_dir.glob("*.md") if p.is_file()]
            + [p for p in source_dir.glob("*.toml") if p.is_file()]
        )

    def _render_for_target(self, source: Path) -> tuple[str, str]:
        """Translate one source agent file to ``(agent.toml, config_fragment)``.

        Per-file source-tool inference: ``.toml`` is Codex (round-trip),
        anything else is parsed as Claude markdown via subagents.api. The
        config_fragment is the ``[agents.<name>]`` block PR #46's Codex
        emitter produces — used by :meth:`_write_codex_config_fragment` to
        register the agent globally in ``~/.codex/config.toml``.
        """
        from_tool = "codex" if source.suffix == ".toml" else _infer_source_tool(source.parent)
        return _translate_codex_agent(
            content=source.read_text(encoding="utf-8"),
            from_tool=from_tool,
            source_path=source,
        )

    def _translate_all(self, source_dir: Path, target_dir: Path, *, dry_run: bool) -> SyncResult:
        sources = self._source_files(source_dir)
        if not sources:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="no agents to translate",
            )

        target_existed = target_dir.is_dir()
        action: Literal["created", "updated"] = "updated" if target_existed else "created"
        wrote_any = False
        skipped_all = True

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            write_managed_marker(target_dir)

        # Stale cleanup — remove .toml outputs whose source is gone.
        if not dry_run and target_dir.is_dir():
            wanted = {f"{src.stem}.toml" for src in sources}
            for existing in target_dir.glob("*.toml"):
                if existing.name not in wanted:
                    existing.unlink()
                    logger.info("agents.stale_removed", path=str(existing))

        for src in sources:
            agent_toml, _config_fragment = self._render_for_target(src)
            dest = target_dir / f"{src.stem}.toml"
            if dest.is_file():
                try:
                    if (
                        hashlib.sha256(dest.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                        == hashlib.sha256(agent_toml.encode("utf-8")).hexdigest()
                    ):
                        continue
                except OSError:
                    pass
            skipped_all = False
            if not dry_run:
                dest.write_text(agent_toml, encoding="utf-8")
            wrote_any = True

        if skipped_all and target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already translated",
            )

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=target_dir,
            message="translated to TOML" + (" (dry-run)" if dry_run and wrote_any else ""),
        )


class CopilotAgentsWriter(AbstractSyncWriter):
    """Sync agents → .github/agents/ using file-level symlinks (.agent.md extension).

    Copilot requires the ``.agent.md`` extension so we create per-file symlinks
    rather than a directory symlink.  Each sync run also removes stale
    ``.agent.md`` symlinks whose source ``.md`` file no longer exists.
    """

    tool_id = AIToolID.COPILOT
    concern = SyncConcern.AGENTS
    _target_rel = ".github/agents"

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        if data.agents_source is None:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no agents source detected",
            )

        source_dir = project_root / data.agents_source
        if not source_dir.exists():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source directory not found: {data.agents_source}",
            )
        if not source_dir.is_dir():
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="error",
                message=f"source path is not a directory: {data.agents_source}",
            )

        target_dir = project_root / self._target_rel

        # If the target exists as a symlink, error by default to avoid writing into the
        # symlink target (which may be outside the project). With --force, replace the
        # symlink with a real directory under the project root.
        if target_dir.is_symlink():
            if not force:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    message=(
                        f"{self._target_rel} exists as a symlink. Remove it or use --force "
                        "to replace it with a real directory before syncing agents."
                    ),
                )
            if not dry_run:
                target_dir.unlink()
                logger.info("agents.symlink_replaced", path=str(target_dir))
        # For Copilot, the target is always a real directory containing per-file symlinks.
        # Ownership is decided by the ``.crossby-managed`` marker — a directory full
        # of ``.agent.md`` files is otherwise indistinguishable from a hand-curated
        # native Copilot agents dir, so the older "all files end in .agent.md = managed"
        # heuristic could wipe user content on stale cleanup.
        elif target_dir.is_dir():
            contents = [f for f in target_dir.iterdir() if f.name != MANAGED_MARKER_NAME]
            if contents and not has_managed_marker(target_dir):
                if not force:
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="error",
                        message=(
                            f"{self._target_rel} exists as a directory but is not "
                            "crossby-managed (no .crossby-managed marker). "
                            f"Migrate its contents to {data.agents_source} first, "
                            "or use --force to back it up and replace it."
                        ),
                    )
                # force=True: back up and clear the directory before re-syncing
                if not dry_run:
                    bak = backup_path(target_dir)
                    shutil.copytree(str(target_dir), str(bak))
                    shutil.rmtree(str(target_dir))
                    logger.info(
                        "agents.dir_backed_up",
                        original=str(target_dir),
                        backup=str(bak),
                    )

        if data.agents_strategy == "copy":
            return self._sync_copy(source_dir, target_dir, dry_run=dry_run)

        if data.agents_strategy == "translate":
            return self._sync_translate(source_dir, target_dir, dry_run=dry_run)

        return self._sync_symlinks(source_dir, target_dir, dry_run=dry_run, force=force)

    def _sync_translate(self, source_dir: Path, target_dir: Path, *, dry_run: bool) -> SyncResult:
        """Per-file translate to Copilot ``.agent.md`` format.

        Parallel to :meth:`_BaseAgentsWriter._sync_translate` but emits the
        Copilot-specific ``<stem>.agent.md`` filename and renders through the
        Copilot emitter so target-specific lossy fields (Claude
        ``permissionMode: plan``, ``allowed-tools``, etc.) become a
        ``crossby:manual-fix`` block instead of being silently dropped.

        Before this path existed, ``--strategy translate`` on the Copilot
        writer fell through to the symlink path, producing source-shape
        symlinks instead of rendered Copilot frontmatter.
        """
        from_tool = _infer_source_tool(source_dir)
        source_files = sorted(source_dir.glob("*.md"))
        target_existed = target_dir.is_dir()
        action: Literal["created", "updated"] = "updated" if target_existed else "created"

        if not source_files and not target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="no agents to translate",
            )

        def _target_name(src: Path) -> str:
            # Copilot's native ``.agent.md`` survives a Copilot→Copilot
            # round-trip; everything else gets the ``.agent.md`` suffix.
            if src.name.endswith(".agent.md"):
                return src.name
            return f"{src.stem}.agent.md"

        if dry_run:
            from crossby.sync.manual_fix import has_manual_fix_block

            manual_fix_count = 0
            for src in source_files:
                rendered = _translate_markdown_agent(
                    content=src.read_text(encoding="utf-8"),
                    from_tool=from_tool,
                    to_tool="copilot",
                    source_path=src,
                )
                if has_manual_fix_block(rendered):
                    manual_fix_count += 1
            message = (
                f"translated (dry-run, {manual_fix_count} manual-fix)"
                if manual_fix_count
                else "translated (dry-run)"
            )
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=target_dir,
                message=message,
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        write_managed_marker(target_dir)
        wanted = {_target_name(src) for src in source_files}
        removed_any = False
        # Stale cleanup: managed *.agent.md outputs whose source is gone.
        for existing in target_dir.glob("*.agent.md"):
            if existing.name not in wanted:
                if existing.is_symlink():
                    os.unlink(existing)
                else:
                    existing.unlink()
                logger.info("agents.stale_removed", path=str(existing))
                removed_any = True

        wrote_any = False
        for src in source_files:
            rendered = _translate_markdown_agent(
                content=src.read_text(encoding="utf-8"),
                from_tool=from_tool,
                to_tool="copilot",
                source_path=src,
            )
            target_file = target_dir / _target_name(src)
            if target_file.is_symlink():
                # Translate replaces symlinks (from a prior symlink-strategy run)
                # with regular files so the rendered Copilot frontmatter persists.
                os.unlink(target_file)
            if target_file.is_file() and target_file.read_text(encoding="utf-8") == rendered:
                continue
            target_file.write_text(rendered, encoding="utf-8")
            wrote_any = True

        if not wrote_any and not removed_any and target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already translated",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=target_dir,
            message="translated",
        )

    def _sync_symlinks(
        self, source_dir: Path, target_dir: Path, *, dry_run: bool, force: bool
    ) -> SyncResult:
        """Create/update per-file .agent.md symlinks; clean up stale ones."""
        dir_newly_created = False
        if not dry_run and not target_dir.is_dir():
            target_dir.mkdir(parents=True, exist_ok=True)
            dir_newly_created = True
        if not dry_run and target_dir.is_dir():
            write_managed_marker(target_dir)

        source_stems = {f.stem for f in source_dir.glob("*.md")}

        # Stale cleanup: remove managed *.agent.md outputs whose source is gone.
        # The .agent.md extension is crossby-specific; both symlinks and regular
        # files (copy-fallback outputs) are treated as managed and eligible for removal.
        if not dry_run and target_dir.is_dir():
            for link in list(target_dir.glob("*.agent.md")):
                original_stem = link.name.removesuffix(".agent.md")
                if original_stem not in source_stems:
                    os.unlink(link)
                    logger.info("agents.stale_removed", link=str(link))

        # Create/update symlinks for each source file
        created_count = 0
        for src in source_dir.glob("*.md"):
            link = target_dir / f"{src.stem}.agent.md"
            try:
                if create_symlink(src, link, force=force, dry_run=dry_run):
                    created_count += 1
                elif link.is_symlink() and link.resolve() != src.resolve():
                    return SyncResult(
                        tool_id=self.tool_id,
                        concern=self.concern,
                        action="error",
                        file_path=link,
                        message=(
                            f"{link.name} symlink points to a different location; "
                            "use --force to replace"
                        ),
                    )
                elif link.exists() and not link.is_symlink():
                    # Regular file at the link path — treat as a managed copy-fallback
                    # output (.agent.md is crossby-specific) and keep it up to date.
                    if not dry_run:
                        _copy_agent_file(src, link, "copilot")
                    created_count += 1
            except OSError:
                # Fallback: copy the file
                if not dry_run:
                    _copy_agent_file(src, link, "copilot")
                    created_count += 1

        if created_count == 0 and not dir_newly_created:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already linked",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action="created",
            file_path=target_dir,
        )

    def _sync_copy(self, source_dir: Path, target_dir: Path, *, dry_run: bool) -> SyncResult:
        target_existed = target_dir.is_dir()
        action: Literal["created", "updated"] = "updated" if target_existed else "created"
        if dry_run:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action=action,
                file_path=target_dir,
                message="copy (dry-run)",
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        write_managed_marker(target_dir)
        wrote_any = False
        for src in source_dir.glob("*.md"):
            dest = target_dir / f"{src.stem}.agent.md"
            if _copy_agent_file(src, dest, "copilot"):
                wrote_any = True
        if not wrote_any and target_existed:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=target_dir,
                message="already copied",
            )
        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=target_dir,
        )
