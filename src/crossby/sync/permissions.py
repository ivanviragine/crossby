"""Permission sync writers for Claude, Cursor, and Gemini.

Contains the core allowlist write/check logic.  The legacy module-level
functions in ``config/claude_allowlist.py`` and ``config/cursor_allowlist.py``
are backward-compatible shims that delegate here.

Gemini uses a TOML-based Policy Engine instead of JSON allowlists.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Literal

import structlog

from crossby.config.allowlist_util import configure_json_allowlist
from crossby.models.ai import AIToolID
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncData, SyncResult

logger = structlog.get_logger()

# Module-level global Cursor config path — monkeypatchable in tests.
_GLOBAL_CURSOR_CONFIG_PATH = Path.home() / ".cursor" / "cli-config.json"


# ---------------------------------------------------------------------------
# Pattern translators (public — re-exported by config shims)
# ---------------------------------------------------------------------------


def canonical_to_claude(pattern: str) -> str:
    """Convert a canonical command pattern to Claude Code allowlist syntax.

    Canonical patterns use ``"cmd:args"`` notation (colon-separated).
    Claude expects ``"Bash(cmd:args)"`` — the command string wrapped in ``Bash(…)``.

    Examples::

        "myapp:*"                 → "Bash(myapp:*)"
        "./scripts/check.sh:*"    → "Bash(./scripts/check.sh:*)"
        "./scripts/check.sh"      → "Bash(./scripts/check.sh)"
    """
    return f"Bash({pattern})"


def canonical_to_cursor(pattern: str) -> str:
    """Convert a canonical command pattern to Cursor CLI allowlist syntax.

    Canonical patterns use ``"cmd:args"`` notation (colon-separated).
    Cursor expects ``"Shell(cmd:args)"`` — the command string wrapped in
    ``Shell(…)``.

    Examples::

        "myapp:*"                 → "Shell(myapp:*)"
        "./scripts/check.sh:*"    → "Shell(./scripts/check.sh:*)"
    """
    return f"Shell({pattern})"


def _cursor_config_path(project_root: Path | None) -> Path:
    """Return the Cursor CLI config file path for the given scope."""
    if project_root is not None:
        return project_root / ".cursor" / "cli.json"
    return _GLOBAL_CURSOR_CONFIG_PATH


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudePermissionWriter(AbstractSyncWriter):
    """Sync ``permissions.allowed_commands`` → ``.claude/settings.json``."""

    tool_id = AIToolID.CLAUDE
    concern = SyncConcern.PERMISSIONS

    @staticmethod
    def check(project_root: Path, patterns: list[str]) -> bool:
        """Return True if ALL patterns are present in .claude/settings.json."""
        settings_path = project_root / ".claude" / "settings.json"
        if not settings_path.is_file():
            return False
        with contextlib.suppress(json.JSONDecodeError, OSError):
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                permissions = raw.get("permissions")
                if not isinstance(permissions, dict):
                    return False
                allow = permissions.get("allow", [])
                if not isinstance(allow, list):
                    return False
                claude_patterns = [canonical_to_claude(p) for p in patterns]
                return all(cp in allow for cp in claude_patterns)
        return False

    @staticmethod
    def write(project_root: Path, patterns: list[str]) -> tuple[str, str | None]:
        """Add patterns to .claude/settings.json. Idempotent, non-destructive.

        Returns ``(action, error_message)`` so callers can surface parse
        failures without overwriting a malformed file.
        """
        settings_path = project_root / ".claude" / "settings.json"
        return configure_json_allowlist(
            settings_path,
            patterns,
            pattern_converter=canonical_to_claude,
            log_event="claude_allowlist.configured",
        )

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        patterns = data.allowed_commands
        if not patterns:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no allowed_commands detected",
            )

        settings_path = project_root / ".claude" / "settings.json"

        if self.check(project_root, patterns):
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=settings_path,
                message="already configured",
            )

        action: Literal["created", "updated"] = (
            "created" if not settings_path.is_file() else "updated"
        )

        if not dry_run:
            written_action, error = self.write(project_root, patterns)
            if error is not None:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    file_path=settings_path,
                    message=error,
                )
            if written_action == "created":
                action = "created"
            elif written_action == "updated":
                action = "updated"

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=settings_path,
        )


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


class CursorPermissionWriter(AbstractSyncWriter):
    """Sync ``permissions.allowed_commands`` → ``.cursor/cli.json`` (or global).

    Args:
        scope: ``"project"`` (default) writes to
            ``<project_root>/.cursor/cli.json``.  ``"global"`` writes to
            ``~/.cursor/cli-config.json``.  The ``run_sync()`` orchestrator
            always uses project scope; global scope is available for direct
            callers.
    """

    tool_id = AIToolID.CURSOR
    concern = SyncConcern.PERMISSIONS

    def __init__(self, scope: Literal["project", "global"] = "project") -> None:
        self.scope = scope

    @staticmethod
    def check(
        project_root: Path | None = None,
        patterns: list[str] | None = None,
    ) -> bool:
        """Return True if ALL patterns are in the Cursor allowlist.

        When ``project_root`` is given, checks the per-project config.
        Otherwise checks the global config.
        """
        if not patterns:
            return True
        config_file = _cursor_config_path(project_root)
        if not config_file.is_file():
            return False
        with contextlib.suppress(json.JSONDecodeError, OSError):
            raw = json.loads(config_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                permissions = raw.get("permissions")
                if not isinstance(permissions, dict):
                    return False
                allow = permissions.get("allow", [])
                if not isinstance(allow, list):
                    return False
                cursor_patterns = [canonical_to_cursor(p) for p in patterns]
                return all(cp in allow for cp in cursor_patterns)
        return False

    @staticmethod
    def write(
        project_root: Path | None = None,
        patterns: list[str] | None = None,
    ) -> tuple[str, str | None]:
        """Add patterns to the Cursor CLI allowlist. Idempotent.

        Returns ``(action, error_message)``; ``error_message`` is set when
        the existing file is malformed (in which case nothing is written).
        """
        return configure_json_allowlist(
            _cursor_config_path(project_root),
            patterns or [],
            pattern_converter=canonical_to_cursor,
            log_event="cursor_allowlist.configured",
        )

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        patterns = data.allowed_commands
        if not patterns:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no allowed_commands detected",
            )

        scope_root = project_root if self.scope == "project" else None
        config_path = _cursor_config_path(scope_root)

        if self.check(scope_root, patterns):
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=config_path,
                message="already configured",
            )

        action: Literal["created", "updated"] = (
            "created" if not config_path.is_file() else "updated"
        )

        if not dry_run:
            written_action, error = self.write(scope_root, patterns)
            if error is not None:
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="error",
                    file_path=config_path,
                    message=error,
                )
            if written_action == "created":
                action = "created"
            elif written_action == "updated":
                action = "updated"

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=config_path,
        )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

_GEMINI_POLICY_PRIORITY = 100


def _canonical_to_gemini_rule(pattern: str) -> str:
    """Convert a canonical ``"cmd:args"`` pattern to a TOML ``[[rule]]`` block.

    Only the binary name (prefix before the first ``:``) is used as the
    ``commandPrefix`` — Gemini's Policy Engine matches by command prefix,
    not by argument globs.
    """
    binary = pattern.split(":", 1)[0]
    escaped = _escape_toml_value(binary)
    return (
        "[[rule]]\n"
        f'toolName = "run_shell_command"\n'
        f'commandPrefix = "{escaped}"\n'
        f'decision = "allow"\n'
        f"priority = {_GEMINI_POLICY_PRIORITY}\n"
    )


def _escape_toml_value(binary: str) -> str:
    """Escape a binary name for use in a TOML quoted string."""
    return binary.replace("\\", "\\\\").replace('"', '\\"')


def _gemini_policy_path(project_root: Path) -> Path:
    return project_root / ".gemini" / "policies" / "crossby.toml"


class GeminiPermissionWriter(AbstractSyncWriter):
    """Sync ``permissions.allowed_commands`` → ``.gemini/policies/crossby.toml``.

    Gemini CLI uses the Policy Engine (TOML files) instead of JSON allowlists
    or ``--allowed-tools`` CLI flags.  Each canonical command pattern becomes a
    ``[[rule]]`` block with ``commandPrefix`` set to the binary name.

    The managed policy file is removed when no commands remain, so reused
    workspaces do not keep stale allow rules.
    """

    tool_id = AIToolID.GEMINI
    concern = SyncConcern.PERMISSIONS

    @staticmethod
    def check(project_root: Path, patterns: list[str]) -> bool:
        """Return True if ALL patterns are present in the managed policy file."""
        valid = [p for p in patterns if p.strip()]
        if not valid:
            return not _gemini_policy_path(project_root).exists()
        policy_file = _gemini_policy_path(project_root)
        if not policy_file.is_file():
            return False
        with contextlib.suppress(OSError):
            content = policy_file.read_text(encoding="utf-8")
            for pat in valid:
                binary = pat.split(":", 1)[0]
                escaped = _escape_toml_value(binary)
                if f'commandPrefix = "{escaped}"' not in content:
                    return False
            return True
        return False

    @staticmethod
    def write(project_root: Path, patterns: list[str]) -> None:
        """Write ``.gemini/policies/crossby.toml``. Overwrites previous content.

        Removes the policy file when *patterns* is empty to prevent stale rules.
        """
        policy_file = _gemini_policy_path(project_root)

        valid = [p for p in patterns if p.strip()]
        if not valid:
            if policy_file.exists():
                policy_file.unlink()
                logger.info("gemini_policy.removed", path=str(policy_file))
            return

        rules = [_canonical_to_gemini_rule(p) for p in valid]
        content = "\n".join(rules)

        policy_file.parent.mkdir(parents=True, exist_ok=True)
        policy_file.write_text(content, encoding="utf-8")
        logger.info("gemini_policy.written", path=str(policy_file), rules=len(rules))

    def sync(
        self,
        data: SyncData,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        patterns = data.allowed_commands
        policy_file = _gemini_policy_path(project_root)

        if not patterns:
            if not policy_file.is_file():
                return SyncResult(
                    tool_id=self.tool_id,
                    concern=self.concern,
                    action="skipped",
                    message="no allowed_commands detected",
                )
            if not dry_run:
                self.write(project_root, [])
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="updated",
                file_path=policy_file,
                message="removed stale policy",
            )

        if self.check(project_root, patterns):
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                file_path=policy_file,
                message="already configured",
            )

        action: Literal["created", "updated"] = (
            "created" if not policy_file.is_file() else "updated"
        )

        if not dry_run:
            self.write(project_root, patterns)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=policy_file,
        )
