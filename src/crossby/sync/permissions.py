"""Permission sync writers for Claude and Cursor.

Contains the core allowlist write/check logic.  The legacy module-level
functions in ``config/claude_allowlist.py`` and ``config/cursor_allowlist.py``
are backward-compatible shims that delegate here.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Literal

import structlog

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig
from crossby.sync.base import AbstractSyncWriter, SyncConcern, SyncResult
from crossby.sync.json_utils import read_json_file, write_json_file

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
    def write(project_root: Path, patterns: list[str]) -> None:
        """Add patterns to .claude/settings.json. Idempotent, non-destructive."""
        settings_path = project_root / ".claude" / "settings.json"

        data, _error, _was_new = read_json_file(settings_path)
        existing: dict[str, object] = data if data is not None else {}

        permissions = existing.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            permissions = {}
            existing["permissions"] = permissions

        allow_list = permissions.setdefault("allow", [])
        if not isinstance(allow_list, list):
            allow_list = []
            permissions["allow"] = allow_list

        changed = False
        for pat in [canonical_to_claude(p) for p in patterns]:
            if pat not in allow_list:
                allow_list.append(pat)
                changed = True

        if not changed:
            return

        write_json_file(settings_path, existing)
        logger.info("claude_allowlist.configured", path=str(settings_path))

    def sync(
        self,
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        patterns = config.permissions.allowed_commands
        if not patterns:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no allowed_commands configured",
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
            self.write(project_root, patterns)

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
    ) -> None:
        """Add patterns to the Cursor CLI allowlist. Idempotent."""
        if not patterns:
            return

        config_file = _cursor_config_path(project_root)

        data, _error, _was_new = read_json_file(config_file)
        existing: dict[str, object] = data if data is not None else {}

        permissions = existing.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            permissions = {}
            existing["permissions"] = permissions

        allow_list = permissions.setdefault("allow", [])
        if not isinstance(allow_list, list):
            allow_list = []
            permissions["allow"] = allow_list

        changed = False
        for pat in [canonical_to_cursor(p) for p in patterns]:
            if pat not in allow_list:
                allow_list.append(pat)
                changed = True

        if not changed:
            return

        write_json_file(config_file, existing)
        logger.info("cursor_allowlist.configured", path=str(config_file))

    def sync(
        self,
        config: CrossbyConfig,
        project_root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncResult:
        patterns = config.permissions.allowed_commands
        if not patterns:
            return SyncResult(
                tool_id=self.tool_id,
                concern=self.concern,
                action="skipped",
                message="no allowed_commands configured",
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
            self.write(scope_root, patterns)

        return SyncResult(
            tool_id=self.tool_id,
            concern=self.concern,
            action=action,
            file_path=config_path,
        )
