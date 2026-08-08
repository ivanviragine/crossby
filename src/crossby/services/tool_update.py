"""Update service for installed AI tools — ``crossby tools update`` backend.

Small and testable, with no CLI/UI concerns. Every function is
**failure-safe**: an updater that crashes, times out, or is missing, and a
version probe that can't read a ``--version``, all become report data rather
than an exception. The CLI runs the selected tools sequentially and renders the
:class:`UpdateResult` rows; nothing here prints.
"""

from __future__ import annotations

import subprocess

import structlog
from pydantic import BaseModel

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolCapabilities, AIToolID, AIToolType
from crossby.utils import process
from crossby.utils.versioning import detect_binary_version

logger = structlog.get_logger()

# Generous ceiling so a slow download (npm/brew fetching a release) is not cut
# off, while still bounding an updater that blocks on its own prompt — the v1
# safety net in place of injected non-interactive flags.
UPDATE_TIMEOUT_S = 600

# Lines of combined stdout/stderr kept for the report's failure detail.
OUTPUT_TAIL_LINES = 20


class UpdateResult(BaseModel):
    """Outcome of running one tool's updater. Never carries an exception."""

    tool_id: AIToolID
    display_name: str
    command: tuple[str, ...]
    success: bool
    exit_code: int | None
    output_tail: str
    before_version: str | None
    after_version: str | None
    unchanged: bool = False
    """Success, but the probed version did not change — a known footgun when an
    updater touches a different install than the one on PATH. Informational; it
    never flips ``success``."""
    error: str | None = None
    """Always set on failure (``success is False``) so the report never shows a
    bare ✗ with no reason."""


def updatable_tools() -> list[AIToolID]:
    """Installed tools that crossby can update, in a stable registry order.

    A tool qualifies only when it is installed **and** its capability declares
    both an ``update_command`` and ``tool_type == TERMINAL``. The terminal
    requirement is a defensive structural invariant: a future GUI adapter that
    *accidentally* set an ``update_command`` is still never offered, independent
    of the ``None`` default convention.
    """
    installed = set(AbstractAITool.detect_installed())
    result: list[AIToolID] = []
    for tool_id in AbstractAITool.available_tools():
        if tool_id not in installed:
            continue
        if _is_updatable(AbstractAITool.get(tool_id).capabilities()):
            result.append(tool_id)
    return result


def probe_version(binary: str) -> str | None:
    """Best-effort ``"major.minor.patch"`` string for *binary*, or ``None``.

    Thin wrapper over :func:`crossby.utils.versioning.detect_binary_version`;
    never raises. ``None`` means the version is unknown (missing binary,
    unparseable ``--version``, timeout, …) — always informational, never a
    failure.
    """
    version = detect_binary_version(binary)
    if version is None:
        return None
    return ".".join(str(part) for part in version)


def run_update(tool_id: AIToolID) -> UpdateResult:
    """Run one tool's static updater and return a report row. Never raises.

    ``success`` is decided **only** by the updater's exit code; a missing or
    unchanged version is informational. Raises :class:`ValueError` only for
    misuse — a tool whose ``update_command`` is ``None`` or whose ``tool_type``
    is not ``TERMINAL`` (the CLI only ever passes updatable tools).
    """
    caps = AbstractAITool.get(tool_id).capabilities()
    if not _is_updatable(caps):
        raise ValueError(f"{tool_id} is not updatable (no update_command or not a terminal tool)")
    command = caps.update_command
    assert command is not None  # narrowed by _is_updatable; for the type checker

    binary = caps.binary
    before_version = probe_version(binary)

    exit_code: int | None
    output_tail = ""
    error: str | None = None
    try:
        completed = process.run(list(command), check=False, capture=True, timeout=UPDATE_TIMEOUT_S)
        exit_code = completed.returncode
        output_tail = _tail(_combine(completed.stdout, completed.stderr))
        if exit_code != 0:
            # Populate error even on an empty non-zero exit so no bare ✗ ships.
            error = f"exited with code {exit_code}"
    except process.CommandError as exc:
        # Binary-not-found (127) or another failure — preserve the return code.
        exit_code = exc.returncode
        output_tail = _tail(exc.stderr)
        error = str(exc)
    except subprocess.TimeoutExpired as exc:
        # Partial output may be bytes or None — decode/guard before tailing.
        exit_code = None
        output_tail = _tail(_combine(exc.stdout, exc.stderr))
        error = f"timed out after {UPDATE_TIMEOUT_S}s"
    except OSError as exc:
        # Permission denied / not executable and similar exec failures.
        exit_code = None
        error = str(exc)

    success = exit_code == 0
    after_version = probe_version(binary)
    unchanged = (
        success
        and before_version is not None
        and after_version is not None
        and before_version == after_version
    )
    logger.debug(
        "tool_update.run",
        tool=str(tool_id),
        exit_code=exit_code,
        success=success,
        unchanged=unchanged,
    )
    return UpdateResult(
        tool_id=tool_id,
        display_name=caps.display_name,
        command=command,
        success=success,
        exit_code=exit_code,
        output_tail=output_tail,
        before_version=before_version,
        after_version=after_version,
        unchanged=unchanged,
        error=error,
    )


def _is_updatable(caps: AIToolCapabilities) -> bool:
    """A capability qualifies iff it has an update command and is a terminal tool."""
    return caps.update_command is not None and caps.tool_type is AIToolType.TERMINAL


def _decode(data: str | bytes | None) -> str:
    """Coerce subprocess output (str, bytes, or None) to text."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _combine(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    """Join non-empty stdout/stderr into one blob for tailing."""
    parts = [part for part in (_decode(stdout), _decode(stderr)) if part.strip()]
    return "\n".join(parts)


def _tail(text: str, lines: int = OUTPUT_TAIL_LINES) -> str:
    """Keep the last *lines* non-empty-trailing lines of *text*."""
    stripped = text.strip("\n")
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])
