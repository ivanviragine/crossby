"""CLI version detection — shell out to a tool binary and parse its semver.

No version-gating infrastructure existed in crossby before scenes: installed
tools are detected purely by PATH presence (``AbstractAITool.detect_installed``),
never by querying a version. Some scene DECLARE keys, however, only work on
recent-enough builds — Claude ``skillOverrides`` behaves correctly only on
``claude >= 2.1.129`` — so the activator must gate on the real version and warn
when it cannot confirm one.

Detection is best-effort: a missing binary, a slow/failing ``--version`` call,
or unparseable output all yield ``None`` (unknown), and callers treat "unknown"
conservatively rather than crashing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from crossby.models.ai import AIToolID

logger = structlog.get_logger()

# First dotted-number run anywhere in the ``--version`` output. Tolerates the
# varied shapes tools emit: "2.1.218 (Claude Code)", "codex-cli 0.146.0",
# "GitHub Copilot CLI 1.0.77.", "1.1.10".
_SEMVER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

# Bound the subprocess so a hung binary can never stall a scene apply.
_VERSION_TIMEOUT_S = 5.0

# Minimum Claude build on which ``skillOverrides`` is honoured (landmine in the
# plan; the key is undocumented and silently ignored on older builds).
CLAUDE_SKILL_OVERRIDES_MIN = (2, 1, 129)


def parse_semver(text: str) -> tuple[int, int, int] | None:
    """Return the first ``(major, minor, patch)`` triple in *text*, or ``None``.

    A missing patch component reads as ``0`` (``"2.1"`` → ``(2, 1, 0)``).
    """
    match = _SEMVER_RE.search(text)
    if match is None:
        return None
    major, minor, patch = match.group(1), match.group(2), match.group(3)
    return (int(major), int(minor), int(patch) if patch is not None else 0)


def detect_binary_version(binary: str) -> tuple[int, int, int] | None:
    """Run ``<binary> --version`` and parse a semver from its output.

    Returns ``None`` when the binary is absent from PATH, the invocation fails
    or times out, or no semver can be parsed. Never raises.
    """
    if shutil.which(binary) is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("scene.version_probe_failed", binary=binary, error=str(exc))
        return None
    if proc.returncode != 0:
        # A failed --version is "unknown", not a version: error output can carry a
        # version-shaped string that would otherwise pass the Claude feature gate.
        logger.debug("scene.version_probe_nonzero", binary=binary, returncode=proc.returncode)
        return None
    # Some tools print the version to stderr; check stdout first, then stderr.
    return parse_semver(proc.stdout) or parse_semver(proc.stderr)


def detect_tool_version(tool_id: AIToolID) -> tuple[int, int, int] | None:
    """Detect the installed version of *tool_id* via its capability binary.

    Resolves the binary name from the tool's adapter, then delegates to
    :func:`detect_binary_version`. Returns ``None`` for unknown/uninstalled
    tools or unparseable output.
    """
    from crossby.ai_tools.base import AbstractAITool

    try:
        binary = AbstractAITool.get(tool_id).capabilities().binary
    except (ValueError, KeyError):
        return None
    return detect_binary_version(binary)


def at_least(version: tuple[int, int, int] | None, minimum: tuple[int, int, int]) -> bool:
    """True when *version* is known and ``>= minimum``.

    An unknown version (``None``) returns ``False`` — callers gate optional,
    silently-ignored keys, so "cannot confirm" must fail closed (skip the key
    and warn) rather than write something the tool will drop.
    """
    if version is None:
        return False
    return version >= minimum
