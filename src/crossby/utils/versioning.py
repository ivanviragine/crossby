"""Generic CLI version probing — shell out to a tool binary and parse its semver.

Best-effort by design: a missing binary, a slow/failing ``--version`` call, or
unparseable output all yield ``None`` (unknown), and callers treat "unknown"
conservatively rather than crashing. Extracted from ``scenes/versioning.py`` so
both the scene activator and the ``crossby tools update`` service share one
failure-safe probe instead of duplicating it.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import structlog

logger = structlog.get_logger()

# First dotted-number run anywhere in the ``--version`` output. Tolerates the
# varied shapes tools emit: "2.1.218 (Claude Code)", "codex-cli 0.146.0",
# "GitHub Copilot CLI 1.0.77.", "1.1.10".
_SEMVER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

# Bound the subprocess so a hung binary can never stall a caller.
_VERSION_TIMEOUT_S = 5.0


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
        logger.debug("version.probe_failed", binary=binary, error=str(exc))
        return None
    if proc.returncode != 0:
        # A failed --version is "unknown", not a version: error output can carry a
        # version-shaped string that would otherwise pass a feature/version gate.
        logger.debug("version.probe_nonzero", binary=binary, returncode=proc.returncode)
        return None
    # Some tools print the version to stderr; check stdout first, then stderr.
    return parse_semver(proc.stdout) or parse_semver(proc.stderr)
