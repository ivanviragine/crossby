"""Generic CLI version probing — shell out to a tool binary and parse its semver.

Best-effort by design: a missing binary, a slow/failing ``--version`` call, or
unparseable output all yield ``None`` (unknown), and callers treat "unknown"
conservatively rather than crashing. Extracted from ``scenes/versioning.py`` so
both the scene activator and the ``crossby tools update`` service share one
failure-safe probe instead of duplicating it.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from contextlib import suppress
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# First dotted-number run anywhere in the ``--version`` output. Tolerates the
# varied shapes tools emit: "2.1.218 (Claude Code)", "codex-cli 0.146.0",
# "GitHub Copilot CLI 1.0.77.", "1.1.10".
_SEMVER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")

# Bound the subprocess so a hung binary can never stall a caller.
_VERSION_TIMEOUT_S = 5.0
_VERSION_CLEANUP_GRACE_S = 0.2


@dataclass(frozen=True)
class BinaryVersion:
    """One bounded version probe with comparison and provenance forms."""

    normalized: tuple[int, int, int]
    text: str

    @property
    def raw(self) -> str:
        """Compatibility spelling for callers that describe the text as raw."""
        return self.text


def parse_semver(text: str) -> tuple[int, int, int] | None:
    """Return the first ``(major, minor, patch)`` triple in *text*, or ``None``.

    A missing patch component reads as ``0`` (``"2.1"`` → ``(2, 1, 0)``).
    """
    match = _SEMVER_RE.search(text)
    if match is None:
        return None
    major, minor, patch = match.group(1), match.group(2), match.group(3)
    return (int(major), int(minor), int(patch) if patch is not None else 0)


def _kill_version_process_group(proc: subprocess.Popen[str]) -> None:
    """Kill the probe-owned process group so helper processes cannot escape."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    if proc.poll() is None:
        with suppress(OSError):
            proc.kill()


def _run_version_probe(binary: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Capture one version probe and clean every process in its owned group."""
    proc = subprocess.Popen(
        [binary, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        _kill_version_process_group(proc)
        with suppress(OSError, subprocess.SubprocessError, UnicodeError):
            proc.communicate(timeout=_VERSION_CLEANUP_GRACE_S)
        raise
    _kill_version_process_group(proc)
    assert proc.returncode is not None
    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def detect_binary_version_info(
    binary: str, *, timeout_seconds: float = _VERSION_TIMEOUT_S
) -> BinaryVersion | None:
    """Run one bounded probe and retain normalized and original version text.

    Returns ``None`` when the binary is absent from PATH, the invocation fails
    or times out, or no semver can be parsed. ``timeout_seconds`` can shorten,
    but never extend, the normal probe bound. Never raises.
    """
    if shutil.which(binary) is None:
        return None
    if timeout_seconds <= 0:
        return None
    try:
        proc = _run_version_probe(
            binary,
            timeout=min(timeout_seconds, _VERSION_TIMEOUT_S),
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        logger.debug("version.probe_failed", binary=binary, error=str(exc))
        return None
    if proc.returncode != 0:
        # A failed --version is "unknown", not a version: error output can carry a
        # version-shaped string that would otherwise pass a feature/version gate.
        logger.debug("version.probe_nonzero", binary=binary, returncode=proc.returncode)
        return None
    # Some tools print the version to stderr. Preserve the first complete line
    # that actually carried a parseable version rather than reconstructing a
    # lossy ``x.y.z`` string (date/build suffixes are meaningful provenance).
    for stream in (proc.stdout, proc.stderr):
        for line in stream.splitlines():
            text = line.strip()
            normalized = parse_semver(text)
            if normalized is not None:
                return BinaryVersion(normalized=normalized, text=text)
    return None


def detect_binary_version(binary: str) -> tuple[int, int, int] | None:
    """Compatibility tuple view over :func:`detect_binary_version_info`."""
    detected = detect_binary_version_info(binary)
    return detected.normalized if detected is not None else None
