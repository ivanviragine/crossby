"""Scene-specific CLI version gating.

No version-gating infrastructure existed in crossby before scenes: installed
tools are detected purely by PATH presence (``AbstractAITool.detect_installed``),
never by querying a version. Some scene DECLARE keys, however, only work on
recent-enough builds — Claude ``skillOverrides`` behaves correctly only on
``claude >= 2.1.129`` — so the activator must gate on the real version and warn
when it cannot confirm one.

The generic, failure-safe probe (:func:`parse_semver`,
:func:`detect_binary_version`) lives in :mod:`crossby.utils.versioning` and is
re-exported here so this module's scene-specific API stays source-compatible for
existing callers and tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-exported for source compatibility — the generic probe moved to
# ``crossby.utils.versioning`` so the ``crossby tools update`` service can reuse
# it without duplicating a second implementation.
from crossby.utils.versioning import detect_binary_version, parse_semver

if TYPE_CHECKING:
    from crossby.models.ai import AIToolID

__all__ = [
    "CLAUDE_SKILL_OVERRIDES_MIN",
    "at_least",
    "detect_binary_version",
    "detect_tool_version",
    "parse_semver",
]

# Minimum Claude build on which ``skillOverrides`` is honoured (landmine in the
# plan; the key is undocumented and silently ignored on older builds).
CLAUDE_SKILL_OVERRIDES_MIN = (2, 1, 129)


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
