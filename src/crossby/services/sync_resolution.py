"""Resolution helpers for ``crossby sync`` defaults.

Mirrors the fallback-chain style of :mod:`crossby.services.ai_resolution` so
every wizard-flavoured command reads explicit CLI flags → ``.crossby.yml``
defaults → auto-detection in a consistent order.

``from`` / ``to`` / ``concern`` are all optional at every layer — returning
``None`` from a resolver means "no preference, let the caller pick at render
time".
"""

from __future__ import annotations

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig
from crossby.sync.base import SyncConcern


def resolve_sync_from(
    from_tool: str | None,
    config: CrossbyConfig,
    *,
    auto_detect: bool = True,
) -> AIToolID | None:
    """Resolve the sync *source* tool.

    Order: explicit CLI flag → ``config.sync_defaults.from`` →
    first auto-detected tool (when ``auto_detect`` is True).
    Returns ``None`` when nothing can be resolved — the caller decides whether
    that is an error (non-TTY) or a prompt trigger (TTY).
    """
    if from_tool:
        return AIToolID(from_tool)

    configured = config.get_sync_from()
    if configured is not None:
        return configured

    if auto_detect:
        installed = AbstractAITool.detect_installed()
        if installed:
            return installed[0]

    return None


def resolve_sync_to(
    to_tool: str | None,
    config: CrossbyConfig,
) -> AIToolID | None:
    """Resolve the sync *target* tool.

    Order: explicit CLI flag → ``config.sync_defaults.to`` → ``None``.
    ``None`` means "sync to all installed tools other than the source".
    """
    if to_tool:
        return AIToolID(to_tool)
    return config.get_sync_to()


def resolve_sync_concern(
    concern: str | None,
    config: CrossbyConfig,
) -> SyncConcern | None:
    """Resolve the sync concern filter.

    Order: explicit CLI flag → ``config.sync_defaults.concern`` → ``None``.
    ``None`` means "all concerns".
    """
    raw = concern or config.get_sync_concern()
    if raw is None:
        return None
    return SyncConcern(raw)
