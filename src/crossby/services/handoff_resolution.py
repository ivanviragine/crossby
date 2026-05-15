"""Resolution helpers for ``crossby handoff`` defaults.

Same shape as :mod:`sync_resolution` — CLI flag → ``.crossby.yml`` →
fallback. No auto-detect for tools, since handoff always needs an explicit
source *and* target (choosing either automatically would surprise users).
"""

from __future__ import annotations

from crossby.models.ai import AIToolID
from crossby.models.config import CrossbyConfig


def resolve_handoff_from(
    from_tool: str | None,
    config: CrossbyConfig,
) -> AIToolID | None:
    """Resolve the handoff *source* tool.

    Order: explicit CLI flag → ``config.handoff_defaults.from`` → ``None``.
    ``None`` means "ask the user" — the caller's wizard handles prompting.
    """
    if from_tool:
        return AIToolID(from_tool)
    return config.get_handoff_from()


def resolve_handoff_to(
    to_tool: str | None,
    config: CrossbyConfig,
) -> AIToolID | None:
    """Resolve the handoff *target* tool.

    Order: explicit CLI flag → ``config.handoff_defaults.to`` → ``None``.
    """
    if to_tool:
        return AIToolID(to_tool)
    return config.get_handoff_to()


def resolve_handoff_preset(
    prompt_preset: str | None,
    config: CrossbyConfig,
    *,
    fallback: str = "default",
) -> str:
    """Resolve the summarization prompt preset.

    Order: explicit CLI flag (any non-None value) →
    ``config.handoff_defaults.prompt_preset`` → ``fallback``.

    Callers must pass ``None`` when the user did not provide ``--prompt-preset``
    so the resolver can distinguish "not provided" from "explicitly set to the
    fallback value." The Typer flag default in ``cli/handoff.py`` is ``None``
    for exactly this reason.
    """
    if prompt_preset is not None:
        return prompt_preset
    configured = config.get_handoff_preset()
    if configured:
        return configured
    return fallback


def resolve_handoff_token_budget(
    token_budget: int | None,
    config: CrossbyConfig,
    *,
    fallback: int = 32_000,
) -> int:
    """Resolve the token budget.

    Order: explicit CLI value (any non-None value) →
    ``config.handoff_defaults.token_budget`` → ``fallback``.

    Callers must pass ``None`` when the user did not provide ``--token-budget``
    so the resolver can distinguish "not provided" from "explicitly set to the
    fallback value."
    """
    if token_budget is not None:
        return token_budget
    configured = config.get_handoff_token_budget()
    if configured is not None:
        return configured
    return fallback
