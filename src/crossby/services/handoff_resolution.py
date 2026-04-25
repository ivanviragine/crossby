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
    cli_default: str = "default",
) -> str:
    """Resolve the summarization prompt preset.

    Order: explicit CLI flag (anything other than the CLI default) →
    ``config.handoff_defaults.prompt_preset`` → ``cli_default``.

    The CLI flag defaults to ``"default"`` — we only treat it as explicit
    when the user actively sets a different preset.
    """
    if prompt_preset is not None and prompt_preset != cli_default:
        return prompt_preset
    configured = config.get_handoff_preset()
    if configured:
        return configured
    return prompt_preset if prompt_preset is not None else cli_default


def resolve_handoff_token_budget(
    token_budget: int | None,
    config: CrossbyConfig,
    *,
    cli_default: int = 32_000,
) -> int:
    """Resolve the token budget.

    Order: explicit CLI value (anything other than ``cli_default``) →
    ``config.handoff_defaults.token_budget`` → ``cli_default``.
    """
    if token_budget is not None and token_budget != cli_default:
        return token_budget
    configured = config.get_handoff_token_budget()
    if configured is not None:
        return configured
    return token_budget if token_budget is not None else cli_default
