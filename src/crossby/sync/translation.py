"""Cross-provider translation primitives used by sync writers.

Owns the small set of cross-provider mappings that any concern (agents,
skills, instructions, mcp, hooks) might need: model family + reasoning effort
between Claude and Codex/GPT, and Claude permission modes ↔ Codex sandbox
modes. Concrete writers compose these helpers; nothing here knows about a
specific concern's control flow.

Mappings are bidirectional. Forward (claude → codex) is well-defined by
family prefix; reverse (codex → claude) is lossy and picks a sensible
default per family.
"""

from __future__ import annotations

from dataclasses import dataclass

from crossby.models.ai import EffortLevel

# ---------------------------------------------------------------------------
# Permission modes
# ---------------------------------------------------------------------------

# Claude permissionMode → Codex sandbox_mode
PERMISSION_MODE_CLAUDE_TO_CODEX: dict[str, str] = {
    "acceptEdits": "workspace-write",
    "readOnly": "read-only",
    "bypassPermissions": "danger-full-access",
}

# Inverse for codex → claude. ``danger-full-access`` is intentionally not
# mapped back to ``bypassPermissions`` because users rarely write that mode
# into an agent file by hand and we want a friendly default that won't trip
# Claude's safety prompts.
PERMISSION_MODE_CODEX_TO_CLAUDE: dict[str, str] = {
    "workspace-write": "acceptEdits",
    "read-only": "readOnly",
}

# Claude modes that have no Codex equivalent — preserved as manual notes.
CLAUDE_PERMISSION_MODES_UNMAPPED: frozenset[str] = frozenset(
    {
        "default",
        "dontAsk",
        "plan",
    }
)


def map_permission_mode_claude_to_codex(mode: str | None) -> str | None:
    """Map a Claude `permissionMode` value to a Codex `sandbox_mode`.

    Returns ``None`` for empty input or unmapped modes — callers are expected
    to emit a manual-fix note for the unmapped case so the lossy edge stays
    visible.
    """
    if not mode:
        return None
    return PERMISSION_MODE_CLAUDE_TO_CODEX.get(mode)


def map_permission_mode_codex_to_claude(mode: str | None) -> str | None:
    """Map a Codex `sandbox_mode` value to a Claude `permissionMode`."""
    if not mode:
        return None
    return PERMISSION_MODE_CODEX_TO_CLAUDE.get(mode)


# ---------------------------------------------------------------------------
# Model family + effort
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFamilyMapping:
    """A single forward Claude → Codex/GPT family mapping with effort bias."""

    claude_prefix: str
    codex_model: str
    # Forward effort mapping: Claude effort → Codex effort. Tuple entries are
    # consulted in order; first match wins.
    effort_forward: tuple[tuple[EffortLevel, EffortLevel], ...]


# Family mapping with effort-bias-by-family, plus an XHIGH ↔ MAX
# round-trip so reverse(forward(x)) ≈ x for the common cases.
MODEL_FAMILY_MAPPINGS: tuple[ModelFamilyMapping, ...] = (
    ModelFamilyMapping(
        claude_prefix="claude-opus",
        codex_model="gpt-5.4",
        effort_forward=(
            (EffortLevel.LOW, EffortLevel.LOW),
            (EffortLevel.MEDIUM, EffortLevel.MEDIUM),
            (EffortLevel.HIGH, EffortLevel.HIGH),
            (EffortLevel.XHIGH, EffortLevel.XHIGH),
            (EffortLevel.MAX, EffortLevel.XHIGH),
        ),
    ),
    # Sonnet is biased one tier higher when crossing into Codex to match
    # coding-agent behavior expectations.
    ModelFamilyMapping(
        claude_prefix="claude-sonnet",
        codex_model="gpt-5.4-mini",
        effort_forward=(
            (EffortLevel.LOW, EffortLevel.MEDIUM),
            (EffortLevel.MEDIUM, EffortLevel.HIGH),
            (EffortLevel.HIGH, EffortLevel.XHIGH),
            (EffortLevel.XHIGH, EffortLevel.XHIGH),
            (EffortLevel.MAX, EffortLevel.XHIGH),
        ),
    ),
    ModelFamilyMapping(
        claude_prefix="claude-haiku",
        codex_model="gpt-5.4-mini",
        effort_forward=(
            (EffortLevel.LOW, EffortLevel.LOW),
            (EffortLevel.MEDIUM, EffortLevel.MEDIUM),
            (EffortLevel.HIGH, EffortLevel.HIGH),
            (EffortLevel.XHIGH, EffortLevel.XHIGH),
            (EffortLevel.MAX, EffortLevel.XHIGH),
        ),
    ),
)


# Reverse Codex → Claude defaults. The reverse direction is genuinely lossy
# (multiple Claude families collapse to one Codex family), so we pick a
# sensible default per Codex family and let users override.
CODEX_TO_CLAUDE_DEFAULTS: dict[str, str] = {
    "gpt-5.4": "claude-opus-4.7",
    "gpt-5.4-mini": "claude-sonnet-4.6",
}


def find_claude_family(model: str) -> ModelFamilyMapping | None:
    """Return the family mapping that owns ``model``, or None."""
    for mapping in MODEL_FAMILY_MAPPINGS:
        if model.startswith(mapping.claude_prefix):
            return mapping
    return None


def map_model_claude_to_codex(model: str) -> str:
    """Translate a Claude model id into the Codex family default.

    Returns the input unchanged if no mapping is known — the caller is
    responsible for flagging that as a manual-fix note.
    """
    mapping = find_claude_family(model)
    if mapping is None:
        return model
    return mapping.codex_model


def map_model_codex_to_claude(model: str) -> str:
    """Translate a Codex/GPT model id into a Claude family default."""
    return CODEX_TO_CLAUDE_DEFAULTS.get(model, model)


def map_effort_claude_to_codex(
    claude_model: str | None, effort: EffortLevel | str | None
) -> EffortLevel | None:
    """Map a Claude (model, effort) pair to the Codex effort level.

    The mapping is family-aware: Sonnet shifts up one tier vs Opus/Haiku.
    """
    if effort is None:
        return None
    effort_enum = _coerce_effort(effort)
    if effort_enum is None:
        return None
    if not claude_model:
        return effort_enum
    mapping = find_claude_family(claude_model)
    if mapping is None:
        return effort_enum
    for source, target in mapping.effort_forward:
        if source == effort_enum:
            return target
    return effort_enum


def map_effort_codex_to_claude(
    codex_model: str | None,
    target_claude_model: str | None,
    effort: EffortLevel | str | None,
) -> EffortLevel | None:
    """Reverse of :func:`map_effort_claude_to_codex`.

    Returns the *first* Claude effort whose forward mapping yields the input
    Codex effort. Lossy in cases where multiple Claude tiers collapse into
    the same Codex tier (e.g. ``xhigh``); we pick the lowest such Claude
    tier so users don't accidentally over-bill.
    """
    if effort is None:
        return None
    effort_enum = _coerce_effort(effort)
    if effort_enum is None:
        return None
    target_model = target_claude_model
    if target_model is None and codex_model is not None:
        target_model = map_model_codex_to_claude(codex_model)
    if target_model is None:
        return effort_enum
    mapping = find_claude_family(target_model)
    if mapping is None:
        return effort_enum
    for source, target in mapping.effort_forward:
        if target == effort_enum:
            return source
    return effort_enum


def _coerce_effort(value: EffortLevel | str) -> EffortLevel | None:
    if isinstance(value, EffortLevel):
        return value
    try:
        return EffortLevel(value)
    except ValueError:
        return None


__all__ = [
    "CLAUDE_PERMISSION_MODES_UNMAPPED",
    "CODEX_TO_CLAUDE_DEFAULTS",
    "MODEL_FAMILY_MAPPINGS",
    "PERMISSION_MODE_CLAUDE_TO_CODEX",
    "PERMISSION_MODE_CODEX_TO_CLAUDE",
    "ModelFamilyMapping",
    "find_claude_family",
    "map_effort_claude_to_codex",
    "map_effort_codex_to_claude",
    "map_model_claude_to_codex",
    "map_model_codex_to_claude",
    "map_permission_mode_claude_to_codex",
    "map_permission_mode_codex_to_claude",
]
