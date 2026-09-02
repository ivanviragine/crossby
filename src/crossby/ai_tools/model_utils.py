"""Model utility functions — tier classification, date suffix detection."""

from __future__ import annotations

import re

from crossby.models.ai import ModelTier


def has_date_suffix(model_id: str) -> bool:
    """Check if a model ID has a YYYYMMDD date suffix.

    Examples:
        claude-haiku-4-5-20251001 → True
        claude-haiku-4-5 → False
        gemini-2.0-flash → False
    """
    return bool(re.search(r"-\d{8}$", model_id))


def _has_component(model_id: str, keyword: str) -> bool:
    """Check if keyword appears as a model ID component (delimited by '-' or '.')."""
    # Match keyword at start, end, or between delimiters
    return bool(re.search(rf"(?:^|[-.]){re.escape(keyword)}(?:[-.]|$)", model_id))


def classify_tier_universal(model_id: str) -> ModelTier:
    """Classify any model into a tier using universal keywords.

    Used when processing raw model IDs from scraping/probing.

    Tier mapping (matches Bash _init_probe_models_for_tool):
        easy         — haiku, flash, spark, mini, luna
        complex      — sonnet, terra, or unrecognized mid-tier models
        very_complex — opus, fable, pro, sol, ultra, max, and the documented
                       bare Copilot complex-reasoning ID gpt-5.4

    A model ID can carry more than one keyword, so the checks are ordered by
    how strongly each keyword identifies the underlying model:

    1. A *variant* marker (``pro``, ``sol``, …) upgrades the family it is
       attached to — ``gpt-5.6-luna-pro`` is a more capable model than
       ``gpt-5.6-luna``, so it must not be read as the fast luna tier.
    2. A *family* marker otherwise decides the tier.
    3. ``max`` is checked last because it is an effort level far more often
       than a family, so any family keyword outranks it: ``gpt-5.6-luna-max``
       is luna at max effort (fast) and ``gpt-5.6-terra-max`` is terra at max
       effort (balanced). ``max`` only decides a tier on its own, for an ID
       carrying no family keyword at all.

    Note: uses component-level matching to avoid false positives like
    "gemini" matching "mini". Keywords must appear as distinct components
    separated by '-' or '.'.
    """
    lower = model_id.lower()
    if lower == "gpt-5.4" or lower.endswith("/gpt-5.4"):
        return ModelTier.POWERFUL
    if any(_has_component(lower, kw) for kw in ("opus", "fable", "pro", "sol", "ultra")):
        return ModelTier.POWERFUL
    if any(_has_component(lower, kw) for kw in ("haiku", "flash", "spark", "mini", "luna")):
        return ModelTier.FAST
    if any(_has_component(lower, kw) for kw in ("sonnet", "terra")):
        return ModelTier.BALANCED
    if _has_component(lower, "max"):
        return ModelTier.POWERFUL
    # Default: unrecognized models go to balanced tier
    return ModelTier.BALANCED
