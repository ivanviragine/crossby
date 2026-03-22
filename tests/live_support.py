"""Shared helpers for env-gated live test lanes."""

from __future__ import annotations

import re
from collections.abc import Iterable


def parse_selected_tools(
    raw: str | None,
    supported_tools: tuple[str, ...],
    *,
    fallback: Iterable[str] | None = None,
) -> set[str]:
    """Parse and validate a comma-separated tool selection env var."""
    if raw:
        selected = {item.strip() for item in raw.split(",") if item.strip()}
    elif fallback is not None:
        selected = {item for item in fallback if item}
    else:
        selected = set()

    unknown = selected - set(supported_tools)
    if unknown:
        supported = ", ".join(sorted(supported_tools))
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown tool selection: {unknown_str}. Supported: {supported}")

    return selected


def is_prerequisite_failure(output: str) -> bool:
    """Return True when output looks like missing auth/session/workspace setup."""
    lowered = output.lower()
    patterns = (
        r"\bnot logged in\b",
        r"\blogin required\b",
        r"\bplease log in\b",
        r"\bsign in\b",
        r"\bauthentication required\b",
        r"\bauthorization failed\b",
        r"\bplease authenticate\b",
        r"\breauthenticate\b",
        r"\bmissing api key\b",
        r"\binvalid api key\b",
        r"\bno api key\b",
        r"\bcredential(s)? missing\b",
        r"\boauth\b",
        r"\bworkspace trust\b",
        r"\btrust this workspace\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)
