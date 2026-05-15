"""Transcript truncation — keep the last N turns under a token budget."""

from __future__ import annotations

from crossby.handoff.models import ConversationTranscript, ConversationTurn


def approx_tokens(text: str) -> int:
    """Very rough token estimate — 1 token ≈ 4 characters.

    The summarizer only uses this to decide *whether* to truncate before
    calling the LLM, not to account for billing. A precise tokenizer would
    add a heavy dependency for no practical benefit.
    """
    return max(1, len(text) // 4)


def turn_tokens(turn: ConversationTurn) -> int:
    """Approximate token cost of a single turn."""
    total = approx_tokens(turn.content)
    for call in turn.tool_calls:
        total += approx_tokens(call.name)
        for key, value in call.arguments.items():
            total += approx_tokens(key) + approx_tokens(str(value))
    return total


def truncate_transcript(
    transcript: ConversationTranscript,
    token_budget: int,
) -> ConversationTranscript:
    """Return a copy of ``transcript`` trimmed to fit ``token_budget``.

    Strategy: keep the most recent turns (they carry the current state);
    drop older turns from the head. ``truncated`` is set to ``True`` when
    anything was dropped.

    Note: at least one turn is always kept, even if it alone exceeds
    ``token_budget``. In the rare case of a single oversized turn (e.g. a
    multi-megabyte tool result), the returned transcript will be larger
    than the budget. This is preferable to returning an empty transcript,
    which the summarizer could not work with.

    Raises ``ValueError`` if ``token_budget`` is not positive — callers
    must pick a real budget; silently returning an unbounded transcript
    would defeat the purpose of truncation.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")

    kept: list[ConversationTurn] = []
    running = 0
    for turn in reversed(transcript.turns):
        cost = turn_tokens(turn)
        if running + cost > token_budget and kept:
            break
        kept.append(turn)
        running += cost

    kept.reverse()
    was_truncated = len(kept) < len(transcript.turns)
    return ConversationTranscript(
        session_ref=transcript.session_ref,
        turns=kept,
        truncated=transcript.truncated or was_truncated,
    )
