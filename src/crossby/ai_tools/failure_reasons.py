"""Recognize why a native session failed, from evidence kept private.

Adapters hand native failure wording, status codes and structurally detected
causes to :meth:`HeadlessRuntimeContext.note_failure_evidence`; the runtime
classifies them here once, at completion, into a fixed
:class:`~crossby.models.ai.HeadlessFailureReason`. Only that reason is ever
exposed, never the native text.

Every pattern below was observed live from the named tool (see
``docs/unattended-headless-verification.md#failure-reasons``); one that was
not is marked as inferred, with what it was inferred from. Unrecognized
evidence classifies as ``None``: no reason, not "no failure".
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from crossby.models.ai import AIToolID, HeadlessFailureReason

_R = HeadlessFailureReason

# Native HTTP status codes with one meaning on every tool that reports them.
_HTTP_STATUSES: dict[int, HeadlessFailureReason] = {
    401: _R.NOT_SIGNED_IN,
    429: _R.RATE_LIMITED,
}

_PATTERNS: dict[AIToolID, tuple[tuple[HeadlessFailureReason, re.Pattern[str]], ...]] = {
    AIToolID.CLAUDE: (
        # "Not logged in · Please run /login" (empty config dir, 2.1.317) and
        # "Invalid API key · Fix external API key" (api_error_status 401).
        (_R.NOT_SIGNED_IN, re.compile(r"\bNot logged in\b|\bInvalid API key\b")),
        # "There's an issue with the selected model (…)" with
        # api_error_status 404, and "[claude-code:unrecognized_model]" on stderr.
        (_R.MODEL_UNAVAILABLE, re.compile(r"issue with the selected model|unrecognized_model")),
    ),
    AIToolID.CODEX: (
        # "unexpected status 401 Unauthorized: Missing bearer or basic
        # authentication in header" in turn.failed (empty CODEX_HOME, 0.166.0).
        (_R.NOT_SIGNED_IN, re.compile(r"\bunexpected status 401\b")),
        # Inferred from the same "unexpected status <code>" template; a 429
        # was not captured live.
        (_R.RATE_LIMITED, re.compile(r"\bunexpected status 429\b")),
        # "The '<model>' model is not supported when using Codex with a
        # ChatGPT account."
        (_R.MODEL_UNAVAILABLE, re.compile(r"model is not supported when using Codex")),
        # "Not inside a trusted directory and --skip-git-repo-check was not
        # specified." on stderr, before any event.
        (_R.UNTRUSTED_FOLDER, re.compile(r"Not inside a trusted directory")),
    ),
    AIToolID.CURSOR: (
        # "Warning: The provided API key is invalid." on stderr (2026.11.05).
        (_R.NOT_SIGNED_IN, re.compile(r"\bAPI key is invalid\b")),
        # "Cannot use this model: <model>. Available models: …" on stderr.
        (_R.MODEL_UNAVAILABLE, re.compile(r"\bCannot use this model\b")),
    ),
    AIToolID.COPILOT: (
        # 'Execution failed: Model "<model>" is not available.' as a
        # session.error frame (1.0.105 and 1.0.109).
        (_R.MODEL_UNAVAILABLE, re.compile(r'\bModel "[^"]*" is not available\b')),
    ),
    AIToolID.OPENCODE: (
        # "permission requested: external_directory (…); auto-rejecting" on
        # stderr, after which `opencode run` exits without its stop event
        # (1.19.5). An unknown model and a missing login both surface only as
        # "UnknownError … Unexpected server error", so neither is recognized.
        (_R.ACTION_DENIED, re.compile(r"permission requested: [a-z_]+ .*auto-rejecting")),
    ),
    AIToolID.ANTIGRAVITY_CLI: (
        # "Error: authentication required. Run 'agy' to log in, then retry." on
        # stderr and "authentication failed or timed out" in the result (1.5.4).
        (_R.NOT_SIGNED_IN, re.compile(r"\bauthentication (?:required|failed)\b")),
        # "invalid model selection (--model …): model … is not recognized as a
        # known model or custom model in settings".
        (_R.MODEL_UNAVAILABLE, re.compile(r"\binvalid model selection\b")),
    ),
}


def classify_failure(
    tool: AIToolID,
    *,
    texts: Sequence[str] = (),
    http_statuses: Sequence[int] = (),
    reasons: Sequence[HeadlessFailureReason] = (),
) -> HeadlessFailureReason | None:
    """The recognized reason for a failure, or ``None``.

    A cause the adapter detected structurally wins, then a native HTTP status,
    then the tool's own wording, each in the order observed.
    """
    if reasons:
        return reasons[0]
    for status in http_statuses:
        if status in _HTTP_STATUSES:
            return _HTTP_STATUSES[status]
    rules = _PATTERNS.get(tool, ())
    for text in texts:
        for reason, pattern in rules:
            if pattern.search(text):
                return reason
    return None


__all__ = ["classify_failure"]
