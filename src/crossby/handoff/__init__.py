"""Cross-tool AI session handoff.

Reads a source tool's session transcript, summarizes it into a structured
``HandoffDocument``, writes it to ``.crossby/handoffs/``, and (optionally)
launches a target tool with the handoff file path as its initial prompt.
"""

from __future__ import annotations

from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    HandoffDocument,
    SessionRef,
    ToolCall,
)
from crossby.handoff.picker import pick_latest_session
from crossby.handoff.writer import HandoffWriter

__all__ = [
    "ConversationTranscript",
    "ConversationTurn",
    "HandoffDocument",
    "HandoffWriter",
    "SessionRef",
    "ToolCall",
    "pick_latest_session",
]
