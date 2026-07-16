"""Runtime hook I/O contract.

The counterpart to ``crossby.sync.hooks`` (which *writes* per-tool hook config):
this package parses a tool's hook stdin into a normalized :class:`HookEvent` and
serializes a :class:`HookDecision` back into each tool's stdout/exit dialect.
"""

from __future__ import annotations

from crossby.hooks.runtime import (
    HookDecision,
    HookEmission,
    HookEvent,
    emit_decision,
    parse_event,
)

__all__ = [
    "HookDecision",
    "HookEmission",
    "HookEvent",
    "emit_decision",
    "parse_event",
]
