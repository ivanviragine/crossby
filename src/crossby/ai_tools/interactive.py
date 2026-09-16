"""Public startup events for interactive launches that support deferred input.

Callbacks run synchronously and must not read or write the terminal. Native
questions and all interaction after startup remain in the tool's own UI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from crossby.models.ai import InteractiveLaunchEvent


class InteractiveSession(Protocol):
    """One-shot initial input port, valid only during the PLAN_READY callback."""

    def send_message(self, message: str) -> None:
        """Queue the first task message; duplicate or late submissions raise ValueError."""
        ...


InteractiveLaunchHandler = Callable[[InteractiveLaunchEvent, InteractiveSession], None]
