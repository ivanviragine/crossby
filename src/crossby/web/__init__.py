"""Local browser UI — launch a terminal AI tool and drive it from a web page.

``crossby ui`` starts a loopback-only HTTP server that serves a small xterm.js
page and brokers a pseudo-terminal per session. The browser plays the terminal
emulator; :class:`crossby.utils.pty_runner.PtySession` gives the AI tool a real
TTY on the other end, so it renders and behaves exactly as it does in a shell.

Layering note: this package talks to :class:`~crossby.ai_tools.base.AbstractAITool`
directly — ``build_launch_command()`` is a public adapter hook returning plain
argv, and adapters import nothing from ``crossby.ui``. The Typer layer in
``cli/launch.py`` is deliberately *not* reused: it resolves interactively,
prints Rich markup, and raises ``typer.Exit``, none of which survives a browser.
Richer launch behaviour (scenes, profiles, transcripts) arrives by extracting
that orchestration into ``services/``, not by calling the command function.
"""

from __future__ import annotations

from crossby.web.server import CrossbyUIServer, serve
from crossby.web.sessions import LaunchRequest, SessionManager, SessionNotFoundError

__all__ = [
    "CrossbyUIServer",
    "LaunchRequest",
    "SessionManager",
    "SessionNotFoundError",
    "serve",
]
