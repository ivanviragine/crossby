"""Whether this platform can allocate a pseudo-terminal.

Deliberately separate from :mod:`crossby.utils.pty_runner`, which imports
``fcntl``, ``pty`` and ``termios`` at module scope — none of which exist on
Windows. ``cli/main.py`` imports ``cli/ui.py`` to register the ``ui`` command,
so asking this question had to be possible without loading the backend that
answers it: otherwise importing the CLI at all raised ``ModuleNotFoundError``
on Windows and *every* command died, ``--help`` included, long before the
friendly "needs POSIX pseudo-terminal support" message could run.
"""

from __future__ import annotations

import os


def pty_supported() -> bool:
    """Whether this platform can allocate a pseudo-terminal.

    ``pty`` is POSIX-only. Windows would need ConPTY (``pywinpty``), which
    :mod:`crossby.utils.pty_runner` does not implement — the same gap ``script``
    already leaves in :func:`crossby.utils.process.run_with_transcript`.
    """
    return os.name == "posix"
