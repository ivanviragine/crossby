"""The CLI must import on a platform with no POSIX pseudo-terminal support.

``cli/main.py`` imports ``cli/ui.py`` to register the ``ui`` command, so
anything that module pulls in at import time is loaded by *every* crossby
invocation. ``pty_runner`` imports ``fcntl``, ``pty`` and ``termios`` at module
scope, none of which exist on Windows: importing it from ``cli/ui.py`` meant
``crossby --help``, ``sync`` and ``launch`` all died with ``ModuleNotFoundError``
before ``ui()`` could print its "needs POSIX pseudo-terminal support" message.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Setting a name to None in sys.modules makes `import name` raise, which is the
# closest we get to Windows from here without a Windows runner.
_SIMULATE_WINDOWS = textwrap.dedent(
    """
    import sys

    for name in ("fcntl", "pty", "termios"):
        sys.modules[name] = None

    import crossby.cli.main  # noqa: F401  — registers every command, `ui` included

    assert "crossby.utils.pty_runner" not in sys.modules, (
        "importing the CLI pulled in the POSIX pty backend"
    )
    print("ok")
    """
)


def test_the_cli_imports_without_the_posix_pty_modules() -> None:
    """A fresh interpreter, so this sees the real import graph rather than
    whatever the test session has already cached in ``sys.modules``."""
    result = subprocess.run(
        [sys.executable, "-c", _SIMULATE_WINDOWS],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_pty_support_is_importable_without_the_backend() -> None:
    """The platform predicate has to be answerable without loading the backend
    that answers it — that is the whole reason it lives in its own module."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "for name in ('fcntl', 'pty', 'termios'):\n"
            "    sys.modules[name] = None\n"
            "from crossby.utils.pty_support import pty_supported\n"
            "print(pty_supported())\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"True", "False"}


def test_running_the_ui_command_reports_unsupported_before_importing_the_backend() -> None:
    """The friendly message must be reachable on the platform it describes.

    `ui()` deferred `from crossby.web import serve` to call time, but placed it
    *above* the `pty_supported()` guard — and `crossby.web` reaches
    `pty_runner`. So `crossby ui` on Windows still died with
    ModuleNotFoundError instead of the message written for exactly that case.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                for name in ("fcntl", "pty", "termios"):
                    sys.modules[name] = None

                from typer.testing import CliRunner

                import crossby.cli.ui as ui_module
                from crossby.cli.main import app

                # Blocking those modules does not move os.name, so the predicate
                # is forced here instead. `cli/ui.py` binds it by name, so the
                # rebind has to happen on that module.
                ui_module.pty_supported = lambda: False

                outcome = CliRunner().invoke(app, ["ui"])
                assert outcome.exit_code == 1, outcome.output
                assert "POSIX pseudo-terminal support" in outcome.output, outcome.output
                assert "crossby.utils.pty_runner" not in sys.modules
                print("ok")
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
