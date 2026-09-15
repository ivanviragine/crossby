"""``crossby ui`` — serve the local browser terminal."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

import typer

from crossby.ui.console import console
from crossby.utils.pty_runner import pty_supported


def ui(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Project directory sessions run in."),
    port: int = typer.Option(0, "--port", help="Port to bind (0 picks a free one)."),
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback address to bind."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the UI in your default browser."
    ),
) -> None:
    """Launch AI tools from a browser terminal.

    Starts a loopback-only web server that runs each AI tool on a real
    pseudo-terminal, so the tool's full-screen interface works in the browser
    exactly as it does in a shell. The printed URL carries an access token that
    every request must present, so treat the URL as a secret. The token lasts as
    long as the server; stopping it invalidates the URL.

    Sessions always run in ``--path``; the page cannot choose another directory.
    """
    from crossby.web import serve

    if not pty_supported():
        console.error("The browser terminal needs POSIX pseudo-terminal support.")
        console.hint("Windows would require a ConPTY backend, which crossby does not ship yet.")
        raise typer.Exit(1)

    project_root = path.expanduser().resolve()
    if not project_root.is_dir():
        console.error(f"Not a directory: {project_root}")
        raise typer.Exit(1)

    try:
        server = serve(project_root, host=host, port=port)
    except ValueError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.error(f"Could not bind {host}:{port} — {exc}")
        raise typer.Exit(1) from exc

    url = server.url()
    console.header("crossby ui")
    console.kv("Project", str(project_root))
    console.kv("URL", url)
    console.empty()
    console.hint("The URL contains an access token — anyone with it can run AI tools here.")
    console.hint("Press Ctrl-C to stop the server and close every session.")
    console.empty()

    if open_browser:
        # Opening can block on a cold browser start, so never hold up serving.
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    with console.status("Serving…"):
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            console.empty()
            console.info("Shutting down…")
        finally:
            server.shutdown()
            server.server_close()

    console.success("Server stopped.")
