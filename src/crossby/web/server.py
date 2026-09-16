"""Loopback HTTP server backing the ``crossby ui`` browser terminal.

Transport is deliberately dependency-free: a stdlib :class:`ThreadingHTTPServer`
streams terminal output over Server-Sent Events and takes keystrokes back as
small POSTs. On loopback the round trip is sub-millisecond, and the project
keeps its current dependency set.

**One stream carries every session.** A browser allows roughly six HTTP/1.1
connections per origin, and a stream per session spends one apiece; measured,
the sixth open terminal exhausts the pool and stalls every other request —
including the POSTs carrying keystrokes, so typing stops working. Multiplexing
removes tab count as a transport concern entirely.

Terminal output is **base64-encoded** inside each SSE frame. Raw output is
bytes, and a read boundary regularly lands mid-UTF-8-sequence or mid-escape;
base64 sidesteps both that and the newline-framing rules of the SSE wire format
at a cost of a third more bytes over a local socket.

Security posture — this process spawns AI tools with filesystem access, so the
server is hostile-browser-aware by default:

- binds loopback only;
- requires a ``secrets``-generated token on every request, compared with
  :func:`secrets.compare_digest`;
- validates the ``Host`` header, which is what defeats DNS rebinding (a remote
  page resolving its own hostname to 127.0.0.1);
- rejects any cross-origin ``Origin``, so no third-party page can drive the
  terminal even if it somehow learned the port;
- pins every session's working directory to the project root the server was
  started with — the browser never supplies a path.

**The static shell is served without a token; the API is not.** The page's
``<link>`` and ``<script>`` subresources cannot attach a header, and the cookie
that once covered them proved fragile in exactly the way that matters: cookies
ignore the port, so they outlive a restart and leak between concurrent servers,
and every ``crossby ui`` run mints a fresh token. A reload from a bookmark, or a
cached shell after a restart, then 401s the stylesheet and script while the HTML
loads — a broken page with no explanation.

The shell holds nothing worth protecting: vendored xterm.js, this project's own
CSS and JS, and an HTML page that does nothing at all without a token. What
needs guarding is the API, which spawns processes, and that still demands the
token on every request. An unauthenticated fetch of the shell tells an attacker
only that crossby is listening, which a 401 would equally reveal. ``Host`` and
``Origin`` are still enforced on it, so no foreign page can embed it.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import secrets
import sys
import threading
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import structlog

from crossby import data as _data
from crossby.config.json_utils import PathContainmentError, assert_within
from crossby.utils.pty_runner import PtyUnsupportedError
from crossby.web.sessions import (
    LaunchRequest,
    LaunchValidationError,
    MultiplexedStream,
    SessionManager,
    SessionNotFoundError,
    StreamEvent,
    describe_tools,
    window_size_from_payload,
)

logger = structlog.get_logger()

STATIC_ROOT = Path(_data.__file__).parent / "ui"

# Largest JSON body accepted. Terminal input arrives a keystroke or a paste at a
# time; this bounds a hostile or wedged client.
MAX_BODY_BYTES = 1024 * 1024

# Emitted every 15s on an idle stream so proxies and sleeping tabs keep the
# connection open. SSE comments are ignored by EventSource.
SSE_KEEPALIVE_SECONDS = 15.0

# A browser resets connections as a matter of course — closing a tab, reloading,
# reaping an idle pooled connection. None of these are server faults.
_CLIENT_DISCONNECT_ERRORS = (
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    TimeoutError,
)

# Hosts a request's `Host` header may name. IPv6 loopback is valid for an
# incoming header even though the server itself binds IPv4.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Hosts `serve()` will bind. `ThreadingHTTPServer` is `AF_INET`, so offering
# ``::1`` here promised an address it could not actually listen on.
_BINDABLE_HOSTS = frozenset({"127.0.0.1", "localhost"})


class CrossbyUIServer(ThreadingHTTPServer):
    """Threaded loopback server holding the session registry and access token."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        project_root: Path,
        token: str,
        allowed_roots: Sequence[Path] | None = None,
    ) -> None:
        # Populate before super().__init__(), which binds the socket and calls
        # server_close() if that fails. The override below would then touch
        # attributes that did not exist yet, and the AttributeError masked the
        # real bind error — an in-use port crashed the CLI instead of printing
        # "could not bind".
        self.sessions = SessionManager(project_root, allowed_roots)
        self.token = token
        self.project_root = project_root.resolve()
        super().__init__(address, _RequestHandler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def url(self) -> str:
        """The page URL, token included, ready to open in a browser."""
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    def allowed_origins(self) -> frozenset[str]:
        return frozenset(
            {
                f"http://127.0.0.1:{self.port}",
                f"http://localhost:{self.port}",
                f"http://[::1]:{self.port}",
            }
        )

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Keep routine client disconnects out of the operator's terminal.

        ``socketserver`` prints a full traceback for any exception escaping a
        handler, and a browser produces these constantly through no fault of
        its own: closing a tab, reloading, or reaping an idle pooled connection
        resets the socket while the server is parked in ``readline()`` waiting
        for the next keep-alive request. The result is a wall of
        ``ConnectionResetError`` tracebacks that look like a crash and bury any
        real error. These are expected, so they are logged at debug; everything
        else still goes to the default handler.
        """
        error = sys.exc_info()[1]
        if isinstance(error, _CLIENT_DISCONNECT_ERRORS):
            logger.debug(
                "web.client_disconnected",
                client=client_address[0] if client_address else None,
                error=type(error).__name__,
            )
            return
        super().handle_error(request, client_address)

    def server_close(self) -> None:
        # Tolerate a half-constructed server: this runs during a failed bind too.
        sessions = getattr(self, "sessions", None)
        if sessions is not None:
            sessions.shutdown()
        super().server_close()


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "crossby-ui"
    protocol_version = "HTTP/1.1"

    @property
    def ui(self) -> CrossbyUIServer:
        server = self.server
        assert isinstance(server, CrossbyUIServer)
        return server

    # -- logging ----------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:
        """Route access logs through structlog instead of stderr."""
        logger.debug("web.request", client=self.client_address[0], message=format % args)

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        route = parsed.path.rstrip("/") or "/"

        # The static shell carries no secrets and is deliberately unauthenticated;
        # see the module docstring. Host/Origin still apply.
        if route == "/":
            if self._same_origin():
                self._serve_static("index.html")
            return
        if route.startswith("/assets/"):
            if self._same_origin():
                self._serve_static(route[len("/assets/") :])
            return

        if not self._authorize(query):
            return

        if route == "/api/tools":
            self._send_json(
                HTTPStatus.OK,
                {
                    "tools": describe_tools(),
                    "project_root": str(self.ui.project_root),
                    "roots": [str(root) for root in self.ui.sessions.allowed_roots],
                },
            )
        elif route == "/api/sessions":
            self._send_json(HTTPStatus.OK, {"sessions": self.ui.sessions.list_sessions()})
        elif route == "/api/directories":
            self._browse(next(iter(query.get("path", [])), None))
        elif route == "/api/stream":
            self._stream()
        elif route.startswith("/api/sessions/"):
            self._with_session(route[len("/api/sessions/") :], self._describe)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self._authorize(parse_qs(parsed.query)):
            return
        route = parsed.path.rstrip("/") or "/"

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if route == "/api/sessions":
            self._create_session(payload)
        elif route.startswith("/api/sessions/") and route.endswith("/input"):
            self._write_input(route[len("/api/sessions/") : -len("/input")], payload)
        elif route.startswith("/api/sessions/") and route.endswith("/resize"):
            self._resize(route[len("/api/sessions/") : -len("/resize")], payload)
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if not self._authorize(parse_qs(parsed.query)):
            return
        route = parsed.path.rstrip("/") or "/"
        if not route.startswith("/api/sessions/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        session_id = route[len("/api/sessions/") :]
        try:
            self.ui.sessions.close(session_id)
        except SessionNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such session"})
            return
        self._send_json(HTTPStatus.OK, {"closed": session_id})

    # -- handlers ---------------------------------------------------------
    def _create_session(self, payload: dict[str, Any]) -> None:
        try:
            request = LaunchRequest.from_payload(payload)
            session = self.ui.sessions.create(request)
        except LaunchValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except PtyUnsupportedError as exc:
            self._send_json(HTTPStatus.NOT_IMPLEMENTED, {"error": str(exc)})
            return
        except FileNotFoundError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": f"tool binary not found: {exc.filename}"}
            )
            return
        self._send_json(HTTPStatus.CREATED, self.ui.sessions.describe(session.id))

    def _browse(self, requested: str | None) -> None:
        try:
            listing = self.ui.sessions.browse(requested)
        except LaunchValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, listing)

    def _describe(self, session_id: str) -> None:
        self._send_json(HTTPStatus.OK, self.ui.sessions.describe(session_id))

    def _write_input(self, session_id: str, payload: dict[str, Any]) -> None:
        data = payload.get("data")
        if not isinstance(data, str):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "'data' must be a string"})
            return
        try:
            session = self.ui.sessions.get(session_id)
        except SessionNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such session"})
            return
        session.write(data.encode("utf-8"))
        self._send_json(HTTPStatus.OK, {"written": len(data)})

    def _resize(self, session_id: str, payload: dict[str, Any]) -> None:
        try:
            size = window_size_from_payload(payload)
            session = self.ui.sessions.get(session_id)
        except LaunchValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except SessionNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such session"})
            return
        session.resize(size)
        self._send_json(HTTPStatus.OK, {"cols": size.cols, "rows": size.rows})

    def _stream(self) -> None:
        """Stream every session's output to the browser as Server-Sent Events.

        One connection carries all sessions. Per-session streams would each hold
        an HTTP/1.1 connection, and a browser allows only ~6 per origin — the
        sixth terminal would stall every other request, keystrokes included.
        """
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        heartbeat = _Heartbeat(self.wfile, SSE_KEEPALIVE_SECONDS)
        heartbeat.start()
        try:
            with MultiplexedStream(self.ui.sessions) as stream:
                for event in stream:
                    heartbeat.write(_sse_frame(event))
                if stream.desynced:
                    # Close without a terminal marker: EventSource reconnects on
                    # its own and repaints every session from fresh scrollback.
                    logger.warning("web.stream.desync")
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("web.stream.disconnected")
        finally:
            heartbeat.stop()

    # -- static -----------------------------------------------------------
    def _serve_static(self, relative: str) -> None:
        candidate = STATIC_ROOT / relative
        try:
            assert_within(STATIC_ROOT, candidate)
        except PathContainmentError:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        if not candidate.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        content_type, _ = mimetypes.guess_type(candidate.name)
        self._send_bytes(
            HTTPStatus.OK,
            candidate.read_bytes(),
            content_type or "application/octet-stream",
            # Never cache the shell: a stale copy after a restart would hide a
            # token change and leave the page silently unable to reach the API.
            cache=False,
        )

    # -- plumbing ---------------------------------------------------------
    def _same_origin(self) -> bool:
        """Reject requests that are not loopback and same-origin.

        Splitting this out lets the unauthenticated static shell keep the
        rebinding and cross-origin barriers that the API also relies on.
        """
        host = self.headers.get("Host", "")
        hostname = host.rsplit(":", 1)[0].strip("[]") if host else ""
        if hostname not in _LOOPBACK_HOSTS:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid host"})
            return False

        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.ui.allowed_origins():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "cross-origin request refused"})
            return False
        return True

    def _authorize(self, query: dict[str, list[str]]) -> bool:
        """Reject anything that is not a same-origin, correctly-tokened request."""
        if not self._same_origin():
            return False
        supplied = self.headers.get("X-Crossby-Token") or next(iter(query.get("token", [])), "")
        if not secrets.compare_digest(supplied, self.ui.token):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "invalid or missing token"})
            return False
        return True

    def _with_session(self, session_id: str, handler: Any) -> None:
        try:
            handler(session_id)
        except SessionNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such session"})

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        length = int(raw_length) if raw_length and raw_length.isdigit() else 0
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        if length == 0:
            return {}
        body = self.rfile.read(length)
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON body: {exc.msg}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("request body must be a JSON object")
        return decoded

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        cache: bool = True,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store, must-revalidate")
        # The page is same-origin and token-gated; deny framing and sniffing so a
        # hostile tab cannot wrap or reinterpret it.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        with_body = self.command != "HEAD"
        if with_body:
            self.wfile.write(body)


def _sse_frame(event: StreamEvent) -> bytes:
    """Encode one multiplexed event as an SSE frame.

    Output travels base64-encoded: terminal bytes split UTF-8 sequences and
    escapes at arbitrary boundaries, and SSE is a newline-framed text protocol.
    """
    payload: dict[str, Any]
    if event.kind == "output":
        payload = {
            "session": event.session_id,
            "chunk": base64.b64encode(event.chunk).decode("ascii"),
        }
        return f"event: output\ndata: {json.dumps(payload)}\n\n".encode()
    payload = {
        "session": event.session_id,
        "exit_code": event.exit_code,
        "stopped": event.stopped,
        "exit_signal": event.exit_signal,
    }
    return f"event: exit\ndata: {json.dumps(payload)}\n\n".encode()


class _Heartbeat:
    """Serializes SSE writes and emits a comment frame while the stream is idle."""

    def __init__(self, stream: Any, interval: float) -> None:
        self._stream = stream
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def write(self, frame: bytes) -> None:
        with self._lock:
            self._stream.write(frame)
            self._stream.flush()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.write(b": keepalive\n\n")
            except (BrokenPipeError, ConnectionResetError, ValueError, OSError):
                return


def serve(
    project_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
    allowed_roots: Sequence[Path] | None = None,
) -> CrossbyUIServer:
    """Create (but do not run) a UI server bound to a loopback address.

    ``port=0`` lets the OS pick a free port — read it back from
    :attr:`CrossbyUIServer.port`. Binding is refused for non-loopback hosts:
    this server spawns AI tools with filesystem access and is not built to face
    a network.
    """
    if host not in _BINDABLE_HOSTS:
        raise ValueError(
            f"refusing to bind {host!r}: the crossby UI binds loopback IPv4 only "
            f"({', '.join(sorted(_BINDABLE_HOSTS))})"
        )
    return CrossbyUIServer(
        (host, port), project_root, token or secrets.token_urlsafe(32), allowed_roots
    )
