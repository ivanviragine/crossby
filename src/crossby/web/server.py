"""Loopback HTTP server backing the ``crossby ui`` browser terminal.

Transport is deliberately dependency-free: a stdlib :class:`ThreadingHTTPServer`
streams terminal output over Server-Sent Events and takes keystrokes back as
small POSTs. On loopback the round trip is sub-millisecond, and the project
keeps its current dependency set.

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

The page's own ``<link>`` and ``<script>`` subresources cannot attach a header,
so loading ``/?token=…`` issues a ``SameSite=Strict`` cookie that authenticates
**static assets only**. API routes keep refusing it and demand the token
explicitly, so cookie-driven CSRF cannot reach anything that spawns a process —
the ``Origin`` check above is then a second, independent barrier rather than the
only one.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import secrets
import threading
from contextlib import suppress
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
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
    SessionManager,
    SessionNotFoundError,
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

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Authenticates static assets only — never an API route.
SESSION_COOKIE = "crossby_ui_token"


class CrossbyUIServer(ThreadingHTTPServer):
    """Threaded loopback server holding the session registry and access token."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], project_root: Path, token: str) -> None:
        super().__init__(address, _RequestHandler)
        self.sessions = SessionManager(project_root)
        self.token = token
        self.project_root = project_root.resolve()

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

    def server_close(self) -> None:
        self.sessions.shutdown()
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
        is_static = route == "/" or route.startswith("/assets/")

        if not self._authorize(query, allow_cookie=is_static):
            return

        if route == "/":
            # Hand the page a cookie so its subresources authenticate themselves.
            self._serve_static("index.html", set_cookie=True)
        elif route.startswith("/assets/"):
            self._serve_static(route[len("/assets/") :])
        elif route == "/api/tools":
            self._send_json(
                HTTPStatus.OK,
                {"tools": describe_tools(), "project_root": str(self.ui.project_root)},
            )
        elif route == "/api/sessions":
            self._send_json(HTTPStatus.OK, {"sessions": self.ui.sessions.list_sessions()})
        elif route.startswith("/api/sessions/") and route.endswith("/stream"):
            self._stream(route[len("/api/sessions/") : -len("/stream")])
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

    def _stream(self, session_id: str) -> None:
        """Stream a session's output to the browser as Server-Sent Events."""
        try:
            session = self.ui.sessions.get(session_id)
        except SessionNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such session"})
            return

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
            for chunk in session.subscribe():
                payload = base64.b64encode(chunk).decode("ascii")
                heartbeat.write(f"event: output\ndata: {payload}\n\n".encode())
            heartbeat.write(
                f"event: exit\ndata: {json.dumps({'exit_code': session.exit_code})}\n\n".encode()
            )
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("web.stream.disconnected", session=session_id)
        finally:
            heartbeat.stop()

    # -- static -----------------------------------------------------------
    def _serve_static(self, relative: str, *, set_cookie: bool = False) -> None:
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
            cookie=self.ui.token if set_cookie else None,
        )

    # -- plumbing ---------------------------------------------------------
    def _authorize(self, query: dict[str, list[str]], *, allow_cookie: bool = False) -> bool:
        """Reject anything that is not a same-origin, correctly-tokened request.

        ``allow_cookie`` is set only for static assets, whose ``<link>``/
        ``<script>`` requests cannot carry a header. API routes never enable it.
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

        supplied = self.headers.get("X-Crossby-Token") or next(iter(query.get("token", [])), "")
        if not supplied and allow_cookie:
            supplied = self._cookie_token()
        if not secrets.compare_digest(supplied, self.ui.token):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "invalid or missing token"})
            return False
        return True

    def _with_session(self, session_id: str, handler: Any) -> None:
        try:
            handler(session_id)
        except SessionNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such session"})

    def _cookie_token(self) -> str:
        """Read the access token from the request's cookie header, if present."""
        jar = SimpleCookie()
        with suppress(CookieError):
            jar.load(self.headers.get("Cookie", ""))
        morsel = jar.get(SESSION_COOKIE)
        return morsel.value if morsel else ""

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
        cookie: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cookie is not None:
            # No Secure flag: loopback HTTP would drop it. HttpOnly because the
            # page reads its token from the URL and never needs script access.
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={cookie}; Path=/; HttpOnly; SameSite=Strict",
            )
        # The page is same-origin and token-gated; deny framing and sniffing so a
        # hostile tab cannot wrap or reinterpret it.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        with_body = self.command != "HEAD"
        if with_body:
            self.wfile.write(body)


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
) -> CrossbyUIServer:
    """Create (but do not run) a UI server bound to a loopback address.

    ``port=0`` lets the OS pick a free port — read it back from
    :attr:`CrossbyUIServer.port`. Binding is refused for non-loopback hosts:
    this server spawns AI tools with filesystem access and is not built to face
    a network.
    """
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind {host!r}: the crossby UI serves loopback addresses only"
        )
    return CrossbyUIServer((host, port), project_root, token or secrets.token_urlsafe(32))
