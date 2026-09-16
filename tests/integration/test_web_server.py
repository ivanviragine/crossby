"""Integration tests for the ``crossby ui`` HTTP server.

The security boundary here is load-bearing — this server spawns AI tools with
filesystem access — so the auth tests drive a real socket rather than calling
handler methods directly.
"""

from __future__ import annotations

import http.client
import json
import socket
import struct
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.models.ai import AIToolID
from crossby.utils.pty_runner import WindowSize
from crossby.web import serve
from crossby.web.server import CrossbyUIServer
from crossby.web.sessions import LaunchRequest, MultiplexedStream

TOKEN = "test-token-not-a-secret"


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CrossbyUIServer]:
    instance = serve(tmp_path, port=0, token=TOKEN)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def request(
    server: CrossbyUIServer,
    method: str,
    path: str,
    *,
    token: str | None = TOKEN,
    host: str | None = None,
    origin: str | None = None,
    cookie: str | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    """Issue one request, controlling every header the auth layer inspects."""
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host or f"127.0.0.1:{server.port}")
        if token is not None:
            conn.putheader("X-Crossby-Token", token)
        if origin is not None:
            conn.putheader("Origin", origin)
        if cookie is not None:
            conn.putheader("Cookie", cookie)
        payload = b"" if body is None else json.dumps(body).encode()
        if body is not None:
            conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(payload)))
        conn.endheaders()
        if payload:
            conn.send(payload)
        response = conn.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        conn.close()


# Echoes stdin forever. These tests exercise session and stream machinery, not
# the adapters, so they must not require a real AI tool on PATH — CI has none,
# which is precisely how they passed locally and failed there.
_STAND_IN = [
    sys.executable,
    "-u",
    "-c",
    "import sys\nfor line in sys.stdin:\n    sys.stdout.write(line)\n    sys.stdout.flush()\n",
]


@pytest.fixture
def stub_tool() -> Iterator[None]:
    """Make every launch spawn a harmless local process instead of an AI tool."""
    with patch.object(AbstractAITool, "build_launch_command", return_value=list(_STAND_IN)):
        yield


def open_session(server: CrossbyUIServer) -> Any:
    return server.sessions.create(
        LaunchRequest(tool=AIToolID.CLAUDE, size=WindowSize(cols=80, rows=24))
    )


class TestTokenAuth:
    def test_missing_token_is_unauthorized(self, server: CrossbyUIServer) -> None:
        assert request(server, "GET", "/api/tools", token=None)[0] == 401

    def test_wrong_token_is_unauthorized(self, server: CrossbyUIServer) -> None:
        assert request(server, "GET", "/api/tools", token="wrong")[0] == 401

    def test_correct_token_is_accepted(self, server: CrossbyUIServer) -> None:
        assert request(server, "GET", "/api/tools")[0] == 200

    def test_token_may_travel_as_a_query_parameter(self, server: CrossbyUIServer) -> None:
        """EventSource cannot set headers, so the stream route needs this."""
        status, _, _ = request(server, "GET", f"/api/tools?token={TOKEN}", token=None)
        assert status == 200


class TestRebindingAndCsrfDefences:
    def test_foreign_host_header_is_refused(self, server: CrossbyUIServer) -> None:
        """A DNS-rebinding attack arrives with an attacker-controlled Host."""
        assert request(server, "GET", "/api/tools", host="attacker.example")[0] == 403

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
    def test_loopback_hosts_are_accepted(self, server: CrossbyUIServer, host: str) -> None:
        status, _, _ = request(server, "GET", "/api/tools", host=f"{host}:{server.port}")
        assert status == 200

    def test_cross_origin_request_is_refused(self, server: CrossbyUIServer) -> None:
        status, _, _ = request(server, "GET", "/api/tools", origin="https://evil.example")
        assert status == 403

    def test_same_origin_request_is_accepted(self, server: CrossbyUIServer) -> None:
        origin = f"http://127.0.0.1:{server.port}"
        assert request(server, "GET", "/api/tools", origin=origin)[0] == 200


class TestStaticAssets:
    """The shell is deliberately unauthenticated.

    `<link>` and `<script>` cannot attach a header, and the cookie that once
    covered them was fragile in the way that mattered: cookies ignore the port,
    so they outlive a restart and leak between concurrent servers, while every
    run mints a fresh token. A reload from a bookmark then 401'd the stylesheet
    and script while the HTML loaded from cache — a broken page, no explanation.
    The shell holds nothing worth guarding; the API still demands the token.
    """

    def test_index_is_served_without_a_token(self, server: CrossbyUIServer) -> None:
        status, body, _ = request(server, "GET", "/", token=None)
        assert status == 200
        assert b"<title>crossby</title>" in body

    def test_index_is_never_cached(self, server: CrossbyUIServer) -> None:
        """A stale shell would hide a token change after a restart."""
        _, _, headers = request(server, "GET", "/", token=None)
        assert "no-store" in headers.get("Cache-Control", "")

    def test_assets_are_served_without_a_token(self, server: CrossbyUIServer) -> None:
        status, body, _ = request(server, "GET", "/assets/app.js", token=None)
        assert status == 200
        assert b"crossby" in body

    def test_shell_still_refuses_cross_origin(self, server: CrossbyUIServer) -> None:
        """Unauthenticated does not mean unprotected: no foreign page may embed it."""
        for path in ("/", "/assets/app.js"):
            status, _, _ = request(server, "GET", path, token=None, origin="https://evil.example")
            assert status == 403, path

    def test_shell_still_refuses_a_foreign_host(self, server: CrossbyUIServer) -> None:
        for path in ("/", "/assets/app.js"):
            status, _, _ = request(server, "GET", path, token=None, host="attacker.example")
            assert status == 403, path

    def test_api_still_requires_the_token(self, server: CrossbyUIServer) -> None:
        """The shell opening up must not have opened up anything that spawns."""
        for path in ("/api/tools", "/api/sessions", "/api/stream"):
            status, _, _ = request(server, "GET", path, token=None)
            assert status == 401, path

    @pytest.mark.parametrize(
        "path",
        [
            "/assets/../../../../etc/passwd",
            "/assets/../server.py",
            "/assets/..%2f..%2fpyproject.toml",
        ],
    )
    def test_path_traversal_is_blocked(self, server: CrossbyUIServer, path: str) -> None:
        assert request(server, "GET", path)[0] in (403, 404)

    def test_vendored_assets_are_present(self, server: CrossbyUIServer) -> None:
        for asset in ("vendor/xterm.js", "vendor/xterm.css", "vendor/addon-fit.js"):
            status, body, _ = request(server, "GET", f"/assets/{asset}")
            assert status == 200, asset
            assert body, asset


class TestApiContract:
    def test_tools_payload_shape(self, server: CrossbyUIServer) -> None:
        _, body, _ = request(server, "GET", "/api/tools")
        payload = json.loads(body)
        assert "tools" in payload
        assert payload["project_root"] == str(server.project_root)

    def test_unknown_route_is_not_found(self, server: CrossbyUIServer) -> None:
        assert request(server, "GET", "/api/nope")[0] == 404

    def test_per_session_stream_route_is_gone(self, server: CrossbyUIServer) -> None:
        """Superseded by the multiplexed stream; see TestMultiplexedStream."""
        assert request(server, "GET", "/api/sessions/any/stream")[0] == 404

    def test_unknown_session_input_is_not_found(self, server: CrossbyUIServer) -> None:
        status, _, _ = request(server, "POST", "/api/sessions/missing/input", body={"data": "x"})
        assert status == 404

    def test_malformed_json_is_rejected(self, server: CrossbyUIServer) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
        try:
            conn.request(
                "POST",
                "/api/sessions",
                body=b"{not json",
                headers={"X-Crossby-Token": TOKEN, "Content-Type": "application/json"},
            )
            assert conn.getresponse().status == 400
        finally:
            conn.close()

    def test_unknown_tool_is_rejected(self, server: CrossbyUIServer) -> None:
        status, body, _ = request(server, "POST", "/api/sessions", body={"tool": "not-a-real-tool"})
        assert status == 400
        assert "unknown tool" in json.loads(body)["error"]

    def test_empty_session_list(self, server: CrossbyUIServer) -> None:
        _, body, _ = request(server, "GET", "/api/sessions")
        assert json.loads(body) == {"sessions": []}


class TestBindingPolicy:
    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com"])
    def test_refuses_non_loopback_bind(self, tmp_path: Path, host: str) -> None:
        """This server spawns AI tools; it must never face a network."""
        with pytest.raises(ValueError, match="loopback"):
            serve(tmp_path, host=host)


class TestMultiplexedStream:
    """One SSE connection carries every session.

    A browser allows roughly six HTTP/1.1 connections per origin. With a stream
    per session the sixth terminal exhausts the pool and stalls every other
    request — including the POSTs carrying keystrokes — so tab count would
    silently become a transport limit.
    """

    def test_stream_requires_a_token(self, server: CrossbyUIServer) -> None:
        assert request(server, "GET", "/api/stream", token=None)[0] == 401

    def test_stream_rejects_cross_origin(self, server: CrossbyUIServer) -> None:
        status, _, _ = request(server, "GET", "/api/stream", origin="https://evil.example")
        assert status == 403

    def test_stream_carries_every_session_tagged(
        self, server: CrossbyUIServer, stub_tool: None
    ) -> None:
        """Two sessions, one connection, frames tagged by session id."""
        first = open_session(server)
        second = open_session(server)
        try:
            seen: set[str] = set()
            with MultiplexedStream(server.sessions) as stream:
                first.write(b"echo first\n")
                second.write(b"echo second\n")
                deadline = time.monotonic() + 10
                for event in stream:
                    seen.add(event.session_id)
                    if {first.id, second.id} <= seen or time.monotonic() > deadline:
                        break
            assert {first.id, second.id} <= seen
        finally:
            first.close()
            second.close()

    def test_stream_picks_up_sessions_created_later(
        self, server: CrossbyUIServer, stub_tool: None
    ) -> None:
        """A tab opened after the stream is live must still receive output."""
        with MultiplexedStream(server.sessions) as stream:
            later = open_session(server)
            try:
                later.write(b"echo later\n")
                deadline = time.monotonic() + 10
                for event in stream:
                    if event.session_id == later.id:
                        break
                    assert time.monotonic() < deadline, "no frame for the later session"
            finally:
                later.close()

    def test_closing_the_stream_releases_idle_pumps(
        self, server: CrossbyUIServer, stub_tool: None
    ) -> None:
        """A pump parked on a silent session must not outlive its stream."""
        session = open_session(server)
        try:
            before = threading.active_count()
            stream = MultiplexedStream(server.sessions)
            stream.__enter__()
            time.sleep(0.4)
            stream.close()
            time.sleep(0.6)
            assert threading.active_count() <= before + 1, "pump thread leaked"
        finally:
            session.close()


class TestExitProvenance:
    def test_stopped_session_is_labelled_stopped_not_exit_minus_one(
        self, server: CrossbyUIServer, stub_tool: None
    ) -> None:
        """`close()` kills with SIGHUP; a raw returncode of -1 reads as nonsense."""
        session = open_session(server)
        session.close()
        described = server.sessions.describe(session.id)
        assert described["stopped"] is True
        assert described["exit_signal"] == "SIGHUP"


class TestClientDisconnectNoise:
    """A browser resets connections constantly and it is never a server fault.

    `socketserver` prints a full traceback for anything escaping a handler, so
    closing a tab, reloading, or reaping an idle pooled connection produced a
    wall of `ConnectionResetError` tracebacks that looked like a crash and
    buried real errors.
    """

    def test_abrupt_client_reset_is_not_reported(
        self, server: CrossbyUIServer, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Reset a keep-alive connection while the server awaits the next request."""
        conn = socket.create_connection(("127.0.0.1", server.port), timeout=10)
        request = (
            f"GET /api/tools HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.port}\r\n"
            f"X-Crossby-Token: {TOKEN}\r\n\r\n"
        ).encode()
        conn.sendall(request)
        conn.recv(65536)

        # SO_LINGER with a zero timeout sends RST rather than FIN — exactly what
        # a browser tearing down a pooled connection looks like.
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()
        time.sleep(0.6)

        captured = capfd.readouterr()
        assert "Traceback" not in captured.err, captured.err
        assert "ConnectionResetError" not in captured.err

    def test_unexpected_errors_are_still_reported(
        self, server: CrossbyUIServer, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Only disconnects are quiet — a genuine fault must still be visible."""
        try:
            raise ValueError("a real failure")
        except ValueError:
            server.handle_error(None, ("127.0.0.1", 12345))

        captured = capfd.readouterr()
        assert "ValueError" in captured.err
        assert "a real failure" in captured.err
