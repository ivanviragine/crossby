"""Integration tests for the ``crossby ui`` HTTP server.

The security boundary here is load-bearing — this server spawns AI tools with
filesystem access — so the auth tests drive a real socket rather than calling
handler methods directly.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from crossby.web import serve
from crossby.web.server import SESSION_COOKIE, CrossbyUIServer

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
    def test_index_is_served_and_issues_a_cookie(self, server: CrossbyUIServer) -> None:
        status, body, headers = request(server, "GET", "/")
        assert status == 200
        assert b"<title>crossby</title>" in body
        assert SESSION_COOKIE in headers.get("Set-Cookie", "")
        assert "HttpOnly" in headers["Set-Cookie"]
        assert "SameSite=Strict" in headers["Set-Cookie"]

    def test_assets_authenticate_by_cookie(self, server: CrossbyUIServer) -> None:
        """A <script> tag cannot send a header, so the cookie must carry it."""
        status, body, _ = request(
            server, "GET", "/assets/app.js", token=None, cookie=f"{SESSION_COOKIE}={TOKEN}"
        )
        assert status == 200
        assert b"crossby" in body

    def test_assets_reject_a_forged_cookie(self, server: CrossbyUIServer) -> None:
        status, _, _ = request(
            server, "GET", "/assets/app.js", token=None, cookie=f"{SESSION_COOKIE}=forged"
        )
        assert status == 401

    def test_api_never_accepts_the_cookie(self, server: CrossbyUIServer) -> None:
        """Cookie auth is scoped to static files; the API demands the token."""
        status, _, _ = request(
            server, "GET", "/api/tools", token=None, cookie=f"{SESSION_COOKIE}={TOKEN}"
        )
        assert status == 401

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

    def test_unknown_session_stream_is_not_found(self, server: CrossbyUIServer) -> None:
        assert request(server, "GET", "/api/sessions/missing/stream")[0] == 404

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
