"""Tests for MCP transport rewrites (Claude/Cursor → Codex)."""

from __future__ import annotations

import pytest

from crossby.sync.mcp_transports import (
    has_default_fallback,
    parse_bearer_env_var,
    parse_env_var_ref,
    rewrite_env_for_codex,
    rewrite_headers_for_codex,
)


class TestParseEnvVarRef:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("${TOKEN}", "TOKEN"),
            ("${API_KEY_2}", "API_KEY_2"),
            ("${TOKEN:-default}", "TOKEN"),  # default discarded
            ("${_underscore}", "_underscore"),
        ],
    )
    def test_matches(self, value: str, expected: str) -> None:
        assert parse_env_var_ref(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "literal",
            "Bearer ${TOKEN}",  # not a standalone reference
            "prefix ${VAR}",
            "${VAR} suffix",
            "$VAR",
            "${1BAD}",  # leading digit invalid
        ],
    )
    def test_no_match(self, value: str) -> None:
        assert parse_env_var_ref(value) is None


class TestParseBearerEnvVar:
    def test_matches(self) -> None:
        assert parse_bearer_env_var("Bearer ${API_TOKEN}") == "API_TOKEN"

    def test_matches_with_default_fallback(self) -> None:
        # Fallback default is discarded.
        assert parse_bearer_env_var("Bearer ${TOKEN:-x}") == "TOKEN"

    def test_no_match_without_bearer(self) -> None:
        assert parse_bearer_env_var("${API_TOKEN}") is None

    def test_no_match_static(self) -> None:
        assert parse_bearer_env_var("Bearer literal-token") is None


class TestHasDefaultFallback:
    def test_true(self) -> None:
        assert has_default_fallback("${VAR:-fallback}") is True

    def test_false_for_pure_var(self) -> None:
        assert has_default_fallback("${VAR}") is False

    def test_false_for_literal(self) -> None:
        assert has_default_fallback("literal") is False


class TestRewriteHeadersForCodex:
    def test_bearer_authorization(self) -> None:
        headers = {"Authorization": "Bearer ${API_TOKEN}"}
        result = rewrite_headers_for_codex(headers)
        assert result.bearer_token_env_var == "API_TOKEN"
        assert result.http_headers == {}
        assert result.env_http_headers == {}

    def test_authorization_case_insensitive(self) -> None:
        result = rewrite_headers_for_codex({"authorization": "Bearer ${TOKEN}"})
        assert result.bearer_token_env_var == "TOKEN"

    def test_env_var_header(self) -> None:
        result = rewrite_headers_for_codex({"X-API-Key": "${API_KEY}"})
        assert result.env_http_headers == {"X-API-Key": "API_KEY"}
        assert result.http_headers == {}
        assert result.bearer_token_env_var is None

    def test_static_header(self) -> None:
        result = rewrite_headers_for_codex({"X-Project": "northstar"})
        assert result.http_headers == {"X-Project": "northstar"}

    def test_mixed_headers(self) -> None:
        result = rewrite_headers_for_codex({
            "Authorization": "Bearer ${TOKEN}",
            "X-Tenant": "${TENANT_ID}",
            "X-Static": "fixed",
        })
        assert result.bearer_token_env_var == "TOKEN"
        assert result.env_http_headers == {"X-Tenant": "TENANT_ID"}
        assert result.http_headers == {"X-Static": "fixed"}

    def test_default_fallbacks_tracked(self) -> None:
        # ${VAR:-default} loses its default — track which keys had one.
        result = rewrite_headers_for_codex({"X-Tenant": "${TENANT:-default}"})
        assert "X-Tenant" in result.dropped_default_fallbacks

    def test_authorization_static_falls_through(self) -> None:
        # An Authorization header that isn't Bearer-${VAR} stays as a static
        # header; Codex won't rewrite it.
        result = rewrite_headers_for_codex({"Authorization": "Basic abc=="})
        assert result.bearer_token_env_var is None
        assert result.http_headers == {"Authorization": "Basic abc=="}


class TestRewriteEnvForCodex:
    def test_self_reference_passthrough(self) -> None:
        result = rewrite_env_for_codex({"GITHUB_TOKEN": "${GITHUB_TOKEN}"})
        assert result.env_vars == ["GITHUB_TOKEN"]
        assert result.env == {}

    def test_literal_value(self) -> None:
        result = rewrite_env_for_codex({"LOG_LEVEL": "info"})
        assert result.env == {"LOG_LEVEL": "info"}
        assert result.env_vars == []

    def test_other_var_reference_stays_literal(self) -> None:
        # KEY = "${OTHER}" is not a self-reference; keep as literal env entry
        # so the source tool can still interpolate it.
        result = rewrite_env_for_codex({"KEY": "${OTHER}"})
        assert result.env == {"KEY": "${OTHER}"}
        assert result.env_vars == []

    def test_mixed(self) -> None:
        result = rewrite_env_for_codex({
            "TOKEN": "${TOKEN}",
            "DEBUG": "true",
        })
        assert result.env_vars == ["TOKEN"]
        assert result.env == {"DEBUG": "true"}

    def test_default_fallbacks_tracked(self) -> None:
        result = rewrite_env_for_codex({"TOKEN": "${TOKEN:-fallback}"})
        # Self-reference is still detected (default ignored) — env_vars wins.
        assert result.env_vars == ["TOKEN"]
        assert "TOKEN" in result.dropped_default_fallbacks
