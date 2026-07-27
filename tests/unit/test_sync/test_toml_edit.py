"""Tests for the targeted TOML splicers.

These back the comment-preservation guarantee for ``.codex/config.toml``. The
splicers are best-effort by design: anything they can't interpret must return
``None`` so the caller falls back to a full (lossy but correct) round trip,
never produce a document that says something other than what was intended.
"""

from __future__ import annotations

import tomllib

import pytest

from crossby.sync.toml_edit import remove_table, set_scalar, splice_or_none, upsert_table


class TestSetScalar:
    def test_adds_key_to_existing_table(self) -> None:
        text = "# lead\n[features]\nother = 1  # trailing\n"
        out = set_scalar(text, ("features",), "codex_hooks", "true")
        assert out is not None
        assert "# lead" in out
        assert "# trailing" in out
        assert tomllib.loads(out)["features"] == {"other": 1, "codex_hooks": True}

    def test_replaces_existing_key(self) -> None:
        text = "[features]\ncodex_hooks = false\nkeep = 2\n"
        out = set_scalar(text, ("features",), "codex_hooks", "true")
        assert out is not None
        parsed = tomllib.loads(out)
        assert parsed["features"]["codex_hooks"] is True
        assert parsed["features"]["keep"] == 2

    def test_creates_missing_table(self) -> None:
        text = 'model = "gpt-5.5"\n'
        out = set_scalar(text, ("features",), "codex_hooks", "true")
        assert out is not None
        assert tomllib.loads(out) == {"model": "gpt-5.5", "features": {"codex_hooks": True}}

    def test_empty_document(self) -> None:
        out = set_scalar("", ("features",), "codex_hooks", "true")
        assert out is not None
        assert tomllib.loads(out) == {"features": {"codex_hooks": True}}

    def test_ignores_headers_inside_multiline_strings(self) -> None:
        text = 'doc = """\n[features]\ncodex_hooks = false\n"""\n'
        out = set_scalar(text, ("features",), "codex_hooks", "true")
        assert out is not None
        parsed = tomllib.loads(out)
        assert parsed["features"]["codex_hooks"] is True
        assert "[features]" in parsed["doc"], "the string literal was modified"

    def test_ignores_brackets_in_multiline_arrays(self) -> None:
        text = "nested = [\n  [1, 2],\n]\n[features]\nx = 1\n"
        out = set_scalar(text, ("features",), "codex_hooks", "true")
        assert out is not None
        parsed = tomllib.loads(out)
        assert parsed["nested"] == [[1, 2]]
        assert parsed["features"] == {"x": 1, "codex_hooks": True}

    def test_bails_on_unterminated_string(self) -> None:
        assert set_scalar('a = "oops\n[features]\n', ("features",), "k", "true") is None


class TestUpsertTable:
    def test_appends_new_table(self) -> None:
        text = "# top\nmodel = 1\n"
        out = upsert_table(text, ("mcp_servers", "srv"), '[mcp_servers.srv]\ncommand = "x"\n')
        assert out is not None
        assert "# top" in out
        assert tomllib.loads(out)["mcp_servers"]["srv"]["command"] == "x"

    def test_replaces_existing_table_and_its_children(self) -> None:
        text = '[mcp_servers.srv]\ncommand = "old"\n\n[mcp_servers.srv.env]\nK = "v"\n'
        out = upsert_table(text, ("mcp_servers", "srv"), '[mcp_servers.srv]\ncommand = "new"\n')
        assert out is not None
        parsed = tomllib.loads(out)
        assert parsed["mcp_servers"]["srv"] == {"command": "new"}
        assert out.count("[mcp_servers.srv]") == 1

    def test_keeps_the_comment_introducing_the_next_table(self) -> None:
        text = '[mcp_servers.srv]\ncommand = "old"\n\n# my profiles\n[profiles.fast]\nx = 1\n'
        out = upsert_table(text, ("mcp_servers", "srv"), '[mcp_servers.srv]\ncommand = "new"\n')
        assert out is not None
        assert "# my profiles" in out
        assert tomllib.loads(out)["profiles"]["fast"] == {"x": 1}

    def test_leaves_unrelated_tables_untouched(self) -> None:
        text = "[a]\nx = 1  # note\n\n[b]\ny = 2\n"
        out = upsert_table(text, ("c",), "[c]\nz = 3\n")
        assert out is not None
        assert "# note" in out
        assert tomllib.loads(out) == {"a": {"x": 1}, "b": {"y": 2}, "c": {"z": 3}}


class TestRemoveTable:
    def test_removes_table_and_children(self) -> None:
        text = '[mcp_servers.a]\nc = "x"\n\n[mcp_servers.a.env]\nK = "v"\n\n[other]\ny = 1\n'
        out = remove_table(text, ("mcp_servers", "a"))
        assert out is not None
        parsed = tomllib.loads(out)
        assert "mcp_servers" not in parsed
        assert parsed["other"] == {"y": 1}

    def test_missing_table_is_a_no_op(self) -> None:
        text = "[a]\nx = 1\n"
        assert remove_table(text, ("nope",)) == text


class TestSpliceOrNone:
    def test_accepts_a_matching_splice(self) -> None:
        assert splice_or_none("a = 1\n", {"a": 1}) == "a = 1\n"

    def test_rejects_a_mismatching_splice(self) -> None:
        assert splice_or_none("a = 2\n", {"a": 1}) is None

    def test_rejects_invalid_toml(self) -> None:
        assert splice_or_none("a = = 1\n", {"a": 1}) is None

    def test_passes_none_through(self) -> None:
        assert splice_or_none(None, {"a": 1}) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a = 1\n",
        "[a]\nb = 1",  # no trailing newline
        "# c\r\n[features]\r\nx = 1\r\n",  # CRLF
        "[[jobs]]\nname = 'a'\n\n[[jobs]]\nname = 'b'\n",  # array of tables
        "[features]\nopts = { a = 1, b = 2 }\n",  # inline table
        "path = '''\n[not_a_table]\n'''\n[features]\nx = 1\n",  # literal multiline
    ],
)
def test_set_scalar_never_corrupts_a_valid_document(text: str) -> None:
    """Whatever the input shape, the result parses and keeps the original data."""
    before = tomllib.loads(text)
    out = set_scalar(text, ("features",), "codex_hooks", "true")
    assert out is not None
    after = tomllib.loads(out)
    assert after["features"]["codex_hooks"] is True
    for key, value in before.items():
        if key == "features":
            continue
        assert after[key] == value
