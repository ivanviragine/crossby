"""Tests for the targeted TOML splicers.

These back the comment-preservation guarantee for ``.codex/config.toml``. The
splicers are best-effort by design: anything they can't interpret must return
``None`` so the caller falls back to a full (lossy but correct) round trip,
never produce a document that says something other than what was intended.
"""

from __future__ import annotations

import random
import tomllib

import pytest
import tomli_w

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


class TestImplicitlyDefinedTables:
    """A table can exist purely by implication, via a sub-table header.

    ``[mcp_servers.alpha.env]`` with no ``[mcp_servers.alpha]`` still defines
    ``alpha``. Both orderings are legal TOML, including the child written above
    its parent.
    """

    def test_remove_takes_an_implicit_parent_with_it(self) -> None:
        text = '# keep\n[mcp_servers.alpha.env]\nK = "v"\n'
        out = remove_table(text, ("mcp_servers", "alpha"))
        assert out is not None
        assert "mcp_servers" not in tomllib.loads(out)

    def test_remove_clears_every_descendant(self) -> None:
        text = (
            '[mcp_servers.alpha]\nc = "a"\n'
            '[mcp_servers.alpha.env]\nK = "v"\n'
            "[other]\nx = 1\n"
            '[mcp_servers.alpha.headers]\nH = "v"\n'
        )
        out = remove_table(text, ("mcp_servers", "alpha"))
        assert out is not None
        parsed = tomllib.loads(out)
        assert "mcp_servers" not in parsed
        assert parsed["other"] == {"x": 1}

    def test_upsert_bails_when_a_child_sits_above_its_parent(self) -> None:
        """Replacing the parent block alone would strand the earlier child."""
        text = '[mcp_servers.alpha.env]\nK = "v"\n[mcp_servers.alpha]\ncommand = "a"\n'
        assert upsert_table(text, ("mcp_servers", "alpha"), "[mcp_servers.alpha]\n") is None

    def test_upsert_bails_when_a_child_is_detached_after_its_parent(self) -> None:
        text = (
            '[mcp_servers.alpha]\ncommand = "a"\n'
            "[unrelated]\nx = 1\n"
            '[mcp_servers.alpha.env]\nK = "v"\n'
        )
        assert upsert_table(text, ("mcp_servers", "alpha"), "[mcp_servers.alpha]\n") is None


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


# ---------------------------------------------------------------------------
# Differential property test
# ---------------------------------------------------------------------------

_FRAGMENTS = (
    "# a comment\n",
    "\n",
    "top = 1\n",
    'name = "x"\n',
    'multi = """\n[features]\nfake = 1\n"""\n',
    "lit = '''\n[nope]\n'''\n",
    "arr = [\n  [1, 2],\n  [3, 4],\n]\n",
    "inline = { a = 1, b = 2 }\n",
    '[sandbox]\nmode = "rw"  # trailing\n',
    "[features]\nother = true\n",
    '[mcp_servers.alpha]\ncommand = "a"\n',
    '[mcp_servers.alpha.env]\nK = "v"\n',
    '[mcp_servers.beta]\nurl = "https://x"\n',
    '[profiles.fast]\nmodel = "m"\n',
    '[[jobs]]\nname = "j"\n',
    '[[mcp_servers.alpha.jobs]]\nname = "j"\n',
    "dotted.key = 5\n",
    '"quoted key" = 6\n',
)


def _documents(seed: int, count: int) -> list[str]:
    """Deterministically assemble pseudo-random TOML documents."""
    rng = random.Random(seed)
    docs = []
    for _ in range(count):
        docs.append("".join(rng.choice(_FRAGMENTS) for _ in range(rng.randint(1, 7))))
    return docs


def test_splices_never_disagree_with_the_intended_data() -> None:
    """The core guarantee: a splice either matches intent exactly, or bails.

    Every edit is compared against the data a full ``tomli_w`` round trip would
    have produced. Anything else is a corrupted or silently-wrong config file,
    which is the failure mode this module exists to avoid. Returning ``None``
    (fall back to the round trip) is always an acceptable answer.
    """
    checked = 0
    for text in _documents(seed=1234, count=600):
        try:
            before = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            continue  # only valid documents carry a meaningful expectation

        cases: list[tuple[str | None, dict[str, object]]] = []

        features = {**(before.get("features") or {}), "codex_hooks": True}
        cases.append(
            (
                set_scalar(text, ("features",), "codex_hooks", "true"),
                {**before, "features": features},
            )
        )

        entry = {"command": "new"}
        servers = {**(before.get("mcp_servers") or {}), "alpha": entry}
        cases.append(
            (
                upsert_table(
                    text,
                    ("mcp_servers", "alpha"),
                    tomli_w.dumps({"mcp_servers": {"alpha": entry}}),
                ),
                {**before, "mcp_servers": servers},
            )
        )

        remaining = {k: v for k, v in (before.get("mcp_servers") or {}).items() if k != "alpha"}
        expected_removed = {k: v for k, v in before.items() if k != "mcp_servers"}
        if remaining:
            expected_removed["mcp_servers"] = remaining
        cases.append((remove_table(text, ("mcp_servers", "alpha")), expected_removed))

        for spliced, expected in cases:
            if spliced is None:
                continue  # bailing is always allowed
            assert tomllib.loads(spliced) == expected, f"input:\n{text}\ngot:\n{spliced}"
            checked += 1

    assert checked > 500, f"property test degenerated to {checked} real assertions"


class TestRemoveTableBailsOnHeaderlessDefinitions:
    """A table defined without a header of its own can't be removed textually.

    Returning the unchanged document would be indistinguishable from a real
    removal, so these must bail and let the caller fall back.
    """

    @pytest.mark.parametrize(
        ("label", "text"),
        [
            ("array of tables", '[[mcp_servers.alpha]]\ncommand = "a"\n'),
            ("inline table", '[mcp_servers]\nalpha = { command = "a" }\n'),
            ("dotted key", '[mcp_servers]\nalpha.command = "a"\n'),
        ],
    )
    def test_bails(self, label: str, text: str) -> None:
        assert remove_table(text, ("mcp_servers", "alpha")) is None, label

    def test_genuinely_absent_is_still_a_no_op(self) -> None:
        text = "[other]\nx = 1\n"
        assert remove_table(text, ("mcp_servers", "alpha")) == text


def test_remove_bails_on_a_detached_array_of_tables_descendant() -> None:
    """``[[a.b.jobs]]`` split from ``[a.b]`` can't be folded into the removal.

    Removing only the table block would leave the array behind, so ``a.b`` would
    still be defined — a half-removal reported as success.
    """
    text = (
        '[mcp_servers.alpha]\nx = 1\n[unrelated]\ny = 2\n[[mcp_servers.alpha.jobs]]\nname = "j"\n'
    )
    assert remove_table(text, ("mcp_servers", "alpha")) is None


def test_remove_handles_a_contiguous_array_of_tables_descendant() -> None:
    """When it trails its parent directly, the extent already covers it."""
    text = '[mcp_servers.alpha]\nx = 1\n[[mcp_servers.alpha.jobs]]\nname = "j"\n[other]\ny = 2\n'
    out = remove_table(text, ("mcp_servers", "alpha"))
    assert out is not None
    parsed = tomllib.loads(out)
    assert "mcp_servers" not in parsed
    assert parsed["other"] == {"y": 2}
