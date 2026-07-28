"""Targeted textual edits for TOML documents.

Round-tripping through ``tomllib`` + ``tomli_w`` is lossy: parsing returns plain
data, so every comment, blank line and key ordering in the user's file is
discarded when it's dumped back. ``.codex/config.toml`` is a file crossby
*merges into* rather than owns — the user's own model, sandbox and profile
settings live there — so rewriting it to add one boolean or one MCP server is
not an acceptable trade.

crossby only ever needs three edits, so these helpers splice the document
textually and leave everything they don't own byte-for-byte intact:

- :func:`set_scalar` — set one key inside one table (``[features].codex_hooks``)
- :func:`upsert_table` — add or replace one table (``[mcp_servers.<name>]``)
- :func:`remove_table` — drop one table

Every function is best-effort and returns ``None`` when it can't apply the edit
confidently (an unterminated string, a table defined inline, a non-contiguous
child table). They guarantee two things: the return value is either ``None`` or
valid TOML, and they never silently report a no-op as a completed edit. Callers
still verify the *data* against their intent with :func:`splice_or_none` before
writing, and fall back to the full round-trip when it disagrees.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class _Header:
    """One ``[table]`` / ``[[array]]`` header and where it sits in the text."""

    parts: tuple[str, ...]
    is_array: bool
    start: int  # offset of the start of the header's line
    body_start: int  # offset just past the header line's newline


@dataclass(frozen=True)
class _Assign:
    """One ``key = value`` assignment and the span of text it occupies."""

    parts: tuple[str, ...]
    start: int  # offset of the first character of the key
    end: int  # offset just past the assignment's final newline


def _find_string_end(text: str, i: int) -> int | None:
    """Return the offset just past the single-line string starting at *i*."""
    quote = text[i]
    j = i + 1
    while j < len(text):
        c = text[j]
        if c == "\n":
            return None
        if quote == '"' and c == "\\":
            j += 2
            continue
        if c == quote:
            return j + 1
        j += 1
    return None


def _find_multiline_end(text: str, start: int, delim: str) -> int | None:
    """Return the offset just past the multi-line string closing *delim*."""
    j = start
    while j < len(text):
        if delim == '"""' and text[j] == "\\":
            j += 2
            continue
        if text.startswith(delim, j):
            return j + 3
        j += 1
    return None


def _unquote(raw: str) -> str:
    """Strip the surrounding quotes from a quoted key segment."""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def _split_key(raw: str) -> tuple[str, ...]:
    """Split a (possibly dotted, possibly quoted) key into its parts."""
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c in "\"'":
            end = _find_string_end(raw, i)
            if end is None:
                buf.append(c)
                i += 1
                continue
            parts.append(_unquote(raw[i:end]))
            i = end
            continue
        if c == ".":
            if "".join(buf).strip():
                parts.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return tuple(parts)


def _parse_header(text: str, i: int) -> tuple[tuple[str, ...], bool, int] | None:
    """Parse the table header starting at *i*; return (parts, is_array, end)."""
    is_array = text.startswith("[[", i)
    j = i + (2 if is_array else 1)
    parts: list[str] = []
    buf: list[str] = []
    while j < len(text):
        c = text[j]
        if c == "\n":
            return None  # a header must close on its own line
        if c in "\"'":
            end = _find_string_end(text, j)
            if end is None:
                return None
            parts.append(_unquote(text[j:end]))
            j = end
            continue
        if c == ".":
            if "".join(buf).strip():
                parts.append("".join(buf).strip())
            buf = []
            j += 1
            continue
        if c == "]":
            if "".join(buf).strip():
                parts.append("".join(buf).strip())
            closer = "]]" if is_array else "]"
            if not text.startswith(closer, j):
                return None
            j += len(closer)
            newline = text.find("\n", j)
            end_offset = len(text) if newline == -1 else newline + 1
            return tuple(parts), is_array, end_offset
        buf.append(c)
        j += 1
    return None


def _scan(text: str) -> tuple[list[_Header], list[_Assign]] | None:
    """Locate every table header and top-level key assignment in *text*.

    Returns ``None`` when the document contains something the scanner can't
    interpret (an unterminated string, a malformed header) — the signal for
    callers to fall back rather than splice blind.
    """
    headers: list[_Header] = []
    assigns: list[_Assign] = []
    i, n = 0, len(text)
    depth = 0  # nesting inside a value: [ ] arrays and { } inline tables
    pending_value = False
    at_line_start = True
    line_start = 0
    key_start: int | None = None
    assign_key: str | None = None

    while i < n:
        ch = text[i]

        if ch == "\n":
            i += 1
            if depth == 0:
                if assign_key is not None and key_start is not None:
                    assigns.append(_Assign(_split_key(assign_key), key_start, i))
                    assign_key = None
                pending_value = False
                key_start = None
            line_start = i
            at_line_start = True
            continue

        if ch in " \t\r":
            i += 1
            continue

        if ch == "#":
            newline = text.find("\n", i)
            i = n if newline == -1 else newline
            continue

        if text.startswith('"""', i) or text.startswith("'''", i):
            end = _find_multiline_end(text, i + 3, text[i : i + 3])
            if end is None:
                return None
            i = end
            at_line_start = False
            continue

        if ch in "\"'":
            if at_line_start and depth == 0 and not pending_value and key_start is None:
                key_start = i
            end = _find_string_end(text, i)
            if end is None:
                return None
            i = end
            at_line_start = False
            continue

        if ch == "[" and at_line_start and depth == 0 and not pending_value:
            parsed = _parse_header(text, i)
            if parsed is None:
                return None
            parts, is_array, end = parsed
            headers.append(_Header(parts, is_array, line_start, end))
            i = end
            line_start = end
            at_line_start = True
            continue

        if ch in "[{":
            depth += 1
            i += 1
            at_line_start = False
            continue

        if ch in "]}":
            depth = max(0, depth - 1)
            i += 1
            at_line_start = False
            continue

        if ch == "=" and depth == 0 and not pending_value:
            pending_value = True
            if key_start is not None:
                assign_key = text[key_start:i].strip()
            i += 1
            at_line_start = False
            continue

        if at_line_start and depth == 0 and not pending_value and key_start is None:
            key_start = i
        i += 1
        at_line_start = False

    if assign_key is not None and key_start is not None:
        assigns.append(_Assign(_split_key(assign_key), key_start, n))
    return headers, assigns


def _find_table(headers: list[_Header], parts: tuple[str, ...]) -> int | None:
    """Return the index of the ``[parts]`` table header, or None if absent."""
    for idx, header in enumerate(headers):
        if not header.is_array and header.parts == parts:
            return idx
    return None


def _detach_trailing_comments(text: str, end: int, floor: int) -> int:
    """Hand back the run of blank/comment lines sitting just before *end*.

    A comment directly above a table header reads as an introduction to that
    header, so replacing or removing the *preceding* block must not consume it.
    Never trims past *floor* (the block's own header line).
    """
    if end >= len(text):
        return end
    boundary = end
    while boundary > floor:
        line_start = text.rfind("\n", 0, boundary - 1) + 1
        if line_start < floor:
            break
        stripped = text[line_start:boundary].strip()
        if stripped and not stripped.startswith("#"):
            break
        boundary = line_start
    return boundary


def _extent(text: str, headers: list[_Header], idx: int) -> tuple[int, int]:
    """Return the (start, end) offsets of the block owned by ``headers[idx]``.

    A table's block runs through any child tables that immediately follow it —
    ``[mcp_servers.foo]`` owns the ``[mcp_servers.foo.env]`` that trails it —
    but stops short of the blank lines and comments introducing the next table.
    """
    parts = headers[idx].parts
    start = headers[idx].start
    for later in headers[idx + 1 :]:
        if len(later.parts) > len(parts) and later.parts[: len(parts)] == parts:
            continue
        return start, _detach_trailing_comments(text, later.start, headers[idx].body_start)
    return start, len(text)


def _has_detached_child(
    headers: list[_Header], parts: tuple[str, ...], span: tuple[int, int]
) -> bool:
    """True when a child table lives outside its parent's block *span*.

    Replacing the parent would then leave a stale definition behind. The child
    can sit either after the parent (separated by an unrelated table) or above
    it — ``[a.b.c]`` before ``[a.b]`` is legal TOML — so the whole document is
    checked rather than just what follows. The caller falls back to a full
    rewrite when this fires.
    """
    start, end = span
    return any(
        not (start <= header.start < end)
        and len(header.parts) > len(parts)
        and header.parts[: len(parts)] == parts
        for header in headers
    )


def _render_key(parts: tuple[str, ...]) -> str:
    """Render a dotted key, quoting any segment that isn't a bare key."""
    return ".".join(p if _BARE_KEY.match(p) else '"' + p.replace('"', '\\"') + '"' for p in parts)


def _append_block(text: str, block: str) -> str:
    """Append *block* to *text*, separated by exactly one blank line."""
    if not text:
        return block
    separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return text + separator + block


def _still_parses(result: str) -> str | None:
    """Return *result* only if it's still valid TOML, else ``None``.

    The last line of defence for this module's contract. Appending a
    ``[features]`` header to a document that already defines ``features`` as an
    inline table or a dotted key produces a redefinition error, and a caller
    that trusted the returned string would write a broken config. Every splice
    is cheap to re-parse, and returning ``None`` costs only a fallback.
    """
    try:
        tomllib.loads(result)
    except tomllib.TOMLDecodeError:
        return None
    return result


def upsert_table(text: str, parts: tuple[str, ...], rendered: str) -> str | None:
    """Add or replace the ``[parts]`` table with *rendered* (header included)."""
    scan = _scan(text)
    if scan is None:
        return None
    headers, _ = scan
    if not rendered.endswith("\n"):
        rendered += "\n"

    idx = _find_table(headers, parts)
    if idx is None:
        if _has_detached_child(headers, parts, (0, 0)):
            return None  # child tables exist without their parent — too odd to splice
        return _still_parses(_append_block(text, rendered))

    start, end = _extent(text, headers, idx)
    if _has_detached_child(headers, parts, (start, end)):
        return None
    return _still_parses(text[:start] + rendered + text[end:])


def _is_defined(text: str, parts: tuple[str, ...]) -> bool:
    """True when *parts* resolves to something in the parsed document.

    A table doesn't need a header of its own to exist: ``[[a.b]]``,
    ``[a]`` + ``b = { ... }``, and ``[a]`` + ``b.c = 1`` all define ``a.b``.
    Only the parsed data can settle it.
    """
    try:
        node: object = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return False
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def remove_table(text: str, parts: tuple[str, ...]) -> str | None:
    """Remove the ``[parts]`` table along with every child table of it.

    A table can exist purely by implication: ``[mcp_servers.alpha.env]`` with
    no ``[mcp_servers.alpha]`` header still defines ``alpha``. Treating that as
    "not present" would leave the server behind, so every descendant block is
    removed too — from last to first, so earlier offsets stay valid.

    Returns ``None`` when the table is defined by something other than a
    header — an array of tables, an inline table, a dotted key — since a
    textual block removal can't express that. Reporting the unchanged document
    as success would look identical to a real removal to the caller.

    A comment sitting above a removed header is left in place rather than
    deleted with it — an orphaned comment is a much smaller surprise than
    silently deleting a line the user wrote.
    """
    scan = _scan(text)
    if scan is None:
        return None
    headers, _ = scan

    targets = [
        idx
        for idx, header in enumerate(headers)
        if not header.is_array and header.parts[: len(parts)] == parts
    ]
    if not targets:
        return None if _is_defined(text, parts) else text

    spans = sorted({_extent(text, headers, idx) for idx in targets}, reverse=True)

    # An array-of-tables descendant — ``[[mcp_servers.alpha.jobs]]`` — isn't a
    # removal target of its own, and ``_extent`` only swallows children that sit
    # contiguously after their parent. One separated by an unrelated table would
    # survive and keep ``parts`` defined, so bail instead of half-removing.
    if any(
        header.is_array
        and len(header.parts) >= len(parts)
        and header.parts[: len(parts)] == parts
        and not any(start <= header.start < end for start, end in spans)
        for header in headers
    ):
        return None

    # Removing back-to-front keeps the offsets of earlier blocks valid. Nested
    # descendants are already swallowed by their parent's extent, so skip any
    # block that a later (earlier-starting) removal will cover.
    result = text
    covered_from = len(text)
    for start, end in spans:
        if start >= covered_from:
            continue
        result = result[:start] + result[min(end, covered_from) :]
        covered_from = start
    return _still_parses(result)


def set_scalar(text: str, table: tuple[str, ...], key: str, literal: str) -> str | None:
    """Set ``key = literal`` inside the ``[table]`` table, creating it if needed.

    Only the one key's line is rewritten; every comment, blank line and key
    ordering elsewhere in the document survives untouched.
    """
    scan = _scan(text)
    if scan is None:
        return None
    headers, assigns = scan
    line = f"{_render_key((key,))} = {literal}\n"

    idx = _find_table(headers, table)
    if idx is None:
        return _still_parses(_append_block(text, f"[{_render_key(table)}]\n{line}"))

    header = headers[idx]
    _, end = _extent(text, headers, idx)
    for assign in assigns:
        if assign.parts == (key,) and header.body_start <= assign.start < end:
            return _still_parses(text[: assign.start] + line + text[assign.end :])

    return _still_parses(text[: header.body_start] + line + text[header.body_start :])


def splice_or_none(spliced: str | None, expected: dict[str, object]) -> str | None:
    """Return *spliced* only if it parses and matches *expected* exactly.

    The splicers work on text, so this is the check that keeps a clever edit
    from producing a document that no longer says what crossby meant. Anything
    short of an exact match sends the caller back to the lossy-but-certain
    ``tomli_w`` round-trip.
    """
    if spliced is None:
        return None
    try:
        if tomllib.loads(spliced) == expected:
            return spliced
    except (tomllib.TOMLDecodeError, ValueError):
        return None
    return None
