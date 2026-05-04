"""Tests for manual-fix block formatting."""

from __future__ import annotations

from crossby.sync.manual_fix import (
    MANUAL_FIX_END,
    MANUAL_FIX_HEADING,
    MANUAL_FIX_START,
    ManualFixNote,
    append_manual_fix_block,
    find_manual_fix_blocks,
    format_manual_fix_block,
    has_manual_fix_block,
    strip_manual_fix_blocks,
)


class TestFormatManualFixBlock:
    def test_empty_returns_empty_string(self) -> None:
        assert format_manual_fix_block([]) == ""

    def test_single_string_note(self) -> None:
        block = format_manual_fix_block(["Translate this manually."])
        assert MANUAL_FIX_START in block
        assert MANUAL_FIX_HEADING in block
        assert "- Translate this manually." in block
        assert MANUAL_FIX_END in block

    def test_multiple_notes_render_as_bullets(self) -> None:
        block = format_manual_fix_block(["First note.", "Second note."])
        assert "- First note." in block
        assert "- Second note." in block

    def test_manual_fix_note_dataclass(self) -> None:
        note = ManualFixNote(message="Fix it", category="permissionMode")
        block = format_manual_fix_block([note])
        assert "- Fix it" in block

    def test_blank_strings_skipped(self) -> None:
        block = format_manual_fix_block(["Real note.", "", "   "])
        assert "Real note." in block
        # Exactly one bullet line.
        bullet_lines = [line for line in block.splitlines() if line.startswith("- ")]
        assert bullet_lines == ["- Real note."]

    def test_all_blank_returns_empty(self) -> None:
        assert format_manual_fix_block(["", "  "]) == ""

    def test_strips_whitespace(self) -> None:
        block = format_manual_fix_block(["  trimmed  "])
        assert "- trimmed" in block
        assert "- trimmed  " not in block


class TestAppendManualFixBlock:
    def test_no_notes_is_noop(self) -> None:
        body = "Some body content."
        assert append_manual_fix_block(body, []) == body

    def test_appends_with_blank_line(self) -> None:
        result = append_manual_fix_block("Body.", ["Note."])
        assert result.startswith("Body.\n\n<!-- crossby:manual-fix:start -->")
        assert result.endswith(f"{MANUAL_FIX_END}\n")

    def test_strips_trailing_whitespace_on_body(self) -> None:
        result = append_manual_fix_block("Body.\n\n\n", ["Note."])
        # Should not have triple blank lines between body and block.
        assert "\n\n\n" not in result

    def test_empty_body_yields_just_block(self) -> None:
        result = append_manual_fix_block("", ["Note."])
        assert result.startswith(MANUAL_FIX_START)
        assert "Body" not in result


class TestStripManualFixBlocks:
    def test_strips_single_block(self) -> None:
        body = (
            "Original content.\n\n"
            f"{MANUAL_FIX_START}\n"
            f"{MANUAL_FIX_HEADING}\n\n"
            "- Note.\n"
            f"{MANUAL_FIX_END}\n"
        )
        result = strip_manual_fix_blocks(body)
        assert MANUAL_FIX_START not in result
        assert MANUAL_FIX_END not in result
        assert "Original content." in result

    def test_strips_multiple_blocks(self) -> None:
        body = (
            f"{MANUAL_FIX_START}\nA\n{MANUAL_FIX_END}\n"
            "Middle.\n"
            f"{MANUAL_FIX_START}\nB\n{MANUAL_FIX_END}\n"
        )
        result = strip_manual_fix_blocks(body)
        assert MANUAL_FIX_START not in result
        assert "Middle." in result

    def test_no_blocks_is_noop(self) -> None:
        body = "Just plain content.\n"
        assert strip_manual_fix_blocks(body) == body

    def test_collapses_extra_blank_lines(self) -> None:
        body = (
            "Top.\n\n"
            f"{MANUAL_FIX_START}\nNote\n{MANUAL_FIX_END}\n\n"
            "Bottom.\n"
        )
        result = strip_manual_fix_blocks(body)
        assert "\n\n\n" not in result
        assert "Top." in result
        assert "Bottom." in result


class TestFindManualFixBlocks:
    def test_finds_block_with_markers(self) -> None:
        body = f"{MANUAL_FIX_START}\nA\n{MANUAL_FIX_END}"
        blocks = find_manual_fix_blocks(body)
        assert len(blocks) == 1
        assert MANUAL_FIX_START in blocks[0]
        assert MANUAL_FIX_END in blocks[0]

    def test_finds_multiple(self) -> None:
        body = (
            f"{MANUAL_FIX_START}\nA\n{MANUAL_FIX_END}\n"
            f"{MANUAL_FIX_START}\nB\n{MANUAL_FIX_END}\n"
        )
        assert len(find_manual_fix_blocks(body)) == 2

    def test_empty_when_none(self) -> None:
        assert find_manual_fix_blocks("just text") == []


class TestHasManualFixBlock:
    def test_true_when_present(self) -> None:
        assert has_manual_fix_block(
            f"x{MANUAL_FIX_START}\nA\n{MANUAL_FIX_END}y"
        )

    def test_false_when_absent(self) -> None:
        assert has_manual_fix_block("plain text") is False

    def test_false_when_only_one_marker(self) -> None:
        # Half-open block must not register as a present block.
        assert has_manual_fix_block(f"only {MANUAL_FIX_START} no end") is False


class TestRoundTrip:
    def test_strip_then_append_is_clean(self) -> None:
        original = "Body.\n"
        with_block = append_manual_fix_block(original, ["Note."])
        stripped = strip_manual_fix_blocks(with_block)
        assert "Note" not in stripped
        assert "Body." in stripped
        # Re-appending yields a single block, not two.
        re_appended = append_manual_fix_block(stripped, ["Note."])
        assert re_appended.count(MANUAL_FIX_START) == 1
