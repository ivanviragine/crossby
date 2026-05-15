"""Tests for HandoffWriter / render_markdown."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from crossby.handoff.models import HandoffDocument, RawHandoff, SessionRef
from crossby.handoff.writer import HandoffWriter, render_markdown, render_markdown_raw
from crossby.models.ai import AIToolID


def _doc(**overrides: object) -> HandoffDocument:
    data: dict[str, object] = {
        "source_tool": AIToolID.CLAUDE,
        "target_tool": AIToolID.CODEX,
        "session_ref": SessionRef(
            tool_id=AIToolID.CLAUDE,
            session_id="abc",
            path=Path("/tmp/abc.jsonl"),
            started_at=datetime(2026, 3, 1, 10, 0, 0),
            cwd=Path("/Users/tester/proj"),
        ),
        "current_task": "Refactor auth",
        "key_decisions": ["Drop shared cache", "Migrate to stripe adapter"],
        "modified_files": [Path("auth.py"), Path("payments.py")],
        "blockers": ["Need prod credentials"],
        "next_steps": ["Write migration"],
        "critical_context": "Session cache is load-bearing; avoid cold-start flushes.",
        "created_at": datetime(2026, 3, 1, 12, 0, 0),
    }
    data.update(overrides)
    return HandoffDocument.model_validate(data)


def test_render_includes_all_sections_in_order() -> None:
    rendered = render_markdown(_doc())
    for heading in (
        "# Handoff: claude → codex",
        "## Current Task",
        "## Key Decisions",
        "## Modified Files",
        "## Blockers",
        "## Next Steps",
        "## Critical Context",
    ):
        assert heading in rendered
    # Ordering
    positions = [
        rendered.index(h)
        for h in (
            "## Current Task",
            "## Key Decisions",
            "## Modified Files",
            "## Blockers",
            "## Next Steps",
            "## Critical Context",
        )
    ]
    assert positions == sorted(positions)


def test_empty_lists_render_placeholder() -> None:
    rendered = render_markdown(
        _doc(key_decisions=[], modified_files=[], blockers=[], next_steps=[])
    )
    # Each empty list becomes "_(none)_"
    assert rendered.count("_(none)_") >= 4


def test_writer_writes_under_crossby_handoffs(tmp_path: Path) -> None:
    writer = HandoffWriter(tmp_path)
    path = writer.write(_doc())
    assert path.exists()
    assert path.is_relative_to(tmp_path / ".crossby" / "handoffs")
    assert path.name.startswith("HANDOFF-")
    assert path.suffix == ".md"
    content = path.read_text(encoding="utf-8")
    assert "Refactor auth" in content


def test_writer_respects_explicit_output_path(tmp_path: Path) -> None:
    writer = HandoffWriter(tmp_path)
    target = tmp_path / "custom" / "out.md"
    path = writer.write(_doc(), output_path=target)
    assert path == target
    assert target.exists()


def _raw(**overrides: object) -> RawHandoff:
    data: dict[str, object] = {
        "source_tool": AIToolID.CLAUDE,
        "target_tool": AIToolID.CODEX,
        "session_ref": SessionRef(
            tool_id=AIToolID.CLAUDE,
            session_id="abc",
            path=Path("/tmp/abc.jsonl"),
            started_at=datetime(2026, 3, 1, 10, 0, 0),
            cwd=Path("/Users/tester/proj"),
        ),
        "body": "<analysis>chosen approach</analysis>\n<summary>done</summary>",
        "prompt_source": "cc-compact",
        "created_at": datetime(2026, 3, 1, 12, 0, 0),
    }
    data.update(overrides)
    return RawHandoff.model_validate(data)


def test_render_markdown_raw_includes_metadata_and_body() -> None:
    rendered = render_markdown_raw(_raw())
    assert "# Handoff: claude → codex (raw)" in rendered
    assert "- **Prompt**: cc-compact" in rendered
    assert "<analysis>chosen approach</analysis>" in rendered
    # None of the structured section headings should appear.
    assert "## Current Task" not in rendered
    assert "## Key Decisions" not in rendered


def test_writer_writes_rawhandoff(tmp_path: Path) -> None:
    writer = HandoffWriter(tmp_path)
    path = writer.write(_raw())
    assert path.exists()
    assert path.is_relative_to(tmp_path / ".crossby" / "handoffs")
    content = path.read_text(encoding="utf-8")
    assert "(raw)" in content
    assert "cc-compact" in content
