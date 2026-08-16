"""Tests for HandoffSummarizer parsing + behavior."""

from __future__ import annotations

import errno
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff import summarizer as summarizer_mod
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    RawHandoff,
    SessionRef,
    ToolCall,
)
from crossby.handoff.summarizer import (
    HandoffSummarizer,
    SummarizerParseError,
    SummarizerToolNotInstalledError,
    estimate_prompt_tokens,
    parse_markdown_sections,
)
from crossby.handoff.truncate import approx_tokens, turn_tokens
from crossby.models.ai import AIToolID


def _ref() -> SessionRef:
    return SessionRef(
        tool_id=AIToolID.CLAUDE,
        session_id="s",
        path=Path("/tmp/s.jsonl"),
        started_at=datetime(2026, 3, 1),
        cwd=Path("/Users/tester/proj"),
    )


def _transcript(n_turns: int = 2) -> ConversationTranscript:
    turns = [ConversationTurn(role="user", content=f"turn-{i}") for i in range(n_turns)]
    return ConversationTranscript(session_ref=_ref(), turns=turns)


def _make_summarizer_tool(
    *,
    json_schema_args: list[str] | None = None,
    tool_id: AIToolID = AIToolID.CLAUDE,
    stdin_args: list[str] | None = None,
) -> MagicMock:
    tool = MagicMock(spec=AbstractAITool)
    tool.TOOL_ID = tool_id
    tool.structured_output_args = MagicMock(return_value=json_schema_args or [])

    # Echo the argv-delivered prompt back into the command so byte-length
    # assertions on the launched command are meaningful. On the stdin path the
    # caller passes ``prompt=None`` and the prompt is absent from argv.
    def _fake_build_launch_command(**kwargs: Any) -> list[str]:
        cmd = ["fake"]
        prompt = kwargs.get("prompt")
        if prompt is not None:
            cmd += ["--prompt", prompt]
        return cmd

    tool.build_launch_command = MagicMock(side_effect=_fake_build_launch_command)
    tool.unwrap_structured_output = MagicMock(side_effect=lambda raw: raw)
    # Default to the argv delivery path; stdin tests override this explicitly.
    # Without this, ``MagicMock(spec=...)`` returns a truthy MagicMock and every
    # test would silently take the stdin branch.
    tool.headless_prompt_stdin_args = MagicMock(return_value=stdin_args)
    return tool


def test_ensure_installed_raises_when_tool_missing() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")
    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[]),
        pytest.raises(SummarizerToolNotInstalledError),
    ):
        summarizer.ensure_installed()


def test_summarize_parses_json_payload() -> None:
    tool = _make_summarizer_tool(json_schema_args=["--output-format", "json"])
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")
    json_stdout = (
        '{"current_task": "Refactor auth", '
        '"key_decisions": ["drop cache"], '
        '"modified_files": ["auth.py"], '
        '"blockers": [], '
        '"next_steps": ["write migration"], '
        '"critical_context": "cache is load-bearing"}'
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=json_stdout, stderr="")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc) as run,
    ):
        doc = summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    assert doc.current_task == "Refactor auth"
    assert doc.key_decisions == ["drop cache"]
    assert doc.modified_files == [Path("auth.py")]
    assert doc.next_steps == ["write migration"]
    assert doc.critical_context == "cache is load-bearing"
    assert doc.source_tool == AIToolID.CLAUDE
    assert doc.target_tool == AIToolID.CODEX
    # build_launch_command must have been called with a json_schema.
    _, kwargs = tool.build_launch_command.call_args
    assert kwargs["json_schema"] is not None
    run.assert_called_once()


def test_summarize_falls_back_to_markdown_when_no_json_support() -> None:
    tool = _make_summarizer_tool(json_schema_args=[])  # no JSON flags
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")
    markdown = (
        "## Current Task\nShip the thing.\n\n"
        "## Key Decisions\n- use stripe\n- drop cache\n\n"
        "## Modified Files\n- foo.py\n\n"
        "## Blockers\n\n"
        "## Next Steps\n- run tests\n\n"
        "## Critical Context\nDon't break billing.\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc),
    ):
        doc = summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    assert doc.current_task == "Ship the thing."
    assert doc.key_decisions == ["use stripe", "drop cache"]
    assert doc.modified_files == [Path("foo.py")]
    assert doc.blockers == []
    assert doc.next_steps == ["run tests"]
    assert "billing" in doc.critical_context
    # JSON schema must NOT have been passed.
    _, kwargs = tool.build_launch_command.call_args
    assert kwargs["json_schema"] is None


def test_summarize_raises_when_output_is_unparseable() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")
    fake_proc = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="plain text with no structure", stderr=""
    )

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc),
        pytest.raises(SummarizerParseError),
    ):
        summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )


def test_summarize_raises_when_subprocess_times_out() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT", timeout_seconds=7)
    timeout_exc = subprocess.TimeoutExpired(cmd=["fake"], timeout=7)

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", side_effect=timeout_exc),
        pytest.raises(SummarizerParseError, match="7s"),
    ):
        summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )


def test_summarize_raises_when_subprocess_oserror() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch(
            "crossby.handoff.summarizer.subprocess.run",
            side_effect=OSError("fork failed"),
        ),
        pytest.raises(SummarizerParseError, match="fork failed"),
    ):
        summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )


def test_summarize_raises_when_tool_exits_nonzero() -> None:
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")
    fake_proc = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc),
        pytest.raises(SummarizerParseError, match="boom"),
    ):
        summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )


def test_summarize_calls_on_truncate_when_transcript_trimmed() -> None:
    tool = _make_summarizer_tool()
    # Tiny budget forces truncation.
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT", token_budget=5)
    big_turns = [ConversationTurn(role="user", content="x" * 400) for _ in range(4)]
    transcript = ConversationTranscript(session_ref=_ref(), turns=big_turns)
    markdown = (
        "## Current Task\nt\n## Key Decisions\n## Modified Files\n"
        "## Blockers\n## Next Steps\n## Critical Context\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    captured: list[tuple[int, int]] = []

    def on_trunc(original: int, kept: int) -> None:
        captured.append((original, kept))

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc),
    ):
        summarizer.summarize_structured(
            transcript,
            source_tool=AIToolID.CLAUDE,
            target_tool=AIToolID.CODEX,
            on_truncate=on_trunc,
        )

    assert captured
    original, kept = captured[0]
    assert original == 4
    assert kept < original


def test_parse_markdown_sections_returns_none_without_headings() -> None:
    assert parse_markdown_sections("plain text") is None


def test_parse_markdown_sections_extracts_bullets_and_text() -> None:
    text = (
        "## Current Task\nDo the work.\n"
        "## Key Decisions\n- a\n- b\n"
        "## Modified Files\n"
        "## Blockers\n* legacy dep\n"
        "## Next Steps\n- ship\n"
        "## Critical Context\nmind the cache\n"
    )
    payload = parse_markdown_sections(text)
    assert payload is not None
    assert payload["current_task"] == "Do the work."
    assert payload["key_decisions"] == ["a", "b"]
    assert payload["modified_files"] == []
    assert payload["blockers"] == ["legacy dep"]
    assert payload["next_steps"] == ["ship"]
    assert payload["critical_context"] == "mind the cache"


def test_summarize_raw_returns_rawhandoff_and_skips_json_schema() -> None:
    tool = _make_summarizer_tool(json_schema_args=["--output-format", "json"])
    summarizer = HandoffSummarizer(tool, prompt_template="CUSTOM")
    free_form = "  <analysis>...</analysis>\n<summary>...</summary>  "
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=free_form, stderr="")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc),
    ):
        doc = summarizer.summarize_raw(
            _transcript(),
            source_tool=AIToolID.CLAUDE,
            target_tool=AIToolID.CODEX,
            prompt_source="cc-compact",
        )

    assert isinstance(doc, RawHandoff)
    assert doc.body == free_form.strip()
    assert doc.prompt_source == "cc-compact"
    # Raw mode must never pass a json_schema — the fixed schema doesn't fit custom prompts.
    _, kwargs = tool.build_launch_command.call_args
    assert kwargs["json_schema"] is None


def test_codex_summarizer_in_worktree_gets_writable_roots_and_cwd(tmp_path: Path) -> None:
    """A Codex summarizer run in a linked worktree grants its git-metadata
    writable roots and runs with a cwd inside the worktree."""
    from crossby.ai_tools.codex import CodexAdapter

    # Real linked worktree so build_launch_command's composer resolves metadata.
    def _git(cwd: Path, *args: str) -> None:
        subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.co")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("x\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "feature")

    summarizer = HandoffSummarizer(CodexAdapter(), prompt_template="TEST PROMPT", working_dir=wt)
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
    real_run = subprocess.run

    # ``patch(...summarizer.subprocess.run)`` swaps the attribute on the shared
    # ``subprocess`` module, which would also intercept the resolver's real
    # ``git rev-parse`` — so delegate git calls to the real subprocess and mock
    # only the summarizer (codex) invocation.
    def _dispatch(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if cmd and cmd[0] == "git":
            return real_run(cmd, **kwargs)
        return fake_proc

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CODEX]),
        patch("crossby.handoff.summarizer.subprocess.run", side_effect=_dispatch) as run,
    ):
        summarizer.summarize_raw(
            _transcript(),
            source_tool=AIToolID.CODEX,
            target_tool=AIToolID.CLAUDE,
            prompt_source="default",
        )

    codex_call = next(c for c in run.call_args_list if c.args[0][0] == "codex")
    launched_cmd = codex_call.args[0]
    assert "--sandbox" in launched_cmd
    assert "workspace-write" in launched_cmd
    # Metadata dirs granted additively via --add-dir (the common .git dir here).
    assert "--add-dir" in launched_cmd
    assert str((repo / ".git").resolve()) in launched_cmd
    assert "sandbox_workspace_write.network_access=false" in launched_cmd
    # The subprocess cwd matches the worktree whose metadata it granted writable.
    assert codex_call.kwargs["cwd"] == wt


# ---------------------------------------------------------------------------
# argv byte-ceiling: re-truncation, preflight, and E2BIG backstop
# ---------------------------------------------------------------------------


def _run_summarize(summarizer: HandoffSummarizer, transcript, mode, on_truncate=None):  # type: ignore[no-untyped-def]
    """Drive either ``summarize_structured`` or ``summarize_raw``."""
    if mode == "structured":
        return summarizer.summarize_structured(
            transcript,
            source_tool=AIToolID.CLAUDE,
            target_tool=AIToolID.CODEX,
            on_truncate=on_truncate,
        )
    return summarizer.summarize_raw(
        transcript,
        source_tool=AIToolID.CLAUDE,
        target_tool=AIToolID.CODEX,
        prompt_source="cc-compact",
        on_truncate=on_truncate,
    )


@pytest.mark.parametrize("mode", ["structured", "raw"])
def test_argv_re_truncation_fits_ceiling_and_fires_callback_once(mode: str, monkeypatch) -> None:
    """A None-stdin tool whose assembled prompt exceeds the ceiling re-truncates:
    it succeeds, fits the byte ceiling, and fires ``on_truncate`` once as
    ``(total, kept)`` — for both the structured and raw paths."""
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 600)
    tool = _make_summarizer_tool()  # stdin_args defaults to None → argv path
    # Large budget so the *byte* ceiling, not token truncation, drops the turns.
    summarizer = HandoffSummarizer(tool, prompt_template="TEST", token_budget=10_000_000)
    turns = [ConversationTurn(role="user", content="x" * 200) for _ in range(8)]
    transcript = ConversationTranscript(session_ref=_ref(), turns=turns)
    markdown = (
        "## Current Task\nt\n## Key Decisions\n## Modified Files\n"
        "## Blockers\n## Next Steps\n## Critical Context\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    captured: list[tuple[int, int]] = []

    def _cap(total: int, kept: int) -> None:
        captured.append((total, kept))

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc) as run,
    ):
        _run_summarize(summarizer, transcript, mode, on_truncate=_cap)

    # Fired exactly once, with (total, kept) and a real drop.
    assert len(captured) == 1
    total, kept = captured[0]
    assert total == 8
    assert 0 < kept < 8

    # The launched argv carries the fitted prompt, under the byte ceiling.
    run.assert_called_once()
    cmd = run.call_args.args[0]
    prompt_arg = cmd[-1]
    assert len(prompt_arg.encode("utf-8")) <= summarizer_mod._MAX_PROMPT_BYTES
    # Never surfaces a raw errno.
    assert "[Errno 7]" not in prompt_arg


def test_argv_preflight_single_oversized_turn_raises_before_spawn(monkeypatch) -> None:
    """A lone turn over the ceiling can't be re-truncated → preflight raises
    before ``subprocess.run`` is ever called."""
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 300)
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST", token_budget=10_000_000)
    transcript = ConversationTranscript(
        session_ref=_ref(), turns=[ConversationTurn(role="user", content="x" * 500)]
    )

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run") as run,
        pytest.raises(SummarizerParseError) as exc_info,
    ):
        summarizer.summarize_structured(
            transcript, source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    run.assert_not_called()
    msg = str(exc_info.value)
    assert "argv" in msg
    # Preflight is the irreducible case: it must recommend stdin delivery and
    # say lowering --token-budget will not help — never a raw errno.
    assert "--summarizer-tool" in msg
    assert "will not help" in msg
    assert "[Errno 7]" not in msg


def test_argv_preflight_oversized_prompt_template_raises(monkeypatch) -> None:
    """A huge ``--prompt`` template a smaller budget can't shrink → preflight
    raises even with a trivial transcript, recommending stdin over --token-budget."""
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 300)
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="T" * 500, token_budget=10_000_000)
    transcript = _transcript(2)

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run") as run,
        pytest.raises(SummarizerParseError, match="--summarizer-tool"),
    ):
        summarizer.summarize_structured(
            transcript, source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    run.assert_not_called()


def test_argv_e2big_oserror_maps_to_actionable_error() -> None:
    """``subprocess.run`` raising ``OSError(E2BIG)`` is mapped to the actionable
    ``--token-budget`` message, never a raw ``[Errno 7]``."""
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")
    e2big = OSError(errno.E2BIG, "Argument list too long")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", side_effect=e2big),
        pytest.raises(SummarizerParseError) as exc_info,
    ):
        summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    msg = str(exc_info.value)
    assert "--token-budget" in msg
    assert "[Errno 7]" not in msg


def test_argv_utf8_uses_byte_length_not_char_count(monkeypatch) -> None:
    """A single turn whose char-count is under the ceiling but whose UTF-8 byte
    count is over it must trip the preflight — proving byte, not char, counting."""
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 300)
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="T", token_budget=10_000_000)
    # 200 '€' chars = 200 chars but 600 bytes; assembled stays under 300 chars,
    # comfortably over 300 bytes.
    transcript = ConversationTranscript(
        session_ref=_ref(), turns=[ConversationTurn(role="user", content="€" * 200)]
    )
    prompt = summarizer._build_prompt(transcript)
    assert len(prompt) < summarizer_mod._MAX_PROMPT_BYTES  # char count under ceiling
    assert len(prompt.encode("utf-8")) > summarizer_mod._MAX_PROMPT_BYTES  # bytes over

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run") as run,
        pytest.raises(SummarizerParseError),
    ):
        summarizer.summarize_structured(
            transcript, source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    run.assert_not_called()


def test_argv_render_length_posix_uses_raw_utf8_bytes(monkeypatch) -> None:
    """On POSIX, argv is passed straight to ``execve`` with no command-line
    rendering, so the render-length helper is just the raw UTF-8 byte count."""
    monkeypatch.setattr(summarizer_mod.sys, "platform", "darwin")
    prompt = 'héllo \\" world'
    assert summarizer_mod._argv_render_length(prompt) == len(prompt.encode("utf-8"))


def test_argv_render_length_windows_uses_quoted_rendering(monkeypatch) -> None:
    """On Windows, the helper measures the ``CreateProcess``-quoted rendering
    (via ``list2cmdline``), not the raw byte count — embedded quotes and
    backslashes expand under that quoting."""
    monkeypatch.setattr(summarizer_mod.sys, "platform", "win32")
    prompt = '\\"' * 50  # backslash-quote pairs double in length when quoted
    rendered = summarizer_mod._argv_render_length(prompt)
    assert rendered == len(subprocess.list2cmdline([prompt]))
    assert rendered > len(prompt.encode("utf-8"))


def test_argv_render_length_windows_counts_utf16_units_not_code_points(monkeypatch) -> None:
    """Windows' 32,767-char ``CreateProcess`` limit is in UTF-16 code *units*,
    not Python ``str`` code points. A non-BMP character (most emoji) is a
    surrogate pair — 2 units — but a single Python code point, so counting
    ``len()`` on the rendered string directly would undercount by ~2x for
    emoji-heavy prompts."""
    monkeypatch.setattr(summarizer_mod.sys, "platform", "win32")
    prompt = "\U0001f600" * 50  # non-BMP emoji: 1 code point, 2 UTF-16 units each
    rendered = subprocess.list2cmdline([prompt])
    assert summarizer_mod._argv_render_length(prompt) == len(rendered.encode("utf-16-le")) // 2
    assert summarizer_mod._argv_render_length(prompt) > len(rendered)


def test_argv_windows_non_bmp_expansion_trips_preflight_under_code_point_ceiling(
    monkeypatch,
) -> None:
    """A prompt whose rendered *code-point* length fits the ceiling can still
    overflow the actual Windows command line once non-BMP emoji (surrogate
    pairs) are counted in UTF-16 units, as ``CreateProcess`` does — the
    preflight must catch that case too."""
    monkeypatch.setattr(summarizer_mod.sys, "platform", "win32")
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 400)
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="T", token_budget=10_000_000)
    content = "\U0001f600" * 250
    transcript = ConversationTranscript(
        session_ref=_ref(), turns=[ConversationTurn(role="user", content=content)]
    )
    prompt = summarizer._build_prompt(transcript)
    rendered = subprocess.list2cmdline([prompt])
    assert len(rendered) < summarizer_mod._MAX_PROMPT_BYTES  # code points fit
    assert len(rendered.encode("utf-16-le")) // 2 > summarizer_mod._MAX_PROMPT_BYTES  # units don't

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run") as run,
        pytest.raises(SummarizerParseError),
    ):
        summarizer.summarize_structured(
            transcript, source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    run.assert_not_called()


def test_argv_windows_quoting_expansion_trips_preflight_under_raw_byte_ceiling(
    monkeypatch,
) -> None:
    """A prompt whose raw UTF-8 byte count fits the ceiling can still overflow
    the actual Windows command line once ``CreateProcess`` quoting expands its
    embedded backslash-quote pairs — the preflight must catch that case using
    the rendered length, not the raw byte count."""
    monkeypatch.setattr(summarizer_mod.sys, "platform", "win32")
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 400)
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="T", token_budget=10_000_000)
    content = '\\"' * 140
    transcript = ConversationTranscript(
        session_ref=_ref(), turns=[ConversationTurn(role="user", content=content)]
    )
    prompt = summarizer._build_prompt(transcript)
    assert len(prompt.encode("utf-8")) < summarizer_mod._MAX_PROMPT_BYTES  # raw bytes fit
    rendered = len(subprocess.list2cmdline([prompt]))
    assert rendered > summarizer_mod._MAX_PROMPT_BYTES  # but the quoted command line doesn't

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run") as run,
        pytest.raises(SummarizerParseError),
    ):
        summarizer.summarize_structured(
            transcript, source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    run.assert_not_called()


def test_stdin_delivery_pipes_prompt_and_skips_argv_ceiling(monkeypatch) -> None:
    """A stdin-enabled tool feeds the full prompt via ``input=`` and appends its
    stdin args last; the prompt is absent from argv and never re-truncated."""
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 600)
    tool = _make_summarizer_tool(stdin_args=["--print"])
    summarizer = HandoffSummarizer(tool, prompt_template="TEST", token_budget=10_000_000)
    turns = [ConversationTurn(role="user", content=f"turn-{i}-" + "x" * 200) for i in range(8)]
    transcript = ConversationTranscript(session_ref=_ref(), turns=turns)
    markdown = (
        "## Current Task\nt\n## Key Decisions\n## Modified Files\n"
        "## Blockers\n## Next Steps\n## Critical Context\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    captured: list[tuple[int, int]] = []

    def _cap(total: int, kept: int) -> None:
        captured.append((total, kept))

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc) as run,
    ):
        summarizer.summarize_structured(
            transcript,
            source_tool=AIToolID.CLAUDE,
            target_tool=AIToolID.CODEX,
            on_truncate=_cap,
        )

    run.assert_called_once()
    cmd = run.call_args.args[0]
    piped = run.call_args.kwargs["input"]

    # stdin args appended last; the prompt was NOT placed on argv.
    assert cmd == ["fake", "--print"]
    assert tool.build_launch_command.call_args.kwargs["prompt"] is None
    # The whole prompt went through the pipe — including the oldest turn, and
    # over the argv byte ceiling (proving no re-truncation on this path).
    assert "turn-0-" in piped
    assert "turn-7-" in piped
    assert len(piped.encode("utf-8")) > summarizer_mod._MAX_PROMPT_BYTES
    # No truncation happened, so the callback never fired.
    assert captured == []


def test_truncated_source_does_not_refire_callback_on_byte_retruncation(
    monkeypatch,
) -> None:
    """A source transcript already ``truncated`` must not fire ``on_truncate``,
    even when byte re-truncation drops further turns."""
    monkeypatch.setattr(summarizer_mod, "_MAX_PROMPT_BYTES", 600)
    tool = _make_summarizer_tool()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST", token_budget=10_000_000)
    turns = [ConversationTurn(role="user", content="x" * 200) for _ in range(8)]
    transcript = ConversationTranscript(session_ref=_ref(), turns=turns, truncated=True)
    markdown = (
        "## Current Task\nt\n## Key Decisions\n## Modified Files\n"
        "## Blockers\n## Next Steps\n## Critical Context\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    captured: list[tuple[int, int]] = []

    def _cap(total: int, kept: int) -> None:
        captured.append((total, kept))

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc) as run,
    ):
        summarizer.summarize_structured(
            transcript,
            source_tool=AIToolID.CLAUDE,
            target_tool=AIToolID.CODEX,
            on_truncate=_cap,
        )

    # Re-truncation still fitted the ceiling, but the callback stayed silent.
    run.assert_called_once()
    assert len(run.call_args.args[0][-1].encode("utf-8")) <= summarizer_mod._MAX_PROMPT_BYTES
    assert captured == []


# ---------------------------------------------------------------------------
# Per-adapter argv vectors in stdin mode (real ClaudeAdapter / CodexAdapter)
# ---------------------------------------------------------------------------


def test_claude_stdin_argv_vector_omits_prompt_and_appends_print() -> None:
    """Real ClaudeAdapter in stdin mode: exact argv with model + schema flags,
    ``--print`` appended last, and the prompt absent from argv (piped instead)."""
    from crossby.ai_tools.claude import ClaudeAdapter

    tool = ClaudeAdapter()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT", model="claude-haiku-4.5")
    marker = "UNIQUE-CLAUDE-TRANSCRIPT-BODY"
    transcript = ConversationTranscript(
        session_ref=_ref(), turns=[ConversationTurn(role="user", content=marker)]
    )
    fake_proc = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(_FULL_PAYLOAD), stderr=""
    )

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc) as run,
    ):
        summarizer.summarize_structured(
            transcript, source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    cmd = run.call_args.args[0]
    schema_json = json.dumps(summarizer_mod._HANDOFF_JSON_SCHEMA)
    assert cmd == [
        "claude",
        "--model",
        "claude-haiku-4-5",
        "--output-format",
        "json",
        "--json-schema",
        schema_json,
        "--print",
    ]
    piped = run.call_args.kwargs["input"]
    assert marker in piped
    assert all(marker not in a for a in cmd)
    assert cmd[-1] == "--print"


def test_codex_stdin_argv_vector_preserves_model_exec_ordering() -> None:
    """Real CodexAdapter in stdin mode: ``codex --model ... exec`` ordering, with
    ``exec`` appended last and the prompt piped rather than placed on argv."""
    from crossby.ai_tools.codex import CodexAdapter

    tool = CodexAdapter()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT", model="gpt-5-codex")
    marker = "UNIQUE-CODEX-TRANSCRIPT-BODY"
    transcript = ConversationTranscript(
        session_ref=_ref(), turns=[ConversationTurn(role="user", content=marker)]
    )
    markdown = (
        "## Current Task\nt\n## Key Decisions\n## Modified Files\n"
        "## Blockers\n## Next Steps\n## Critical Context\n"
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=markdown, stderr="")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CODEX]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc) as run,
    ):
        summarizer.summarize_structured(
            transcript, source_tool=AIToolID.CODEX, target_tool=AIToolID.CLAUDE
        )

    cmd = run.call_args.args[0]
    # Codex has no structured-output flags, so schema is absent; exec is last.
    assert cmd == ["codex", "--model", "gpt-5-codex", "exec"]
    piped = run.call_args.kwargs["input"]
    assert marker in piped
    assert all(marker not in a for a in cmd)
    assert cmd[-1] == "exec"


# ---------------------------------------------------------------------------
# estimate_prompt_tokens alignment with turn_tokens
# ---------------------------------------------------------------------------


def test_estimate_prompt_tokens_matches_turn_tokens_and_counts_tool_calls() -> None:
    """The estimate equals the truncation accounting (``turn_tokens``) and is
    strictly larger than the old content-only sum for a tool-heavy turn."""
    turn = ConversationTurn(
        role="assistant",
        content="did the thing",
        tool_calls=[
            ToolCall(name="run_tests", arguments={"path": "tests/", "verbose": True}),
            ToolCall(name="read_file", arguments={"file": "src/app.py"}),
        ],
    )
    transcript = ConversationTranscript(session_ref=_ref(), turns=[turn, turn])

    estimate = estimate_prompt_tokens(transcript)
    assert estimate == sum(turn_tokens(t) for t in transcript.turns)
    content_only = sum(approx_tokens(t.content) for t in transcript.turns)
    assert estimate > content_only


# Suppress unused-import linter complaints in some configs.
_ = Any


# ---------------------------------------------------------------------------
# ClaudeAdapter.unwrap_structured_output unit tests
# ---------------------------------------------------------------------------

_FULL_PAYLOAD: dict[str, Any] = {
    "current_task": "Refactor auth",
    "key_decisions": ["drop cache"],
    "modified_files": ["auth.py"],
    "blockers": [],
    "next_steps": ["write migration"],
    "critical_context": "cache is load-bearing",
}


def _claude_envelope(
    payload: dict[str, Any],
    *,
    is_error: bool = False,
    include_structured_output: bool = True,
) -> str:
    env: dict[str, Any] = {
        "type": "result",
        "subtype": "error" if is_error else "success",
        "is_error": is_error,
        "result": payload.get("result", json.dumps(payload)) if is_error else json.dumps(payload),
        "session_id": "test-session",
    }
    if include_structured_output and not is_error:
        env["structured_output"] = payload
    return json.dumps(env)


def test_claude_unwrap_extracts_structured_output() -> None:
    from crossby.ai_tools.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    raw = _claude_envelope(_FULL_PAYLOAD, include_structured_output=True)
    result = adapter.unwrap_structured_output(raw)
    assert json.loads(result) == _FULL_PAYLOAD


def test_claude_unwrap_falls_back_to_result_when_no_structured_output() -> None:
    from crossby.ai_tools.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    raw = _claude_envelope(_FULL_PAYLOAD, include_structured_output=False)
    result = adapter.unwrap_structured_output(raw)
    assert json.loads(result) == _FULL_PAYLOAD


def test_claude_unwrap_raises_on_is_error() -> None:
    from crossby.ai_tools.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    error_envelope = json.dumps(
        {
            "type": "result",
            "subtype": "error",
            "is_error": True,
            "result": "Model refused to respond",
            "session_id": "test-session",
        }
    )
    with pytest.raises(SummarizerParseError, match="Model refused to respond"):
        adapter.unwrap_structured_output(error_envelope)


def test_claude_unwrap_passthrough_for_plain_json() -> None:
    from crossby.ai_tools.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    plain = json.dumps(_FULL_PAYLOAD)
    assert adapter.unwrap_structured_output(plain) == plain


def test_claude_unwrap_passthrough_for_markdown() -> None:
    from crossby.ai_tools.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    markdown = "## Current Task\nBuild thing.\n## Key Decisions\n- a\n"
    assert adapter.unwrap_structured_output(markdown) == markdown


# ---------------------------------------------------------------------------
# End-to-end regression: Claude envelope → summarizer produces populated doc
# ---------------------------------------------------------------------------


def test_summarize_structured_unwraps_claude_envelope() -> None:
    """Regression: Claude's JSON envelope is unwrapped before _parse_output."""
    from crossby.ai_tools.claude import ClaudeAdapter

    tool = ClaudeAdapter()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")

    envelope_stdout = _claude_envelope(_FULL_PAYLOAD, include_structured_output=True)
    fake_proc = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=envelope_stdout, stderr=""
    )

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc),
    ):
        doc = summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )

    assert doc.current_task == "Refactor auth"
    assert doc.key_decisions == ["drop cache"]
    assert doc.next_steps == ["write migration"]
    assert doc.critical_context == "cache is load-bearing"


def test_summarize_structured_raises_on_claude_is_error() -> None:
    """Regression: Claude is_error envelope propagates as SummarizerParseError."""
    from crossby.ai_tools.claude import ClaudeAdapter

    tool = ClaudeAdapter()
    summarizer = HandoffSummarizer(tool, prompt_template="TEST PROMPT")

    error_stdout = json.dumps(
        {
            "type": "result",
            "subtype": "error",
            "is_error": True,
            "result": "context limit exceeded",
            "session_id": "test-session",
        }
    )
    fake_proc = subprocess.CompletedProcess(args=[], returncode=0, stdout=error_stdout, stderr="")

    with (
        patch.object(AbstractAITool, "detect_installed", return_value=[AIToolID.CLAUDE]),
        patch("crossby.handoff.summarizer.subprocess.run", return_value=fake_proc),
        pytest.raises(SummarizerParseError, match="context limit exceeded"),
    ):
        summarizer.summarize_structured(
            _transcript(), source_tool=AIToolID.CLAUDE, target_tool=AIToolID.CODEX
        )
