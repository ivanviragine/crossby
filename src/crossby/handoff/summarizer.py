"""Summarize a ConversationTranscript into a structured HandoffDocument."""

from __future__ import annotations

import errno
import json
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    HandoffDocument,
    RawHandoff,
    SessionRef,
)
from crossby.handoff.truncate import truncate_transcript, turn_tokens
from crossby.models.ai import AIToolID

logger = structlog.get_logger()

DEFAULT_TOKEN_BUDGET = 32_000
DEFAULT_TIMEOUT_SECONDS = 120

# Byte ceiling for the assembled prompt on the argv delivery path. Linux caps a
# *single* argv string at MAX_ARG_STRLEN = 131,072 bytes, independent of total
# ARG_MAX, so we stay comfortably below that (UTF-8/margin) on POSIX — mirroring
# cli/handoff.py's 4 KB launch-message cap. Windows CreateProcess instead caps the
# *whole* command line at 32,767 chars, so 120 KB is unsafe there; use a
# conservative value and steer Windows users toward a stdin-capable summarizer.
# stdin delivery (Claude/Codex) bypasses this limit entirely.
#
# On Windows the ceiling is compared against the *rendered* (quoted/escaped)
# length, not the raw byte count: CreateProcess quoting can expand embedded
# quotes and backslashes well past the source length (see
# ``_argv_render_length``), so a raw byte count under the ceiling does not
# guarantee the rendered command line is too. POSIX argv is passed to
# ``execve`` directly with no such rendering, so raw UTF-8 bytes are exact there.
_MAX_PROMPT_BYTES = 30_000 if sys.platform.startswith("win") else 120_000


def _argv_render_length(prompt: str) -> int:
    """Length of ``prompt`` as it actually lands on the process command line.

    On Windows, ``CreateProcess`` receives one quoted/escaped string built
    from all argv elements — embedded quotes and backslashes in ``prompt``
    expand under that quoting (each backslash immediately before a quote is
    doubled, each quote is escaped), so the rendered length can exceed the raw
    byte count substantially. ``subprocess.list2cmdline`` implements the same
    quoting Python uses when it launches the process, so measuring a single
    quoted element here matches what will actually be spawned. The 32,767
    limit itself is in UTF-16 code *units*, not Python ``str`` code points, so
    non-BMP characters (most emoji) — encoded as a surrogate pair, 2 units —
    would be undercounted by a factor of ~2 if we used ``len()`` directly;
    encoding to UTF-16 and halving the byte count gives the exact unit count.
    POSIX has no such rendering step, so the raw UTF-8 byte count is exact.
    """
    if sys.platform.startswith("win"):
        rendered = subprocess.list2cmdline([prompt])
        return len(rendered.encode("utf-16-le")) // 2
    return len(prompt.encode("utf-8"))


# Preflight hint: the assembled prompt is over the ceiling *after* re-truncation,
# so it is irreducible by truncation — a lone oversized turn (truncation always
# keeps ≥1) or an oversized custom --prompt template. Lowering --token-budget
# cannot help either, so the hint says so and points at the fixes that do.
_ARGV_PREFLIGHT_HINT = (
    "Lowering --token-budget will not help — truncation always keeps at least one "
    "turn and cannot shrink the --prompt template. Use a stdin-capable summarizer "
    "(--summarizer-tool claude or codex), or shorten the oversized turn or custom "
    "--prompt template."
)

# E2BIG hint: a defensive backstop for an argv overflow the preflight did not
# anticipate; the cause is unknown, so a smaller budget (fewer turns) *may* help
# alongside stdin delivery.
_ARGV_E2BIG_HINT = (
    "Use a stdin-capable summarizer (--summarizer-tool claude or codex), lower "
    "--token-budget, or shorten the transcript."
)

_HANDOFF_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "current_task",
        "key_decisions",
        "modified_files",
        "blockers",
        "next_steps",
        "critical_context",
    ],
    "properties": {
        "current_task": {"type": "string"},
        "key_decisions": {"type": "array", "items": {"type": "string"}},
        "modified_files": {"type": "array", "items": {"type": "string"}},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "critical_context": {"type": "string"},
    },
}


class SummarizerToolNotInstalledError(RuntimeError):
    """Raised when the chosen summarizer tool is not in PATH."""


class SummarizerParseError(RuntimeError):
    """Raised when the summarizer returns output we cannot parse."""


class HandoffSummarizer:
    """LLM-backed summarizer that produces a :class:`HandoffDocument` or :class:`RawHandoff`."""

    def __init__(
        self,
        summarizer_tool: AbstractAITool,
        prompt_template: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        model: str | None = None,
    ) -> None:
        self.summarizer_tool = summarizer_tool
        self.prompt_template = prompt_template
        self.token_budget = token_budget
        self.timeout_seconds = timeout_seconds
        self.model = model

    def ensure_installed(self) -> None:
        """Raise if the summarizer tool is not available on this system."""
        installed = AbstractAITool.detect_installed()
        if self.summarizer_tool.TOOL_ID not in installed:
            raise SummarizerToolNotInstalledError(
                f"Summarizer tool '{self.summarizer_tool.TOOL_ID}' is not installed "
                "on this system. Install it, or pass --summarizer-tool with a tool "
                "crossby detected."
            )

    def summarize_structured(
        self,
        transcript: ConversationTranscript,
        source_tool: AIToolID,
        target_tool: AIToolID,
        on_truncate: Callable[[int, int], None] | None = None,
    ) -> HandoffDocument:
        """Summarize ``transcript`` into a parsed :class:`HandoffDocument`."""
        stdin_args = self.summarizer_tool.headless_prompt_stdin_args()
        prepared, prompt = self._prepare_prompt(
            transcript, uses_stdin=stdin_args is not None, on_truncate=on_truncate
        )
        json_schema = _HANDOFF_JSON_SCHEMA if self._supports_json() else None
        raw = self._invoke_tool(prompt, json_schema, stdin_args)
        payload = self._parse_output(raw)
        return self._build_document(payload, prepared.session_ref, source_tool, target_tool)

    def summarize_raw(
        self,
        transcript: ConversationTranscript,
        source_tool: AIToolID,
        target_tool: AIToolID,
        prompt_source: str,
        on_truncate: Callable[[int, int], None] | None = None,
    ) -> RawHandoff:
        """Summarize ``transcript`` and return the tool's raw output unchanged."""
        stdin_args = self.summarizer_tool.headless_prompt_stdin_args()
        prepared, prompt = self._prepare_prompt(
            transcript, uses_stdin=stdin_args is not None, on_truncate=on_truncate
        )
        raw = self._invoke_tool(prompt, None, stdin_args)
        return RawHandoff(
            source_tool=source_tool,
            target_tool=target_tool,
            session_ref=prepared.session_ref,
            body=raw.strip(),
            prompt_source=prompt_source,
            created_at=datetime.now(tz=UTC),
        )

    def _prepare_prompt(
        self,
        transcript: ConversationTranscript,
        *,
        uses_stdin: bool,
        on_truncate: Callable[[int, int], None] | None,
    ) -> tuple[ConversationTranscript, str]:
        """Truncate, assemble, and (for argv delivery) byte-fit the prompt.

        Returns ``(final_transcript, prompt)``. On the argv path (``uses_stdin``
        is ``False``) the assembled prompt is re-truncated turn-by-turn until it
        fits ``_MAX_PROMPT_BYTES``; a prompt that still overflows — a lone
        oversized turn or an oversized custom ``--prompt`` template, neither of
        which a smaller budget can shrink — raises :class:`SummarizerParseError`
        *before* any process is spawned. The stdin path skips the byte ceiling
        entirely (the prompt never touches ``argv``).

        ``on_truncate`` fires at most once, after *all* truncation is complete,
        with ``(total, kept)`` turn counts — and is skipped when the source
        transcript was already ``truncated``.
        """
        self.ensure_installed()
        prepared = truncate_transcript(transcript, self.token_budget)
        prompt = self._build_prompt(prepared)

        if not uses_stdin:
            prepared, prompt = self._fit_argv_bytes(prepared, prompt)
            rendered_len = _argv_render_length(prompt)
            if rendered_len > _MAX_PROMPT_BYTES:
                raise SummarizerParseError(
                    f"Assembled summarizer prompt is {rendered_len} (rendered) against the "
                    f"{_MAX_PROMPT_BYTES} argv ceiling after transcript truncation. "
                    + _ARGV_PREFLIGHT_HINT
                )

        if prepared.truncated and not transcript.truncated and on_truncate is not None:
            on_truncate(len(transcript.turns), len(prepared.turns))
        return prepared, prompt

    def _fit_argv_bytes(
        self,
        transcript: ConversationTranscript,
        prompt: str,
    ) -> tuple[ConversationTranscript, str]:
        """Drop the oldest kept turns until the argv prompt fits the argv ceiling.

        Finds the drop count by binary search rather than dropping one turn at a
        time: rendered prompt length is monotonically non-increasing as more
        oldest turns are dropped (every turn contributes a non-negative amount
        of rendered text — see :meth:`_build_prompt`), so bisecting on the drop
        count is valid and rebuilds the prompt O(log n) times instead of O(n),
        matching ``truncate_transcript``'s keep-most-recent policy. At least one
        turn is always kept — a lone oversized turn is handled by the preflight
        in :meth:`_prepare_prompt`, not here. The token budget is deliberately
        *not* shrunk: a smaller budget need not remove another turn
        (``truncate_transcript`` keeps ≥1 turn) and could rebuild the same
        transcript forever. Fit is judged by :func:`_argv_render_length`, which
        is the Windows-quoted rendered length on Windows (not the raw byte
        count) and the raw UTF-8 byte count elsewhere.
        """
        turns = list(transcript.turns)
        if _argv_render_length(prompt) <= _MAX_PROMPT_BYTES or len(turns) <= 1:
            return transcript, prompt

        def build(drop: int) -> tuple[ConversationTranscript, str]:
            kept = turns[drop:]
            t = ConversationTranscript(
                session_ref=transcript.session_ref,
                turns=kept,
                truncated=True,
            )
            return t, self._build_prompt(t)

        # Binary search for the smallest drop count that fits. `hi` (dropping
        # down to exactly one turn) is the most it will ever drop, mirroring
        # the old loop's stopping condition of `len(turns) > 1`.
        lo, hi = 0, len(turns) - 1
        best_transcript, best_prompt = build(hi)
        if _argv_render_length(best_prompt) > _MAX_PROMPT_BYTES:
            return best_transcript, best_prompt

        while lo < hi:
            mid = (lo + hi) // 2
            mid_transcript, mid_prompt = build(mid)
            if _argv_render_length(mid_prompt) <= _MAX_PROMPT_BYTES:
                hi = mid
                best_transcript, best_prompt = mid_transcript, mid_prompt
            else:
                lo = mid + 1
        return best_transcript, best_prompt

    def _supports_json(self) -> bool:
        return bool(self.summarizer_tool.structured_output_args(_HANDOFF_JSON_SCHEMA))

    def _build_prompt(self, transcript: ConversationTranscript) -> str:
        lines = [self.prompt_template, ""]
        if transcript.truncated:
            # Neutral wording: turns are dropped either to fit the token budget or
            # to fit the argv byte ceiling (the re-truncation path), so don't
            # attribute the drop specifically to the token budget.
            lines.append(
                "Note: transcript was truncated — earlier turns were dropped to fit "
                "the summarizer's size limit. Rely on the last turns for current state.\n"
            )
        lines.append("--- Transcript ---\n")
        for turn in transcript.turns:
            lines.append(render_turn(turn))
            lines.append("")
        return "\n".join(lines)

    def _invoke_tool(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None,
        stdin_args: list[str] | None,
    ) -> str:
        """Run the summarizer with the already-fitted ``prompt``.

        ``stdin_args`` is the value :meth:`AbstractAITool.headless_prompt_stdin_args`
        returned for this tool (selected once up front, never re-queried here).
        When non-``None`` the prompt is delivered through stdin — the command is
        built with ``prompt=None`` and ``stdin_args`` appended after the
        model/schema flags — so no argv byte ceiling applies. Otherwise the
        prompt rides on ``argv`` (already fitted to ``_MAX_PROMPT_BYTES``).
        """
        if stdin_args is not None:
            cmd = self.summarizer_tool.build_launch_command(
                model=self.model,
                prompt=None,
                json_schema=json_schema,
            )
            cmd = [*cmd, *stdin_args]
            stdin_input: str | None = prompt
        else:
            cmd = self.summarizer_tool.build_launch_command(
                model=self.model,
                prompt=prompt,
                json_schema=json_schema,
            )
            stdin_input = None
        logger.info(
            "handoff.summarize.launch",
            tool=str(self.summarizer_tool.TOOL_ID),
            json_schema=bool(json_schema),
            stdin=stdin_input is not None,
        )
        try:
            result = subprocess.run(
                cmd,
                input=stdin_input,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SummarizerParseError(
                f"Summarizer tool timed out after {self.timeout_seconds}s."
            ) from exc
        except OSError as exc:
            if exc.errno == errno.E2BIG:
                raise SummarizerParseError(
                    "Summarizer command line was too long (argument list too long). "
                    + _ARGV_E2BIG_HINT
                ) from exc
            raise SummarizerParseError(f"Summarizer tool failed to run: {exc}") from exc
        if result.returncode != 0:
            raise SummarizerParseError(
                f"Summarizer tool exited {result.returncode}: {result.stderr.strip()}"
            )
        try:
            return self.summarizer_tool.unwrap_structured_output(result.stdout)
        except SummarizerParseError:
            raise
        except Exception as exc:
            raise SummarizerParseError(f"Failed to unwrap tool output: {exc}") from exc

    def _parse_output(self, raw: str) -> dict[str, Any]:
        payload = _try_parse_json(raw)
        if payload is not None:
            return payload
        fallback = parse_markdown_sections(raw)
        if fallback is not None:
            return fallback
        raise SummarizerParseError(
            "Summarizer output did not match JSON or markdown-section format."
        )

    def _build_document(
        self,
        payload: dict[str, Any],
        session_ref: SessionRef,
        source_tool: AIToolID,
        target_tool: AIToolID,
    ) -> HandoffDocument:
        return HandoffDocument(
            source_tool=source_tool,
            target_tool=target_tool,
            session_ref=session_ref,
            current_task=str(payload.get("current_task", "")),
            key_decisions=_as_str_list(payload.get("key_decisions")),
            modified_files=[Path(p) for p in _as_str_list(payload.get("modified_files"))],
            blockers=_as_str_list(payload.get("blockers")),
            next_steps=_as_str_list(payload.get("next_steps")),
            critical_context=str(payload.get("critical_context", "")),
            created_at=datetime.now(tz=UTC),
        )


def render_turn(turn: ConversationTurn) -> str:
    """Render a transcript turn for inclusion in the summarizer prompt."""
    header = f"[{turn.role}]"
    if turn.timestamp is not None:
        header += f" {turn.timestamp.isoformat()}"
    body = turn.content or ""
    tool_calls = ""
    if turn.tool_calls:
        call_lines = []
        for call in turn.tool_calls:
            call_lines.append(f"  - tool_call: {call.name}({_short_args(call.arguments)})")
        tool_calls = "\n" + "\n".join(call_lines)
    file_refs = ""
    if turn.file_refs:
        file_refs = "\n  files: " + ", ".join(str(p) for p in turn.file_refs)
    return f"{header}\n{body}{tool_calls}{file_refs}".rstrip()


def _short_args(args: dict[str, Any], max_len: int = 200) -> str:
    rendered = json.dumps(args, default=str, sort_keys=True)
    if len(rendered) > max_len:
        return rendered[: max_len - 3] + "..."
    return rendered


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    stripped = raw.strip()
    if not stripped:
        return None
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return None
    candidate = stripped[first_brace : last_brace + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


_SECTION_HEADINGS = {
    "current_task": "Current Task",
    "key_decisions": "Key Decisions",
    "modified_files": "Modified Files",
    "blockers": "Blockers",
    "next_steps": "Next Steps",
    "critical_context": "Critical Context",
}

_LIST_KEYS = {"key_decisions", "modified_files", "blockers", "next_steps"}


def parse_markdown_sections(text: str) -> dict[str, Any] | None:
    """Parse a free-form markdown handoff into a payload dict.

    Accepts ``## Heading`` lines (case-insensitive, trimmed). Bulleted list
    sections keep each ``- item`` as a list element; other sections collapse
    to a single string.
    """
    if "##" not in text:
        return None

    payload: dict[str, Any] = {}
    heading_to_key = {v.lower(): k for k, v in _SECTION_HEADINGS.items()}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_key is None:
            return
        if current_key in _LIST_KEYS:
            items = _extract_bullets(current_lines)
            payload[current_key] = items
        else:
            payload[current_key] = "\n".join(current_lines).strip()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            if heading in heading_to_key:
                flush()
                current_key = heading_to_key[heading]
                current_lines = []
                continue
            if current_key is not None:
                current_lines.append(line)
            continue
        if current_key is not None:
            current_lines.append(line)

    flush()
    if not payload:
        return None
    for key in _SECTION_HEADINGS:
        payload.setdefault(key, [] if key in _LIST_KEYS else "")
    return payload


def _extract_bullets(lines: list[str]) -> list[str]:
    bullets: list[str] = []
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^[-*]\s+(.*)$", stripped)
        if match:
            value = match.group(1).strip()
            if value:
                bullets.append(value)
    return bullets


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def estimate_prompt_tokens(transcript: ConversationTranscript) -> int:
    """Rough token cost the truncation budget measures for ``transcript``.

    Sums :func:`turn_tokens` per turn — content **plus** tool-call names and
    argument strings — so the estimate matches exactly what
    ``truncate_transcript`` counts when deciding whether to drop turns. This is
    deliberately *not* the assembled prompt's byte size: the prompt template,
    per-turn ``[role]`` headers, ISO timestamps, ``files:`` lines, and the
    rendered/truncated tool-call representation all sit outside this figure.
    """
    return sum(turn_tokens(turn) for turn in transcript.turns)
