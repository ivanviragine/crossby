"""Summarize a ConversationTranscript into a structured HandoffDocument."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.handoff.models import (
    ConversationTranscript,
    ConversationTurn,
    HandoffDocument,
    SessionRef,
)
from crossby.handoff.truncate import approx_tokens, truncate_transcript
from crossby.models.ai import AIToolID

logger = structlog.get_logger()

DEFAULT_TOKEN_BUDGET = 32_000
DEFAULT_TIMEOUT_SECONDS = 120

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

_PROMPT_HEADER = (
    "You are generating a structured handoff document so another AI coding CLI "
    "can continue work without losing context.\n\n"
    "Produce exactly these sections. For JSON output, return a single JSON "
    "object with the keys listed below and no prose around it. For markdown, "
    "use the exact headings shown and bulleted lists under list-valued "
    "sections.\n\n"
    "Sections:\n"
    "- current_task (string)\n"
    "- key_decisions (list of strings)\n"
    "- modified_files (list of file paths)\n"
    "- blockers (list of strings)\n"
    "- next_steps (list of strings)\n"
    "- critical_context (string)\n\n"
    "Markdown headings must be exactly: '## Current Task', '## Key Decisions', "
    "'## Modified Files', '## Blockers', '## Next Steps', '## Critical Context'.\n"
)


class SummarizerToolNotInstalled(RuntimeError):
    """Raised when the chosen summarizer tool is not in PATH."""


class SummarizerParseError(RuntimeError):
    """Raised when the summarizer returns output we cannot parse."""


class HandoffSummarizer:
    """LLM-backed summarizer that produces a :class:`HandoffDocument`."""

    def __init__(
        self,
        summarizer_tool: AbstractAITool,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        model: str | None = None,
    ) -> None:
        self.summarizer_tool = summarizer_tool
        self.token_budget = token_budget
        self.timeout_seconds = timeout_seconds
        self.model = model

    def ensure_installed(self) -> None:
        """Raise if the summarizer tool is not available on this system."""
        installed = AbstractAITool.detect_installed()
        if self.summarizer_tool.TOOL_ID not in installed:
            raise SummarizerToolNotInstalled(
                f"Summarizer tool '{self.summarizer_tool.TOOL_ID}' is not installed "
                "on this system. Install it, or pass --summarizer-tool with a tool "
                "crossby detected."
            )

    def summarize(
        self,
        transcript: ConversationTranscript,
        source_tool: AIToolID,
        target_tool: AIToolID,
        on_truncate: Callable[[int, int], None] | None = None,
    ) -> HandoffDocument:
        """Summarize ``transcript`` into a HandoffDocument."""
        self.ensure_installed()

        prepared = truncate_transcript(transcript, self.token_budget)
        if prepared.truncated and not transcript.truncated and on_truncate is not None:
            on_truncate(len(transcript.turns), len(prepared.turns))

        prompt = self._build_prompt(prepared)
        json_schema = _HANDOFF_JSON_SCHEMA if self._supports_json() else None
        raw = self._invoke_tool(prompt, json_schema)
        payload = self._parse_output(raw)
        return self._build_document(payload, prepared.session_ref, source_tool, target_tool)

    def _supports_json(self) -> bool:
        return bool(self.summarizer_tool.structured_output_args(_HANDOFF_JSON_SCHEMA))

    def _build_prompt(self, transcript: ConversationTranscript) -> str:
        lines = [_PROMPT_HEADER, ""]
        if transcript.truncated:
            lines.append(
                "Note: transcript was truncated — earlier turns were dropped to fit "
                "the token budget. Rely on the last turns for current state.\n"
            )
        lines.append("--- Transcript ---\n")
        for turn in transcript.turns:
            lines.append(render_turn(turn))
            lines.append("")
        return "\n".join(lines)

    def _invoke_tool(
        self, prompt: str, json_schema: dict[str, Any] | None
    ) -> str:
        cmd = self.summarizer_tool.build_launch_command(
            model=self.model,
            prompt=prompt,
            json_schema=json_schema,
        )
        logger.info(
            "handoff.summarize.launch",
            tool=str(self.summarizer_tool.TOOL_ID),
            json_schema=bool(json_schema),
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise SummarizerParseError(
                f"Summarizer tool exited {result.returncode}: {result.stderr.strip()}"
            )
        return result.stdout

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
            created_at=datetime.now(tz=timezone.utc),
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
    """Rough token cost of the full transcript (before truncation)."""
    total = 0
    for turn in transcript.turns:
        total += approx_tokens(turn.content)
    return total
