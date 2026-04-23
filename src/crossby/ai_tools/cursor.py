"""Cursor CLI adapter."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

import structlog

from crossby.ai_tools.base import AbstractAITool
from crossby.data import get_models_for_tool
from crossby.handoff.models import ConversationTranscript, SessionRef
from crossby.handoff.readers import cursor as cursor_reader
from crossby.models.ai import (
    AIToolCapabilities,
    AIToolID,
    AIToolType,
    EffortLevel,
)

logger = structlog.get_logger()

# Cursor model IDs that already encode an effort level in their name — e.g.
# "claude-opus-4-7-high", "claude-opus-4-7-thinking-xhigh". Appending
# "-thinking" to these would produce invalid IDs.
_EFFORT_LEVEL_SUFFIXES = frozenset({"-low", "-medium", "-high", "-xhigh", "-max"})

# Models that have no "-thinking" variant — appending the suffix produces an invalid ID.
_NO_THINKING_MODELS: frozenset[str] = frozenset({"auto"})


class CursorAdapter(AbstractAITool):
    """Adapter for Cursor CLI (``agent`` binary).

    Cursor is an AI-powered IDE with a terminal CLI that supports plan mode,
    model selection, headless execution, and skill discovery.

    Cursor uses its own model ID namespace — e.g. ``sonnet-4.6``, ``opus-4.6``,
    ``gpt-5.3-codex`` — so no format normalization is needed.

    For high/max effort, Cursor uses thinking model variants (e.g.,
    ``sonnet-4.6-thinking``) rather than a separate effort flag.
    """

    TOOL_ID: ClassVar[AIToolID] = AIToolID.CURSOR

    def capabilities(self) -> AIToolCapabilities:
        return AIToolCapabilities(
            tool_id=AIToolID.CURSOR,
            display_name="Cursor",
            binary="agent",
            tool_type=AIToolType.TERMINAL,
            supports_model_flag=True,
            headless_flag="--print",
            supports_headless=True,
            supports_effort=True,
            supports_yolo=True,
        )

    def initial_message_args(self, prompt: str) -> list[str]:
        """Cursor accepts the initial message as a positional argument."""
        return [prompt]

    def locate_sessions(self, project_path: Path) -> list[SessionRef]:
        return cursor_reader.locate_sessions(project_path)

    def read_session(self, ref: SessionRef) -> ConversationTranscript:
        return cursor_reader.read_session(ref)

    def is_model_compatible(self, model: str) -> bool:
        """Cursor accepts all model IDs."""
        return True

    def plan_mode_args(self) -> list[str]:
        """Cursor supports ``--mode plan``."""
        return ["--mode", "plan"]

    def yolo_args(self) -> list[str]:
        """Cursor uses ``--force`` (``--yolo`` is an alias)."""
        return ["--force"]

    def resolve_effort_model(self, model: str | None, effort: EffortLevel) -> str | None:
        """For high/xhigh/max effort, append ``-thinking`` to the model ID.

        Models that already encode effort (e.g. ``-high``, ``-xhigh``) or
        thinking mode (``-thinking``) in their name are returned unchanged.

        The constructed ``<model>-thinking`` ID is validated against the
        bundled Cursor model registry. If the registry has no matching
        entry (e.g. the model has no thinking variant, or is unknown to
        crossby), the original ``model`` is returned unchanged so the
        Cursor CLI receives a valid ID.
        """
        if (
            effort in (EffortLevel.HIGH, EffortLevel.XHIGH, EffortLevel.MAX)
            and model
            and model not in _NO_THINKING_MODELS
            and not model.endswith("-thinking")
            and not model.endswith(tuple(_EFFORT_LEVEL_SUFFIXES))
        ):
            candidate = f"{model}-thinking"
            if candidate in get_models_for_tool(AIToolID.CURSOR):
                return candidate
        return model

    def preserve_session_data(self, working_dir: Path, main_checkout_path: Path) -> bool:
        """Copy Cursor session data from source directory to target's project dir.

        Cursor stores sessions in ``~/.cursor/projects/<encoded-path>/``.
        The path encoding strips the leading ``/`` then replaces remaining
        ``/`` with ``-``, so ``/Users/foo/bar`` becomes ``Users-foo-bar``
        (note: no leading dash, unlike Claude Code).

        Files are copied without overwriting any that already exist in the
        target's session directory, so existing data is preserved.
        """
        cursor_projects_dir = Path.home() / ".cursor" / "projects"

        wt_encoded = str(working_dir).lstrip("/").replace("/", "-")
        main_encoded = str(main_checkout_path).lstrip("/").replace("/", "-")

        wt_session_dir = cursor_projects_dir / wt_encoded
        main_session_dir = cursor_projects_dir / main_encoded

        if not wt_session_dir.exists():
            logger.debug(
                "cursor.preserve_session_data.no_source",
                working_dir=str(working_dir),
            )
            return True

        main_session_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for item in wt_session_dir.iterdir():
            dest = main_session_dir / item.name
            if dest.exists():
                continue
            if item.is_file():
                shutil.copy2(item, dest)
                copied += 1
            elif item.is_dir():
                shutil.copytree(item, dest)
                copied += 1

        logger.info(
            "cursor.preserve_session_data.copied",
            working_dir=str(working_dir),
            main=str(main_checkout_path),
            items=copied,
        )
        return True

    def session_data_dirs(self) -> list[str]:
        return [".cursor"]
