"""Load summarizer and launch prompts from bundled .md files or user paths."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"

PRESETS: dict[str, str] = {
    "default": "summarize_default.md",
    "cc-compact": "summarize_cc_compact.md",
}

_LAUNCH_TEMPLATE_FILE = "launch_initial_message.md"


class PromptNotFoundError(RuntimeError):
    """Raised when a prompt preset or user-supplied prompt cannot be loaded."""


def load_preset(name: str) -> str:
    """Return the contents of a bundled prompt preset."""
    filename = PRESETS.get(name)
    if filename is None:
        valid = ", ".join(sorted(PRESETS))
        raise PromptNotFoundError(
            f"Unknown prompt preset {name!r}. Valid presets: {valid}."
        )
    path = _PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptNotFoundError(
            f"Bundled prompt {filename!r} is missing at {path}. This is a packaging bug."
        ) from exc


def load_user_prompt(path: Path) -> str:
    """Return the contents of a user-supplied prompt file at an absolute path."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptNotFoundError(f"Prompt file not found: {path}") from exc
    except PermissionError as exc:
        raise PromptNotFoundError(f"Cannot read prompt file {path}: {exc}") from exc


def load_launch_template() -> str:
    """Return the launch-time initial-message template with a ``{path}`` placeholder."""
    path = _PROMPTS_DIR / _LAUNCH_TEMPLATE_FILE
    try:
        return path.read_text(encoding="utf-8").rstrip()
    except FileNotFoundError as exc:
        raise PromptNotFoundError(
            f"Bundled launch template missing at {path}. This is a packaging bug."
        ) from exc
