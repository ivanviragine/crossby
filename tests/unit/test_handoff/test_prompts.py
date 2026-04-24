"""Tests for the bundled-prompt loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossby.handoff.prompts import (
    PRESETS,
    PromptNotFoundError,
    load_launch_template,
    load_preset,
    load_user_prompt,
)


def test_load_preset_default_returns_nonempty_string() -> None:
    body = load_preset("default")
    assert body
    # Must retain the 6-section schema so the structured parser still works.
    assert "current_task" in body


def test_load_preset_cc_compact_returns_nonempty_string() -> None:
    body = load_preset("cc-compact")
    assert body
    assert "<analysis>" in body
    assert "<summary>" in body


def test_load_preset_unknown_raises() -> None:
    with pytest.raises(PromptNotFoundError, match="Unknown prompt preset"):
        load_preset("bogus")


def test_load_user_prompt_reads_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "custom.md"
    target.write_text("my custom prompt\n", encoding="utf-8")
    assert load_user_prompt(target) == "my custom prompt\n"


def test_load_user_prompt_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PromptNotFoundError, match="not found"):
        load_user_prompt(tmp_path / "does_not_exist.md")


def test_load_launch_template_contains_path_placeholder() -> None:
    template = load_launch_template()
    assert "{path}" in template


def test_presets_dict_lists_known_presets() -> None:
    assert set(PRESETS) == {"default", "cc-compact"}
