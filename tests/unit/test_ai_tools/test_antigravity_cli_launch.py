"""Antigravity CLI launch-command coverage: yolo, effort, initial message, build_launch_command."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossby.ai_tools.antigravity_cli import AntigravityCLIAdapter
from crossby.models.ai import EffortLevel, TokenUsage


def _model_of(cmd: list[str]) -> str | None:
    """Return the value passed to --model, or None if the flag is absent."""
    return cmd[cmd.index("--model") + 1] if "--model" in cmd else None


class TestAntigravityCLIYolo:
    def test_yolo_skips_permissions_without_sandbox(self) -> None:
        # agy's --sandbox is a terminal restriction (blocks every shell command),
        # not a write sandbox, so yolo must NOT pair it with skip-permissions.
        args = AntigravityCLIAdapter().yolo_args()
        assert args == ["--dangerously-skip-permissions"]

    def test_supports_yolo_capability(self) -> None:
        assert AntigravityCLIAdapter().capabilities().supports_yolo is True


class TestAntigravityCLIEffort:
    """agy bakes effort into the model ID and never emits a separate --effort."""

    def test_supports_effort_capability(self) -> None:
        assert AntigravityCLIAdapter().capabilities().supports_effort is True

    def test_supported_efforts_is_low_medium_high(self) -> None:
        caps = AntigravityCLIAdapter().capabilities()
        assert caps.supported_efforts == (
            EffortLevel.LOW,
            EffortLevel.MEDIUM,
            EffortLevel.HIGH,
        )

    @pytest.mark.parametrize("effort", [EffortLevel.LOW, EffortLevel.MEDIUM, EffortLevel.HIGH])
    def test_gemini_3_8_bakes_each_supported_effort(self, effort: EffortLevel) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(model="gemini-3.8-flash", effort=effort)
        assert _model_of(cmd) == f"gemini-3.8-flash-{effort.value}"
        assert "--effort" not in cmd

    def test_gemini_3_8_no_effort_defaults_to_medium(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(model="gemini-3.8-flash", effort=None)
        assert _model_of(cmd) == "gemini-3.8-flash-medium"

    @pytest.mark.parametrize("effort", [EffortLevel.XHIGH, EffortLevel.MAX])
    def test_gemini_3_8_xhigh_and_max_normalize_to_high(self, effort: EffortLevel) -> None:
        with pytest.warns(UserWarning, match="only low/medium/high"):
            resolved = AntigravityCLIAdapter().resolve_effort_model("gemini-3.8-flash", effort)
        assert resolved == "gemini-3.8-flash-high"

    def test_gemini_3_8_suffix_wins_and_minimal_is_never_emitted(self) -> None:
        resolved = AntigravityCLIAdapter().resolve_effort_model(
            "gemini-3.8-flash-low", EffortLevel.HIGH
        )
        assert resolved == "gemini-3.8-flash-low"
        assert "minimal" not in resolved

    def test_base_plus_effort_bakes_suffix_and_omits_effort_flag(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(
            model="gemini-3.6-flash", effort=EffortLevel.HIGH
        )
        assert _model_of(cmd) == "gemini-3.6-flash-high"
        assert "--effort" not in cmd

    def test_idempotent_suffixed_model_plus_matching_effort(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(
            model="gemini-3.6-flash-high", effort=EffortLevel.HIGH
        )
        assert _model_of(cmd) == "gemini-3.6-flash-high"
        assert "--effort" not in cmd

    def test_suffix_wins_over_conflicting_effort(self) -> None:
        # Precedence: an effort baked into the ID beats a separately supplied one,
        # and no --effort is emitted — so a stored suffixed model stays valid.
        cmd = AntigravityCLIAdapter().build_launch_command(
            model="gemini-3.6-flash-high", effort=EffortLevel.LOW
        )
        assert _model_of(cmd) == "gemini-3.6-flash-high"
        assert "--effort" not in cmd

    @pytest.mark.parametrize("effort", [EffortLevel.XHIGH, EffortLevel.MAX])
    def test_xhigh_and_max_normalize_to_high_with_warning(self, effort: EffortLevel) -> None:
        with pytest.warns(UserWarning, match="only low/medium/high"):
            resolved = AntigravityCLIAdapter().resolve_effort_model("gemini-3.6-flash", effort)
        assert resolved == "gemini-3.6-flash-high"

    def test_per_model_gap_snaps_to_nearest_tier_with_warning(self) -> None:
        # gemini-3.1-pro supports only low/high; medium snaps up (ties → higher).
        with pytest.warns(UserWarning, match="no 'medium' effort tier"):
            resolved = AntigravityCLIAdapter().resolve_effort_model(
                "gemini-3.1-pro", EffortLevel.MEDIUM
            )
        assert resolved == "gemini-3.1-pro-high"

    def test_invalid_stored_suffix_is_repaired(self) -> None:
        # gemini-3.1-pro-medium is a tier agy rejects — snap to a valid one.
        with pytest.warns(UserWarning, match="no 'medium' effort tier"):
            resolved = AntigravityCLIAdapter().resolve_effort_model("gemini-3.1-pro-medium", None)
        assert resolved == "gemini-3.1-pro-high"

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini-3.6-flash-xhigh", "gemini-3.6-flash-high"),
            ("gemini-3.1-pro-max", "gemini-3.1-pro-high"),
        ],
    )
    def test_stored_xhigh_max_suffix_normalizes_to_high(self, model: str, expected: str) -> None:
        # agy never emits a -xhigh/-max ID; a hand-written one must normalize
        # down rather than pass through as an invalid command.
        with pytest.warns(UserWarning, match="only low/medium/high"):
            resolved = AntigravityCLIAdapter().resolve_effort_model(model, None)
        assert resolved == expected

    def test_no_effort_defaults_to_medium_when_supported(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(model="gemini-3.6-flash", effort=None)
        assert _model_of(cmd) == "gemini-3.6-flash-medium"
        assert "--effort" not in cmd

    def test_no_effort_defaults_to_nearest_when_medium_unsupported(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(model="gemini-3.1-pro", effort=None)
        assert _model_of(cmd) == "gemini-3.1-pro-high"
        assert "--effort" not in cmd

    def test_effort_without_model_emits_no_model_flag(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(model=None, effort=EffortLevel.HIGH)
        assert "--model" not in cmd

    @pytest.mark.parametrize(
        "model",
        ["gpt-oss-120b", "claude-sonnet-4-6", "claude-opus-4-6-thinking"],
    )
    def test_bare_models_ignore_effort_and_pass_through(self, model: str) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(model=model, effort=EffortLevel.HIGH)
        assert _model_of(cmd) == model
        assert "--effort" not in cmd

    def test_fixed_gpt_oss_suffix_is_preserved_exactly(self) -> None:
        # `agy models` reports this exact ID. Its suffix is part of the model
        # name, so even a separately requested effort must not rewrite it.
        cmd = AntigravityCLIAdapter().build_launch_command(
            model="gpt-oss-120b-medium", effort=EffortLevel.HIGH
        )
        assert _model_of(cmd) == "gpt-oss-120b-medium"
        assert "--effort" not in cmd


class TestAntigravityCLIInitialMessage:
    def test_initial_message_uses_prompt_interactive_flag(self) -> None:
        args = AntigravityCLIAdapter().initial_message_args("do the thing")
        assert args == ["--prompt-interactive", "do the thing"]

    def test_supports_initial_message_capability(self) -> None:
        assert AntigravityCLIAdapter().capabilities().supports_initial_message is True


class TestAntigravityCLIPlanMode:
    def test_plan_dir_args_uses_add_dir(self) -> None:
        args = AntigravityCLIAdapter().plan_dir_args("/tmp/plan")
        assert args == ["--add-dir", "/tmp/plan"]


class TestAntigravityCLIParseTranscript:
    def test_parse_transcript_always_empty(self, tmp_path: Path) -> None:
        transcript = tmp_path / "session.txt"
        transcript.write_text("anything at all, this is never parsed")
        usage = AntigravityCLIAdapter().parse_transcript(transcript)
        assert usage == TokenUsage()


class TestAntigravityCLIBuildLaunchCommand:
    def test_combines_model_effort_and_yolo(self) -> None:
        # A base model + effort bakes into one suffixed --model with NO --effort
        # flag (agy rejects the two together), alongside the yolo flags.
        cmd = AntigravityCLIAdapter().build_launch_command(
            model="gemini-3.6-flash",
            effort=EffortLevel.HIGH,
            yolo=True,
        )
        assert cmd[0] == "agy"
        assert _model_of(cmd) == "gemini-3.6-flash-high"
        assert "--effort" not in cmd
        assert "--dangerously-skip-permissions" in cmd
        # Regression guard: agy's --sandbox is a terminal restriction that blocks
        # shell commands, so a yolo launch must never emit it.
        assert "--sandbox" not in cmd

    def test_initial_message_is_first_positional(self) -> None:
        cmd = AntigravityCLIAdapter().build_launch_command(initial_message="hello there")
        assert cmd == ["agy", "--prompt-interactive", "hello there"]
