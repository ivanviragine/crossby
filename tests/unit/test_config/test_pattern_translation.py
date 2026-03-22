"""Tests for shared pattern translation helpers."""

from __future__ import annotations

from crossby.config.patterns import (
    canonical_to_shell,
    canonical_to_wrapped,
    shell_to_canonical,
    wrapped_to_canonical,
)


class TestCanonicalToWrapped:
    def test_wraps_pattern_in_requested_wrapper(self) -> None:
        assert canonical_to_wrapped("crossby:*", "Bash") == "Bash(crossby:*)"
        assert (
            canonical_to_wrapped("./scripts/check.sh:*", "Shell") == "Shell(./scripts/check.sh:*)"
        )

    def test_preserves_bare_commands(self) -> None:
        assert canonical_to_wrapped("git", "Bash") == "Bash(git)"


class TestWrappedToCanonical:
    def test_strips_known_wrapper(self) -> None:
        assert wrapped_to_canonical("Bash(crossby:*)", "Bash") == "crossby:*"
        assert (
            wrapped_to_canonical("Shell(./scripts/check.sh:*)", "Shell") == "./scripts/check.sh:*"
        )

    def test_passthrough_for_non_matching_wrapper(self) -> None:
        assert wrapped_to_canonical("shell(crossby:*)", "Bash") == "shell(crossby:*)"


class TestShellHelpers:
    def test_canonical_to_shell(self) -> None:
        assert canonical_to_shell("crossby:*") == "shell(crossby:*)"
        assert canonical_to_shell("git") == "shell(git)"

    def test_shell_to_canonical(self) -> None:
        assert shell_to_canonical("shell(crossby:*)") == "crossby:*"
        assert shell_to_canonical("shell(git)") == "git"

    def test_shell_to_canonical_passthrough_for_plain_pattern(self) -> None:
        assert shell_to_canonical("crossby:*") == "crossby:*"
