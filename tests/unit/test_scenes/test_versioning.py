"""CLI version detection and gating helpers."""

from __future__ import annotations

import subprocess

import pytest

from crossby.scenes import versioning


class TestParseSemver:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2.1.218 (Claude Code)", (2, 1, 218)),
            ("codex-cli 0.146.0", (0, 146, 0)),
            ("GitHub Copilot CLI 1.0.77.", (1, 0, 77)),
            ("1.1.10", (1, 1, 10)),
            ("v2.1", (2, 1, 0)),
            ("no version here", None),
        ],
    )
    def test_parse(self, text: str, expected: tuple[int, int, int] | None) -> None:
        assert versioning.parse_semver(text) == expected


class TestAtLeast:
    def test_meets_floor(self) -> None:
        assert versioning.at_least((2, 1, 218), (2, 1, 129)) is True

    def test_below_floor(self) -> None:
        assert versioning.at_least((2, 1, 100), (2, 1, 129)) is False

    def test_unknown_fails_closed(self) -> None:
        assert versioning.at_least(None, (2, 1, 129)) is False


class TestDetectBinaryVersion:
    def test_missing_binary_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("crossby.scenes.versioning.shutil.which", lambda _b: None)
        assert versioning.detect_binary_version("nope") is None

    def test_parses_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("crossby.scenes.versioning.shutil.which", lambda _b: "/usr/bin/claude")

        def fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="2.1.218 (Claude Code)\n", stderr=""
            )

        monkeypatch.setattr("crossby.scenes.versioning.subprocess.run", fake_run)
        assert versioning.detect_binary_version("claude") == (2, 1, 218)

    def test_falls_back_to_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("crossby.scenes.versioning.shutil.which", lambda _b: "/x")

        def fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="v0.9.1")

        monkeypatch.setattr("crossby.scenes.versioning.subprocess.run", fake_run)
        assert versioning.detect_binary_version("x") == (0, 9, 1)

    def test_nonzero_exit_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A failed --version whose error output still carries a version-shaped
        # string must read as unknown, not sneak past the feature gate.
        monkeypatch.setattr("crossby.scenes.versioning.shutil.which", lambda _b: "/x")

        def fake_run(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="error: unknown flag near v9.9.9"
            )

        monkeypatch.setattr("crossby.scenes.versioning.subprocess.run", fake_run)
        assert versioning.detect_binary_version("x") is None

    def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("crossby.scenes.versioning.shutil.which", lambda _b: "/x")

        def boom(*_a: object, **_k: object) -> object:
            raise subprocess.TimeoutExpired(cmd="x", timeout=5)

        monkeypatch.setattr("crossby.scenes.versioning.subprocess.run", boom)
        assert versioning.detect_binary_version("x") is None
