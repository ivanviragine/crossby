"""Codex project-trust detection from ~/.codex/config.toml."""

from __future__ import annotations

from pathlib import Path

from crossby.scenes.trust import codex_trusts_project


def _write_codex_config(home: Path, body: str) -> None:
    (home / ".codex").mkdir(parents=True, exist_ok=True)
    (home / ".codex" / "config.toml").write_text(body, encoding="utf-8")


def test_trusted_project_detected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    _write_codex_config(home, f'[projects."{project}"]\ntrust_level = "trusted"\n')
    assert codex_trusts_project(project, home=home) is True


def test_untrusted_project_not_detected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    _write_codex_config(home, f'[projects."{project}"]\ntrust_level = "untrusted"\n')
    assert codex_trusts_project(project, home=home) is False


def test_absent_project_not_detected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "proj"
    project.mkdir()
    _write_codex_config(home, '[projects."/some/other/path"]\ntrust_level = "trusted"\n')
    assert codex_trusts_project(project, home=home) is False


def test_missing_config_returns_false(tmp_path: Path) -> None:
    assert codex_trusts_project(tmp_path / "proj", home=tmp_path / "home") is False


def test_malformed_config_returns_false(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_codex_config(home, "this is not = valid toml [[[")
    assert codex_trusts_project(tmp_path, home=home) is False
