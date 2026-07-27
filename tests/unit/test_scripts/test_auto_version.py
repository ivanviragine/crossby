"""Tests for scripts/auto_version.py's lockfile handling.

``scripts/`` is not an importable package, so the module is loaded from its
file path (matching :mod:`test_probe_models`). Only :func:`update_lockfile` is
exercised — the version-arithmetic and git plumbing around it are a developer
utility.

Regression context: the bump used to stage only ``pyproject.toml`` and
``__init__.py``, so every release shipped a ``uv.lock`` still asserting the
previous version.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "auto_version.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("auto_version", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def auto_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The script with its paths redirected into a scratch project."""
    module = _load()
    monkeypatch.setattr(module, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(module, "LOCK_FILE", tmp_path / "uv.lock")
    return module


def test_returns_false_without_a_lockfile(auto_version: ModuleType) -> None:
    assert auto_version.update_lockfile() is False


def test_reports_a_changed_lockfile(
    auto_version: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = auto_version.LOCK_FILE
    lock.write_text('version = "0.1.0"\n', encoding="utf-8")

    def _fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        lock.write_text('version = "0.2.0"\n', encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert auto_version.update_lockfile() is True


def test_reports_an_unchanged_lockfile(
    auto_version: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    auto_version.LOCK_FILE.write_text('version = "0.1.0"\n', encoding="utf-8")
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0, stderr="")
    )
    assert auto_version.update_lockfile() is False


def test_a_failing_uv_lock_is_not_fatal(
    auto_version: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale lockfile must not abort a release."""
    auto_version.LOCK_FILE.write_text('version = "0.1.0"\n', encoding="utf-8")
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=1, stderr="boom")
    )

    assert auto_version.update_lockfile() is False
    assert "warning" in capsys.readouterr().out.lower()


def test_a_missing_uv_binary_is_not_fatal(
    auto_version: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    auto_version.LOCK_FILE.write_text('version = "0.1.0"\n', encoding="utf-8")

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("uv not found")

    monkeypatch.setattr(subprocess, "run", _raise)

    assert auto_version.update_lockfile() is False
    assert "warning" in capsys.readouterr().out.lower()
