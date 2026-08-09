"""Tests for the shared checked-write helper used by init and scene authoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossby.config.loader import ConfigError, parse_config_file
from crossby.config.safe_write import (
    ConfigWriteError,
    resolve_config_target,
    write_config_checked,
)

VALID = "version: 1\nai:\n  default_tool: claude\n"
BROKEN = "version: 1\nscenes:\n  x:\n  bad: [unclosed\n"


def test_writes_and_reparses_valid_config(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    write_config_checked(target, VALID)
    assert parse_config_file(target).ai.default_tool == "claude"
    # No backup is left behind on success.
    assert not list(tmp_path.glob("*.bak*"))


def test_invalid_render_restores_backup_byte_for_byte(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    original = "version: 1\nprofiles:\n  ccyolo:\n    tool: claude\n"
    target.write_text(original, encoding="utf-8")

    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(target, BROKEN)

    assert exc.value.restored is True
    assert target.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak*"))


def test_invalid_render_with_no_prior_file_removes_it(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(target, BROKEN)
    assert exc.value.restored is False
    assert not target.exists()


def test_validator_failure_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    original = "version: 1\n"
    target.write_text(original, encoding="utf-8")

    def _reject(_config: object) -> None:
        raise ConfigError("nope")

    with pytest.raises(ConfigWriteError):
        write_config_checked(target, VALID, validate=_reject)
    assert target.read_text(encoding="utf-8") == original


def test_validator_runs_on_parsed_config(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    seen: list[str | None] = []
    write_config_checked(target, VALID, validate=lambda cfg: seen.append(cfg.ai.default_tool))
    assert seen == ["claude"]


def test_symlinked_config_write_preserves_link(tmp_path: Path) -> None:
    real = tmp_path / "real.crossby.yml"
    original = "version: 1\n"
    real.write_text(original, encoding="utf-8")
    target = tmp_path / ".crossby.yml"
    target.symlink_to(real)

    write_config_checked(target, VALID)

    assert target.is_symlink()
    assert target.resolve() == real.resolve()
    assert real.read_text(encoding="utf-8") == VALID
    assert not list(tmp_path.glob("*.bak*"))


# --- resolve_config_target: direct unit tests -------------------------------


def test_resolve_regular_file_passthrough(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    target.write_text(VALID, encoding="utf-8")
    assert resolve_config_target(target) == target


def test_resolve_not_yet_existing_plain_path(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    assert not target.exists()
    assert resolve_config_target(target) == target


def test_resolve_dangling_symlink_returns_intended_target(tmp_path: Path) -> None:
    real = tmp_path / "real.crossby.yml"
    target = tmp_path / ".crossby.yml"
    target.symlink_to(real)
    assert target.is_symlink()
    assert not target.exists()

    assert resolve_config_target(target) == real.resolve()


def test_resolve_self_loop_raises(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    target.symlink_to(target)

    with pytest.raises(ConfigWriteError) as exc:
        resolve_config_target(target)
    assert exc.value.restored is False


def test_resolve_two_link_cycle_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.crossby.yml"
    b = tmp_path / "b.crossby.yml"
    a.symlink_to(b)
    b.symlink_to(a)

    with pytest.raises(ConfigWriteError) as exc:
        resolve_config_target(a)
    assert exc.value.restored is False


def test_resolve_symlink_to_directory_raises(tmp_path: Path) -> None:
    some_dir = tmp_path / "some_dir"
    some_dir.mkdir()
    target = tmp_path / ".crossby.yml"
    target.symlink_to(some_dir)

    with pytest.raises(ConfigWriteError) as exc:
        resolve_config_target(target)
    assert exc.value.restored is False
    assert "non-regular-file" in str(exc.value.original)


def test_resolve_plain_directory_raises(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    target.mkdir()

    with pytest.raises(ConfigWriteError) as exc:
        resolve_config_target(target)
    assert exc.value.restored is False
    assert "non-regular-file" in str(exc.value.original)


# --- write_config_checked: cyclic / non-file target integration -------------


def test_write_self_referential_symlink_preserves_link(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    target.symlink_to(target)

    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(target, VALID)

    assert exc.value.restored is False
    # The link is preserved, never clobbered into a regular file.
    assert target.is_symlink()
    assert not list(tmp_path.glob("*.bak*"))


def test_write_two_link_cycle_preserves_links(tmp_path: Path) -> None:
    a = tmp_path / ".crossby.yml"
    b = tmp_path / "b.crossby.yml"
    a.symlink_to(b)
    b.symlink_to(a)

    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(a, VALID)

    assert exc.value.restored is False
    assert a.is_symlink()
    assert b.is_symlink()
    assert not list(tmp_path.glob("*.bak*"))


def test_write_symlink_to_directory_raises_clean_error(tmp_path: Path) -> None:
    some_dir = tmp_path / "some_dir"
    some_dir.mkdir()
    existing = some_dir / "keep.txt"
    existing.write_text("keep", encoding="utf-8")
    target = tmp_path / ".crossby.yml"
    target.symlink_to(some_dir)

    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(target, VALID)

    # A clean ConfigWriteError, not a raw IsADirectoryError.
    assert not isinstance(exc.value.original, IsADirectoryError)
    assert exc.value.restored is False
    # Directory untouched, symlink preserved, no backup created.
    assert target.is_symlink()
    assert list(some_dir.iterdir()) == [existing]
    assert not list(tmp_path.glob("*.bak*"))


def test_write_direct_directory_target_raises_clean_error(tmp_path: Path) -> None:
    target = tmp_path / ".crossby.yml"
    target.mkdir()
    existing = target / "keep.txt"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(target, VALID)

    assert not isinstance(exc.value.original, IsADirectoryError)
    assert exc.value.restored is False
    # Directory untouched, no backup created.
    assert target.is_dir()
    assert list(target.iterdir()) == [existing]
    assert not list(tmp_path.glob("*.bak*"))


def test_symlinked_config_failed_write_restores_real_target(tmp_path: Path) -> None:
    real = tmp_path / "real.crossby.yml"
    original = "version: 1\nprofiles:\n  ccyolo:\n    tool: claude\n"
    real.write_text(original, encoding="utf-8")
    target = tmp_path / ".crossby.yml"
    target.symlink_to(real)

    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(target, BROKEN)

    assert exc.value.restored is True
    assert target.is_symlink()
    assert real.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak*"))


def test_broken_symlink_valid_write_creates_resolved_target(tmp_path: Path) -> None:
    real = tmp_path / "real.crossby.yml"
    target = tmp_path / ".crossby.yml"
    target.symlink_to(real)
    assert target.is_symlink()
    assert not target.exists()

    write_config_checked(target, VALID)

    assert target.is_symlink()
    assert target.resolve() == real.resolve()
    assert real.read_text(encoding="utf-8") == VALID


def test_broken_symlink_failed_write_removes_only_created_target(tmp_path: Path) -> None:
    real = tmp_path / "real.crossby.yml"
    target = tmp_path / ".crossby.yml"
    target.symlink_to(real)

    with pytest.raises(ConfigWriteError) as exc:
        write_config_checked(target, BROKEN)

    assert exc.value.restored is False
    assert target.is_symlink()
    assert not target.exists()
    assert not real.exists()


def test_write_through_link_outside_project_root_succeeds(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real = outside / "real.crossby.yml"
    real.write_text("version: 1\n", encoding="utf-8")
    target = project / ".crossby.yml"
    target.symlink_to(real)

    write_config_checked(target, VALID)

    assert target.is_symlink()
    assert target.resolve() == real.resolve()
    assert real.read_text(encoding="utf-8") == VALID
