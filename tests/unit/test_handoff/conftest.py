"""Shared fixtures for handoff reader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "handoff"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture()
def project_path() -> Path:
    # A stable project root that matches the ``cwd`` fields in our fixtures.
    return Path("/Users/tester/proj")
