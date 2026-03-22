"""Fixtures for deterministic subprocess E2E tests."""

from __future__ import annotations

import pytest
from tests.e2e._support import E2EContext, make_e2e_context


@pytest.fixture
def e2e_context(tmp_path) -> E2EContext:
    return make_e2e_context(tmp_path)
