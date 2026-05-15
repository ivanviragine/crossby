"""Tests for crossby.logging.setup."""

from __future__ import annotations

import logging

from crossby.logging.setup import configure


def test_default_level_is_warning() -> None:
    configure(verbose=False)
    assert logging.getLogger().level == logging.WARNING


def test_verbose_sets_debug() -> None:
    configure(verbose=True)
    assert logging.getLogger().level == logging.DEBUG
