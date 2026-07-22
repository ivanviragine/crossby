"""Regression tests for ``_confirm_sync_defaults`` change closures.

``src/crossby/cli/sync.py`` uses ``from __future__ import annotations`` and
imports ``AIToolID`` / ``SyncConcern`` only under ``if TYPE_CHECKING:``. The
``_change_from`` / ``_change_to`` / ``_change_concern`` closures call those
names as *runtime* constructors, so before the fix (issue #72) picking any
"Change ..." entry in the confirm-defaults menu raised
``NameError: name 'AIToolID' is not defined``. These tests drive each closure
and assert the returned ``(source, target, concern)`` tuple.

Mechanics mirror ``tests/unit/test_services/test_confirm.py``: patch
``crossby.ui.prompts.is_tty`` -> ``True`` and ``crossby.ui.prompts.select``
with a scripted queue. One patch covers both the confirm-defaults menu select
and the in-closure ``prompts.select`` — both resolve the name at call time via
a lazy ``from crossby.ui import prompts``.
"""

from __future__ import annotations

import pytest

from crossby.cli.sync import _confirm_sync_defaults
from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern

_INSTALLED = [AIToolID.CLAUDE, AIToolID.CURSOR, AIToolID.CODEX]


class _PromptRecorder:
    """Stand-in for ``prompts.select`` that plays back a scripted queue.

    Each scripted entry is the 0-based index returned for the next call.
    ``self.calls`` records ``(title, items)`` per invocation for assertions.
    """

    def __init__(self, responses: list[int]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, title: str, items: list[str], default: int = 0) -> int:
        self.calls.append((title, list(items)))
        if not self.responses:
            raise AssertionError(
                f"prompts.select called beyond scripted responses: title={title!r}"
            )
        return self.responses.pop(0)


@pytest.fixture()
def tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: True)


def _install_select(monkeypatch: pytest.MonkeyPatch, recorder: _PromptRecorder) -> None:
    monkeypatch.setattr("crossby.ui.prompts.select", recorder)


class TestConfirmSyncDefaultsChangePaths:
    def test_change_source_tool_path(self, tty: None, monkeypatch: pytest.MonkeyPatch) -> None:
        # menu: Change source tool (1) -> select "cursor" (1) -> Proceed (0)
        recorder = _PromptRecorder([1, 1, 0])
        _install_select(monkeypatch, recorder)

        source, target, concern = _confirm_sync_defaults(
            source_tool=AIToolID.CLAUDE,
            target_tool=None,
            sync_concern=None,
            installed_tools=_INSTALLED,
            from_explicit=False,
            to_explicit=False,
            concern_explicit=False,
        )

        assert source == AIToolID.CURSOR
        assert isinstance(source, AIToolID)
        assert target is None
        assert concern is None
        assert ("Source tool", ["claude", "cursor", "codex"]) in recorder.calls

    def test_change_target_tool_path(self, tty: None, monkeypatch: pytest.MonkeyPatch) -> None:
        # menu: Change target tool (2) -> select "cursor" (1) -> Proceed (0)
        recorder = _PromptRecorder([2, 1, 0])
        _install_select(monkeypatch, recorder)

        source, target, concern = _confirm_sync_defaults(
            source_tool=AIToolID.CLAUDE,
            target_tool=None,
            sync_concern=None,
            installed_tools=_INSTALLED,
            from_explicit=False,
            to_explicit=False,
            concern_explicit=False,
        )

        # Target choices exclude the source tool (claude).
        assert source == AIToolID.CLAUDE
        assert target == AIToolID.CURSOR
        assert isinstance(target, AIToolID)
        assert concern is None
        assert ("Target tool", ["(all installed)", "cursor", "codex"]) in recorder.calls

    def test_change_concern_path(self, tty: None, monkeypatch: pytest.MonkeyPatch) -> None:
        # Derive the menu index so the test survives SyncConcern reordering.
        chosen = SyncConcern.PERMISSIONS
        concern_idx = list(SyncConcern).index(chosen) + 1  # +1 for "(all concerns)"
        # menu: Change concern (3) -> select the concern -> Proceed (0)
        recorder = _PromptRecorder([3, concern_idx, 0])
        _install_select(monkeypatch, recorder)

        source, target, concern = _confirm_sync_defaults(
            source_tool=AIToolID.CLAUDE,
            target_tool=None,
            sync_concern=None,
            installed_tools=_INSTALLED,
            from_explicit=False,
            to_explicit=False,
            concern_explicit=False,
        )

        assert source == AIToolID.CLAUDE
        assert target is None
        assert concern == chosen
        assert isinstance(concern, SyncConcern)

    def test_change_all_three_paths(self, tty: None, monkeypatch: pytest.MonkeyPatch) -> None:
        # Derive the concern menu index so the test survives SyncConcern reordering.
        chosen_concern = SyncConcern.RULES
        concern_idx = list(SyncConcern).index(chosen_concern) + 1  # +1 for "(all concerns)"
        # menu Change source (1) -> "cursor" (1)
        # menu Change target (2) -> "claude" (1, choices exclude cursor)
        # menu Change concern (3) -> chosen concern
        # menu Proceed (0)
        recorder = _PromptRecorder([1, 1, 2, 1, 3, concern_idx, 0])
        _install_select(monkeypatch, recorder)

        source, target, concern = _confirm_sync_defaults(
            source_tool=AIToolID.CLAUDE,
            target_tool=None,
            sync_concern=None,
            installed_tools=_INSTALLED,
            from_explicit=False,
            to_explicit=False,
            concern_explicit=False,
        )

        assert (source, target, concern) == (
            AIToolID.CURSOR,
            AIToolID.CLAUDE,
            chosen_concern,
        )
        # After switching source to cursor, the target picker excludes cursor.
        assert ("Target tool", ["(all installed)", "claude", "codex"]) in recorder.calls

    def test_change_source_to_match_target_resets_target(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Changing the source to the already-set target clears the target.

        The target picker never offers the current source, so leaving a stale
        ``target == source`` would schedule a redundant tool -> itself sync.
        """
        # menu: Change source tool (1) -> select "cursor" (1) -> Proceed (0)
        recorder = _PromptRecorder([1, 1, 0])
        _install_select(monkeypatch, recorder)

        source, target, concern = _confirm_sync_defaults(
            source_tool=AIToolID.CLAUDE,
            target_tool=AIToolID.CURSOR,
            sync_concern=None,
            installed_tools=_INSTALLED,
            from_explicit=False,
            to_explicit=False,
            concern_explicit=False,
        )

        assert source == AIToolID.CURSOR
        assert target is None
        assert concern is None

    def test_change_target_to_all_installed_clears_target(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # menu: Change target tool (2) -> select "(all installed)" (0) -> Proceed (0)
        recorder = _PromptRecorder([2, 0, 0])
        _install_select(monkeypatch, recorder)

        source, target, concern = _confirm_sync_defaults(
            source_tool=AIToolID.CLAUDE,
            target_tool=AIToolID.CURSOR,
            sync_concern=None,
            installed_tools=_INSTALLED,
            from_explicit=False,
            to_explicit=False,
            concern_explicit=False,
        )

        assert source == AIToolID.CLAUDE
        assert target is None
        assert concern is None

    def test_change_concern_to_all_clears_concern(
        self, tty: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # menu: Change concern (3) -> select "(all concerns)" (0) -> Proceed (0)
        recorder = _PromptRecorder([3, 0, 0])
        _install_select(monkeypatch, recorder)

        source, target, concern = _confirm_sync_defaults(
            source_tool=AIToolID.CLAUDE,
            target_tool=None,
            sync_concern=SyncConcern.RULES,
            installed_tools=_INSTALLED,
            from_explicit=False,
            to_explicit=False,
            concern_explicit=False,
        )

        assert source == AIToolID.CLAUDE
        assert target is None
        assert concern is None
