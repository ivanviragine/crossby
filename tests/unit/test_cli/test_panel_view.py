"""What ``crossby panel`` shows: the opening, the live line, messages and the outcome."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from crossby.cli import panel as panel_cli
from crossby.cli.panel_view import PanelView, _Activity, _Seat, activity_text, version_number
from crossby.services.converse import StopReason
from crossby.services.journal import ConversationEvent, EventType
from crossby.services.panel import (
    HumanBackend,
    InputRequest,
    Member,
    PanelResult,
    describe_panel_stop,
)
from tests.unit.test_cli.test_panel_cmd import RECIPE, _invoke, _project, _use
from tests.unit.test_services.test_panel import PanelRunner, _spec


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("2.1.317 (Claude Code)", "2.1.317"),
        ("codex-cli 0.166.0", "0.166.0"),
        ("2026.11.10-731fb78", "2026.11.10-731fb78"),
        ("GitHub Copilot CLI 1.0.109.", "1.0.109"),
        ("opencode 1.19.5\n", "1.19.5"),
        ("no digits here", "no digits here"),
        (None, None),
    ],
)
def test_a_tool_version_is_shown_without_its_words(text: str | None, number: str | None) -> None:
    assert version_number(text) == number


def test_the_live_line_shows_each_addressed_member() -> None:
    seats = (
        _Seat("ops", "writing", since=100.0),
        _Seat("design", "answered", seconds=14.2),
        _Seat("dana", "waiting", since=130.0),
        _Seat("support", "failed"),
        _Seat("buyer", "queued"),
    )
    text = activity_text(_Activity("members", 100.0, seats=seats), now=142.0)
    assert text == (
        "@ops writing 42s · @design ✓ 14s · @dana waiting for a reply 12s · @support ✗ · "
        "@buyer queued"
    )
    assert (
        activity_text(_Activity("lead", 0.0, label="[3] lead (claude)"), now=65.0)
        == "[3] lead (claude) is writing… 1m 05s"
    )
    assert (
        activity_text(_Activity("checking", 0.0, label="claude, codex"), now=2.0)
        == "checking claude, codex… 2s"
    )


def _result(tmp_path: Path, reason: StopReason, dispatches: int, **extra: Any) -> PanelResult:
    now = datetime.now(tz=UTC)
    return PanelResult(
        spec=_spec(tmp_path, max_rounds=3),
        started_at=now,
        finished_at=now,
        stop_reason=reason,
        dispatches=dispatches,
        **extra,
    )


@pytest.mark.parametrize(
    ("reason", "dispatches", "extra", "outcome"),
    [
        (
            StopReason.ANSWERED,
            1,
            {},
            "answered: the lead gave its final answer after 1 message to the members",
        ),
        (
            StopReason.ANSWERED,
            0,
            {},
            "answered: the lead gave its final answer without asking the members",
        ),
        (
            StopReason.ANSWERED,
            3,
            {"answered_at_limit": True},
            "answered: the lead gave its final answer after 3 messages to the members "
            "at the message limit",
        ),
        (
            StopReason.ROUND_LIMIT,
            3,
            {},
            "message limit reached (3 messages) and the lead gave no final answer",
        ),
        (
            StopReason.TIMEOUT,
            1,
            {"error": "panel time limit reached during message 1"},
            "stopped on the time limit after 1 message to the members: "
            "panel time limit reached during message 1",
        ),
        (StopReason.INTERRUPTED, 2, {}, "interrupted by the user after 2 messages to the members"),
        (
            StopReason.FAILED,
            0,
            {"error": "lead turn 1: not signed in"},
            "failed before the lead asked the members: lead turn 1: not signed in",
        ),
    ],
)
def test_the_outcome_reads_as_one_sentence(
    tmp_path: Path, reason: StopReason, dispatches: int, extra: dict[str, Any], outcome: str
) -> None:
    assert describe_panel_stop(_result(tmp_path, reason, dispatches, **extra)) == outcome


def _event(seq: int, type_: EventType, **data: Any) -> ConversationEvent:
    return ConversationEvent(run_id="r", seq=seq, ts=datetime.now(tz=UTC), type=type_, data=data)


def _human_spec(tmp_path: Path) -> Any:
    spec = _spec(tmp_path)
    return spec.model_copy(
        update={
            "members": (*spec.members, Member(id="dana", name="Dana", backend=HumanBackend())),
        }
    )


def test_nothing_is_printed_while_a_person_types(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    view = PanelView(_spec(tmp_path), live=False, typed_here=True)
    with view.hold():
        view(_event(1, EventType.TURN_FINISHED, participant="cto", error="it broke"))
        assert "it broke" not in capsys.readouterr().err
    assert "@cto: it broke" in " ".join(capsys.readouterr().err.split())


def test_a_persons_own_reply_is_not_printed_twice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _human_spec(tmp_path)
    committed = {"turn": 3, "recipients": ["lead"], "duration_seconds": 12.0, "denials": []}
    PanelView(spec, live=False, typed_here=True)(
        _event(1, EventType.MESSAGE_COMMITTED, participant="dana", text="Teal.", **committed)
    )
    typed = capsys.readouterr().err
    assert "[3] @dana (Dana) → lead" in typed
    assert "typed above · 12s" in typed
    assert "Teal." not in typed
    # Replies piped in were never on screen, so they are shown.
    PanelView(spec, live=False, typed_here=False)(
        _event(1, EventType.MESSAGE_COMMITTED, participant="dana", text="Teal.", **committed)
    )
    assert "Teal." in capsys.readouterr().err


def test_a_seat_is_prompted_once_its_message_is_on_screen(tmp_path: Path) -> None:
    view = PanelView(_human_spec(tmp_path), live=False, typed_here=True)
    assert view.wait_announced("r-1", timeout=0.05) is False
    later = threading.Timer(
        0.1,
        view,
        args=(
            _event(
                1, EventType.INPUT_REQUESTED, request_id="r-1", participant="dana", kind="member"
            ),
        ),
    )
    later.start()
    assert view.wait_announced("r-1", timeout=5) is True
    later.join()


def test_the_prompt_points_at_a_message_already_shown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    from crossby.services.panel import HumanReply

    request = InputRequest("r1", "dana", "member", 2, "What is your favorite color?")
    monkeypatch.setattr("sys.stdin", io.StringIO("Teal.\n.\n"))
    assert panel_cli._read_terminal_reply(request, message_shown=True) == HumanReply("Teal.")
    shown = " ".join(capsys.readouterr().err.split())
    assert "@dana, your turn: reply to the lead's message above" in shown
    assert "favorite color" not in shown
    monkeypatch.setattr("sys.stdin", io.StringIO("Teal.\n.\n"))
    panel_cli._read_terminal_reply(InputRequest("r2", "dana", "member", 2, "Color?"))
    assert "Color?" in capsys.readouterr().err


def test_a_human_lead_is_offered_the_member_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    from crossby.services.panel import HumanReply

    monkeypatch.setattr("sys.stdin", io.StringIO("Price?\n.\n@cto\n"))
    request = InputRequest("r1", "lead", "lead", 1, None)
    reply = panel_cli._read_terminal_reply(request, members=("cto", "founder"))
    assert reply == HumanReply("Price?", recipients=("@cto",))
    shown = " ".join(capsys.readouterr().err.split())
    assert "lead, your turn: write your first message to the panel" in shown
    assert "Send it to whom? @cto @founder, @all, or done" in shown


def test_a_log_reads_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, RECIPE)
    _use(
        monkeypatch,
        PanelRunner(
            {
                root.name: ["Would it sell?\nNEXT: ASK @all", "Both say yes.\nNEXT: DONE"],
                "cto": ["Yes, at $20."],
                "founder": [None],
            }
        ),
    )
    result = _invoke(root, "pricing", "-g", "g", "--output-dir", str(root / "out"))
    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    expected = [
        "▸ claude leads a private panel of 2 members:",
        "@cto Rui, codex",
        "@founder cursor",
        "members answer in parallel; agents are read-only",
        "[1] lead (claude) is writing…",
        "[1] lead (claude) → @cto @founder",
        "Would it sell?",
        "asking @cto, @founder…",
        "! @founder: the session ended with status failed",
        "[2] @cto (Rui, codex) → lead",
        "Yes, at $20.",
        "✓ answered: the lead gave its final answer after 1 message to the members",
        "no answer from @founder to message 1: the session ended with status failed",
    ]
    flat = [" ".join(line.split()) for line in lines]
    positions = [flat.index(line) for line in expected]
    assert positions == sorted(positions), flat
    # stdout carries only the final answer, once.
    assert result.stdout == "Both say yes.\n"
