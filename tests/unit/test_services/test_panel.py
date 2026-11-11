"""Panels: a lead consults several read-only members, by id."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from crossby.models.ai import (
    AIToolID,
    HeadlessEvent,
    HeadlessEventKind,
    HeadlessSessionRequest,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
    RestrictionChannel,
)
from crossby.services.ask import CliBackend, ProjectWorkspace
from crossby.services.converse import DEPTH_ENV, StopReason
from crossby.services.journal import EventType, Journal
from crossby.services.panel import (
    FINAL_NUDGE,
    Dispatch,
    Lead,
    Member,
    PanelSpec,
    ProtocolError,
    parse_dispatch,
    render_notice,
    render_panel_transcript,
    run_panel,
)

IDS = ["cto", "founder", "buyer"]


@pytest.fixture(autouse=True)
def _top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "dispatch"),
    [
        ("Q?\nNEXT: ASK @founder @cto", Dispatch(False, ("cto", "founder"))),
        ("Q?\nNEXT: ASK @all", Dispatch(False, tuple(IDS))),
        ("Q?\n**NEXT: ASK @buyer**", Dispatch(False, ("buyer",))),
        ("Q?\nnext: ask @CTO, @buyer", Dispatch(False, ("cto", "buyer"))),
        ("Answer.\nNEXT: DONE", Dispatch(True)),
    ],
)
def test_the_lead_addresses_members_on_its_last_line(text: str, dispatch: Dispatch) -> None:
    body, parsed = parse_dispatch(text, IDS)
    assert parsed == dispatch
    assert body == text.splitlines()[0]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Q?\nNEXT: ASK @nobody", "no member is named @nobody"),
        ("Q?\nNEXT: ASK @cto @cto", "named twice"),
        ("Q?\nNEXT: ASK @all @cto", "cannot be combined"),
        ("Done.\nNEXT: DONE @cto", "takes no members"),
        ("Q?\nNEXT: ASK", "must name members"),
        ("Q?\nNEXT: ASK cto", "named as `@id`"),
        ("Just a question?", "last line is not"),
        ("", "empty"),
    ],
)
def test_an_invalid_last_line_is_a_protocol_error(text: str, message: str) -> None:
    with pytest.raises(ProtocolError, match=message):
        parse_dispatch(text, IDS)


def test_a_one_member_panel_reads_like_a_consultation() -> None:
    assert parse_dispatch("Q?\nNEXT: ASK", ["solo"]) == ("Q?", Dispatch(False, ("solo",)))
    assert parse_dispatch("No marker", ["solo"]) == ("No marker", Dispatch(False, ("solo",)))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _ok(tool: AIToolID, text: str, session: str) -> HeadlessSessionResult:
    status = HeadlessTerminalStatus.SUCCEEDED
    return HeadlessSessionResult(
        tool=tool,
        version="9.9.9",
        status=status,
        events=(
            HeadlessEvent(sequence=1, kind=HeadlessEventKind.TERMINAL, terminal_status=status),
        ),
        final_text=text,
        session_id=session,
        duration_seconds=0.1,
    )


def _failed(tool: AIToolID) -> HeadlessSessionResult:
    status = HeadlessTerminalStatus.FAILED
    return HeadlessSessionResult(
        tool=tool,
        version="9.9.9",
        status=status,
        events=(
            HeadlessEvent(sequence=1, kind=HeadlessEventKind.TERMINAL, terminal_status=status),
        ),
        duration_seconds=0.1,
    )


class PanelRunner:
    """Replies per participant folder; records every request and overlap."""

    def __init__(self, script: dict[str, list[str | None]], *, delay: float = 0.0) -> None:
        self.script = {key: list(value) for key, value in script.items()}
        self.requests: dict[str, list[HeadlessSessionRequest]] = {key: [] for key in script}
        self.delay = delay
        self.active = 0
        self.most_active = 0
        self.lock = threading.Lock()
        self.preflights: list[HeadlessSessionRequest] = []

    def preflight(self, tool: AIToolID, request: HeadlessSessionRequest) -> None:
        self.preflights.append(request)

    def run(
        self,
        tool: AIToolID,
        request: HeadlessSessionRequest,
        cancel_event: threading.Event,
        on_event: Any = None,
    ) -> HeadlessSessionResult:
        who = request.working_dir.name
        with self.lock:
            self.requests[who].append(request)
            self.active += 1
            self.most_active = max(self.most_active, self.active)
            reply = self.script[who].pop(0)
        try:
            time.sleep(self.delay)
        finally:
            with self.lock:
                self.active -= 1
        if reply is None:
            return _failed(tool)
        return _ok(tool, reply, request.resume_id or f"{who}-session")


def _spec(tmp_path: Path, **updates: Any) -> PanelSpec:
    folders = {}
    for name in ["lead", *IDS]:
        folders[name] = tmp_path / name
        folders[name].mkdir(exist_ok=True)
    values: dict[str, Any] = {
        "goal": "Would this pricing page convert?",
        "lead": Lead(backend=CliBackend(tool="claude"), folder=folders["lead"]),
        "members": tuple(
            Member(
                id=name,
                name=name.upper(),
                backend=CliBackend(tool="codex"),
                workspace=ProjectWorkspace(path=folders[name]),
            )
            for name in IDS
        ),
    }
    values.update(updates)
    return PanelSpec(**values)


def test_a_private_panel_relays_addressed_messages_in_roster_order(tmp_path: Path) -> None:
    runner = PanelRunner(
        {
            "lead": ["Price?\nNEXT: ASK @founder @cto", "Verdict: no.\nNEXT: DONE"],
            "cto": ["Too high."],
            "founder": ["Fine."],
            "buyer": [],
        }
    )
    result = run_panel(_spec(tmp_path), runner=runner)

    assert result.answered
    assert result.final_answer == "Verdict: no."
    assert [turn.sender for turn in result.turns] == ["lead", "cto", "founder", "lead"]
    # Members get the lead's message exactly, read-only; the buyer is not asked.
    assert [r.prompt for r in runner.requests["cto"]] == ["Price?"]
    assert all(r.read_only for r in runner.requests["cto"] + runner.requests["founder"])
    assert runner.requests["buyer"] == []
    lead_second = runner.requests["lead"][1]
    assert lead_second.prompt == "@cto:\nToo high.\n\n@founder:\nFine."
    assert lead_second.resume_id == "lead-session"
    assert not lead_second.read_only
    # The lead's opening names the roster and the limit.
    opening = runner.requests["lead"][0].prompt
    assert opening.startswith("Goal: Would this pricing page convert?")
    assert "@cto (CTO), @founder (FOUNDER), @buyer (BUYER)" in opening
    assert result.sessions["cto"] == "cto-session"


def test_members_resume_and_a_shared_panel_queues_missed_replies(tmp_path: Path) -> None:
    runner = PanelRunner(
        {
            "lead": [
                "First?\nNEXT: ASK @cto @founder",
                "And you, CTO?\nNEXT: ASK @cto",
                "Done.\nNEXT: DONE",
            ],
            "cto": ["A1", "A2"],
            "founder": ["B1"],
            "buyer": [],
        }
    )
    result = run_panel(_spec(tmp_path, visibility="shared"), runner=runner)

    assert result.answered
    second = runner.requests["cto"][1]
    assert second.resume_id == "cto-session"
    # The founder's reply, which the CTO had not seen, comes first, labeled.
    assert second.prompt == "@founder:\nB1\n\n@lead:\nAnd you, CTO?"


def test_a_private_member_sees_only_what_is_addressed_to_it(tmp_path: Path) -> None:
    runner = PanelRunner(
        {
            "lead": ["First?\nNEXT: ASK @all", "Again?\nNEXT: ASK @cto", "Done.\nNEXT: DONE"],
            "cto": ["A1", "A2"],
            "founder": ["B1"],
            "buyer": ["C1"],
        }
    )
    run_panel(_spec(tmp_path), runner=runner)
    assert runner.requests["cto"][1].prompt == "Again?"


def test_a_failed_member_is_a_note_and_all_failing_stops_the_panel(tmp_path: Path) -> None:
    runner = PanelRunner(
        {
            "lead": ["Q?\nNEXT: ASK @cto @founder", "Q2?\nNEXT: ASK @founder", "x\nNEXT: DONE"],
            "cto": ["Yes."],
            "founder": [None, None],
            "buyer": [],
        }
    )
    result = run_panel(_spec(tmp_path), runner=runner)

    note = runner.requests["lead"][1].prompt
    assert note.startswith(
        "@cto:\nYes.\n\n@founder (no answer: the session ended with status failed"
    )
    assert result.stop_reason is StopReason.FAILED
    assert "every member addressed in message 2 failed" in (result.error or "")
    assert [member for _dispatch, member, _reason in result.member_failures] == [
        "founder",
        "founder",
    ]


def test_a_protocol_error_sends_nothing(tmp_path: Path) -> None:
    runner = PanelRunner({"lead": ["Q?\nNEXT: ASK @ceo"], "cto": [], "founder": [], "buyer": []})
    result = run_panel(_spec(tmp_path), runner=runner)
    assert result.stop_reason is StopReason.FAILED
    assert "no member is named @ceo; nothing was sent" in (result.error or "")
    assert all(not runner.requests[name] for name in IDS)


def test_max_rounds_counts_dispatches_plus_one_final_turn(tmp_path: Path) -> None:
    runner = PanelRunner(
        {
            "lead": ["Q1\nNEXT: ASK @cto", "Q2\nNEXT: ASK @cto", "Q3\nNEXT: ASK @cto"],
            "cto": ["A1", "A2"],
            "founder": [],
            "buyer": [],
        }
    )
    result = run_panel(_spec(tmp_path, max_rounds=2), runner=runner)
    assert result.stop_reason is StopReason.ROUND_LIMIT
    # Two were sent; the third was the final turn, which ended the panel.
    assert result.dispatches == 2
    assert runner.requests["lead"][2].prompt.endswith(FINAL_NUDGE.format(max_rounds=2))


@pytest.mark.parametrize(("concurrency", "expected"), [(1, 1), (3, 3)])
def test_concurrency_caps_parallel_members(tmp_path: Path, concurrency: int, expected: int) -> None:
    runner = PanelRunner(
        {
            "lead": ["Q\nNEXT: ASK @all", "x\nNEXT: DONE"],
            "cto": ["a"],
            "founder": ["b"],
            "buyer": ["c"],
        },
        delay=0.2,
    )
    run_panel(_spec(tmp_path, concurrency=concurrency), runner=runner)
    assert runner.most_active == expected


def test_instructions_and_requirements_reach_members_only(tmp_path: Path) -> None:
    spec = _spec(tmp_path, require_blocked=frozenset({RestrictionChannel.FILE_EDITS}))
    members = (
        spec.members[0].model_copy(
            update={
                "workspace": ProjectWorkspace(path=tmp_path / "cto", instructions="You are a CTO.")
            }
        ),
        *spec.members[1:],
    )
    runner = PanelRunner(
        {
            "lead": ["Q\nNEXT: ASK @cto", "x\nNEXT: DONE"],
            "cto": ["a"],
            "founder": [],
            "buyer": [],
        }
    )
    run_panel(spec.model_copy(update={"members": members}), runner=runner)
    cto = runner.requests["cto"][0]
    assert cto.prompt == "<instructions>\nYou are a CTO.\n</instructions>\n\nQ"
    assert cto.require_blocked == frozenset({RestrictionChannel.FILE_EDITS})
    assert runner.requests["lead"][0].require_blocked == frozenset()


def test_the_panel_is_journaled_with_participant_ids(tmp_path: Path) -> None:
    runner = PanelRunner(
        {
            "lead": ["Q\nNEXT: ASK @cto @buyer", "x\nNEXT: DONE"],
            "cto": ["a"],
            "founder": [],
            "buyer": ["c"],
        }
    )
    journal = Journal("panel-1")
    result = run_panel(_spec(tmp_path), runner=runner, journal=journal)
    types = [event.type for event in journal.events]
    assert types[0] is EventType.CONVERSATION_STARTED
    assert journal.events[0].data["kind"] == "panel"
    assert types[-1] is EventType.CONVERSATION_FINISHED
    committed = [e.data for e in journal.events if e.type is EventType.MESSAGE_COMMITTED]
    assert [(item["participant"], item["recipients"]) for item in committed] == [
        ("lead", ["cto", "buyer"]),
        ("cto", ["lead"]),
        ("buyer", ["lead"]),
        ("lead", []),
    ]
    ready = [e.data["participant"] for e in journal.events if e.type is EventType.PARTICIPANT_READY]
    assert ready == ["lead", *IDS]
    transcript = render_panel_transcript(result)
    assert "## Turn 1 · lead (claude) → @cto @buyer" in transcript
    assert "## Turn 2 · @cto → lead" in transcript


def test_the_notice_wording_is_replaceable_but_not_the_grammar(tmp_path: Path) -> None:
    spec = _spec(tmp_path, protocol_notice="{goal} | {roster} | {count} | {max_rounds}")
    assert render_notice(spec) == (
        "Would this pricing page convert? | @cto (CTO), @founder (FOUNDER), @buyer (BUYER) | 3 | 5"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"members": ()},
        {"goal": " "},
    ],
)
def test_invalid_panels_are_refused(tmp_path: Path, updates: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _spec(tmp_path, **updates)


def test_member_ids_are_addressable(tmp_path: Path) -> None:
    for bad in ("all", "lead", "CTO", "a b", "@x", ""):
        with pytest.raises(ValueError):
            Member(
                id=bad, backend=CliBackend(tool="codex"), workspace=ProjectWorkspace(path=tmp_path)
            )
    duplicate = Member(
        id="x", backend=CliBackend(tool="codex"), workspace=ProjectWorkspace(path=tmp_path)
    )
    with pytest.raises(ValueError, match="unique"):
        _spec(tmp_path, members=(duplicate, duplicate))



def test_turn_numbers_match_their_events_when_a_member_fails(tmp_path: Path) -> None:
    runner = PanelRunner(
        {
            "lead": ["Q\nNEXT: ASK @cto @founder", "x\nNEXT: DONE"],
            "cto": [None],
            "founder": ["F"],
            "buyer": [],
        }
    )
    journal = Journal("p")
    result = run_panel(_spec(tmp_path), runner=runner, journal=journal)
    assert [(turn.number, turn.sender) for turn in result.turns] == [
        (1, "lead"),
        (3, "founder"),
        (4, "lead"),
    ]
    started = {
        (e.data["turn"], e.data["participant"])
        for e in journal.events
        if e.type is EventType.TURN_STARTED
    }
    committed = {
        (e.data["turn"], e.data["participant"])
        for e in journal.events
        if e.type is EventType.MESSAGE_COMMITTED
    }
    assert committed <= started
    assert (2, "cto") in started and (2, "cto") not in committed


def test_a_drifted_persona_folder_stops_the_panel_before_anyone_starts(tmp_path: Path) -> None:
    from crossby.services.workspaces import OwnedWorkspaceSpec, provision

    owned = provision(
        OwnedWorkspaceSpec(path=tmp_path / "persona", tool="codex", instructions="You are a CTO.")
    )
    (owned.path / "AGENTS.md").write_text("Someone else.")
    spec = _spec(tmp_path)
    members = (spec.members[0].model_copy(update={"workspace": owned}), *spec.members[1:])
    runner = PanelRunner({"lead": [], "cto": [], "founder": [], "buyer": [], "persona": []})
    result = run_panel(spec.model_copy(update={"members": members}), runner=runner)
    assert result.stop_reason is StopReason.FAILED
    assert "@cto's persona folder changed since it was provisioned: AGENTS.md" in (
        result.error or ""
    )
    assert runner.preflights == []
    assert all(not calls for calls in runner.requests.values())
