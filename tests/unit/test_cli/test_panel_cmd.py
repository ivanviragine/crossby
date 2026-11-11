"""CLI tests for ``crossby panel``: recipes from .crossby.yml, run with a fake runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.services import panel as panel_service
from crossby.services.converse import DEPTH_ENV
from tests.unit.test_services.test_panel import PanelRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def _top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEPTH_ENV, raising=False)


def _project(tmp_path: Path, recipe: str) -> Path:
    for name in ("cto", "founder"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "personas").mkdir(exist_ok=True)
    (tmp_path / "personas" / "cto.md").write_text("You are a CTO.\n")
    (tmp_path / ".crossby.yml").write_text(recipe)
    return tmp_path


RECIPE = """\
version: 1
panels:
  pricing:
    lead: {tool: claude}
    members:
      - {id: cto, name: Rui, tool: codex, folder: cto, instructions: personas/cto.md}
      - {id: founder, tool: cursor, folder: founder}
    max_rounds: 2
"""


def _invoke(root: Path, *args: str) -> Any:
    return runner.invoke(app, ["panel", *args, "--path", str(root)])


def _use(monkeypatch: pytest.MonkeyPatch, fake: PanelRunner) -> PanelRunner:
    monkeypatch.setattr(panel_service, "HeadlessTurnRunner", lambda: fake)
    return fake


def test_panel_runs_a_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, RECIPE)
    lead_folder = root.name
    fake = _use(
        monkeypatch,
        PanelRunner(
            {
                lead_folder: ["Q?\nNEXT: ASK @all", "Both agree.\nNEXT: DONE"],
                "cto": ["Yes."],
                "founder": ["Also yes."],
            }
        ),
    )
    out = root / "out"
    result = _invoke(root, "pricing", "-g", "Would it sell?", "--output-dir", str(out), "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "answered"
    assert payload["final_answer"] == "Both agree."
    assert [m["id"] for m in payload["members"]] == ["cto", "founder"]
    # The CTO's instructions travel in the envelope on its first message.
    assert fake.requests["cto"][0].prompt.startswith("<instructions>\nYou are a CTO.\n")
    assert (out / "transcript.md").read_text().startswith("# Panel")
    assert (out / "events.jsonl").is_file()


def test_a_persona_folder_member_is_provisioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = RECIPE.replace(
        "{id: founder, tool: cursor, folder: founder}",
        "{id: founder, tool: cursor, persona_folder: .crossby/personas/founder, "
        "instructions: personas/cto.md, files: {notes/a.md: personas/cto.md}}",
    )
    root = _project(tmp_path, recipe)
    _use(
        monkeypatch,
        PanelRunner(
            {
                root.name: ["Q?\nNEXT: ASK @founder", "Done.\nNEXT: DONE"],
                "cto": [],
                "founder": ["ok"],
            }
        ),
    )
    result = _invoke(root, "pricing", "-g", "g", "--quiet")
    assert result.exit_code == 0, result.output
    persona = root / ".crossby" / "personas" / "founder"
    assert (persona / "AGENTS.md").read_text() == "You are a CTO.\n"
    assert (persona / "notes" / "a.md").is_file()


@pytest.mark.parametrize(
    ("recipe", "args", "message"),
    [
        (RECIPE, ["missing", "-g", "g"], "No panel named 'missing'"),
        (RECIPE.replace("max_rounds: 2", "max_rounds: 0"), ["pricing", "-g", "g"], "max_rounds"),
        (RECIPE.replace("tool: cursor", "tool: vscode"), ["pricing", "-g", "g"], "vscode"),
        (
            RECIPE.replace("folder: founder}", "folder: founder, skills: [x]}"),
            ["pricing", "-g", "g"],
            "skills and files need a persona_folder",
        ),
        (RECIPE, ["pricing"], "A goal is required"),
    ],
)
def test_a_bad_recipe_or_call_starts_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recipe: str,
    args: list[str],
    message: str,
) -> None:
    root = _project(tmp_path, recipe)
    fake = _use(monkeypatch, PanelRunner({root.name: [], "cto": [], "founder": []}))
    result = _invoke(root, *args)
    assert result.exit_code == 1
    assert message in " ".join(result.stderr.split())
    assert all(not calls for calls in fake.requests.values())


def test_a_panel_never_reuses_a_folder_with_a_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, RECIPE)
    _use(
        monkeypatch,
        PanelRunner(
            {root.name: ["Q?\nNEXT: ASK @cto", "ok\nNEXT: DONE"], "cto": ["a"], "founder": []}
        ),
    )
    out = root / "out"
    assert _invoke(root, "pricing", "-g", "g", "--output-dir", str(out), "--quiet").exit_code == 0
    before = (out / "events.jsonl").read_text()
    fake = _use(monkeypatch, PanelRunner({root.name: [], "cto": [], "founder": []}))
    again = _invoke(root, "pricing", "-g", "g", "--output-dir", str(out))
    assert again.exit_code == 1
    assert "already holds a panel" in " ".join(again.stderr.split())
    assert (out / "events.jsonl").read_text() == before
    assert all(not calls for calls in fake.requests.values())


def test_a_journal_that_could_not_be_saved_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crossby.services import journal as journal_module

    root = _project(tmp_path, RECIPE)
    _use(
        monkeypatch,
        PanelRunner(
            {root.name: ["Q?\nNEXT: ASK @cto", "ok\nNEXT: DONE"], "cto": ["a"], "founder": []}
        ),
    )

    def unwritable(fd: int, data: bytes) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(journal_module, "_write_all", unwritable)
    result = _invoke(root, "pricing", "-g", "g", "--output-dir", str(root / "out"), "--json")
    assert result.exit_code == 0, result.output
    assert "The event journal is incomplete" in " ".join(result.stderr.split())
    assert "No space left" in json.loads(result.stdout)["journal_error"]


def test_a_person_takes_a_seat_from_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = RECIPE.replace(
        "{id: founder, tool: cursor, folder: founder}", "{id: dana, name: Dana, human: true}"
    )
    root = _project(tmp_path, recipe)
    fake = _use(
        monkeypatch,
        PanelRunner({root.name: ["Q?\nNEXT: ASK @all", "ok\nNEXT: DONE"], "cto": ["Yes."]}),
    )
    result = runner.invoke(
        app,
        ["panel", "pricing", "-g", "g", "--path", str(root), "--json"],
        input="I would pay 20.\nAnd no more.\n.\n",
    )
    assert result.exit_code == 0, result.output
    assert fake.requests[root.name][1].prompt == (
        "@cto:\nYes.\n\n@dana:\nI would pay 20.\nAnd no more."
    )
    assert "@dana, your turn" in " ".join(result.stderr.split())


@pytest.mark.parametrize(
    ("member", "message"),
    [
        ("{id: dana, human: true, tool: codex}", "set either tool or human"),
        ("{id: dana}", "set either tool or human"),
        ("{id: dana, human: true, instructions: personas/cto.md}", "human member takes no"),
    ],
)
def test_a_human_seat_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str, message: str
) -> None:
    founder = "{id: founder, tool: cursor, folder: founder}"
    root = _project(tmp_path, RECIPE.replace(founder, member))
    _use(monkeypatch, PanelRunner({root.name: [], "cto": []}))
    result = _invoke(root, "pricing", "-g", "g")
    assert result.exit_code == 1
    assert message in " ".join(result.stderr.split())


def test_terminal_seats_take_turns_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading as threads

    from crossby.cli import panel as panel_cli

    active = {"now": 0, "most": 0}
    lock = threads.Lock()

    def reading(_request: Any, **_shown: Any) -> Any:
        with lock:
            active["now"] += 1
            active["most"] = max(active["most"], active["now"])
        threads.Event().wait(0.1)
        with lock:
            active["now"] -= 1
        return None

    from crossby.services.panel import InputRequest

    monkeypatch.setattr(panel_cli, "_read_terminal_reply", reading)
    requests = [InputRequest(f"r{index}", f"m{index}", "member", index, "Q?") for index in range(3)]
    workers = [threads.Thread(target=panel_cli._terminal_reply, args=(r,)) for r in requests]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert active["most"] == 1


def test_an_expired_seat_is_never_prompted_and_frees_the_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading as threads

    from crossby.cli import panel as panel_cli
    from crossby.services.panel import InputRequest

    holder = InputRequest("r1", "dana", "member", 2, "Q?")
    waiting = InputRequest("r2", "lee", "member", 3, "Q?")
    reached: list[str] = []

    def reading(request: Any, **_shown: Any) -> Any:
        reached.append(request.request_id)
        request.expired.wait(5)  # blocked until its turn expires
        return None

    monkeypatch.setattr(panel_cli, "_read_terminal_reply", reading)
    first = threads.Thread(target=panel_cli._terminal_reply, args=(holder,))
    first.start()
    threads.Event().wait(0.1)
    second = threads.Thread(target=panel_cli._terminal_reply, args=(waiting,))
    second.start()
    waiting.expired.set()  # expires while waiting for the terminal
    second.join(2)
    assert not second.is_alive()
    holder.expired.set()  # expires mid-reply: the reader stops
    first.join(2)
    assert not first.is_alive()
    assert reached == ["r1"]
    assert panel_cli._TERMINAL.acquire(timeout=1)
    panel_cli._TERMINAL.release()


class _Terminal:
    """A terminal-like stdin: lines arrive when a test types them; Ctrl-D is ``""``."""

    def __init__(self) -> None:
        import queue

        self.typed: queue.Queue[str] = queue.Queue()

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return self.typed.get()


def _seat(request_id: str, participant: str = "dana") -> Any:
    from crossby.services.panel import InputRequest

    return InputRequest(request_id, participant, "member", 2, "Q?")


def test_a_pipe_left_open_still_delivers_every_buffered_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import threading as threads

    from crossby.cli import panel as panel_cli
    from crossby.services.panel import HumanReply

    read_end, write_end = os.pipe()
    stream = os.fdopen(read_end, "r")
    monkeypatch.setattr("sys.stdin", stream)
    os.write(write_end, b"answer\n.\nnext seat\n.\n")  # both replies at once; never closed
    seats = [_seat("r1"), _seat("r2", "lee")]
    # A reader that missed buffered lines would wait out these deadlines instead.
    deadlines = [threads.Timer(5, seat.expired.set) for seat in seats]
    for deadline in deadlines:
        deadline.start()
    try:
        first, second = (panel_cli._read_terminal_reply(seat) for seat in seats)
    finally:
        for deadline in deadlines:
            deadline.cancel()
        os.close(write_end)
    assert first == HumanReply("answer")
    assert second == HumanReply("next seat")


def test_an_expired_seat_stops_waiting_and_piped_lines_go_to_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from crossby.cli import panel as panel_cli
    from crossby.services.panel import HumanReply

    read_end, write_end = os.pipe()
    monkeypatch.setattr("sys.stdin", os.fdopen(read_end, "r"))
    expired = _seat("r1")
    expired.expired.set()
    try:
        assert panel_cli._read_terminal_reply(expired) is None
        os.write(write_end, b"for lee\n.\n")
        assert panel_cli._read_terminal_reply(_seat("r2", "lee")) == HumanReply("for lee")
    finally:
        os.close(write_end)


def test_at_a_terminal_lines_typed_before_the_prompt_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading as threads

    from crossby.cli import panel as panel_cli
    from crossby.services.panel import HumanReply

    terminal = _Terminal()
    monkeypatch.setattr("sys.stdin", terminal)
    reader = panel_cli._stdin_lines()
    terminal.typed.put("typed while the agents worked\n")
    for _ in range(50):  # until the reader has queued it
        if not reader._lines.empty():
            break
        threads.Event().wait(0.02)
    replies: list[Any] = []
    seat = threads.Thread(
        target=lambda: replies.append(panel_cli._read_terminal_reply(_seat("r1")))
    )
    seat.start()
    threads.Event().wait(0.2)
    for line in ("meant for the prompt\n", ".\n"):
        terminal.typed.put(line)
    seat.join(5)
    assert replies == [HumanReply("meant for the prompt")]


def test_at_a_terminal_ctrl_d_ends_one_reply_and_the_next_seat_still_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crossby.cli import panel as panel_cli
    from crossby.services.panel import HumanReply

    terminal = _Terminal()
    monkeypatch.setattr("sys.stdin", terminal)
    replies: list[Any] = []

    def answer(request: Any, *typed: str) -> None:
        import threading as threads

        seat = threads.Thread(
            target=lambda: replies.append(panel_cli._read_terminal_reply(request))
        )
        seat.start()
        threads.Event().wait(0.2)
        for line in typed:
            terminal.typed.put(line)
        seat.join(5)

    answer(_seat("r1"), "first\n", "")
    answer(_seat("r2", "lee"), "second\n", ".\n")
    assert replies == [HumanReply("first"), HumanReply("second")]
