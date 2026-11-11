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

    def reading(_request: Any) -> Any:
        with lock:
            active["now"] += 1
            active["most"] = max(active["most"], active["now"])
        threads.Event().wait(0.1)
        with lock:
            active["now"] -= 1
        return None

    monkeypatch.setattr(panel_cli, "_read_terminal_reply", reading)
    workers = [threads.Thread(target=panel_cli._terminal_reply, args=(object(),)) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert active["most"] == 1
