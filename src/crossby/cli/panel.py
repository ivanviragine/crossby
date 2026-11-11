"""crossby panel — run a panel recipe from .crossby.yml.

Like ``converse``, it never prompts, writes messages and progress to stderr,
and reserves stdout for the lead's final answer (or ``--json``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import queue
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import typer

from crossby.cli.converse import (
    _escape,
    _headless_tool,
    _print_json,
    _project_root,
    _read_text_input,
    _stderr,
    _write,
)
from crossby.ui.console import console

if TYPE_CHECKING:
    from crossby.cli.panel_view import PanelView
    from crossby.models.config import PanelConfig
    from crossby.services.panel import PanelResult, PanelSpec


def panel(
    name: str = typer.Argument(..., help="Panel recipe name under `panels:` in .crossby.yml."),
    goal: str | None = typer.Option(
        None, "--goal", "-g", help="What the lead should find out. Only the lead sees it."
    ),
    goal_file: Path | None = typer.Option(
        None, "--goal-file", help="Read the goal from this file instead."
    ),
    max_rounds: int | None = typer.Option(
        None, "--max-rounds", help="Most messages the lead may send (default: the recipe's)."
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for transcript.md and events.jsonl (default: .crossby/panels/<time>/).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print a JSON result object."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Do not print messages."),
    allow_project_hooks: bool = typer.Option(
        False,
        "--allow-project-hooks",
        help="Let read-only members run their folder's own hooks, which can change files.",
    ),
    path: Path = typer.Option(Path("."), "--path", help="Project root (where .crossby.yml is)."),
) -> None:
    """Let a lead agent consult a panel of read-only agents, by id, until it answers.

    The recipe in .crossby.yml names the lead and the members, each with its
    tool and folder. The lead ends each message with `NEXT: ASK @id …` (or
    `@all`), or `NEXT: DONE` for its final answer; addressed members answer
    in parallel, read-only, each resuming its own session, and their replies
    reach the lead together under their @ids. A member with a persona_folder
    gets it provisioned with its instructions and skills in its tool's own
    layout. Exit codes are those of `converse`.
    """
    from crossby.config.loader import ConfigError, load_config
    from crossby.services.journal import (
        JOURNAL_FILE,
        Journal,
        JournalError,
        JournalExistsError,
    )
    from crossby.services.panel import HumanInput, render_panel_transcript, run_panel

    root = _project_root(path)
    try:
        config = load_config(root)
    except ConfigError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None
    recipe = config.panels.get(name)
    if recipe is None:
        known = ", ".join(sorted(config.panels)) or "none defined"
        console.error(f"No panel named {name!r} in .crossby.yml ({known}).")
        raise typer.Exit(1)
    goal_text = _read_text_input(goal, goal_file, what="goal", hint="--goal or --goal-file")
    spec = _spec(
        recipe,
        Path(config.project_root or root),
        goal_text,
        max_rounds=max_rounds,
        allow_project_hooks=allow_project_hooks,
    )

    started = datetime.now(tz=UTC)
    out_dir = (
        output_dir.resolve()
        if output_dir is not None
        else root / ".crossby" / "panels" / started.strftime("%Y%m%dT%H%M%S%f")
    )
    from crossby.cli.panel_view import PanelView, show_opening

    view = (
        None
        if quiet
        else PanelView(spec, live=console.err.is_terminal, typed_here=_isatty(sys.stdin))
    )
    try:
        # Created exclusively: a second run never mixes into this journal.
        journal = Journal(
            uuid.uuid4().hex, path=out_dir / JOURNAL_FILE, sinks=[] if view is None else [view]
        )
    except JournalExistsError:
        console.error(f"{out_dir} already holds a panel; choose another --output-dir.")
        raise typer.Exit(1) from None
    except JournalError as exc:
        console.error(str(exc))
        raise typer.Exit(1) from None
    human = spec.lead.human or any(member.human for member in spec.members)
    member_ids = tuple(member.id for member in spec.members)
    transcript = out_dir / "transcript.md"

    def reply(request: Any) -> Any:
        return _terminal_reply(request, view, member_ids)

    try:
        if view is not None:
            events = str(journal.path) if journal.path is not None else None
            show_opening(spec, transcript=str(transcript), events=events)
            view.start()
        result = run_panel(spec, journal=journal, inputs=HumanInput(reply) if human else None)
    finally:
        journal.close()  # delivers every queued event to the view first
        if view is not None:
            view.close()
    if journal.store_error is not None:
        console.warn(f"The event journal is incomplete: {journal.store_error}")
        result = dataclasses.replace(result, journal_error=journal.store_error)
    _write(transcript, render_panel_transcript(result))
    _report(
        result,
        transcript,
        journal.path,
        journal.run_id,
        as_json=as_json,
        # In a terminal stdout and stderr share the screen, where the final
        # answer was just shown as the last message: print it only once.
        answer_shown=not quiet and sys.stdout.isatty() and console.err.is_terminal,
    )
    if result.exit_code:
        raise typer.Exit(result.exit_code)


def _spec(
    recipe: PanelConfig,
    root: Path,
    goal: str,
    *,
    max_rounds: int | None,
    allow_project_hooks: bool,
) -> PanelSpec:
    """Turn a recipe into a panel, provisioning persona folders it names."""
    from pydantic import ValidationError

    from crossby.models.ai import EffortLevel, RestrictionChannel
    from crossby.services.ask import CliBackend, ProjectWorkspace
    from crossby.services.panel import HumanBackend, Lead, Member, PanelSpec
    from crossby.services.workspaces import OwnedWorkspaceSpec, WorkspaceError, provision

    def backend(tool: str, model: str | None, effort: str | None, role: str) -> CliBackend:
        return CliBackend(
            tool=_headless_tool(tool, role),
            model=model,
            effort=EffortLevel(effort) if effort else None,
        )

    def text(relative: str, what: str) -> str:
        source = root / relative
        try:
            return source.read_text(encoding="utf-8")
        except OSError as exc:
            console.error(f"Cannot read {what} {source}: {exc}")
            raise typer.Exit(1) from None

    try:
        members = []
        for item in recipe.members:
            if item.human:
                members.append(Member(id=item.id, name=item.name, backend=HumanBackend()))
                continue
            assert item.tool is not None
            instructions = text(item.instructions, "instructions") if item.instructions else None
            member_backend = backend(item.tool, item.model, item.effort, f"member {item.id}")
            if item.persona_folder is not None:
                workspace: Any = provision(
                    OwnedWorkspaceSpec(
                        path=root / item.persona_folder,
                        tool=member_backend.tool,
                        instructions=instructions,
                        skills=tuple(root / skill for skill in item.skills),
                        files={key: text(value, "file") for key, value in item.files.items()},
                    )
                )
            else:
                if item.skills or item.files:
                    console.error(f"Member {item.id}: skills and files need a persona_folder.")
                    raise typer.Exit(1)
                workspace = ProjectWorkspace(
                    path=_project_root(root / item.folder, f"member {item.id}'s folder"),
                    instructions=instructions,
                )
            members.append(
                Member(id=item.id, name=item.name, backend=member_backend, workspace=workspace)
            )
        if recipe.lead.human:
            lead = Lead(backend=HumanBackend())
        else:
            assert recipe.lead.tool is not None
            lead = Lead(
                backend=backend(
                    recipe.lead.tool, recipe.lead.model, recipe.lead.effort, "the lead"
                ),
                folder=_project_root(root / recipe.lead.folder, "the lead's folder"),
            )
        return PanelSpec(
            goal=goal,
            lead=lead,
            members=tuple(members),
            visibility=recipe.visibility,
            max_rounds=max_rounds or recipe.max_rounds,
            concurrency=recipe.concurrency,
            timeout_seconds=recipe.timeout,
            turn_timeout_seconds=recipe.turn_timeout,
            require_blocked=frozenset(RestrictionChannel(c) for c in recipe.require_blocked),
            allow_project_hooks=allow_project_hooks,
            protocol_notice=text(recipe.notice, "notice") if recipe.notice else None,
            human_timeout_seconds=recipe.human_timeout,
        )
    except (ValidationError, ValueError, WorkspaceError) as exc:
        console.error(f"The panel recipe is invalid: {exc}")
        raise typer.Exit(1) from None


_TERMINAL = threading.Lock()


def _terminal_reply(
    request: Any, view: PanelView | None = None, members: tuple[str, ...] = ()
) -> Any:
    """One human seat's prompt and reply on the terminal, one seat at a time.

    A request that expires while waiting for the terminal is never prompted,
    and one that expires mid-reply stops reading and frees the terminal.
    While the person types, the panel's live line and other output wait.
    """
    while not _TERMINAL.acquire(timeout=0.2):
        if request.expired.is_set():
            return None
    try:
        if request.expired.is_set():
            return None
        # The messages this seat answers are printed above the prompt, unless
        # the view is slow to catch up; then the prompt shows them itself.
        shown = view is not None and view.wait_announced(request.request_id, timeout=2.0)
        with view.hold() if view is not None else contextlib.nullcontext():
            return _read_terminal_reply(request, message_shown=shown, members=members)
    finally:
        _TERMINAL.release()


class _StdinLines:
    """stdin, read a whole line at a time by one thread of its own.

    A seat waits on the queue, not on the stream, so it can stop the moment
    its request expires. Nothing is ever hidden from it: every line the
    stream buffered is already in the queue, unlike text left in
    ``sys.stdin``'s buffer behind a readiness check of its descriptor. A
    blocking read on a thread works the same on every platform. Lines go to
    seats in order, so piped replies are taken as given; at a terminal, lines
    typed before a prompt appeared are dropped when it does, since no one
    could have meant them for it. The reader is a daemon and never holds up exit.
    """

    def __init__(self, stream: IO[str]) -> None:
        self.stream = stream
        self.interactive = _isatty(stream)
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._closed = threading.Event()
        threading.Thread(target=self._pump, daemon=True, name="crossby-stdin").start()

    def _pump(self) -> None:
        try:
            while True:
                line = self.stream.readline()
                if line:
                    self._lines.put(line)
                    continue
                # End of input. At a terminal it ends one reply (Ctrl-D) and
                # the next prompt can read again; anywhere else it is final.
                self._lines.put(None)
                if not self.interactive:
                    break
        except (OSError, ValueError):
            pass
        self._closed.set()

    def drop_typed_ahead(self) -> None:
        if not self.interactive:
            return
        while True:
            try:
                self._lines.get_nowait()
            except queue.Empty:
                return

    def next(self, expired: threading.Event) -> str | None:
        """The next line, or ``None`` at end of input or once ``expired`` is set."""
        while not expired.is_set():
            try:
                return self._lines.get(timeout=0.1)
            except queue.Empty:
                if self._closed.is_set() and self._lines.empty():
                    return None
        return None


def _isatty(stream: IO[str]) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, OSError, ValueError):
        return False


_STDIN: dict[str, _StdinLines] = {}
_STDIN_LOCK = threading.Lock()


def _stdin_lines() -> _StdinLines:
    """The reader of the current ``sys.stdin``, started on first use."""
    with _STDIN_LOCK:
        reader = _STDIN.get("current")
        if reader is None or reader.stream is not sys.stdin:
            reader = _STDIN["current"] = _StdinLines(sys.stdin)
        return reader


def _readline(request: Any) -> str | None:
    """One line of stdin, or ``None`` at end of input or once the request expired."""
    return _stdin_lines().next(request.expired)


def _read_terminal_reply(
    request: Any, *, message_shown: bool = False, members: tuple[str, ...] = ()
) -> Any:
    """A human seat's reply, read from stdin: lines up to a lone "." or end of input.

    A human lead then names the members (`@cto @founder`, `@all`) or types
    `done` for a final answer. ``message_shown`` means the messages this seat
    answers are already on screen, so the prompt points at them instead.
    """
    from crossby.services.panel import HumanReply

    who = "lead" if request.kind == "lead" else f"@{request.participant}"
    _stdin_lines().drop_typed_ahead()
    if request.kind == "lead" and request.message is None:
        head = "your turn: write your first message to the panel"
    elif request.kind == "lead":
        head = "your turn: the replies are above" if message_shown else "your turn. The replies:"
    else:
        head = (
            "your turn: reply to the lead's message above"
            if message_shown
            else "your turn. The lead's message:"
        )
    _stderr(f"  [step]▸[/] [bold]{_escape(who)}[/], {head}")
    if request.message and not message_shown:
        sys.stderr.write(f"\n{request.message}\n\n")
    _stderr("[dim]      End your reply with a line holding only a period (or Ctrl-D).[/]")
    lines: list[str] = []
    while (line := _readline(request)) is not None:
        if line.rstrip("\n") == ".":
            break
        lines.append(line.rstrip("\n"))
    if request.expired.is_set():
        return None
    text = "\n".join(lines).strip()
    if not text:
        return None
    if request.kind != "lead":
        return HumanReply(text)
    names = " ".join(f"@{member}" for member in members) or "@ids"
    _stderr(
        f"[dim]      Send it to whom? {_escape(names)}, @all, or done for your final answer:[/]"
    )
    target = (_readline(request) or "").strip()
    if target.lower() == "done":
        return HumanReply(text, done=True)
    return HumanReply(text, recipients=tuple(target.replace(",", " ").split()))


def _report(
    result: PanelResult,
    transcript: Path,
    events: Path | None,
    run_id: str,
    *,
    as_json: bool,
    answer_shown: bool,
) -> None:
    from crossby.services.panel import describe_panel_stop

    outcome = describe_panel_stop(result)
    if as_json:
        _print_json(
            {
                "status": result.stop_reason.value,
                "answered": result.answered,
                "final_answer": result.final_answer,
                "outcome": outcome,
                "error": result.error,
                "dispatches": result.dispatches,
                "members": [
                    {
                        "id": member.id,
                        "tool": member.tool_name,
                        "session_id": result.sessions.get(member.id),
                        "replies": sum(1 for t in result.turns if t.sender == member.id),
                    }
                    for member in result.spec.members
                ],
                "member_failures": [
                    {"message": dispatch, "member": member, "reason": reason}
                    for dispatch, member, reason in result.member_failures
                ],
                "transcript_path": str(transcript),
                "events_path": str(events) if events is not None else None,
                "journal_error": result.journal_error,
                "run_id": run_id,
            }
        )
        return
    # stdout carries only the lead's final answer, so it can be piped.
    if result.final_answer is not None and not answer_shown:
        print(result.final_answer)
    if result.answered:
        _stderr(f"  [success]✓[/] {_escape(outcome)}")
    elif result.exit_code == 1:
        _stderr(f"  [error]✗ Error:[/] {_escape(outcome)}")
    else:
        _stderr(f"  [warning]![/] {_escape(outcome)}")
    for dispatch, member, reason in result.member_failures:
        _stderr(f"[dim]      no answer from @{member} to message {dispatch}: {_escape(reason)}[/]")
    _stderr(f"[dim]      transcript: {_escape(str(transcript))}[/]")
