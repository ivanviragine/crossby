"""Command-line interface: `carbrain --help`."""

from __future__ import annotations

import json
import os
from typing import Annotated, Any

import typer

from carbrain.archive import RawArchive
from carbrain.catalog import load_catalog, load_events, open_reviews, resolve_review
from carbrain.config import Settings, load_settings
from carbrain.connectors import CONNECTORS
from carbrain.connectors.base import Connector, RunResult, run_connector
from carbrain.connectors.official import Anfavea, AnpWeekly
from carbrain.connectors.vehicles import FipeDevSample, SenatranFleet
from carbrain.content.base import NotConfiguredError
from carbrain.content.catalog import load_publishers, sync_catalog
from carbrain.content.runner import sync_content, verify_channels
from carbrain.db import connect
from carbrain.http import Http
from carbrain.models import Use
from carbrain.resolve import Resolver
from carbrain.retention import purge_expired
from carbrain.rights import Registry, RightsError
from carbrain.tools import (
    FreshnessInput,
    InformationSourcesInput,
    ToolContext,
    call_tool,
    data_freshness,
    information_sources,
)

app = typer.Typer(help="Brazilian car market evidence database.", no_args_is_help=True)
review_app = typer.Typer(help="Review how source labels map to vehicle families.")
app.add_typer(review_app, name="review")
catalog_app = typer.Typer(help="Publishers (specialist media, creators) and their channels.")
app.add_typer(catalog_app, name="catalog")


class _Env:
    def __init__(self) -> None:
        self.settings: Settings = load_settings()
        self.conn = connect(self.settings.db_path)
        self.registry = Registry.load(self.settings.sources_file)
        self.resolver = Resolver.from_yaml(self.settings.families_file)

    def ctx(self) -> ToolContext:
        return ToolContext(self.conn, self.registry, self.resolver)


def _print_json(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@app.command()
def init() -> None:
    """Create the database and load the vehicle catalog and events."""
    env = _Env()
    families = load_catalog(env.conn, env.resolver)
    events = load_events(env.conn, env.settings.events_file)
    publishers = sync_catalog(env.conn, load_publishers(env.settings.publishers_file))
    typer.echo(f"Database: {env.settings.db_path}")
    typer.echo(f"Loaded {families} vehicle families, {events} events, {publishers} publishers.")


@app.command()
def sources(source_type: Annotated[str | None, typer.Option("--type")] = None) -> None:
    """Data sources grouped by type, with what their rights allow."""
    env = _Env()
    by_type: dict[str, list[Any]] = {}
    for spec in env.registry:
        if source_type is None or spec.type == source_type:
            by_type.setdefault(spec.type, []).append(spec)
    for kind, specs in sorted(by_type.items()):
        typer.echo(f"\n{kind}")
        for spec in specs:
            allowed = [u.value for u in Use if env.registry.allows(spec.id, u)]
            typer.echo(
                f"  {spec.id:17} {spec.cadence:8} allows: {', '.join(allowed) or 'nothing yet'}"
            )


@catalog_app.command("list")
def catalog_list(
    kind: Annotated[str | None, typer.Option()] = None,
    focus: Annotated[str | None, typer.Option()] = None,
    platform: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Specialist media and creators, with their channels and verification status."""
    env = _Env()
    sync_catalog(env.conn, load_publishers(env.settings.publishers_file))
    result = information_sources(
        env.ctx(),
        InformationSourcesInput.model_validate(
            {"kind": kind, "focus": focus, "platform": platform}
        ),
    )
    for pub in result.data or []:
        typer.echo(f"{pub['name']} [{pub['kind']}] focus={','.join(pub['focus'])}")
        for ch in pub["channels"]:
            where = ch["handle"] or ch["url"] or f"channel id {ch['external_id']}"
            typer.echo(f"    {ch['platform']:10} {where:55} {ch['status']}")


@catalog_app.command("verify")
def catalog_verify(platform: str = "rss") -> None:
    """Confirm channels through official interfaces (feeds, YouTube Data API, Instagram Graph)."""
    env = _Env()
    sync_catalog(env.conn, load_publishers(env.settings.publishers_file))
    http = Http()
    try:
        outcomes = verify_channels(env.conn, env.registry, http, platform)
    except (RightsError, NotConfiguredError) as exc:
        typer.echo(f"{platform}: cannot verify — {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        http.close()
    for channel_id, outcome in outcomes:
        typer.echo(f"{channel_id:60} {outcome}")


@app.command("sync-content")
def sync_content_cmd(
    platform: Annotated[list[str] | None, typer.Option(help="rss, youtube, instagram")] = None,
    limit: int = 50,
) -> None:
    """Fetch the latest headlines/videos from catalogued channels (metadata only)."""
    env = _Env()
    load_catalog(env.conn, env.resolver)
    sync_catalog(env.conn, load_publishers(env.settings.publishers_file))
    http = Http()
    try:
        results = sync_content(
            env.conn, env.registry, env.resolver, http, platforms=platform, limit=limit
        )
    finally:
        http.close()
    for r in results:
        line = (
            f"{r.platform:10} {r.status:8} channels={r.channels} "
            f"new={r.items_new} seen={r.items_seen}"
        )
        typer.echo(line + (f" ({r.reason})" if r.reason else ""))
        for err in r.errors:
            typer.echo(f"    {err}", err=True)


@app.command()
def sync(
    sources: Annotated[list[str] | None, typer.Argument(help="Source IDs; default: all")] = None,
    weeks: Annotated[int, typer.Option(help="ANP weeks to fetch")] = 1,
    months: Annotated[int, typer.Option(help="SENATRAN months to fetch")] = 1,
    years: Annotated[list[int] | None, typer.Option(help="ANFAVEA years")] = None,
    reparse: Annotated[bool, typer.Option(help="Parse files even if unchanged")] = False,
) -> None:
    """Check sources and store whatever changed. Safe to run every day."""
    env = _Env()
    load_catalog(env.conn, env.resolver)
    factories: dict[str, Any] = {
        **CONNECTORS,
        "anp_weekly": lambda: AnpWeekly(weeks=weeks),
        "senatran_fleet": lambda: SenatranFleet(months=months),
        "anfavea": lambda: Anfavea(years=years),
    }
    selected = sources or list(CONNECTORS)
    failed = False
    for source_id in selected:
        if source_id not in factories:
            raise typer.BadParameter(f"Unknown source {source_id}. Known: {', '.join(factories)}")
        result = _run(env, factories[source_id](), reparse=reparse)
        failed |= result.status == "error"
    if failed:
        raise typer.Exit(1)


def _run(env: _Env, connector: Connector, **kw: bool) -> RunResult:
    http = Http()
    try:
        result = run_connector(
            connector,
            conn=env.conn,
            http=http,
            archive=RawArchive(env.conn, env.settings.raw_dir),
            registry=env.registry,
            resolver=env.resolver,
            **kw,
        )
    except RightsError as exc:
        typer.echo(f"{connector.source_id}: blocked — {exc}", err=True)
        return RunResult(connector.source_id, "error", error=str(exc))
    finally:
        http.close()
    line = (
        f"{result.source_id:15} {result.status:9} files={result.files_fetched} "
        f"new={result.files_new} inserted={result.inserted} revised={result.revised}"
    )
    typer.echo(line + (f" error={result.error}" if result.error else ""))
    return result


@app.command("fipe-sample")
def fipe_sample(
    family_id: str,
    brand_code: Annotated[str, typer.Argument(help="FIPE brand code, e.g. 59 for VW")],
    max_models: int = 2,
) -> None:
    """Collect a few FIPE lookups for parser development (daily cap; never shown)."""
    env = _Env()
    load_catalog(env.conn, env.resolver)
    connector = FipeDevSample(env.resolver, family_id, brand_code, max_models=max_models)
    result = _run(env, connector, dev_sample=True)
    if result.status == "error":
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """When each source was last checked, and the latest period it covers."""
    env = _Env()
    result = data_freshness(env.ctx(), FreshnessInput())
    for row in result.data or []:
        typer.echo(
            f"{row['source_id']:15} {row['status']:6} checked={row['last_checked'] or '-':25} "
            f"data_until={row['latest_period_end'] or '-'}"
        )
    reviews = len(open_reviews(env.conn, limit=10_000))
    typer.echo(f"Open mapping reviews: {reviews}")


@app.command()
def purge() -> None:
    """Delete text whose retention period has ended (e.g. YouTube comments after 30 days)."""
    env = _Env()
    typer.echo(f"Purged {purge_expired(env.conn, registry=env.registry)} expired items.")


@review_app.command("list")
def review_list(limit: int = 20) -> None:
    """Labels waiting for a person to confirm or reject."""
    env = _Env()
    for item in open_reviews(env.conn, limit):
        typer.echo(
            f"#{item.id:<5} {item.source_id:15} {item.external_key:30} "
            f"-> {item.candidate_id} ({item.confidence or 0:.2f}) {item.reason}"
        )


@review_app.command("confirm")
def review_confirm(review_id: int, family_id: str) -> None:
    env = _Env()
    resolve_review(env.conn, review_id, family_id=family_id)
    typer.echo(f"#{review_id} mapped to {family_id}.")


@review_app.command("reject")
def review_reject(review_id: int) -> None:
    env = _Env()
    resolve_review(env.conn, review_id, family_id=None)
    typer.echo(f"#{review_id} rejected.")


@app.command()
def tool(
    name: str,
    arguments: Annotated[str, typer.Argument(help="JSON object of arguments")] = "{}",
) -> None:
    """Call a chatbot tool directly and print its JSON result."""
    env = _Env()
    try:
        result = call_tool(env.ctx(), name, json.loads(arguments))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def ask(question: str, model: str = "claude-opus-5") -> None:
    """Ask the chatbot (needs the `chat` extra and Anthropic credentials)."""
    try:
        import anthropic
    except ImportError as exc:
        raise typer.BadParameter("Install the chat extra: uv sync --extra chat") from exc
    from carbrain.chat import ChatSession

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        typer.echo("Note: no ANTHROPIC_API_KEY set; relying on other configured credentials.")
    env = _Env()
    session = ChatSession(anthropic.Anthropic().beta.messages, env.ctx(), model=model)
    answer = session.ask(question)
    typer.echo(answer.text)
    if answer.unsupported_numbers:
        typer.echo(
            f"\n[warning] numbers not found in any tool result: {answer.unsupported_numbers}",
            err=True,
        )
