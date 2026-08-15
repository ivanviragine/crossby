"""DECLARE activators — write a tool's own disable key, non-destructively.

Each activator disables a *set* of deselected items on one tool by writing that
tool's native key, records exactly what it wrote in the ownership ledger
(``scene`` section), and can be reverted later touching only crossby-owned
entries. All five surfaces share one provenance-diff model, identical to
revocable sync, so apply is idempotent and a scene switch (A → B) leaves no
trace of A:

    to_add     = desired - present        # write these disable entries
    to_remove  = owned   - desired        # crossby wrote these before; revert
    new_owned  = (owned & desired) | to_add

``desired`` is the deselected set for the scene (``clear_scene`` passes the empty
set, so every owned entry reverts). ``present`` is what is currently disabled
regardless of author, so a user's own disable of the same item is never claimed
and never reverted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from crossby.config.json_utils import read_json_file, write_json_file
from crossby.models.ai import AIToolID
from crossby.scenes import trust, versioning
from crossby.sync.base import SyncConcern, SyncResult
from crossby.sync.ownership import OwnershipLedger, SceneDeclareKey
from crossby.sync.toml_edit import set_scalar, unset_scalar

logger = structlog.get_logger()

_CLAUDE_SETTINGS = Path(".claude") / "settings.json"
_CODEX_CONFIG = Path(".codex") / "config.toml"
_ANTIGRAVITY_MCP = Path(".agents") / "mcp_config.json"

_SKILL_OFF = "off"


@dataclass(frozen=True)
class _Diff:
    to_add: set[str]
    to_remove: set[str]
    new_owned: frozenset[str]

    @property
    def changed(self) -> bool:
        return bool(self.to_add or self.to_remove)


def _diff(owned: frozenset[str], desired: set[str], present: set[str]) -> _Diff:
    to_add = set(desired - present)
    to_remove = set(owned - desired)
    new_owned = frozenset((owned & desired) | to_add)
    return _Diff(to_add=to_add, to_remove=to_remove, new_owned=new_owned)


def _agent_rule(name: str) -> str:
    """The Claude permission-rule string that blocks subagent *name*."""
    return f"Agent({name})"


def _str_list(value: object) -> list[str]:
    """The string members of *value* when it is a list, else ``[]``."""
    return [e for e in value if isinstance(e, str)] if isinstance(value, list) else []


# ---------------------------------------------------------------------------
# Claude — all three surfaces live in .claude/settings.json
# ---------------------------------------------------------------------------


def apply_claude_skill_overrides(
    project_root: Path,
    disable: set[str],
    ledger: OwnershipLedger,
    *,
    dry_run: bool = False,
    version: tuple[int, int, int] | None = None,
) -> SyncResult:
    """Set ``skillOverrides: {"<name>": "off"}`` for each deselected skill.

    Adding a ``skillOverrides`` entry is gated on ``claude >= 2.1.129`` (older
    builds silently ignore the key, and it never affects plugin skills), so on an
    older/unknown build the *additions* are skipped with a warning. Removals are
    never gated: a scene switch or ``clear_scene`` must always be able to revert
    entries crossby wrote earlier, regardless of the version detected now.
    """
    owned = ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES)
    path = project_root / _CLAUDE_SETTINGS

    settings, error, was_new = read_json_file(path)
    if settings is None:
        return _malformed(AIToolID.CLAUDE, SyncConcern.SKILLS, path, error)

    raw_overrides = settings.get("skillOverrides")
    overrides: dict[str, object] = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
    present = {name for name, value in overrides.items() if value == _SKILL_OFF}
    diff = _diff(owned, disable, present)

    version_ok = versioning.at_least(version, versioning.CLAUDE_SKILL_OVERRIDES_MIN)
    adds = diff.to_add if version_ok else set()
    blocked = diff.to_add if not version_ok else set()
    # Never overwrite an override a human set to a non-"off" value — respect their
    # intent (and never claim it), the same way the removal path refuses to.
    protected = {name for name in adds if overrides.get(name) not in (None, _SKILL_OFF)}
    adds -= protected

    for name in adds:
        overrides[name] = _SKILL_OFF
    for name in diff.to_remove:
        # Only drop it if it is still our "off" marker — never clobber a user
        # who re-purposed the same key to another value.
        if overrides.get(name) == _SKILL_OFF:
            del overrides[name]

    # Never claim ownership of an override we couldn't actually write.
    new_owned = frozenset((owned & disable) | adds)

    if not adds and not diff.to_remove:
        # No-write path: record ownership inline (nothing can fail here), which
        # still reconciles a server the user re-enabled out of our ownership.
        ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES, new_owned)
        message = None
        if blocked:
            floor = ".".join(map(str, versioning.CLAUDE_SKILL_OVERRIDES_MIN))
            detected = ".".join(map(str, version)) if version else "unknown"
            message = (
                f"skillOverrides needs claude >= {floor} (detected {detected}); "
                f"{len(blocked)} deselected skill(s) not filtered for Claude"
            )
        elif protected:
            message = f"{len(protected)} deselected skill(s) kept a user override; left untouched"
        return SyncResult(
            tool_id=AIToolID.CLAUDE,
            concern=SyncConcern.SKILLS,
            action="skipped",
            file_path=path,
            message=message or "already applied",
        )
    if not dry_run:
        if overrides:
            settings["skillOverrides"] = overrides
        else:
            settings.pop("skillOverrides", None)
        write_json_file(path, settings)
    # Record ownership only AFTER a successful write (transactional): if
    # write_json_file raises, this never runs and the in-memory ledger keeps the
    # prior ownership for this key. On dry_run no write happened, but recording
    # the planned ownership matches the write path's committed state.
    ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.SKILL_OVERRIDES, new_owned)

    note = f"skillOverrides off: {_names(adds)}"
    if blocked:
        note += f"; {len(blocked)} skill(s) not filtered (claude too old)"
    if protected:
        note += f"; {len(protected)} kept a user override"
    return SyncResult(
        tool_id=AIToolID.CLAUDE,
        concern=SyncConcern.SKILLS,
        action="created" if was_new else "updated",
        file_path=path,
        message=note,
        added=len(adds),
        revoked=len(diff.to_remove),
        created=tuple(sorted(adds)),
    )


def apply_claude_deny_agents(
    project_root: Path,
    disable: set[str],
    ledger: OwnershipLedger,
    *,
    dry_run: bool = False,
) -> SyncResult:
    """Add ``permissions.deny: ["Agent(<name>)"]`` for each deselected agent."""
    owned = ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS)
    desired = {_agent_rule(name) for name in disable}
    path = project_root / _CLAUDE_SETTINGS

    settings, error, was_new = read_json_file(path)
    if settings is None:
        return _malformed(AIToolID.CLAUDE, SyncConcern.AGENTS, path, error)

    raw_perms = settings.get("permissions")
    permissions: dict[str, object] = dict(raw_perms) if isinstance(raw_perms, dict) else {}
    deny: list[str] = _str_list(permissions.get("deny"))

    present = set(deny)
    diff = _diff(owned, desired, present)

    deny = [e for e in deny if e not in diff.to_remove]
    deny.extend(sorted(diff.to_add))

    if not diff.changed:
        # No-write path: record inline (reconciliation only, nothing can fail).
        ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS, diff.new_owned)
        return _skipped(AIToolID.CLAUDE, SyncConcern.AGENTS, path)
    if not dry_run:
        if deny:
            permissions["deny"] = deny
        else:
            permissions.pop("deny", None)
        if permissions:
            settings["permissions"] = permissions
        else:
            settings.pop("permissions", None)
        write_json_file(path, settings)
    # Record ownership only AFTER a successful write (transactional): a raising
    # write_json_file leaves the prior ownership for this key intact in memory.
    ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DENY_AGENTS, diff.new_owned)
    return _written(
        AIToolID.CLAUDE,
        SyncConcern.AGENTS,
        path,
        diff,
        was_new,
        f"permissions.deny: {_names(diff.to_add)}",
    )


def apply_claude_disabled_mcp(
    project_root: Path,
    disable: set[str],
    ledger: OwnershipLedger,
    *,
    dry_run: bool = False,
) -> SyncResult:
    """Add deselected servers to ``disabledMcpjsonServers`` in settings.json."""
    owned = ledger.scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DISABLED_MCP)
    path = project_root / _CLAUDE_SETTINGS

    settings, error, was_new = read_json_file(path)
    if settings is None:
        return _malformed(AIToolID.CLAUDE, SyncConcern.MCP, path, error)

    disabled: list[str] = _str_list(settings.get("disabledMcpjsonServers"))
    present = set(disabled)
    diff = _diff(owned, disable, present)

    disabled = [e for e in disabled if e not in diff.to_remove]
    disabled.extend(sorted(diff.to_add))

    if not diff.changed:
        # No-write path: record inline (reconciliation only, nothing can fail).
        ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DISABLED_MCP, diff.new_owned)
        return _skipped(AIToolID.CLAUDE, SyncConcern.MCP, path)
    if not dry_run:
        if disabled:
            settings["disabledMcpjsonServers"] = sorted(set(disabled))
        else:
            settings.pop("disabledMcpjsonServers", None)
        write_json_file(path, settings)
    # Record ownership only AFTER a successful write (transactional): a raising
    # write_json_file leaves the prior ownership for this key intact in memory.
    ledger.record_scene_declare(AIToolID.CLAUDE, SceneDeclareKey.DISABLED_MCP, diff.new_owned)
    return _written(
        AIToolID.CLAUDE,
        SyncConcern.MCP,
        path,
        diff,
        was_new,
        f"disabledMcpjsonServers: {_names(diff.to_add)}",
    )


# ---------------------------------------------------------------------------
# Codex — .codex/config.toml, mcp_servers.<id>.enabled = false
# ---------------------------------------------------------------------------


def apply_codex_disabled_mcp(
    project_root: Path,
    disable: set[str],
    ledger: OwnershipLedger,
    *,
    dry_run: bool = False,
    trusted: bool | None = None,
) -> SyncResult:
    """Set ``mcp_servers.<id>.enabled = false`` for each deselected server.

    Only touches servers that already have an ``[mcp_servers.<id>]`` table (a
    deselected server crossby can't see there is nothing to disable). When the
    project is untrusted, the write still happens but the row explains the
    toggle won't take effect until Codex trusts the directory.
    """
    owned = ledger.scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED)
    path = project_root / _CODEX_CONFIG
    if trusted is None:
        trusted = trust.codex_trusts_project(project_root)

    if not path.is_file():
        # No config, no server tables — nothing crossby can disable here. Still
        # reconcile ownership so a prior scene's entries don't linger.
        ledger.record_scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED, frozenset())
        return _skipped(AIToolID.CODEX, SyncConcern.MCP, path)

    text = path.read_text(encoding="utf-8")
    present, defined = _codex_mcp_state(text)
    # Only servers with a table can be toggled; narrow the target accordingly.
    target = disable & defined
    diff = _diff(owned, target, present)

    new_text = text
    added: set[str] = set()
    add_failed: set[str] = set()
    for server in sorted(diff.to_add):
        spliced = set_scalar(new_text, ("mcp_servers", server), "enabled", "false")
        if spliced is not None:
            new_text = spliced
            added.add(server)
        else:
            # The disable splice returned None — no file change for this server.
            add_failed.add(server)
    removed_ok: set[str] = set()
    remove_failed: set[str] = set()
    for server in sorted(diff.to_remove):
        # Only revert crossby's own disable. If the user flipped ``enabled`` back
        # to true (so the server is no longer in ``present``), leave their value
        # untouched — that is a clean release, not a failure: ownership drops
        # either way, since a reverted server never re-enters ``new_owned``.
        if server not in present:
            continue
        spliced = unset_scalar(
            new_text,
            ("mcp_servers", server),
            "enabled",
            include_implicit=True,
        )
        # unset_scalar returns the text UNCHANGED (not None) when the marker lives
        # in a representation it can't edit textually. Since ``server in present``
        # means ``enabled = false`` IS in the parsed data, "no change" means the
        # revert did not land: treat it as a failure (retain ownership + error
        # below), not a phantom success that would leave the server disabled on
        # disk with no provenance.
        if spliced is not None and spliced != new_text:
            new_text = spliced
            removed_ok.add(server)
        else:
            remove_failed.add(server)

    # Claim ownership for every server still disabled under our marker: the ones
    # we just wrote, the still-desired ones already disabled, and any removal
    # whose splice failed (still on disk, so still ours to revert next time).
    new_owned = frozenset((owned & target & present) | added | remove_failed)
    splice_failed = bool(add_failed or remove_failed)

    if new_text == text:
        # Nothing spliced — no write can fail, so record inline. Report an error
        # if any planned splice failed (never an "already applied" no-op that
        # claims a toggle that never happened); otherwise an idempotent skip.
        ledger.record_scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED, new_owned)
        if splice_failed:
            return _codex_splice_error(path, add_failed, remove_failed)
        return _skipped(AIToolID.CODEX, SyncConcern.MCP, path)

    if not dry_run:
        from crossby.config.json_utils import atomic_write_text

        atomic_write_text(path, new_text)
    # Record ownership only AFTER the (partial or full) write succeeds: a raising
    # atomic_write_text leaves the prior ownership for this key intact in memory.
    ledger.record_scene_declare(AIToolID.CODEX, SceneDeclareKey.CODEX_MCP_DISABLED, new_owned)

    if splice_failed:
        # Some splices landed and were written, but at least one failed. Report an
        # error (even though the file was partially written) so the CLI keeps
        # scene-state.json and a retry re-attempts the still-owned failures, while
        # the successful changes stay on disk.
        return _codex_splice_error(
            path, add_failed, remove_failed, added=added, revoked=len(removed_ok)
        )

    message = f"mcp_servers enabled=false: {_names(added)}"
    if not trusted:
        message += " — Codex does not trust this project; toggle will not take effect until trusted"
    return SyncResult(
        tool_id=AIToolID.CODEX,
        concern=SyncConcern.MCP,
        action="updated",  # the config file already existed (early-returned otherwise)
        file_path=path,
        message=message,
        added=len(added),
        revoked=len(removed_ok),
        created=tuple(sorted(added)),
    )


def _codex_splice_error(
    path: Path,
    add_failed: set[str],
    remove_failed: set[str],
    *,
    added: set[str] | None = None,
    revoked: int = 0,
) -> SyncResult:
    """An ``error`` row naming the Codex splices that could not be applied.

    ``added`` / ``revoked`` carry the splices that *did* land (a partial success),
    so the report still credits the changes written alongside the failure.
    """
    parts: list[str] = []
    if add_failed:
        parts.append(f"disable {len(add_failed)}")
    if remove_failed:
        parts.append(f"revert {len(remove_failed)}")
    return SyncResult(
        tool_id=AIToolID.CODEX,
        concern=SyncConcern.MCP,
        action="error",
        file_path=path,
        message=f"could not {' and '.join(parts)} mcp server(s) in {path.name}",
        added=len(added) if added else 0,
        revoked=revoked,
        created=tuple(sorted(added)) if added else (),
    )


def _codex_mcp_state(text: str) -> tuple[set[str], set[str]]:
    """Return ``(disabled_now, defined)`` server-id sets from a codex config.

    ``defined`` is every server with an ``[mcp_servers.<id>]`` table; ``disabled_now``
    is the subset whose ``enabled`` reads falsey. A parse failure yields two
    empty sets (best-effort, never raises).
    """
    import tomllib

    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return set(), set()
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return set(), set()
    defined = {str(name) for name in servers}
    disabled = {
        str(name)
        for name, cfg in servers.items()
        if isinstance(cfg, dict) and cfg.get("enabled") is False
    }
    return disabled, defined


# ---------------------------------------------------------------------------
# Antigravity CLI — .agents/mcp_config.json, mcpServers.<name>.disabled = true
# ---------------------------------------------------------------------------


def apply_antigravity_disabled_mcp(
    project_root: Path,
    disable: set[str],
    ledger: OwnershipLedger,
    *,
    dry_run: bool = False,
) -> SyncResult:
    """Set ``mcpServers.<name>.disabled = true`` for each deselected server."""
    owned = ledger.scene_declare(AIToolID.ANTIGRAVITY_CLI, SceneDeclareKey.ANTIGRAVITY_MCP_DISABLED)
    path = project_root / _ANTIGRAVITY_MCP

    settings, error, was_new = read_json_file(path)
    if settings is None:
        return _malformed(AIToolID.ANTIGRAVITY_CLI, SyncConcern.MCP, path, error)

    raw_servers = settings.get("mcpServers")
    servers: dict[str, object] = dict(raw_servers) if isinstance(raw_servers, dict) else {}
    # Only dict-valued entries can carry a ``disabled`` flag; a malformed non-dict
    # entry must never enter the target set, the ledger, or the added count.
    defined = {name for name, cfg in servers.items() if isinstance(cfg, dict)}
    present = {
        name
        for name, cfg in servers.items()
        if isinstance(cfg, dict) and cfg.get("disabled") is True
    }
    target = disable & defined
    diff = _diff(owned, target, present)

    for name in diff.to_add:
        cfg = servers.get(name)
        if isinstance(cfg, dict):
            servers[name] = {**cfg, "disabled": True}
    for name in diff.to_remove:
        cfg = servers.get(name)
        if isinstance(cfg, dict) and cfg.get("disabled") is True:
            servers[name] = {k: v for k, v in cfg.items() if k != "disabled"}

    if not diff.changed:
        # No-write path: record inline (reconciliation only, nothing can fail).
        ledger.record_scene_declare(
            AIToolID.ANTIGRAVITY_CLI, SceneDeclareKey.ANTIGRAVITY_MCP_DISABLED, diff.new_owned
        )
        return _skipped(AIToolID.ANTIGRAVITY_CLI, SyncConcern.MCP, path)
    if not dry_run:
        settings["mcpServers"] = servers
        write_json_file(path, settings)
    # Record ownership only AFTER a successful write (transactional): a raising
    # write_json_file leaves the prior ownership for this key intact in memory.
    ledger.record_scene_declare(
        AIToolID.ANTIGRAVITY_CLI, SceneDeclareKey.ANTIGRAVITY_MCP_DISABLED, diff.new_owned
    )
    return _written(
        AIToolID.ANTIGRAVITY_CLI,
        SyncConcern.MCP,
        path,
        diff,
        was_new,
        f"mcpServers disabled=true: {_names(diff.to_add)}",
    )


# ---------------------------------------------------------------------------
# SyncResult builders
# ---------------------------------------------------------------------------


def _names(items: set[str]) -> str:
    return ", ".join(sorted(items)) if items else "(none)"


def _skipped(tool: AIToolID, concern: SyncConcern, path: Path) -> SyncResult:
    return SyncResult(
        tool_id=tool, concern=concern, action="skipped", file_path=path, message="already applied"
    )


def _malformed(tool: AIToolID, concern: SyncConcern, path: Path, error: str | None) -> SyncResult:
    return SyncResult(
        tool_id=tool,
        concern=concern,
        action="error",
        file_path=path,
        message=f"refusing to edit malformed {path.name}: {error}",
    )


def _written(
    tool: AIToolID,
    concern: SyncConcern,
    path: Path,
    diff: _Diff,
    was_new: bool,
    message: str,
) -> SyncResult:
    return SyncResult(
        tool_id=tool,
        concern=concern,
        action="created" if was_new else "updated",
        file_path=path,
        message=message,
        added=len(diff.to_add),
        revoked=len(diff.to_remove),
        created=tuple(sorted(diff.to_add)),
    )
