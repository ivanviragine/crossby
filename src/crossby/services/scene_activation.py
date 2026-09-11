"""Recoverable persistent scene activation shared by every entry point.

The scene engine intentionally owns filesystem mutation and ledger provenance
only.  This service owns the lifecycle around those mutations: active-state
loading, scope safety, drift checks, outgoing reverts, and the companion scene
state record used by ``scene status`` / ``scene clear``.

Presentation stays at the CLI boundary.  Callers receive structured outcomes
and failures and decide how to render warnings, result rows, and recovery hints.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from crossby.models.ai import AIToolID
from crossby.models.config import SCENE_CONCERNS, SceneConfig
from crossby.scenes import engine
from crossby.scenes.mechanism import base_mechanism
from crossby.scenes.state import (
    SceneState,
    SceneToolRecord,
    clear_scene_state,
    compute_hashes,
    content_hash,
    detect_drift,
    load_scene_state,
    now_iso,
    save_scene_state,
)
from crossby.services.scene_resolution import ResolvedScene
from crossby.sync.base import SyncConcern, SyncResult
from crossby.sync.ownership import LEDGER_PATH, load_ledger_checked


class ActivationFailureKind(StrEnum):
    """Stable categories callers can map to their own presentation."""

    CORRUPT_PROVENANCE = "corrupt_provenance"
    UNSAFE_SWITCH = "unsafe_switch"
    DRIFT = "drift"
    FAILED_REVERT = "failed_revert"
    APPLY_EXCEPTION = "apply_exception"
    STATE_PERSISTENCE = "state_persistence"


class SceneActivationError(Exception):
    """A persistent activation that could not safely complete."""

    def __init__(
        self,
        kind: ActivationFailureKind,
        message: str,
        *,
        hint: str | None = None,
        warnings: Sequence[str] = (),
        drifted: Sequence[str] = (),
        results: Sequence[SyncResult] = (),
        recovery_recorded: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.hint = hint
        self.warnings = tuple(warnings)
        self.drifted = tuple(drifted)
        self.results = tuple(results)
        self.recovery_recorded = recovery_recorded


@dataclass(frozen=True)
class SceneActivationOutcome:
    """The complete result of a persistent activation or preview."""

    scope: tuple[AIToolID, ...]
    results: tuple[SyncResult, ...]
    status: Literal["applied", "partial", "preview"]
    warnings: tuple[str, ...] = ()

    @property
    def has_errors(self) -> bool:
        return any(result.action == "error" for result in self.results)


def expand_shared_scope(
    scope: Iterable[AIToolID], candidates: Iterable[AIToolID]
) -> list[AIToolID]:
    """Include candidate tools affected through a shared skills directory."""
    from crossby.config.skills import SKILLS_DIR

    expanded = list(dict.fromkeys(scope))
    scoped_dirs = {SKILLS_DIR.get(tool) for tool in expanded} - {None}
    for tool in candidates:
        if tool not in expanded and SKILLS_DIR.get(tool) in scoped_dirs:
            expanded.append(tool)
    return expanded


def recorded_tools(active: SceneState) -> list[AIToolID]:
    """Return recognised tools from an on-disk state record."""
    tools: list[AIToolID] = []
    for tool in active.tool_ids:
        with contextlib.suppress(ValueError):
            tools.append(AIToolID(tool))
    return tools


def _outgoing_shared_skill_scope(
    active: SceneState, primary_scope: Sequence[AIToolID]
) -> list[AIToolID]:
    """Return active skills-only co-sharers restored by a scoped clear.

    The incoming scene may not constrain skills, but clearing one tool still
    restores an outgoing shared skills directory. Derive those co-sharers from
    the active record rather than the incoming scene so the switch guard does
    not mistake them for stranded tools.
    """
    primary = {str(tool) for tool in primary_scope}
    return [
        tool
        for tool in expand_shared_scope(primary_scope, recorded_tools(active))
        if str(tool) not in primary
        and "skills" in active.tools.get(str(tool), SceneToolRecord()).mechanisms
    ]


def activate_scene(
    *,
    scene_name: str,
    scene: SceneConfig,
    resolved: ResolvedScene,
    project_root: Path,
    requested_tools: Iterable[AIToolID] | None,
    installed_candidates: Iterable[AIToolID],
    force: bool = False,
    dry_run: bool = False,
) -> SceneActivationOutcome:
    """Apply a persistent scene through the recoverable lifecycle.

    ``requested_tools=None`` means an unscoped ``scene use`` and targets every
    installed candidate. A concrete iterable is a scoped activation (including
    a launch fallback). When the scene narrows skills, installed tools sharing
    the requested tool's skills directory are additionally recorded for that
    shared concern; they are deliberately not passed to the engine's whole-tool
    scope, which would also apply unrelated concerns to them.
    """
    candidates = list(dict.fromkeys(installed_candidates))
    explicitly_scoped = requested_tools is not None
    initial_scope = candidates if requested_tools is None else list(requested_tools)
    shared_skill_scope = (
        expand_shared_scope(initial_scope, candidates)
        if scene.skills is not None
        else initial_scope
    )
    scope = list(dict.fromkeys([*initial_scope, *shared_skill_scope]))
    initial_scope_strings = {str(tool) for tool in initial_scope}

    loaded = load_scene_state(project_root)
    warnings = (loaded.warning,) if loaded.warning else ()
    active = loaded.state
    reverted_scope: list[AIToolID] = []
    outgoing_shared_skill_scope = (
        _outgoing_shared_skill_scope(active, initial_scope)
        if active is not None and explicitly_scoped
        else []
    )
    outgoing_scope_strings = {str(tool) for tool in (*initial_scope, *outgoing_shared_skill_scope)}
    rollback_shared_skill_scope = list(
        dict.fromkeys([*shared_skill_scope, *outgoing_shared_skill_scope])
    )

    if load_ledger_checked(project_root).corrupt:
        raise SceneActivationError(
            ActivationFailureKind.CORRUPT_PROVENANCE,
            f"{LEDGER_PATH.as_posix()} is unreadable — cannot determine what to revert.",
            hint=(
                "Restore a valid owned.json from backup, or manually revert the applied "
                "settings. Do NOT delete owned.json — an empty ledger would let 'clear' "
                "drop the recovery state while leaving settings applied."
            ),
            warnings=warnings,
        )

    # A preview deliberately does not enforce active-scene drift/switch rules:
    # it writes nothing and historically remains useful for inspecting the next
    # apply even while the outgoing scene has drifted.
    if dry_run:
        results = engine.apply_scene(
            resolved, project_root, dry_run=True, force=force, tools=initial_scope
        )
        return SceneActivationOutcome(tuple(scope), tuple(results), "preview", warnings)

    if active is not None and active.scene != scene_name and explicitly_scoped:
        other_tools = [
            tool
            for tool in active.tool_ids
            if tool not in outgoing_scope_strings
            # A mechanism-free recovery record carries only irreversible removals
            # from an earlier replacement. It has no active scene state to strand;
            # _inherit_revocations preserves it in the incoming state instead.
            and not (active.tools[tool].status == "recovery" and not active.tools[tool].mechanisms)
        ]
        other_tools.extend(
            tool
            for tool, record in active.tools.items()
            if tool in outgoing_scope_strings
            and tool not in initial_scope_strings
            and any(concern != "skills" for concern in record.mechanisms)
        )
        if other_tools:
            raise SceneActivationError(
                ActivationFailureKind.UNSAFE_SWITCH,
                (
                    f"Scene {active.scene!r} is active on {', '.join(sorted(other_tools))}; "
                    f"switching to {scene_name!r} with a scoped activation would strand "
                    f"them on {active.scene!r}."
                ),
                hint="Run 'crossby scene clear' first, or activate the scene without a tool scope.",
                warnings=warnings,
            )

    if active is not None:
        recorded = recorded_tools(active)
        outgoing = (
            recorded
            if not explicitly_scoped
            else [tool for tool in recorded if str(tool) in outgoing_scope_strings]
        )
        if outgoing:
            drifted = (
                _detect_scoped_drift(project_root, active, outgoing, initial_scope)
                if explicitly_scoped
                else detect_drift(project_root, active, tools=[str(tool) for tool in outgoing])
            )
            if drifted and not force:
                raise SceneActivationError(
                    ActivationFailureKind.DRIFT,
                    f"Refusing to switch from scene {active.scene!r}: managed files have drifted.",
                    hint=(
                        "Review the changed paths, then use 'crossby scene clear --force' or "
                        "re-run 'crossby scene use' with --force."
                    ),
                    warnings=warnings,
                    drifted=drifted,
                )
            try:
                reverted_scope = outgoing if not explicitly_scoped else initial_scope
                revert_results = engine.clear_scene(project_root, tools=reverted_scope, force=force)
            except Exception as exc:
                raise SceneActivationError(
                    ActivationFailureKind.FAILED_REVERT,
                    f"Scene engine error while reverting {active.scene!r}: {exc}",
                    hint="The existing scene state was left intact; fix the error and retry.",
                    warnings=warnings,
                ) from exc
            if _has_error(revert_results):
                try:
                    reconcile_partial_clear_state(project_root, active, revert_results)
                except Exception as exc:
                    raise SceneActivationError(
                        ActivationFailureKind.FAILED_REVERT,
                        (
                            f"Could not preserve partial recovery state while reverting "
                            f"{active.scene!r}: {exc}"
                        ),
                        hint=(
                            "The outgoing scene may be only partially reverted. Restore write "
                            "access, inspect its scene state, then run "
                            "'crossby scene clear --force'."
                        ),
                        warnings=warnings,
                        results=revert_results,
                    ) from exc
                raise SceneActivationError(
                    ActivationFailureKind.FAILED_REVERT,
                    (
                        f"Could not fully revert {active.scene!r} — aborting; "
                        "remaining recovery state was retained."
                    ),
                    warnings=warnings,
                    results=revert_results,
                )
    else:
        outgoing = []

    outgoing_revocations = _recorded_revocations(active, outgoing)
    retained_revocations = _recorded_revocations(active)

    try:
        results = engine.apply_scene(resolved, project_root, force=force, tools=initial_scope)
    except Exception as exc:
        partial_results = exc.results if isinstance(exc, engine.SceneApplyError) else ()
        recovery_revocations = retained_revocations | _revoked_concerns(partial_results)
        recovery_recorded = _save_recovery_state(
            project_root,
            scene_name,
            scene,
            initial_scope,
            partial_results,
            shared_skill_scope=shared_skill_scope,
            active=active,
        )
        raise SceneActivationError(
            ActivationFailureKind.APPLY_EXCEPTION,
            f"Scene apply failed: {exc}",
            hint=(
                "Run 'crossby scene clear' to revert changes crossby recorded, then "
                "'crossby sync' to restore the previously removed hooks/permissions."
                if recovery_recorded and recovery_revocations
                else "'crossby scene clear' can revert changes crossby recorded."
                if recovery_recorded
                else "Recoverable scene state could not be recorded; inspect the ownership "
                "ledger and tool files before retrying."
            ),
            warnings=warnings,
            results=partial_results,
            recovery_recorded=recovery_recorded,
        ) from exc

    state = _build_state(
        project_root,
        scene_name,
        scene,
        initial_scope,
        results,
        shared_skill_scope=shared_skill_scope,
    )
    if active is not None and active.scene == scene_name:
        _merge_same_scene_tools(state, active, initial_scope)
    _inherit_revocations(state, active)
    try:
        save_scene_state(project_root, state)
    except Exception as exc:
        rollback_results: list[SyncResult] = []
        rollback_error: Exception | None = None
        try:
            rollback_results = engine.clear_scene(project_root, tools=initial_scope, force=True)
        except Exception as rollback_exc:
            rollback_error = rollback_exc
        engine_rolled_back = rollback_error is None and not _has_error(rollback_results)
        if engine_rolled_back:
            try:
                _restore_state_after_rollback(
                    project_root,
                    active,
                    reverted_scope,
                    shared_skill_scope=rollback_shared_skill_scope,
                )
            except Exception as cleanup_exc:
                rollback_error = cleanup_exc
        removed_concerns = outgoing_revocations | _revoked_concerns(results)
        rolled_back = engine_rolled_back and rollback_error is None and not removed_concerns
        if removed_concerns and engine_rolled_back and rollback_error is None:
            remaining = "/".join(sorted(removed_concerns))
            rollback_message = (
                f"The automatic rollback restored reversible changes, but removed {remaining} "
                "remain narrowed."
            )
            rollback_hint = (
                "The launch was aborted. Run 'crossby sync' to restore the removed "
                "hooks/permissions, then fix the state path before retrying."
            )
        elif removed_concerns:
            remaining = "/".join(sorted(removed_concerns))
            rollback_message = (
                f"The automatic rollback did not complete, and removed {remaining} remain narrowed."
            )
            rollback_hint = (
                "The launch was aborted. Run 'crossby sync' to restore removed "
                "hooks/permissions, then inspect .crossby/owned.json, "
                ".crossby/scene-state.json, and the tool files before retrying recovery."
            )
        elif rolled_back:
            rollback_message = "Persistent changes were rolled back."
            rollback_hint = "The launch was aborted; fix the state path before retrying."
        else:
            rollback_message = "The automatic rollback did not complete."
            rollback_hint = (
                "The launch was aborted. Inspect .crossby/owned.json, "
                ".crossby/scene-state.json, and the tool files before retrying recovery."
            )
        raise SceneActivationError(
            ActivationFailureKind.STATE_PERSISTENCE,
            f"Scene recoverable state could not be recorded: {exc}. {rollback_message}",
            hint=rollback_hint,
            warnings=warnings,
            results=(*results, *rollback_results),
        ) from exc

    status: Literal["applied", "partial"] = "partial" if _has_error(results) else "applied"
    return SceneActivationOutcome(tuple(scope), tuple(results), status, warnings)


def _has_error(results: Sequence[SyncResult]) -> bool:
    return any(result.action == "error" for result in results)


def reconcile_partial_clear_state(
    project_root: Path, active: SceneState, results: Sequence[SyncResult]
) -> None:
    """Persist only mechanisms that remain after an incomplete clear.

    PROJECT restoration is keyed by physical path, so one successful row clears
    every sharing tool's bookkeeping. DECLARE rows clear only their own tool and
    concern. Recovery-only revocation records are retained.
    """
    from crossby.config.skills import SKILLS_DIR
    from crossby.sync.agents import _AGENT_TARGET_PATHS

    registered_targets = {*SKILLS_DIR.values(), *_AGENT_TARGET_PATHS.values()}
    project_targets: list[tuple[str, str]] = []
    for result in results:
        if result.action == "error":
            continue
        concern = result.concern.value
        if result.file_path is not None and concern in ("skills", "agents"):
            try:
                rel = result.file_path.relative_to(project_root).as_posix()
            except ValueError:
                continue
            if rel in registered_targets:
                project_targets.append((concern, rel))
                continue
        if result.tool_id is None:
            continue
        record = active.tools.get(str(result.tool_id))
        if record is None or record.mechanisms.get(concern) != "declare":
            continue
        record.mechanisms.pop(concern, None)
        if result.file_path is not None:
            with contextlib.suppress(ValueError):
                record.hashes.pop(result.file_path.relative_to(project_root).as_posix(), None)

    for concern, target in project_targets:
        for tool_name, record in active.tools.items():
            try:
                tool = AIToolID(tool_name)
            except ValueError:
                continue
            tool_target = (
                SKILLS_DIR.get(tool)
                if concern == SyncConcern.SKILLS.value
                else _AGENT_TARGET_PATHS.get(tool_name)
            )
            if tool_target != target or record.mechanisms.get(concern) != "project":
                continue
            record.mechanisms.pop(concern, None)
            record.hashes.pop(target, None)

    # Unsupported cells never changed persistent state and need no recovery.
    for record in active.tools.values():
        record.mechanisms = {
            concern: mechanism
            for concern, mechanism in record.mechanisms.items()
            if mechanism != "unsupported"
        }

    for tool_name in list(active.tools):
        record = active.tools[tool_name]
        if record.mechanisms or record.hashes:
            continue
        if record.revoked_concerns:
            record.status = "recovery"
        else:
            active.tools.pop(tool_name)

    if not active.tools:
        clear_scene_state(project_root)
        return
    active.status = "partial"
    save_scene_state(project_root, active)


def _detect_scoped_drift(
    project_root: Path,
    active: SceneState,
    outgoing: Sequence[AIToolID],
    primary_scope: Sequence[AIToolID],
) -> list[str]:
    """Check primary tools fully and shared-only co-sharers for skills drift."""
    from crossby.config.skills import SKILLS_DIR

    primary = {str(tool) for tool in primary_scope}
    primary_outgoing = [str(tool) for tool in outgoing if str(tool) in primary]
    drifted = detect_drift(project_root, active, tools=primary_outgoing)

    for tool in outgoing:
        tool_name = str(tool)
        if tool_name in primary:
            continue
        shared_path = SKILLS_DIR.get(tool)
        if shared_path is None:
            continue
        expected = active.tools[tool_name].hashes.get(shared_path)
        if expected is not None and content_hash(project_root / shared_path) != expected:
            drifted.append(shared_path)
    return sorted(set(drifted))


def _recorded_revocations(
    active: SceneState | None, tools: Sequence[AIToolID] | None = None
) -> set[str]:
    """Return irreversible revocations for a tool scope, or every active record."""
    if active is None:
        return set()
    if tools is None:
        return {concern for record in active.tools.values() for concern in record.revoked_concerns}
    return {
        concern
        for tool in tools
        for concern in active.tools.get(str(tool), SceneToolRecord()).revoked_concerns
    }


def _revoked_concerns(results: Sequence[SyncResult]) -> set[str]:
    """Return irreversible hook/permission removals completed in result rows."""
    return {
        result.concern.value
        for result in results
        if result.concern.value in ("hooks", "permissions") and result.revoked > 0
    }


def _inherit_revocations(state: SceneState, active: SceneState | None) -> None:
    """Carry still-unrestored outgoing removals into the replacement state."""
    if active is None:
        return
    for tool, previous in active.tools.items():
        if not previous.revoked_concerns:
            continue
        record = state.tools.get(tool)
        if record is None:
            state.tools[tool] = SceneToolRecord(
                status="recovery",
                revoked_concerns=previous.revoked_concerns,
            )
            continue
        record.revoked_concerns = tuple(
            sorted({*record.revoked_concerns, *previous.revoked_concerns})
        )


def _merge_same_scene_tools(
    state: SceneState,
    active: SceneState,
    primary_scope: Sequence[AIToolID],
) -> None:
    """Merge shared-skill records without replacing their unrelated concerns."""
    primary = {str(tool) for tool in primary_scope}
    merged = dict(active.tools)
    for tool, record in state.tools.items():
        previous = active.tools.get(tool)
        if tool in primary or previous is None:
            merged[tool] = record
            continue
        unaffected = set(previous.mechanisms) - set(record.mechanisms)
        merged[tool] = SceneToolRecord(
            mechanisms={**previous.mechanisms, **record.mechanisms},
            status=("failed" if previous.status == "failed" and unaffected else record.status),
            hashes={**previous.hashes, **record.hashes},
            revoked_concerns=tuple(sorted({*previous.revoked_concerns, *record.revoked_concerns})),
        )
    state.tools = merged


def _restore_state_after_rollback(
    project_root: Path,
    active: SceneState | None,
    reverted_scope: Sequence[AIToolID],
    *,
    shared_skill_scope: Sequence[AIToolID],
) -> None:
    """Restore unaffected prior state after rolling back the attempted apply."""
    if active is None:
        clear_scene_state(project_root)
        return

    from crossby.config.skills import SKILLS_DIR

    rolled_back_tools = {str(tool) for tool in reverted_scope}
    shared_only = {
        str(tool): SKILLS_DIR.get(tool)
        for tool in shared_skill_scope
        if str(tool) not in rolled_back_tools
    }
    remaining_tools: dict[str, SceneToolRecord] = {}
    for tool, record in active.tools.items():
        if tool in rolled_back_tools:
            if record.revoked_concerns:
                remaining_tools[tool] = SceneToolRecord(
                    status="recovery",
                    revoked_concerns=record.revoked_concerns,
                )
            continue
        shared_path = shared_only.get(tool)
        if tool not in shared_only:
            remaining_tools[tool] = record
            continue
        mechanisms = dict(record.mechanisms)
        mechanisms.pop("skills", None)
        hashes = dict(record.hashes)
        if shared_path is not None:
            hashes.pop(shared_path, None)
        if mechanisms or hashes or record.revoked_concerns:
            remaining_tools[tool] = SceneToolRecord(
                mechanisms=mechanisms,
                status="recovery" if not mechanisms and record.revoked_concerns else record.status,
                hashes=hashes,
                revoked_concerns=record.revoked_concerns,
            )
    if not remaining_tools:
        clear_scene_state(project_root)
        return

    # ``save_scene_state`` writes before updating .gitignore, so a failed save
    # may have replaced the prior record already. Restore the unaffected tools
    # rather than deleting their recovery state along with the rolled-back scope.
    save_scene_state(
        project_root,
        SceneState(
            scene=active.scene,
            applied_at=active.applied_at,
            status=active.status,
            tools=remaining_tools,
        ),
    )


def _save_recovery_state(
    project_root: Path,
    scene_name: str,
    scene: SceneConfig,
    scope: list[AIToolID],
    results: Sequence[SyncResult],
    *,
    shared_skill_scope: list[AIToolID],
    active: SceneState | None,
) -> bool:
    """Best-effort partial state after an exceptional engine failure."""
    try:
        state = _build_state(
            project_root,
            scene_name,
            scene,
            scope,
            results,
            shared_skill_scope=shared_skill_scope,
        )
        state.status = "partial"
        if active is not None and active.scene == scene_name:
            _merge_same_scene_tools(state, active, scope)
        _inherit_revocations(state, active)
        save_scene_state(project_root, state)
    except Exception:
        return False
    return True


def _build_state(
    project_root: Path,
    scene_name: str,
    scene: SceneConfig,
    scope: list[AIToolID],
    results: Sequence[SyncResult],
    *,
    shared_skill_scope: list[AIToolID] | None = None,
) -> SceneState:
    tools = _tool_mechanisms(scene, scope, shared_skill_scope=shared_skill_scope)
    hashes_by_tool = compute_hashes(project_root, results)
    for tool_name, record in tools.items():
        record.hashes = hashes_by_tool.get(tool_name, {})
    _replicate_shared_hashes(tools, shared_skill_scope or scope)
    for result in results:
        if result.tool_id is None or str(result.tool_id) not in tools:
            continue
        record = tools[str(result.tool_id)]
        if result.action == "error":
            record.status = "failed"
        if result.concern.value in ("hooks", "permissions") and result.revoked > 0:
            record.revoked_concerns = tuple(
                sorted({*record.revoked_concerns, result.concern.value})
            )
    status = "partial" if _has_error(results) else "applied"
    return SceneState(scene=scene_name, applied_at=now_iso(), status=status, tools=tools)


def _replicate_shared_hashes(tools: dict[str, SceneToolRecord], scope: list[AIToolID]) -> None:
    from crossby.config.skills import SKILLS_DIR

    for tool in scope:
        shared_dir = SKILLS_DIR.get(tool)
        record = tools.get(str(tool))
        if shared_dir is None or record is None or shared_dir in record.hashes:
            continue
        for other in scope:
            other_record = tools.get(str(other))
            if other_record is not None and shared_dir in other_record.hashes:
                record.hashes[shared_dir] = other_record.hashes[shared_dir]
                break


def _tool_mechanisms(
    scene: SceneConfig,
    scope: list[AIToolID],
    *,
    shared_skill_scope: list[AIToolID] | None = None,
) -> dict[str, SceneToolRecord]:
    declared = [concern for concern in SCENE_CONCERNS if getattr(scene, concern) is not None]
    tools = {
        str(tool): SceneToolRecord(
            mechanisms={concern: base_mechanism(tool, concern).value for concern in declared}
        )
        for tool in scope
    }
    if scene.skills is not None:
        for tool in shared_skill_scope or scope:
            tools.setdefault(str(tool), SceneToolRecord()).mechanisms["skills"] = base_mechanism(
                tool, "skills"
            ).value
    return tools


__all__ = [
    "ActivationFailureKind",
    "SceneActivationError",
    "SceneActivationOutcome",
    "activate_scene",
    "expand_shared_scope",
    "reconcile_partial_clear_state",
    "recorded_tools",
]
