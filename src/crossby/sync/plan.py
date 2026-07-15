"""Plan and doctor summaries derived from sync results + validators.

``crossby sync --plan`` and ``crossby sync --doctor`` are pre-write
inspection modes. Both ride on a dry-run sync execution; this module
just aggregates the :class:`SyncResult` rows into a stage-by-concern
summary, counts manual-fix and error situations, and turns that into
the readiness score the doctor mode renders.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from crossby.sync.base import SyncConcern, SyncResult
from crossby.sync.validate import ValidationFinding

Readiness = Literal["high", "medium", "low"]


# Heuristic: which sync result messages indicate a manual-fix block
# was injected into the artifact. ``foreign markers`` comes from the
# rules writer; ``translated`` from agents/skills writers. Manual-fix
# blocks are only counted when the action made a write — a "skipped"
# row that mentions translation just means the translation was already
# in place.
_MANUAL_FIX_HINTS = ("foreign markers", "translated", "manual_fix", "manual-fix")
_WRITING_ACTIONS = {"created", "updated"}


@dataclass(frozen=True)
class PlanSummary:
    """Aggregated view of a dry-run :func:`run_sync` invocation.

    ``by_concern`` and ``by_action`` are sparse — only concerns / actions
    that actually appeared in the results show up. ``manual_fix_results``
    holds the original :class:`SyncResult` rows whose message looked like
    a manual-fix annotation, so the renderer can list them with full
    context.
    """

    by_concern: dict[SyncConcern, int]
    by_action: dict[str, int]
    manual_fix_results: list[SyncResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.by_action.values())

    @property
    def error_count(self) -> int:
        return self.by_action.get("error", 0)

    @property
    def manual_fix_count(self) -> int:
        return len(self.manual_fix_results)


def summarize_plan(results: Sequence[SyncResult]) -> PlanSummary:
    """Aggregate dry-run sync results into a :class:`PlanSummary`.

    ``SyncConcern.PLUGINS`` rows always count as manual-review, regardless
    of action or message content: :func:`crossby.sync.plugins.report_plugins`
    only ever emits a row when it found a plugin/marketplace Crossby can't
    migrate, so by construction every such row is undone work. Without this,
    plugin findings (always ``action="skipped"``) never matched
    ``_WRITING_ACTIONS`` and silently dropped out of the doctor readiness
    score.

    Detect-only reporters that share a concern with regular writers (e.g.
    :func:`crossby.sync.mcp_discovery.report_oauth_configs`, concern
    ``MCP``) can't use the same concern-wide carve-out — most ``MCP`` rows
    are normal writes. They're instead recognized by ``file_path is None``
    plus a manual-fix hint in the message: a *real* idempotent "already
    handled" skip always has ``file_path`` set to the target it left alone,
    so ``file_path is None`` here means "this row never had a target to
    write, it's pure detection" — the same signal
    :func:`crossby.sync.report.classify_status` uses for ``Not Added``.
    """
    by_concern: dict[SyncConcern, int] = {}
    by_action: dict[str, int] = {}
    manual_fix: list[SyncResult] = []

    for result in results:
        by_concern[result.concern] = by_concern.get(result.concern, 0) + 1
        by_action[result.action] = by_action.get(result.action, 0) + 1
        is_plugin_finding = result.concern == SyncConcern.PLUGINS
        has_manual_fix_hint = bool(
            result.message and any(hint in result.message.lower() for hint in _MANUAL_FIX_HINTS)
        )
        is_unaddressed_report = (
            result.action == "skipped" and result.file_path is None and has_manual_fix_hint
        )
        if (
            is_plugin_finding
            or is_unaddressed_report
            or (result.action in _WRITING_ACTIONS and has_manual_fix_hint)
        ):
            manual_fix.append(result)

    return PlanSummary(
        by_concern=by_concern,
        by_action=by_action,
        manual_fix_results=manual_fix,
    )


def render_plan(summary: PlanSummary) -> str:
    """Build a human-readable plan summary block."""
    lines: list[str] = ["Migration plan:"]
    if summary.total == 0:
        lines.append("  (no sync rows produced — check source/target/concern flags)")
        return "\n".join(lines)

    lines.append(f"  total rows: {summary.total}")
    if summary.by_action:
        action_summary = ", ".join(
            f"{count} {action}" for action, count in sorted(summary.by_action.items())
        )
        lines.append(f"  actions:    {action_summary}")
    if summary.by_concern:
        concern_summary = ", ".join(
            f"{count} {concern.value}"
            for concern, count in sorted(summary.by_concern.items(), key=lambda kv: kv[0].value)
        )
        lines.append(f"  concerns:   {concern_summary}")
    if summary.manual_fix_results:
        lines.append(f"  manual review: {summary.manual_fix_count} item(s)")
        for result in summary.manual_fix_results:
            tool = str(result.tool_id) if result.tool_id is not None else "crossby"
            path = result.file_path.as_posix() if result.file_path else "(no path)"
            detail = result.message or ""
            lines.append(f"    - [{tool}] {result.concern.value} {path}: {detail}")
    else:
        lines.append("  manual review: none")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorReport:
    """Readiness rating + the inputs that produced it."""

    readiness: Readiness
    plan: PlanSummary
    validation: list[ValidationFinding]

    @property
    def validation_errors(self) -> int:
        return sum(1 for f in self.validation if f.level == "error")

    @property
    def validation_warnings(self) -> int:
        return sum(1 for f in self.validation if f.level == "warning")


def doctor_readiness(plan: PlanSummary, validation: Sequence[ValidationFinding]) -> Readiness:
    """Map plan + validation counts to a coarse readiness rating.

    Static thresholds:

    - ``high`` when nothing demands attention.
    - ``medium`` when there are 1-3 actionable items.
    - ``low`` otherwise (or whenever an error is present).
    """
    error_total = plan.error_count + sum(1 for f in validation if f.level == "error")
    actionable = (
        error_total + plan.manual_fix_count + sum(1 for f in validation if f.level == "warning")
    )
    if error_total > 0:
        return "low"
    if actionable == 0:
        return "high"
    if actionable <= 3:
        return "medium"
    return "low"


def build_doctor(plan: PlanSummary, validation: Sequence[ValidationFinding]) -> DoctorReport:
    return DoctorReport(
        readiness=doctor_readiness(plan, validation),
        plan=plan,
        validation=list(validation),
    )


def render_doctor(report: DoctorReport) -> str:
    lines: list[str] = ["Crossby doctor:", f"  readiness: {report.readiness}"]
    lines.append(f"  manual-review items: {report.plan.manual_fix_count}")
    lines.append(f"  sync errors: {report.plan.error_count}")
    lines.append(f"  validation errors: {report.validation_errors}")
    lines.append(f"  validation warnings: {report.validation_warnings}")
    if report.plan.manual_fix_results:
        lines.append("  manual review:")
        for result in report.plan.manual_fix_results:
            tool = str(result.tool_id) if result.tool_id is not None else "crossby"
            path = result.file_path.as_posix() if result.file_path else "(no path)"
            lines.append(f"    - [{tool}] {result.concern.value} {path}: {result.message}")
    if report.validation_errors or report.validation_warnings:
        lines.append("  validation:")
        for f in report.validation:
            if f.level == "ok":
                continue
            tool = str(f.tool_id) if f.tool_id is not None else "crossby"
            lines.append(f"    - [{tool}] {f.level}: {f.path} — {f.detail}")
    lines.append(
        "  recommended flow: --plan, --dry-run, fix manual items, sync, --validate-target."
    )
    return "\n".join(lines)


__all__ = [
    "DoctorReport",
    "PlanSummary",
    "Readiness",
    "build_doctor",
    "doctor_readiness",
    "render_doctor",
    "render_plan",
    "summarize_plan",
]
