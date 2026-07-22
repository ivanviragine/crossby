"""Tests for plan/doctor summaries."""

from __future__ import annotations

from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncResult
from crossby.sync.plan import (
    build_doctor,
    doctor_readiness,
    render_doctor,
    render_plan,
    summarize_plan,
)
from crossby.sync.validate import ValidationFinding


def _result(
    action: str = "created",
    message: str | None = None,
    concern: SyncConcern = SyncConcern.RULES,
    tool_id: AIToolID | None = AIToolID.CLAUDE,
) -> SyncResult:
    return SyncResult(
        tool_id=tool_id,
        concern=concern,
        action=action,  # type: ignore[arg-type]
        file_path=Path("x"),
        message=message,
    )


class TestSummarize:
    def test_empty(self) -> None:
        s = summarize_plan([])
        assert s.total == 0
        assert s.error_count == 0
        assert s.manual_fix_count == 0

    def test_aggregates_actions_and_concerns(self) -> None:
        results = [
            _result(action="created", concern=SyncConcern.RULES),
            _result(action="created", concern=SyncConcern.MCP),
            _result(action="skipped", concern=SyncConcern.MCP),
        ]
        s = summarize_plan(results)
        assert s.by_action == {"created": 2, "skipped": 1}
        assert s.by_concern == {SyncConcern.RULES: 1, SyncConcern.MCP: 2}
        assert s.total == 3

    def test_manual_fix_detected_from_message(self) -> None:
        results = [
            _result(action="created", message="foreign markers in source"),
            _result(action="updated", message="translated"),
            _result(action="created", message="copy"),
        ]
        s = summarize_plan(results)
        assert s.manual_fix_count == 2

    def test_skipped_with_translate_message_not_counted(self) -> None:
        # A skipped row that mentions "translated" means the translation
        # was already in place — not a fresh manual-fix item.
        results = [_result(action="skipped", message="already translated")]
        s = summarize_plan(results)
        assert s.manual_fix_count == 0

    def test_error_count(self) -> None:
        results = [_result(action="error"), _result(action="created")]
        s = summarize_plan(results)
        assert s.error_count == 1

    def test_unaddressed_mcp_report_counts_as_manual_fix(self) -> None:
        # Regression: crossby.sync.mcp_discovery.report_oauth_configs() emits
        # action="skipped", file_path=None rows (same shape as plugins, but
        # sharing SyncConcern.MCP with regular writer rows, so it can't use
        # a concern-wide carve-out). file_path=None + a manual-fix hint in
        # the message must count even though the action isn't a write.
        results = [
            SyncResult(
                tool_id=None,
                concern=SyncConcern.MCP,
                action="skipped",
                file_path=None,
                message="MCP server `x` has an oauth block; this is a manual-fix.",
            )
        ]
        s = summarize_plan(results)
        assert s.manual_fix_count == 1

    def test_idempotent_skip_with_real_target_not_counted(self) -> None:
        # Contrast case: a real "already handled" skip has file_path set to
        # the target it left alone, even if the message happens to mention
        # "translated" — it must NOT count as fresh manual-review.
        results = [_result(action="skipped", message="already translated")]
        s = summarize_plan(results)
        assert s.manual_fix_count == 0

    def test_plugin_finding_counts_as_manual_fix(self) -> None:
        # Regression: crossby.sync.plugins.report_plugins() always emits
        # action="skipped" rows (never created/updated), so they used to
        # fall outside _WRITING_ACTIONS and silently drop out of the
        # manual-fix tally — meaning --doctor could report readiness: high
        # while an unmigrated Claude plugin sat in the project.
        results = [
            _result(
                action="skipped",
                concern=SyncConcern.PLUGINS,
                tool_id=None,
                message="plugin `team-macros`: needs manual migration",
            )
        ]
        s = summarize_plan(results)
        assert s.manual_fix_count == 1


class TestRenderPlan:
    def test_empty_plan(self) -> None:
        out = render_plan(summarize_plan([]))
        assert "no sync rows" in out

    def test_renders_counts(self) -> None:
        results = [
            _result(action="created", concern=SyncConcern.RULES),
            _result(action="updated", concern=SyncConcern.MCP),
        ]
        out = render_plan(summarize_plan(results))
        assert "total rows: 2" in out
        assert "1 created" in out
        assert "1 updated" in out

    def test_renders_manual_fix_items(self) -> None:
        results = [
            _result(
                action="created",
                concern=SyncConcern.RULES,
                tool_id=AIToolID.ANTIGRAVITY_CLI,
                message="foreign markers in source",
            )
        ]
        out = render_plan(summarize_plan(results))
        assert "manual review: 1 item" in out
        assert "antigravity-cli" in out
        assert "rules" in out

    def test_renders_no_manual_review_when_clean(self) -> None:
        out = render_plan(summarize_plan([_result(action="created")]))
        assert "manual review: none" in out


class TestDoctorReadiness:
    def test_high_when_clean(self) -> None:
        plan = summarize_plan([_result(action="created")])
        assert doctor_readiness(plan, []) == "high"

    def test_low_on_any_error(self) -> None:
        plan = summarize_plan([_result(action="error")])
        assert doctor_readiness(plan, []) == "low"

    def test_low_on_validation_error(self) -> None:
        plan = summarize_plan([])
        validation = [
            ValidationFinding(
                tool_id=None,
                concern=None,
                level="error",
                path=Path("x"),
                detail="bad",
            )
        ]
        assert doctor_readiness(plan, validation) == "low"

    def test_medium_with_one_to_three_actionable(self) -> None:
        plan = summarize_plan(
            [
                _result(action="created", message="foreign markers in source"),
                _result(action="updated", message="translated"),
            ]
        )
        assert doctor_readiness(plan, []) == "medium"

    def test_low_when_more_than_three_actionable(self) -> None:
        plan = summarize_plan(
            [
                _result(action="created", message="foreign markers in source"),
                _result(action="updated", message="translated"),
                _result(action="created", message="manual_fix"),
                _result(action="updated", message="translated"),
            ]
        )
        assert doctor_readiness(plan, []) == "low"


class TestBuildAndRenderDoctor:
    def test_basic_render(self) -> None:
        plan = summarize_plan([_result(action="created", message="foreign markers in source")])
        validation = [
            ValidationFinding(
                tool_id=AIToolID.CODEX,
                concern=SyncConcern.MCP,
                level="warning",
                path=Path(".codex/config.toml"),
                detail="MCP `x` not on PATH",
            )
        ]
        report = build_doctor(plan, validation)
        out = render_doctor(report)
        assert "readiness:" in out
        assert "manual-review items: 1" in out
        assert "validation warnings: 1" in out
        assert "recommended flow" in out
