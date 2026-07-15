"""Tests for sync report rendering and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from crossby.models.ai import AIToolID
from crossby.sync.base import SyncConcern, SyncResult
from crossby.sync.report import (
    REPORT_PATH,
    classify_status,
    render_markdown_table,
    render_persistent_report,
    write_persistent_report,
)


def _r(
    action: str = "created",
    message: str | None = None,
    concern: SyncConcern = SyncConcern.RULES,
    tool_id: AIToolID | None = AIToolID.CLAUDE,
    file_path: Path | None = Path("CLAUDE.md"),
) -> SyncResult:
    return SyncResult(
        tool_id=tool_id,
        concern=concern,
        action=action,  # type: ignore[arg-type]
        file_path=file_path,
        message=message,
    )


class TestClassifyStatus:
    def test_created_added(self) -> None:
        assert classify_status(_r(action="created")) == "Added"

    def test_updated_added(self) -> None:
        assert classify_status(_r(action="updated")) == "Added"

    def test_check_before_using_on_foreign_markers(self) -> None:
        assert (
            classify_status(_r(action="created", message="foreign markers in source"))
            == "Check before using"
        )

    def test_check_before_using_on_translated(self) -> None:
        assert classify_status(_r(action="updated", message="translated")) == "Check before using"

    def test_skipped_with_no_file_path_is_not_added(self) -> None:
        # "no source detected" / "no hooks config" / etc. — writer never
        # identified a target artifact. No file_path is the canonical signal.
        assert (
            classify_status(
                _r(
                    action="skipped",
                    message="no rules source detected",
                    file_path=None,
                )
            )
            == "Not Added"
        )

    def test_skipped_no_hooks_config_is_not_added(self) -> None:
        # Regression: the "no hooks config" message previously fell through
        # to the default "Added" classification because the heuristic only
        # matched messages mentioning "source".
        assert (
            classify_status(_r(action="skipped", message="no hooks config", file_path=None))
            == "Not Added"
        )

    def test_skipped_when_already_in_place_is_added(self) -> None:
        # file_path set + skipped == idempotent re-run.
        assert classify_status(_r(action="skipped", message="already linked")) == "Added"

    def test_error_is_not_added(self) -> None:
        assert classify_status(_r(action="error", message="boom")) == "Not Added"

    def test_mcp_oauth_report_is_not_added(self) -> None:
        # crossby.sync.mcp_discovery.report_oauth_configs() rows: skipped,
        # file_path=None — already covered by the general "skipped + no
        # file_path" rule, unlike the plugins case above. Locked in here
        # under its own name so a future refactor of either function trips
        # this test instead of only the general one.
        assert (
            classify_status(
                _r(
                    action="skipped",
                    concern=SyncConcern.MCP,
                    tool_id=None,
                    file_path=None,
                    message="MCP server `x` has an oauth block; this is a manual-fix.",
                )
            )
            == "Not Added"
        )

    def test_plugin_finding_is_not_added_even_with_file_path(self) -> None:
        # Regression: crossby.sync.plugins.report_plugins() always emits
        # action="skipped" with file_path set to the *undone* source plugin
        # path (not a target artifact already in place), which used to
        # collide with the general "skipped + file_path set == Added"
        # heuristic and mislabel unmigrated plugins as "Added".
        assert (
            classify_status(
                _r(
                    action="skipped",
                    concern=SyncConcern.PLUGINS,
                    tool_id=None,
                    file_path=Path(".claude/plugins/team-macros"),
                    message="plugin `team-macros`: needs manual migration",
                )
            )
            == "Not Added"
        )


class TestRenderMarkdownTable:
    def test_empty_when_no_results(self) -> None:
        assert render_markdown_table([]) == ""

    def test_renders_header(self) -> None:
        out = render_markdown_table([_r()])
        assert "| Status | Item | Notes |" in out
        assert "| --- | --- | --- |" in out

    def test_uses_singular_concern_label(self) -> None:
        out = render_markdown_table([_r(concern=SyncConcern.AGENTS)])
        assert "`Agent`" in out

    def test_status_value_quoted(self) -> None:
        out = render_markdown_table([_r(action="created", message="x")])
        assert "`Added`" in out

    def test_pipes_in_messages_escaped(self) -> None:
        out = render_markdown_table([_r(action="created", message="contains | pipe")])
        assert r"\|" in out

    def test_em_dash_for_empty_message(self) -> None:
        out = render_markdown_table([_r(action="created", message=None)])
        assert "—" in out


class TestRenderPersistentReport:
    def test_includes_timestamp_and_project_name(self) -> None:
        ts = datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC)
        out = render_persistent_report([_r()], project_name="northstar", timestamp=ts)
        assert "2026-05-04T12:00:00Z" in out
        assert "northstar" in out

    def test_no_rows_says_nothing_was_synced(self) -> None:
        out = render_persistent_report([], project_name="x")
        assert "No sync rows" in out

    def test_paths_relativized_against_project_root(self) -> None:
        # Regression: persistent reports used to leak absolute paths
        # (/tmp/<project>/.cursor/cli.json) which makes the report
        # non-portable. With project_root supplied, paths render relative.
        result = _r(
            action="created",
            file_path=Path("/tmp/proj/.cursor/cli.json"),
        )
        out = render_persistent_report(
            [result],
            project_name="proj",
            project_root=Path("/tmp/proj"),
        )
        assert ".cursor/cli.json" in out
        assert "/tmp/proj/.cursor/cli.json" not in out


class TestRenderMarkdownTableRelativePaths:
    def test_passes_project_root_through(self) -> None:
        result = _r(
            action="created",
            file_path=Path("/tmp/proj/.cursor/cli.json"),
        )
        out = render_markdown_table([result], project_root=Path("/tmp/proj"))
        assert "/tmp/proj/" not in out
        assert ".cursor/cli.json" in out

    def test_no_project_root_keeps_absolute_path(self) -> None:
        result = _r(
            action="created",
            file_path=Path("/tmp/proj/.cursor/cli.json"),
        )
        out = render_markdown_table([result])
        assert "/tmp/proj/.cursor/cli.json" in out

    def test_outside_project_root_keeps_absolute(self) -> None:
        # If the file_path isn't under project_root, leave it as-is.
        result = _r(
            action="created",
            file_path=Path("/etc/global-config.json"),
        )
        out = render_markdown_table([result], project_root=Path("/tmp/proj"))
        assert "/etc/global-config.json" in out


class TestWritePersistentReport:
    def test_writes_to_default_path(self, tmp_path: Path) -> None:
        path = write_persistent_report([_r()], tmp_path)
        assert path == tmp_path / REPORT_PATH
        assert path.is_file()
        body = path.read_text(encoding="utf-8")
        assert "crossby sync report" in body
        assert "Status" in body

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        # .crossby/ may not exist yet.
        write_persistent_report([_r()], tmp_path)
        assert (tmp_path / ".crossby").is_dir()

    def test_overwrites_on_re_run(self, tmp_path: Path) -> None:
        write_persistent_report([_r(action="created")], tmp_path)
        write_persistent_report([_r(action="updated")], tmp_path)
        body = (tmp_path / REPORT_PATH).read_text(encoding="utf-8")
        # Only the second run's row should be present.
        assert body.count("Added") == 1
