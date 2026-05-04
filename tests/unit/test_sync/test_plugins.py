"""Tests for Claude plugin / marketplace discovery."""

from __future__ import annotations

import json
from pathlib import Path

from crossby.sync.base import SyncConcern
from crossby.sync.plugins import (
    PluginFinding,
    discover_plugins,
    findings_to_results,
    has_plugin_findings,
    report_plugins,
)


class TestDiscoverPlugins:
    def test_empty_when_nothing_present(self, tmp_path: Path) -> None:
        assert discover_plugins(tmp_path) == []

    def test_finds_plugin_subdirs(self, tmp_path: Path) -> None:
        plugins = tmp_path / ".claude" / "plugins"
        (plugins / "team-macros").mkdir(parents=True)
        (plugins / "release-helper").mkdir(parents=True)
        findings = discover_plugins(tmp_path)
        labels = [f.label for f in findings]
        assert any("team-macros" in label for label in labels)
        assert any("release-helper" in label for label in labels)

    def test_empty_plugins_dir_still_reports(self, tmp_path: Path) -> None:
        (tmp_path / ".claude" / "plugins").mkdir(parents=True)
        findings = discover_plugins(tmp_path)
        assert len(findings) == 1
        assert "empty" in findings[0].detail.lower()

    def test_marketplace_registry_reported(self, tmp_path: Path) -> None:
        marketplace = tmp_path / ".claude" / "plugin-marketplaces.json"
        marketplace.parent.mkdir(parents=True)
        marketplace.write_text(json.dumps({}), encoding="utf-8")
        findings = discover_plugins(tmp_path)
        assert findings
        assert any("registry" in f.label.lower() for f in findings)

    def test_marketplace_manifest_with_named_plugins(self, tmp_path: Path) -> None:
        manifest = tmp_path / ".claude-plugin" / "marketplace.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"plugins": [{"name": "alpha"}, {"name": "beta"}]}),
            encoding="utf-8",
        )
        findings = discover_plugins(tmp_path)
        names = [f.label for f in findings]
        assert any("alpha" in n for n in names)
        assert any("beta" in n for n in names)

    def test_invalid_json_marketplace_still_reports_file(
        self, tmp_path: Path
    ) -> None:
        manifest = tmp_path / ".claude" / "plugin-marketplaces.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{ broken json", encoding="utf-8")
        findings = discover_plugins(tmp_path)
        assert findings


class TestFindingsToResults:
    def test_each_finding_becomes_skipped_row(self) -> None:
        findings = [
            PluginFinding(path=Path("x"), label="a", detail="b"),
            PluginFinding(path=Path("y"), label="c", detail="d"),
        ]
        results = findings_to_results(findings)
        assert len(results) == 2
        for r in results:
            assert r.concern == SyncConcern.PLUGINS
            assert r.action == "skipped"
            assert r.tool_id is None

    def test_message_combines_label_and_detail(self) -> None:
        finding = PluginFinding(path=Path("x"), label="lbl", detail="why")
        result = findings_to_results([finding])[0]
        assert "lbl" in (result.message or "")
        assert "why" in (result.message or "")


class TestReportPlugins:
    def test_returns_results_for_each_finding(self, tmp_path: Path) -> None:
        (tmp_path / ".claude" / "plugins" / "team").mkdir(parents=True)
        results = report_plugins(tmp_path)
        assert results
        assert all(r.concern == SyncConcern.PLUGINS for r in results)


class TestHasPluginFindings:
    def test_true_when_any(self) -> None:
        from crossby.sync.base import SyncResult

        results = [
            SyncResult(
                tool_id=None,
                concern=SyncConcern.PLUGINS,
                action="skipped",
                file_path=Path("x"),
                message="m",
            )
        ]
        assert has_plugin_findings(results) is True

    def test_false_when_none(self) -> None:
        assert has_plugin_findings([]) is False
