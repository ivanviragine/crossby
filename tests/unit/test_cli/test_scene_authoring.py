"""Integration tests for the scene authoring commands.

Covers create (wizard + flag-driven), add/remove, delete, install-starters,
the non-TTY guard, ``--print``, and byte-preservation through the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from crossby.cli.main import app
from crossby.models.ai import AIToolID
from crossby.scenes.state import SceneState, SceneToolRecord, now_iso, save_scene_state
from tests.unit.test_scenes.conftest import populate_project

runner = CliRunner()

INSTALLED = [AIToolID.CLAUDE, AIToolID.CODEX, AIToolID.ANTIGRAVITY_CLI]

# A config carrying comments and sibling sections a splice must never disturb.
CONFIG = (
    "# header comment\n"
    "version: 1\n"
    "\n"
    "# AI launch defaults — keep me\n"
    "ai:\n"
    "  default_tool: claude   # inline comment\n"
    "\n"
    "profiles:\n"
    "  ccyolo:\n"
    "    tool: claude\n"
    "\n"
    "scenes:\n"
    "  base:\n"
    "    skills:\n"
    '      exclude: ["deploy-*"]\n'
)


@pytest.fixture(autouse=True)
def _pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crossby.ai_tools.base.AbstractAITool.detect_installed",
        classmethod(lambda _cls: list(INSTALLED)),
    )
    monkeypatch.setattr("crossby.scenes.versioning.detect_tool_version", lambda _tool: (2, 1, 218))
    monkeypatch.setattr("crossby.scenes.trust.codex_trusts_project", lambda *a, **k: True)
    # Non-interactive by default; wizard tests flip this back on.
    monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: False)


def _project(tmp_path: Path, body: str = CONFIG) -> Path:
    populate_project(tmp_path)
    (tmp_path / ".crossby.yml").write_text(body, encoding="utf-8")
    return tmp_path


def _run(args: list[str], root: Path) -> Result:
    return runner.invoke(app, [*args, "--path", str(root)])


def _scenes(root: Path) -> dict[str, object]:
    data = yaml.safe_load((root / ".crossby.yml").read_text(encoding="utf-8"))
    return data["scenes"]


# ---------------------------------------------------------------------------
# create — flag-driven (non-TTY) and the guard
# ---------------------------------------------------------------------------


class TestCreateNonInteractive:
    def test_flags_write_a_resolvable_scene(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _run(
            [
                "scene",
                "create",
                "pr-review",
                "--skill",
                "review-*",
                "--skill",
                "knowledge",
                "--agent",
                "code-reviewer",
                "--exclude-mcp",
                "linear",
                "--description",
                "Review a pull request",
                "--profile",
                "ccyolo",
            ],
            root,
        )
        assert result.exit_code == 0, result.output
        scene = _scenes(root)["pr-review"]
        assert scene["skills"]["include"] == ["review-*", "knowledge"]
        assert scene["mcp"]["exclude"] == ["linear"]
        assert scene["profile"] == "ccyolo"
        # It resolves through the real pipeline.
        show = _run(["scene", "show", "pr-review"], root)
        assert show.exit_code == 0, show.output
        assert "review-skill" in show.output

    def test_non_tty_without_selectors_fails_and_writes_nothing(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        result = _run(["scene", "create", "empty"], root)
        assert result.exit_code != 0
        assert "TTY" in result.output
        # Never the all-items scene multi_select would return.
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before
        assert "empty" not in _scenes(root)

    def test_refuses_existing_name(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _run(["scene", "create", "base", "--skill", "x"], root)
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_extends_round_trips(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _run(
            ["scene", "create", "child", "--extends", "base", "--agent", "code-reviewer"],
            root,
        )
        assert result.exit_code == 0, result.output
        assert _scenes(root)["child"]["extends"] == "base"

    def test_undefined_extends_is_rejected_and_rolled_back(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        result = _run(["scene", "create", "child", "--extends", "ghost", "--skill", "x"], root)
        assert result.exit_code != 0
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before

    def test_byte_preservation_through_cli(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _run(["scene", "create", "pr-review", "--skill", "review-*"], root)
        text = (root / ".crossby.yml").read_text(encoding="utf-8")
        for fragment in ("# header comment", "# AI launch defaults — keep me", "# inline comment"):
            assert fragment in text
        assert _scenes(root)["base"]["skills"]["exclude"] == ["deploy-*"]

    def test_print_leaves_file_untouched(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        result = _run(["scene", "create", "pr-review", "--skill", "review-*", "--print"], root)
        assert result.exit_code == 0
        assert "pr-review:" in result.output and "review-*" in result.output
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# add / remove
# ---------------------------------------------------------------------------


class TestAddRemove:
    def test_add_appends_and_is_idempotent(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _run(["scene", "add", "base", "--skill", "review-*"], root)
        _run(["scene", "add", "base", "--skill", "review-*"], root)
        assert _scenes(root)["base"]["skills"]["include"].count("review-*") == 1

    def test_add_cross_channel_move_reported(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        # base already excludes deploy-*; add it to include -> move
        result = _run(["scene", "add", "base", "--skill", "deploy-*"], root)
        assert result.exit_code == 0, result.output
        assert "moved from exclude to include" in result.output
        skills = _scenes(root)["base"]["skills"]
        assert "deploy-*" in skills["include"]
        assert "deploy-*" not in skills.get("exclude", [])

    def test_add_unknown_scene_fails(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _run(["scene", "add", "ghost", "--skill", "x"], root)
        assert result.exit_code != 0
        assert "Unknown scene" in result.output

    def test_remove_is_idempotent_noop(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _run(["scene", "remove", "base", "--exclude-skill", "not-there"], root)
        assert result.exit_code == 0
        assert "not present" in result.output

    def test_remove_exclude_round_trip(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _run(["scene", "remove", "base", "--exclude-skill", "deploy-*"], root)
        # base's only selector was that exclude; the concern collapses away.
        assert "skills" not in _scenes(root)["base"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_entry(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _run(["scene", "create", "pr-review", "--skill", "review-*"], root)
        result = _run(["scene", "delete", "pr-review"], root)
        assert result.exit_code == 0, result.output
        assert "pr-review" not in _scenes(root)

    def test_delete_active_refused_without_force(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        save_scene_state(
            root,
            SceneState(
                scene="base",
                applied_at=now_iso(),
                status="applied",
                tools={"claude": SceneToolRecord(mechanisms={"skills": "declare"})},
            ),
        )
        result = _run(["scene", "delete", "base"], root)
        assert result.exit_code != 0
        assert "active" in result.output.lower()
        assert "clear" in result.output
        # --force overrides
        forced = _run(["scene", "delete", "base", "--force"], root)
        assert forced.exit_code == 0, forced.output

    def test_delete_refuses_when_extended(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _run(["scene", "create", "child", "--extends", "base", "--agent", "code-reviewer"], root)
        result = _run(["scene", "delete", "base"], root)
        assert result.exit_code != 0
        assert "extend" in result.output

    def test_delete_unknown_fails(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        result = _run(["scene", "delete", "ghost"], root)
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# install-starters
# ---------------------------------------------------------------------------


class TestInstallStarters:
    def test_install_into_empty_config(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        (tmp_path / ".crossby.yml").write_text("version: 1\n", encoding="utf-8")
        result = _run(["scene", "install-starters"], tmp_path)
        assert result.exit_code == 0, result.output
        scenes = _scenes(tmp_path)
        assert {"pr-review", "deploy-watch", "write-docs", "presentation"} <= set(scenes)

    def test_starters_keep_same_named_user_scene(self, tmp_path: Path) -> None:
        body = CONFIG + (
            "  pr-review:\n"
            "    description: my own review scene\n"
            "    skills:\n"
            '      include: ["knowledge"]\n'
        )
        root = _project(tmp_path, body)
        result = _run(["scene", "install-starters"], root)
        assert result.exit_code == 0, result.output
        assert "Skipped 'pr-review'" in result.output
        assert _scenes(root)["pr-review"]["description"] == "my own review scene"

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        (tmp_path / ".crossby.yml").write_text("version: 1\n", encoding="utf-8")
        _run(["scene", "install-starters"], tmp_path)
        again = _run(["scene", "install-starters"], tmp_path)
        assert "already present" in again.output

    def test_installed_starter_resolves_with_warnings_not_error(self, tmp_path: Path) -> None:
        # A project with none of the presentation starter's named skills.
        populate_project(tmp_path)
        (tmp_path / ".crossby.yml").write_text("version: 1\n", encoding="utf-8")
        _run(["scene", "install-starters"], tmp_path)
        show = _run(["scene", "show", "presentation"], tmp_path)
        assert show.exit_code == 0, show.output
        assert "matched no detected item" in show.output


# ---------------------------------------------------------------------------
# wizard (interactive) — patched prompts
# ---------------------------------------------------------------------------


class TestWizard:
    def test_wizard_builds_scene_from_multiselect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path)
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: True)
        # Pick the first item of every concern offered.
        monkeypatch.setattr("crossby.ui.prompts.multi_select", lambda _t, items: [0])
        # Every single-select (extends/profile pickers + confirm menu) -> index 0.
        monkeypatch.setattr("crossby.ui.prompts.select", lambda *a, **k: 0)
        # Description prompt returns empty.
        monkeypatch.setattr("crossby.ui.prompts.input_prompt", lambda *a, **k: "")

        result = _run(["scene", "create", "wiztest", "--agent", "code-reviewer"], root)
        assert result.exit_code == 0, result.output
        scene = _scenes(root)["wiztest"]
        # A concern was chosen via multi-select, and the seed flag folded in.
        assert scene["agents"]["include"] == ["code-reviewer"]
        assert "skills" in scene


# ---------------------------------------------------------------------------
# rollback on a corrupt render
# ---------------------------------------------------------------------------


class TestRollback:
    def test_corrupt_render_restores_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        monkeypatch.setattr(
            "crossby.scenes.authoring.splice_scene_text",
            lambda *_a, **_k: "version: 1\nscenes:\n  x:\n  bad: [unclosed\n",
        )
        result = _run(["scene", "create", "boom", "--skill", "review-*"], root)
        assert result.exit_code != 0
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before
