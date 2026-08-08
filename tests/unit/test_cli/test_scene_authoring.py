"""Integration tests for the scene authoring commands.

Covers create (wizard + flag-driven), add/remove, delete, install-starters,
the non-TTY guard, ``--print``, and byte-preservation through the CLI.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
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

    def test_add_print_suppresses_move_notice(self, tmp_path: Path) -> None:
        # A contradictory add triggers a cross-channel move; --print must keep the
        # move notice off stdout so the streamed block stays valid YAML.
        root = _project(tmp_path)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        result = _run(["scene", "add", "base", "--skill", "deploy-*", "--print"], root)
        assert result.exit_code == 0, result.output
        assert "moved" not in result.output
        parsed = yaml.safe_load(result.output)
        assert isinstance(parsed, dict)
        assert "deploy-*" in parsed["base"]["skills"]["include"]
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before

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
        # --force overrides, but warns that state still records the deleted scene.
        forced = _run(["scene", "delete", "base", "--force"], root)
        assert forced.exit_code == 0, forced.output
        assert "scene-state.json" in forced.output
        assert "clear" in forced.output

    def test_delete_refuses_when_extended(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _run(["scene", "create", "child", "--extends", "base", "--agent", "code-reviewer"], root)
        result = _run(["scene", "delete", "base"], root)
        assert result.exit_code != 0
        assert "extend" in result.output

    def test_delete_force_warns_about_broken_dependents(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        _run(["scene", "create", "child", "--extends", "base", "--agent", "code-reviewer"], root)
        forced = _run(["scene", "delete", "base", "--force"], root)
        assert forced.exit_code == 0, forced.output
        assert "child" in forced.output
        assert "unresolvable" in forced.output
        assert "base" not in _scenes(root)

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

    def test_print_emits_only_valid_yaml(self, tmp_path: Path) -> None:
        # A same-named user scene forces a "Skipped" notice; --print must keep it
        # off stdout so the streamed blocks stay parseable YAML.
        body = CONFIG + "  pr-review:\n    description: mine\n"
        root = _project(tmp_path, body)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        result = _run(["scene", "install-starters", "--print"], root)
        assert result.exit_code == 0, result.output
        assert "Skipped" not in result.output
        parsed = yaml.safe_load(result.output)
        assert isinstance(parsed, dict)
        # The not-yet-present starters are streamed; the skipped one is not.
        assert "deploy-watch" in parsed
        assert "pr-review" not in parsed
        # --print writes nothing.
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before

    def test_print_ignores_flow_style_scenes(self, tmp_path: Path) -> None:
        # A flow-style scenes block cannot be spliced, but --print never edits, so
        # it must render the starters instead of raising the block-style error.
        body = "version: 1\nscenes: {base: {skills: {exclude: [deploy-*]}}}\n"
        root = _project(tmp_path, body)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        result = _run(["scene", "install-starters", "--print"], root)
        assert result.exit_code == 0, result.output
        assert "block style" not in result.output
        parsed = yaml.safe_load(result.output)
        assert "deploy-watch" in parsed
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before


class TestDirectDirectoryConfigTarget:
    """A plain-directory ``.crossby.yml`` is not returned by ``find_config_entry``
    (neither file nor symlink), so ``load_config`` walks past it rather than
    rejecting it — authoring commands then read the target directly. The shared
    preflight must refuse it cleanly instead of crashing with a raw
    ``IsADirectoryError`` in ``read_bytes()``.
    """

    def test_create_refuses_directory_target(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        target = tmp_path / ".crossby.yml"
        target.mkdir()
        (target / "keep.txt").write_text("keep", encoding="utf-8")

        result = _run(["scene", "create", "foo", "--skill", "x"], tmp_path)

        assert result.exit_code == 1, result.output
        assert "non-regular-file" in result.output
        # No crash, directory untouched.
        assert "IsADirectoryError" not in result.output
        assert target.is_dir()
        assert list(target.iterdir()) == [target / "keep.txt"]

    def test_install_starters_refuses_directory_target(self, tmp_path: Path) -> None:
        populate_project(tmp_path)
        target = tmp_path / ".crossby.yml"
        target.mkdir()

        result = _run(["scene", "install-starters"], tmp_path)

        assert result.exit_code == 1, result.output
        assert "non-regular-file" in result.output
        assert "IsADirectoryError" not in result.output
        assert target.is_dir()


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

    def test_wizard_confirm_reselect_moves_exclude_to_include(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Seed a scene that *excludes* deploy-prod, then have the user re-select
        # deploy-prod into the skills include at the confirm step. The interactive
        # include must win, dropping deploy-prod from exclude (the CrossChannelMove
        # conflict-resolution path in _review_scene, unreachable from the always
        # index-0 test above).
        root = _project(tmp_path)
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: True)

        picking = {"confirm": False}

        def _multi_select(_title: str, items: list[str]) -> list[int]:
            # Pick nothing while walking concerns; only the confirm re-selection of
            # skills chooses deploy-prod.
            if picking["confirm"] and "deploy-prod" in items:
                return [items.index("deploy-prod")]
            return []

        confirm_menu = iter(["Change skills include", "Proceed"])

        def _select(title: str, items: list[str], *_a: object, **_k: object) -> int:
            if title == "Confirm scene":
                choice = next(confirm_menu)
                picking["confirm"] = choice != "Proceed"
                return items.index(choice)
            return 0  # optional extends/profile pickers -> "(none)"

        monkeypatch.setattr("crossby.ui.prompts.multi_select", _multi_select)
        monkeypatch.setattr("crossby.ui.prompts.select", _select)
        monkeypatch.setattr("crossby.ui.prompts.input_prompt", lambda *_a, **_k: "")

        result = _run(["scene", "create", "wiztest", "--exclude-skill", "deploy-prod"], root)
        assert result.exit_code == 0, result.output
        skills = _scenes(root)["wiztest"]["skills"]
        assert skills["include"] == ["deploy-prod"]
        assert "deploy-prod" not in skills.get("exclude", [])

    def test_wizard_scans_config_root_not_invocation_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``create`` run from a subdirectory must still see the root project's
        skills/agents/servers — the wizard's universe is built from ``root``, not
        the (empty) invocation subdirectory."""
        root = _project(tmp_path)
        sub = root / "packages" / "app"
        sub.mkdir(parents=True)
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: True)
        monkeypatch.setattr(
            "crossby.ui.prompts.multi_select", lambda _t, items: [0] if items else []
        )
        monkeypatch.setattr("crossby.ui.prompts.select", lambda *a, **k: 0)
        monkeypatch.setattr("crossby.ui.prompts.input_prompt", lambda *a, **k: "")

        result = _run(["scene", "create", "wiztest"], sub)
        assert result.exit_code == 0, result.output
        scene = _scenes(root)["wiztest"]
        # Only present at all if the universe had items to pick from — an empty
        # (subdirectory-rooted) scan would skip every concern entirely.
        assert "skills" in scene
        assert not (sub / ".crossby.yml").exists()


class TestCreatePrintPurity:
    """``create --print`` must emit only valid YAML on stdout, even though the
    interactive wizard runs first — the whole interactive phase is redirected
    to stderr (issue #121, Part 3)."""

    def _wire_prompts(
        self, monkeypatch: pytest.MonkeyPatch, *, name_prompt: str | None = None
    ) -> None:
        monkeypatch.setattr("crossby.ui.prompts.is_tty", lambda: True)
        monkeypatch.setattr(
            "crossby.ui.prompts.multi_select", lambda _t, items: [0] if items else []
        )
        monkeypatch.setattr("crossby.ui.prompts.select", lambda *a, **k: 0)

        def _fake_input_prompt(prompt: str, *a: object, **k: object) -> str:
            if prompt == "New scene name" and name_prompt is not None:
                return name_prompt
            return ""

        monkeypatch.setattr("crossby.ui.prompts.input_prompt", _fake_input_prompt)

    def test_print_interactive_stdout_is_pure_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project(tmp_path)
        self._wire_prompts(monkeypatch)

        result = _run(["scene", "create", "wiztest", "--print"], root)

        assert result.exit_code == 0, result.output
        parsed = yaml.safe_load(result.stdout)
        assert isinstance(parsed, dict)
        assert "wiztest" in parsed
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == CONFIG
        # confirm_defaults' kv review lines run before the (mocked) menu select,
        # and must land on stderr, never interleaved into the YAML stdout.
        assert "Skills include" in result.stderr
        assert "Skills include" not in result.stdout

    def test_print_with_omitted_name_still_emits_pure_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The name prompt runs *before* the wizard — it must be inside the
        redirected phase too, or a piped stdout would see it leak first."""
        root = _project(tmp_path)
        self._wire_prompts(monkeypatch, name_prompt="prompted-name")

        result = _run(["scene", "create", "--print"], root)

        assert result.exit_code == 0, result.output
        # No leaked prefix ahead of the rendered block — the parsed dict has
        # exactly the one expected key, and the raw text starts with it.
        parsed = yaml.safe_load(result.stdout)
        assert parsed == {"prompted-name": parsed.get("prompted-name")}
        assert result.stdout.lstrip().startswith("prompted-name:")
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == CONFIG
        assert "Skills include" in result.stderr
        assert "Skills include" not in result.stdout


# ---------------------------------------------------------------------------
# path resolution + safety
# ---------------------------------------------------------------------------


class TestPathAndSafety:
    def test_edit_from_subdir_targets_the_walked_up_config(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        sub = root / "packages" / "app"
        sub.mkdir(parents=True)
        result = _run(["scene", "add", "base", "--skill", "review-*"], sub)
        assert result.exit_code == 0, result.output
        # No shadow config written into the subdirectory.
        assert not (sub / ".crossby.yml").exists()
        assert "review-*" in _scenes(root)["base"]["skills"]["include"]

    def test_inline_flow_scenes_block_is_refused(self, tmp_path: Path) -> None:
        body = "version: 1\nscenes: {base: {skills: {exclude: [deploy-*]}}}\n"
        root = _project(tmp_path, body)
        before = (root / ".crossby.yml").read_text(encoding="utf-8")
        result = _run(["scene", "add", "base", "--skill", "review-*"], root)
        assert result.exit_code != 0
        assert "block style" in result.output
        assert (root / ".crossby.yml").read_text(encoding="utf-8") == before

    def test_glob_pattern_not_mangled_by_rich_markup(self, tmp_path: Path) -> None:
        root = _project(tmp_path)
        # A char-class glob would be eaten as Rich markup if not escaped.
        result = _run(["scene", "remove", "base", "--exclude-skill", "[abc]"], root)
        assert result.exit_code == 0, result.output
        assert "[abc]" in result.output

    def test_create_from_subdir_with_broken_root_symlink_writes_through_it(
        self, tmp_path: Path
    ) -> None:
        # A broken (dangling) .crossby.yml symlink at the root is a legitimate,
        # not-yet-populated config identity (write_config_checked supports
        # writing through it) — root discovery must not walk past it just
        # because it isn't parseable yet, or a subdir run shadows it instead.
        root = tmp_path / "project"
        populate_project(root)
        real = tmp_path / "real.crossby.yml"
        (root / ".crossby.yml").symlink_to(real)
        assert not real.exists()
        sub = root / "packages" / "app"
        sub.mkdir(parents=True)

        result = _run(["scene", "create", "wiztest", "--skill", "review-*"], sub)

        assert result.exit_code == 0, result.output
        target = root / ".crossby.yml"
        assert target.is_symlink()
        assert target.resolve() == real.resolve()
        assert real.exists()
        assert "wiztest" in _scenes(root)
        assert not (sub / ".crossby.yml").exists()

    def test_create_from_subdir_prefers_broken_child_symlink_over_ancestor(
        self, tmp_path: Path
    ) -> None:
        # With a dangling child .crossby.yml symlink AND a valid ancestor config,
        # authoring must write THROUGH the child link, never splice into the
        # ancestor: parse discovery now stops at the same broken-symlink boundary
        # root discovery does, so scene create can't target the ancestor while
        # scans/state root at the child.
        ancestor_cfg = tmp_path / ".crossby.yml"
        ancestor_cfg.write_text(CONFIG, encoding="utf-8")
        root = tmp_path / "project"
        populate_project(root)
        real = tmp_path / "real.crossby.yml"
        (root / ".crossby.yml").symlink_to(real)
        assert not real.exists()
        sub = root / "packages" / "app"
        sub.mkdir(parents=True)

        result = _run(["scene", "create", "wiztest", "--skill", "review-*"], sub)

        assert result.exit_code == 0, result.output
        target = root / ".crossby.yml"
        assert target.is_symlink()
        assert target.resolve() == real.resolve()
        assert real.exists()
        assert "wiztest" in _scenes(root)
        # The ancestor config was left untouched — no shadow write to it.
        ancestor_scenes = yaml.safe_load(ancestor_cfg.read_text(encoding="utf-8"))["scenes"]
        assert "wiztest" not in ancestor_scenes
        assert not (sub / ".crossby.yml").exists()

    def test_add_writes_through_symlinked_config(self, tmp_path: Path) -> None:
        # Splice writes must go through the link's resolved target so the
        # symlink itself survives (config/safe_write.py write-through fix).
        root = tmp_path / "project"
        populate_project(root)
        real = tmp_path / "real.crossby.yml"
        real.write_text(CONFIG, encoding="utf-8")
        (root / ".crossby.yml").symlink_to(real)

        result = _run(["scene", "add", "base", "--skill", "review-*"], root)

        assert result.exit_code == 0, result.output
        target = root / ".crossby.yml"
        assert target.is_symlink()
        assert target.resolve() == real.resolve()
        assert "review-*" in _scenes(root)["base"]["skills"]["include"]


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


# ---------------------------------------------------------------------------
# real pty smoke test — the only test that exercises actual questionary
# rendering; monkeypatched prompts (above) render nothing, so they cannot
# catch a prompt-toolkit stdout leak.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="pty not available on this platform")
class TestCreatePrintPtySmoke:
    def test_real_questionary_session_keeps_piped_stdout_pure(self, tmp_path: Path) -> None:
        """stdin is a real pty (``is_tty()`` True); stdout is a plain pipe —
        exactly ``crossby scene create --print > file.yml`` run from a real
        terminal. Before the fix, questionary/prompt-toolkit would render its
        menus straight into that pipe alongside the YAML.
        """
        (tmp_path / ".crossby.yml").write_text("version: 1\n", encoding="utf-8")

        master_fd, slave_fd = os.openpty()
        env = dict(os.environ)
        env["PATH"] = ""  # no AI tool discoverable: fixed, minimal prompt sequence
        env["COLUMNS"] = "80"
        env["LINES"] = "24"

        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from crossby.cli.main import cli_main; cli_main()",
                "scene",
                "create",
                "ptytest",
                "--print",
                "--path",
                str(tmp_path),
            ],
            stdin=slave_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        os.close(slave_fd)
        try:
            # Keystrokes for the fixed sequence with no installed tools and an
            # empty project (no scenes/profiles): Enter skips the optional
            # description prompt, then Enter again accepts the confirm menu's
            # default "Proceed" choice.
            deadline = time.monotonic() + 10
            for _ in range(2):
                time.sleep(0.3)
                os.write(master_fd, b"\r")
            stdout, stderr = proc.communicate(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail(
                f"crossby scene create --print timed out.\nstdout={stdout!r}\nstderr={stderr!r}"
            )
        finally:
            os.close(master_fd)

        assert proc.returncode == 0, stderr.decode("utf-8", errors="replace")
        text = stdout.decode("utf-8", errors="replace")
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict)
        assert "ptytest" in parsed
