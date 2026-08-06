"""PROJECT materialisation and re-point primitives."""

from __future__ import annotations

from pathlib import Path

from crossby.scenes import projection
from crossby.sync.file_utils import MANAGED_MARKER_NAME
from tests.unit.test_scenes.conftest import make_agent, make_skill, symlink_dir


class TestMaterialiseTree:
    def test_links_only_selected_skills_with_marker(self, tmp_path: Path) -> None:
        for name in ("a", "b", "c"):
            make_skill(tmp_path, ".claude/skills", name)
        tree = projection.materialise_tree(tmp_path, "skills", ".claude/skills", {"a", "c"})
        tree_dir = tmp_path / ".crossby" / "scene" / "active" / "skills"
        assert set(tree.linked) == {"a", "c"}
        assert (tree_dir / MANAGED_MARKER_NAME).is_file()
        links = {p.name for p in tree_dir.iterdir() if p.is_symlink()}
        assert links == {"a", "c"}
        # Symlinks are relative and resolve into the real source.
        assert (tree_dir / "a").resolve() == (tmp_path / ".claude" / "skills" / "a").resolve()

    def test_prunes_previous_selection(self, tmp_path: Path) -> None:
        for name in ("a", "b", "c"):
            make_skill(tmp_path, ".claude/skills", name)
        projection.materialise_tree(tmp_path, "skills", ".claude/skills", {"a", "b"})
        projection.materialise_tree(tmp_path, "skills", ".claude/skills", {"c"})
        tree_dir = tmp_path / ".crossby" / "scene" / "active" / "skills"
        assert {p.name for p in tree_dir.iterdir() if p.name != MANAGED_MARKER_NAME} == {"c"}

    def test_agents_matched_by_stem(self, tmp_path: Path) -> None:
        make_agent(tmp_path, ".claude/agents", "code-reviewer.md")
        make_agent(tmp_path, ".claude/agents", "deployer.md")
        tree = projection.materialise_tree(tmp_path, "agents", ".claude/agents", {"code-reviewer"})
        assert set(tree.linked) == {"code-reviewer.md"}


class TestSceneNames:
    def test_enumerates_skill_dir(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            make_skill(tmp_path, ".claude/skills", name)
        assert projection.scene_names(tmp_path, ".claude/skills", "skills") == {"a", "b"}


class TestIsSourceDir:
    def test_symlinked_target_is_not_the_source(self, tmp_path: Path) -> None:
        make_skill(tmp_path, ".claude/skills", "a")
        symlink_dir(tmp_path / ".agents" / "skills", tmp_path / ".claude" / "skills")
        # .agents/skills is a symlink → not the canonical source, safe to re-point.
        assert projection.is_source_dir(tmp_path, ".agents/skills", ".claude/skills") is False

    def test_same_real_dir_is_the_source(self, tmp_path: Path) -> None:
        make_skill(tmp_path, ".claude/skills", "a")
        assert projection.is_source_dir(tmp_path, ".claude/skills", ".claude/skills") is True


class TestPlanTreeDryRun:
    def test_dry_run_does_not_create_anything(self, tmp_path: Path) -> None:
        for name in ("a", "b"):
            make_skill(tmp_path, ".claude/skills", name)
        tree = projection.plan_tree(tmp_path, "skills", ".claude/skills", {"a"}, dry_run=True)
        assert set(tree.linked) == {"a"}
        assert not (tmp_path / ".crossby").exists()
