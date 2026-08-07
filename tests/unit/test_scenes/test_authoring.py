"""Unit tests for scene authoring — rendering, scoped splicing, selector edits."""

from __future__ import annotations

import yaml

from crossby.models.config import SceneConfig, SceneSelector
from crossby.scenes.authoring import (
    SelectorEdit,
    add_selectors,
    remove_scene_text,
    remove_selectors,
    render_scene_entry,
    splice_scene_text,
)

# A document that exercises every region a splice must leave untouched: a header
# comment, an inline comment, a preserved ``ai:`` / ``profiles:`` block, a comment
# documenting a *sibling* scene, and a block scalar whose body contains a line
# that looks like a top-level key.
DOC = (
    "# header comment\n"
    "version: 1\n"
    "\n"
    "# AI launch defaults — keep me\n"
    "ai:\n"
    "  default_tool: claude   # inline comment\n"
    "\n"
    "profiles:\n"
    "  ccyolo:\n"
    "    tool: claude   # profile comment\n"
    "\n"
    "scenes:\n"
    "  base:\n"
    "    skills:\n"
    '      exclude: ["deploy-*"]\n'
    "  # this comment documents pr-review\n"
    "  pr-review:\n"
    "    description: old\n"
    "    skills:\n"
    '      include: ["old"]\n'
    "  weird:\n"
    "    description: |\n"
    "      scenes:\n"
    "        not-a-key: true\n"
    "    mcp:\n"
    '      include: ["x"]\n'
    "handoff_defaults:\n"
    "  from: claude\n"
)


class TestRenderSceneEntry:
    def test_field_order_and_flow_lists(self) -> None:
        scene = SceneConfig(
            description="Review a pull request",
            profile="ccyolo",
            skills=SceneSelector(include=["review-*", "knowledge"]),
            mcp=SceneSelector(include=["github"], exclude=["linear"]),
        )
        entry = render_scene_entry("pr-review", scene)
        assert entry.startswith("  pr-review:\n")
        # flow-style lists, matching the schema examples
        assert "include: [review-*, knowledge]" in entry
        # description precedes selectors
        assert entry.index("description:") < entry.index("skills:")
        parsed = yaml.safe_load(entry)
        assert parsed["pr-review"]["mcp"]["exclude"] == ["linear"]

    def test_empty_selector_dropped(self) -> None:
        scene = SceneConfig(skills=SceneSelector(include=None, exclude=[]))
        entry = render_scene_entry("x", scene)
        assert "skills:" not in entry

    def test_explicit_empty_include_kept(self) -> None:
        scene = SceneConfig(mcp=SceneSelector(include=[]))
        entry = render_scene_entry("x", scene)
        assert yaml.safe_load(entry)["x"]["mcp"]["include"] == []


class TestSpliceBytePreservation:
    def test_replacing_a_scene_touches_only_its_entry(self) -> None:
        new_scene = SceneConfig(
            description="Review a pull request",
            skills=SceneSelector(include=["review-*", "knowledge"]),
        )
        out = splice_scene_text(DOC, "pr-review", new_scene)

        # Everything before and after the pr-review entry is byte-identical.
        before = DOC[: DOC.index("  pr-review:")]
        assert out.startswith(before)
        tail = DOC[DOC.index("  weird:") :]
        assert out.endswith(tail)

        # The sibling comment, ai/profiles comments, and block scalar survive.
        for fragment in (
            "# header comment",
            "# AI launch defaults — keep me",
            "# inline comment",
            "# profile comment",
            "# this comment documents pr-review",
            "not-a-key: true",
        ):
            assert fragment in out, fragment

        data = yaml.safe_load(out)
        assert data["scenes"]["pr-review"]["skills"]["include"] == ["review-*", "knowledge"]
        assert data["scenes"]["base"]["skills"]["exclude"] == ["deploy-*"]
        assert data["scenes"]["weird"]["mcp"]["include"] == ["x"]

    def test_adversarial_block_scalar_not_line_scanned(self) -> None:
        # Editing 'weird' must not be confused by the 'scenes:'-looking line in
        # its own block scalar, nor eat 'handoff_defaults'.
        out = splice_scene_text(DOC, "weird", SceneConfig(mcp=SceneSelector(include=["y"])))
        data = yaml.safe_load(out)
        assert data["scenes"]["weird"]["mcp"]["include"] == ["y"]
        assert data["handoff_defaults"]["from"] == "claude"
        assert "not-a-key" not in data["scenes"]

    def test_insert_new_entry_into_existing_block(self) -> None:
        out = splice_scene_text(DOC, "deploy", SceneConfig(mcp=SceneSelector(include=["linear"])))
        data = yaml.safe_load(out)
        assert set(data["scenes"]) == {"base", "pr-review", "weird", "deploy"}
        # untouched siblings
        assert data["scenes"]["pr-review"]["description"] == "old"

    def test_whole_block_appended_after_last_top_level_key(self) -> None:
        doc = "version: 1\nai:\n  default_tool: claude\n"
        out = splice_scene_text(doc, "x", SceneConfig(skills=SceneSelector(include=["a"])))
        assert out.startswith(doc)
        assert yaml.safe_load(out)["scenes"]["x"]["skills"]["include"] == ["a"]

    def test_empty_scenes_key_replaced_not_duplicated(self) -> None:
        for doc in ("version: 1\nscenes: {}\n", "version: 1\nscenes:\n"):
            out = splice_scene_text(doc, "x", SceneConfig(skills=SceneSelector(include=["a"])))
            data = yaml.safe_load(out)
            assert list(data["scenes"]) == ["x"]
            # exactly one top-level scenes key
            assert out.count("\nscenes:") + out.startswith("scenes:") == 1


class TestRemoveSceneText:
    def test_removes_only_named_entry(self) -> None:
        out, found = remove_scene_text(DOC, "pr-review")
        assert found
        data = yaml.safe_load(out)
        assert set(data["scenes"]) == {"base", "weird"}
        assert "# header comment" in out and "# AI launch defaults — keep me" in out

    def test_absent_scene_is_noop(self) -> None:
        out, found = remove_scene_text(DOC, "nope")
        assert not found and out == DOC


class TestSelectorEdits:
    def test_cross_channel_add_moves_pattern(self) -> None:
        scene = SceneConfig(skills=SceneSelector(include=["a"], exclude=["review-x"]))
        edited, moves = add_selectors(scene, [SelectorEdit("skills", ("review-x",), exclude=False)])
        assert edited.skills is not None
        assert edited.skills.include == ["a", "review-x"]
        assert edited.skills.exclude == []
        assert len(moves) == 1 and moves[0].to_exclude is False

    def test_cross_channel_add_exclude_moves_from_include(self) -> None:
        scene = SceneConfig(skills=SceneSelector(include=["a", "b"]))
        edited, moves = add_selectors(scene, [SelectorEdit("skills", ("b",), exclude=True)])
        assert edited.skills is not None
        assert edited.skills.include == ["a"]
        assert edited.skills.exclude == ["b"]
        assert moves[0].to_exclude is True

    def test_add_is_idempotent(self) -> None:
        scene = SceneConfig(skills=SceneSelector(include=["a"]))
        edited, moves = add_selectors(scene, [SelectorEdit("skills", ("a",), exclude=False)])
        assert edited.skills is not None and edited.skills.include == ["a"]
        assert not moves

    def test_add_include_to_absent_selector_creates_explicit_list(self) -> None:
        edited, _ = add_selectors(SceneConfig(), [SelectorEdit("agents", ("x",), exclude=False)])
        assert edited.agents is not None and edited.agents.include == ["x"]

    def test_add_exclude_only_leaves_include_none(self) -> None:
        edited, _ = add_selectors(SceneConfig(), [SelectorEdit("mcp", ("linear",), exclude=True)])
        assert edited.mcp is not None
        assert edited.mcp.include is None
        assert edited.mcp.exclude == ["linear"]

    def test_remove_reports_missing(self) -> None:
        scene = SceneConfig(skills=SceneSelector(include=["a"]))
        edited, missing = remove_selectors(
            scene, [SelectorEdit("skills", ("a", "gone"), exclude=False)]
        )
        assert edited.skills is not None and edited.skills.include == []
        assert missing == ["gone"]

    def test_remove_collapses_all_default_selector_to_none(self) -> None:
        scene = SceneConfig(skills=SceneSelector(include=None, exclude=["deploy-*"]))
        edited, _ = remove_selectors(scene, [SelectorEdit("skills", ("deploy-*",), exclude=True)])
        assert edited.skills is None
