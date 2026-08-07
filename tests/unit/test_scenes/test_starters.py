"""Every bundled starter scene must load and validate under the schema."""

from __future__ import annotations

from crossby.models.config import SceneConfig
from crossby.scenes.starters import load_starter_scenes

EXPECTED = {"pr-review", "deploy-watch", "write-docs", "presentation"}


def test_all_starters_present() -> None:
    starters = load_starter_scenes()
    assert set(starters) == EXPECTED


def test_every_starter_validates_under_schema() -> None:
    for name, scene in load_starter_scenes().items():
        assert isinstance(scene, SceneConfig), name
        assert scene.description, f"{name} should carry a description"
        # Starters must stay self-contained: no coupling to a user's config.
        assert scene.extends is None, f"{name} must not extend a user scene"
        assert scene.profile is None, f"{name} must not reference a profile"


def test_deploy_watch_uses_excludes() -> None:
    scene = load_starter_scenes()["deploy-watch"]
    assert scene.permissions is not None
    assert scene.permissions.exclude, "deploy-watch is the exclude exemplar"


def test_presentation_has_minimal_mcp() -> None:
    scene = load_starter_scenes()["presentation"]
    assert scene.mcp is not None and scene.mcp.include == []
