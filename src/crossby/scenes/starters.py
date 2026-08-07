"""Bundled starter scenes — opinionated presets shipped with crossby.

Each ``*.yml`` under ``crossby/data/scenes`` holds a single ``<name>: <body>``
mapping whose body validates against :class:`~crossby.models.config.SceneConfig`.
The definitions use glob selectors only (no ``extends``/``profile`` coupling to a
user's config), so ``crossby scene install-starters`` can drop them into any
project and they degrade gracefully where the named skills/servers are absent.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from crossby import data as _data
from crossby.models.config import SceneConfig

_SCENES_DIR = Path(_data.__file__).parent / "scenes"


def load_starter_scenes() -> dict[str, SceneConfig]:
    """Load and validate every bundled starter definition, sorted by name.

    Raises:
        yaml.YAMLError: a bundled file is not valid YAML.
        pydantic.ValidationError: a bundled scene body does not fit the schema.
    """
    scenes: dict[str, SceneConfig] = {}
    for path in sorted(_SCENES_DIR.glob("*.yml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        for name, body in raw.items():
            scenes[str(name)] = SceneConfig.model_validate(body or {})
    return scenes
