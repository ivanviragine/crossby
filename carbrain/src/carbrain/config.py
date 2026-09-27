"""Runtime settings, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    sources_file: Path = PACKAGE_DATA / "sources.yaml"
    families_file: Path = PACKAGE_DATA / "families.yaml"
    events_file: Path = PACKAGE_DATA / "events.yaml"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "carbrain.sqlite"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def load_settings() -> Settings:
    """Settings from `CARBRAIN_DATA_DIR` (default: `./var`)."""
    return Settings(data_dir=Path(os.environ.get("CARBRAIN_DATA_DIR", "var")).resolve())
