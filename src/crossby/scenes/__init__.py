"""Scene activation engine — apply a resolved scene to each installed tool.

The resolver (:mod:`crossby.services.scene_resolution`) decides *what* a scene
selects; this package decides *how* to make each tool honour that selection,
choosing the least-invasive mechanism available per ``(tool, concern)`` cell:

- **DECLARE** — write the tool's own disable key (non-destructive, instantly
  reversible): Claude ``skillOverrides`` / ``permissions.deny`` /
  ``disabledMcpjsonServers``; Codex ``mcp_servers.<id>.enabled = false``;
  Antigravity CLI ``mcpServers.<name>.disabled``.
- **PROJECT** — materialise a scene-filtered source tree of symlinks and point
  the existing sync writers at it, or filter the concern's list and drive
  ``run_sync`` through the revocable-sync removal channel (hooks / permissions).
- **UNSUPPORTED** — the tool has no per-item lever for the concern; report it
  rather than silently pretend the selection took effect.

Public API: :func:`apply_scene` and :func:`clear_scene`, both returning
``list[SyncResult]`` so the CLI can reuse :mod:`crossby.sync.report` unchanged.
"""

from __future__ import annotations

from crossby.scenes.engine import apply_scene, clear_scene
from crossby.scenes.mechanism import ActivationUnit, SceneMechanism, plan_units

__all__ = [
    "ActivationUnit",
    "SceneMechanism",
    "apply_scene",
    "clear_scene",
    "plan_units",
]
