"""Configuration domain models — CrossbyConfig and nested sections.

Matches the .crossby.yml format:

    version: 1
    ai:
      default_tool: claude
      default_model: claude-sonnet-4.6
      effort: medium
      commands:
        plan:
          tool: claude
          model: claude-opus-4.6
        implement:
          tool: copilot
    models:
      claude:
        easy: claude-haiku-4.5
        medium: claude-sonnet-4.6
        complex: claude-sonnet-4.6
        very_complex: claude-opus-4.6
    profiles:
      ccyolo:
        tool: claude
        yolo: true
        effort: max
      quick:
        tool: cursor
        model: haiku
        effort: low
    scenes:
      base:
        skills:
          exclude: ["deploy-*"]
      pr-review:
        description: Review a pull request
        extends: base                 # optional single-parent composition
        profile: ccyolo               # optional default launch profile
        skills:
          include: ["review-*", "knowledge"]
        agents:
          include: ["code-reviewer"]
        mcp:
          include: ["github"]
        hooks:
          include: ["pre_tool_use:*"]
        permissions:
          include: ["git diff:*", "gh pr *"]
    sync_defaults:
      from: claude
      to: null
      concern: null
    handoff_defaults:
      from: claude
      to: codex
      prompt_preset: default
      token_budget: 32000
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crossby.models.ai import AIToolID


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server entry.

    A server must have either ``command`` (stdio transport) or ``url``
    (http/sse transport) — not both, not neither.

    ``headers`` is the source-shaped HTTP-header map (as Claude/Cursor
    write it, with ``${VAR}`` literals preserved). The Codex writer
    refactors this into ``bearer_token_env_var`` / ``http_headers`` /
    ``env_http_headers`` at render time; other writers preserve the
    literal shape.
    """

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    transport: Literal["stdio", "http", "sse"] = "stdio"
    url: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerConfig:
        has_command = self.command is not None
        has_url = self.url is not None
        if has_command and has_url:
            raise ValueError("MCP server must have 'command' or 'url', not both")
        if not has_command and not has_url:
            raise ValueError("MCP server must have either 'command' (stdio) or 'url' (http/sse)")
        if has_command and self.transport != "stdio":
            raise ValueError(
                f"transport must be 'stdio' when 'command' is set, got '{self.transport}'"
            )
        if has_url and self.transport not in {"http", "sse"}:
            raise ValueError(
                f"transport must be 'http' or 'sse' when 'url' is set, got '{self.transport}'"
            )
        return self


class ComplexityModelMapping(BaseModel):
    """Model IDs (and optional per-tier effort) for each complexity tier.

    Values are exact model IDs as returned by the tool's get_models().
    Defaults are None — populated at init time by querying the tool.

    The ``*_effort`` fields are optional per-tier effort overrides, parallel
    to the model fields. They are stored as plain strings that must parse to
    a ``crossby.models.ai.EffortLevel`` when resolved.
    """

    easy: str | None = None
    medium: str | None = None
    complex: str | None = None
    very_complex: str | None = None

    # Per-tier effort overrides — optional, parallel to the model fields.
    easy_effort: str | None = None
    medium_effort: str | None = None
    complex_effort: str | None = None
    very_complex_effort: str | None = None


class CommandConfig(BaseModel):
    """Per-command AI tool and model override."""

    tool: str | None = None
    model: str | None = None
    effort: str | None = None
    yolo: bool | None = None
    accept_edits: bool | None = None
    auto: bool | None = None


class AIConfig(BaseModel):
    """AI tool configuration section — generic command map."""

    default_tool: str | None = None
    default_model: str | None = None
    effort: str | None = None
    yolo: bool | None = None
    accept_edits: bool | None = None
    auto: bool | None = None
    commands: dict[str, CommandConfig] = {}


class HookEntry(BaseModel):
    """A single canonical hook definition (used by sync readers/writers)."""

    event: str
    command: str
    tools: list[str] = Field(default_factory=list)
    description: str = ""
    fail_closed: bool = False
    """Block the action if the hook process itself fails (crash/timeout/invalid
    output) instead of letting it through. Honored only by tools that expose a
    per-hook fail-closed switch — today just Cursor (``failClosed: true``), which
    otherwise defaults to fail-open. Set this on security guards; other writers
    ignore it (their hooks already fail closed, or offer no such switch)."""
    timeout: int | None = Field(default=None, gt=0)
    """Seconds before the tool gives up on the hook process, or ``None`` for the
    tool's own default. Every tool measures this in seconds but spells the key
    differently — Copilot uses ``timeoutSec`` (default 30), the rest use
    ``timeout`` (Cursor default 60, Codex 600, agy 30) — so each writer emits it
    under its own native name.

    Must be positive: Cursor rejects a non-positive timeout outright ("Hook
    script timeout must be a positive number"), and a config it rejects loads no
    hooks at all — so catch it here rather than write a file that silently
    disables the guard.

    Worth setting on a guard that runs on every write: the default is generous
    enough that a hung hook stalls the agent for a noticeable time, and on
    Cursor a timeout is only *blocked* rather than allowed when
    ``fail_closed`` is also set."""


class ProfileConfig(BaseModel):
    """A saved launch profile (stored in .crossby.yml under ``profiles``)."""

    tool: str | None = None
    model: str | None = None
    effort: str | None = None
    yolo: bool | None = None
    accept_edits: bool | None = None
    auto: bool | None = None


class SceneSelector(BaseModel):
    """Include/exclude globs for one concern inside a :class:`SceneConfig`.

    Selection has three states, distinguished by whether ``include`` is set:

    - ``include`` **absent** (``None``) → start from *everything* detected for
      the concern, then drop anything ``exclude`` matches.
    - ``include: []`` → start from *nothing*.
    - ``include: [globs]`` → start from the union of items each glob matches.

    ``exclude`` is applied last and always wins over ``include``. Patterns are
    ``fnmatch`` globs over each item's canonical name; ``extra="forbid"`` turns
    a typo'd key (``includ:``) into an error instead of a silent no-op.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str] | None = None
    exclude: list[str] = Field(default_factory=list)


class SceneConfig(BaseModel):
    """A named, task-shaped bundle of capabilities (stored under ``scenes``).

    Where a :class:`ProfileConfig` answers *how do I launch* (tool/model/effort),
    a scene answers *what is in the room* — which skills, agents, MCP servers,
    hooks and permissions the session should carry. ``rules`` and ``plugins`` are
    deliberately not selectable concerns.

    ``extends`` names a single parent scene to compose from; flattening (with
    per-concern *replace* semantics, cycle detection and undefined-parent
    errors) happens in :meth:`CrossbyConfig.get_scene`, not on this model.
    ``profile`` names a default launch profile and is validated against the
    ``profiles:`` section by the loader. ``extra="forbid"`` rejects a typo'd
    concern key (``skils:``) rather than dropping it.
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    extends: str | None = None
    profile: str | None = None

    skills: SceneSelector | None = None
    agents: SceneSelector | None = None
    mcp: SceneSelector | None = None
    hooks: SceneSelector | None = None
    permissions: SceneSelector | None = None


# Fields inherited across an ``extends`` edge. ``extends`` itself is resolved
# away during flattening, so it is not carried onto the merged scene.
_SCENE_INHERITABLE_FIELDS: tuple[str, ...] = (
    "description",
    "profile",
    "skills",
    "agents",
    "mcp",
    "hooks",
    "permissions",
)

# The subset of the above that are per-concern selectors — used by the resolver.
SCENE_CONCERNS: tuple[str, ...] = ("skills", "agents", "mcp", "hooks", "permissions")


def _merge_scene(parent: SceneConfig, child: SceneConfig) -> SceneConfig:
    """Merge *child* onto *parent* with per-concern **replace** semantics.

    A field the child declares (present in ``model_fields_set``) fully replaces
    the parent's value; a field the child omits is inherited verbatim. This
    holds for the concern selectors and for ``description``/``profile`` alike,
    so e.g. a child that declares its own ``skills:`` does **not** inherit the
    parent's ``skills.exclude``. The result is a flat scene: ``extends`` is not
    an inheritable field, so the merged scene keeps the parent's (``None``).
    """
    declared = child.model_fields_set
    updates = {
        field: getattr(child, field) for field in _SCENE_INHERITABLE_FIELDS if field in declared
    }
    return parent.model_copy(update=updates)


class SyncDefaults(BaseModel):
    """Defaults for ``crossby sync`` — all fields optional.

    The YAML key is ``sync_defaults`` (not ``sync``, which is a
    deprecated-and-ignored legacy key handled by the loader).

    The ``from:`` YAML key maps to the Python field ``from_tool``
    because ``from`` is a reserved keyword.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_tool: AIToolID | None = Field(default=None, alias="from")
    to: AIToolID | None = None
    concern: str | None = None


class HandoffDefaults(BaseModel):
    """Defaults for ``crossby handoff`` — all fields optional.

    See :class:`SyncDefaults` for the ``from`` / ``from_tool`` alias note.
    ``prompt_preset`` is validated by the loader (not on this model) to
    avoid a circular import with ``crossby.handoff.prompts``.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_tool: AIToolID | None = Field(default=None, alias="from")
    to: AIToolID | None = None
    prompt_preset: str | None = None
    token_budget: int | None = Field(default=None, gt=0)


class CrossbyConfig(BaseModel):
    """Full configuration from .crossby.yml.

    This is the validated, structured representation. The config loader
    parses the YAML file and constructs this model.

    Only contains launch preferences (AI defaults, model mappings, profiles).
    Sync data is read directly from tool configs by ``sync.readers``.
    """

    version: int = 1

    ai: AIConfig = AIConfig()
    models: dict[str, ComplexityModelMapping] = {}
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    scenes: dict[str, SceneConfig] = Field(default_factory=dict)
    sync_defaults: SyncDefaults = Field(default_factory=SyncDefaults)
    handoff_defaults: HandoffDefaults = Field(default_factory=HandoffDefaults)

    # Resolved values (set after loading, not in YAML)
    config_path: str | None = Field(default=None, exclude=True)
    project_root: str | None = Field(default=None, exclude=True)

    def get_ai_tool(self, command: str | None = None) -> str | None:
        """Get the AI tool for a command, with fallback chain.

        Fallback: command-specific tool → global default_tool → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.tool:
                return cmd_config.tool
        return self.ai.default_tool

    def get_model(self, command: str | None = None) -> str | None:
        """Get the model for a command, with fallback chain.

        Fallback: command-specific model → ai.default_model → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.model:
                return cmd_config.model
        return self.ai.default_model

    def get_complexity_model(self, tool: str, complexity: str) -> str | None:
        """Get model ID for a tool + complexity combination."""
        mapping = self.models.get(tool)
        if mapping:
            return getattr(mapping, complexity, None)
        return None

    def get_complexity_effort(self, tool: str, complexity: str) -> str | None:
        """Get effort level for a tool + complexity combination."""
        mapping = self.models.get(tool)
        if mapping:
            return getattr(mapping, f"{complexity}_effort", None)
        return None

    def get_effort(self, command: str | None = None) -> str | None:
        """Get the effort level for a command, with fallback chain.

        Fallback: command-specific effort → global ai.effort → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.effort:
                return cmd_config.effort
        return self.ai.effort

    def get_yolo(self, command: str | None = None) -> bool | None:
        """Get the yolo setting for a command, with fallback chain.

        Fallback: command-specific yolo → global ai.yolo → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.yolo is not None:
                return cmd_config.yolo
        return self.ai.yolo

    def get_accept_edits(self, command: str | None = None) -> bool | None:
        """Get the accept-edits setting for a command, with fallback chain.

        Fallback: command-specific accept_edits → global ai.accept_edits → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.accept_edits is not None:
                return cmd_config.accept_edits
        return self.ai.accept_edits

    def get_auto(self, command: str | None = None) -> bool | None:
        """Get the auto (classifier) setting for a command, with fallback chain.

        Fallback: command-specific auto → global ai.auto → None.
        """
        if command and command in self.ai.commands:
            cmd_config = self.ai.commands[command]
            if cmd_config.auto is not None:
                return cmd_config.auto
        return self.ai.auto

    def get_profile(self, name: str) -> ProfileConfig | None:
        """Get a named launch profile."""
        return self.profiles.get(name)

    def get_scene(self, name: str) -> SceneConfig | None:
        """Return the named scene with its ``extends`` chain flattened.

        The returned :class:`SceneConfig` is *flat*: every parent selector has
        already been folded in with per-concern replace semantics (see
        :func:`_merge_scene`), so the resolver never needs sibling scenes.
        Returns ``None`` for an unknown scene — mirroring :meth:`get_profile`.

        Raises:
            ConfigError: if any ``extends`` in the chain names an undefined
                scene, or if the chain forms a cycle (the message names the
                full chain).
        """
        if name not in self.scenes:
            return None
        return self._flatten_scene(name, [])

    def _flatten_scene(self, name: str, chain: list[str]) -> SceneConfig:
        """Recursively fold *name*'s ``extends`` parents into a flat scene.

        *chain* is the list of scenes visited on the way down (most-derived
        first); it drives both cycle detection and the error message.
        """
        # ConfigError lives in the loader, which imports this module — a
        # function-local import keeps that dependency one-directional.
        from crossby.config.loader import ConfigError

        if name in chain:
            cycle = " -> ".join([*chain, name])
            raise ConfigError(f"scene 'extends' cycle detected: {cycle}")

        scene = self.scenes[name]  # callers only recurse into known scenes
        if scene.extends is None:
            return scene
        if scene.extends not in self.scenes:
            raise ConfigError(f"scene {name!r} extends undefined scene {scene.extends!r}")
        parent = self._flatten_scene(scene.extends, [*chain, name])
        return _merge_scene(parent, scene)

    def get_sync_from(self) -> AIToolID | None:
        """Get the default source tool for ``crossby sync``."""
        return self.sync_defaults.from_tool

    def get_sync_to(self) -> AIToolID | None:
        """Get the default target tool for ``crossby sync`` (``None`` = all installed)."""
        return self.sync_defaults.to

    def get_sync_concern(self) -> str | None:
        """Get the default sync concern (``None`` = all concerns)."""
        return self.sync_defaults.concern

    def get_handoff_from(self) -> AIToolID | None:
        """Get the default source tool for ``crossby handoff``."""
        return self.handoff_defaults.from_tool

    def get_handoff_to(self) -> AIToolID | None:
        """Get the default target tool for ``crossby handoff``."""
        return self.handoff_defaults.to

    def get_handoff_preset(self) -> str | None:
        """Get the default summarization prompt preset for ``crossby handoff``."""
        return self.handoff_defaults.prompt_preset

    def get_handoff_token_budget(self) -> int | None:
        """Get the default token budget for ``crossby handoff``."""
        return self.handoff_defaults.token_budget
