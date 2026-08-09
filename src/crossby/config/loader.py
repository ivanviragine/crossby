"""Configuration loader — find + parse .crossby.yml (walk up from CWD)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from crossby.models.config import (
    AIConfig,
    CommandConfig,
    ComplexityModelMapping,
    CrossbyConfig,
    HandoffDefaults,
    ProfileConfig,
    SceneConfig,
    SyncDefaults,
)

CONFIG_FILENAME = ".crossby.yml"

# Sections removed in the stateless sync refactor — silently ignored.
_DEPRECATED_SECTIONS = frozenset(
    {
        "permissions",
        "mcp_servers",
        "rules",
        "sync",
        "agents",
        "hooks",
    }
)


class ConfigError(Exception):
    """Raised when .crossby.yml cannot be parsed or has invalid structure."""


def ensure_yaml_mapping(raw: Any) -> dict[str, Any] | None:
    """Validate that parsed YAML is a dict (mapping).

    Returns:
        The dict if raw is a dict, None if raw is None (empty file).

    Raises:
        ConfigError: If raw is a non-dict, non-None value (list, scalar).
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    raise ConfigError("Config must be a YAML mapping (key: value pairs)")


def find_config_file(start: Path | None = None) -> Path | None:
    """Walk up from start (or CWD) looking for a *readable* .crossby.yml.

    Returns the path to the config file, or None if not found. A *broken*
    (dangling) ``.crossby.yml`` symlink does not count — ``is_file()``
    resolves through the link and requires the target to exist — so this walks
    past one to a parent file (or None). This is a "is there a readable config
    up-tree?" probe (e.g. the interactive menu's Init toggle). Parse discovery
    (:func:`load_config`) and root resolution (:func:`find_config_entry`)
    instead stop *at* a broken symlink so the two never diverge; callers that
    need to know *where* the project's config identity lives, or to actually
    load it, want those functions, not this one.
    """
    current = (start or Path.cwd()).resolve()

    while True:
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break  # Reached filesystem root
        current = parent

    return None


def find_config_entry(start: Path | None = None) -> Path | None:
    """Walk up from start (or CWD) looking for a ``.crossby.yml`` entry.

    Like :func:`find_config_file`, but stops at *any* existing ``.crossby.yml``
    entry — including a broken symlink and a non-regular file (a plain
    directory, fifo, ...) — rather than only a readable regular file. A broken
    symlink is a legitimate config identity (``write_config_checked`` supports
    writing through a not-yet-populated link), and a *direct* non-regular entry
    must be stopped at rather than walked past: a plain-directory
    ``.crossby.yml`` in a subproject is not a readable config, but walking past
    it to an ancestor would silently load — and let an authoring command edit —
    the *wrong* file. :func:`load_config` classifies each: the broken symlink
    becomes an empty config rooted here; the non-regular entry is rejected.
    Returns the entry path itself (not the — possibly nonexistent — resolved
    target), so a caller combining this with ``write_config_checked`` writes
    through a link.

    :func:`load_config` stops at this same boundary, so parse discovery and root
    discovery stay aligned: a subdir run never resolves scenes from an ancestor
    while rooting state/scan at the entry's own directory.
    """
    current = (start or Path.cwd()).resolve()

    while True:
        candidate = current / CONFIG_FILENAME
        # ``is_symlink()`` catches a broken/looping link (``exists()`` is False
        # for both); ``exists()`` catches a regular file *and* a direct
        # non-regular entry (directory, fifo). Together they stop at any entry.
        if candidate.is_symlink() or candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break  # Reached filesystem root
        current = parent

    return None


def load_config(start: Path | None = None) -> CrossbyConfig:
    """Find and parse the project config.

    Returns a CrossbyConfig with defaults if no config file exists.

    Discovery stops at the SAME boundary as :func:`find_config_entry` (and thus
    :func:`~crossby.services.scene_resolution.scene_root`): the nearest ancestor
    holding a ``.crossby.yml`` entry. When that entry is a *broken* (dangling)
    symlink it can't be parsed, but it is still a legitimate, not-yet-populated
    config identity — so it is surfaced as an *empty* config rooted at that
    directory rather than walking past it to an ancestor. That keeps parse
    discovery from diverging from root discovery: without it a subdir run with a
    broken-symlink config would resolve scenes from an ancestor config while
    rooting scene state/scan at the broken-symlink dir (and an authoring command
    would splice into the ancestor). The empty config's ``config_path`` points
    at the link, so authoring writes *through* it (``write_config_checked``
    supports a not-yet-populated symlink).

    Only a *genuinely dangling* symlink — whose target does not exist yet —
    gets that empty-config treatment. Every other non-file entry is rejected as a
    :class:`ConfigError` rather than masked as empty:
      - a *direct* (non-symlink) non-regular entry — a plain directory or fifo
        named ``.crossby.yml`` — is not a readable config; ``find_config_entry``
        stops at it (instead of walking past to an ancestor, which would silently
        load/edit the wrong file), and it is rejected here;
      - a symlink to an existing non-regular target (e.g. a directory) would
        otherwise silently yield an empty config on read, so a read command would
        run against empty config and an authoring command would splice into it;
      - a symlink loop or an unreadable target is likewise not a not-yet-populated
        identity. (``exists()`` alone can't make this call for a symlink — it also
        returns ``False`` for loops, over-long names, and permission errors,
        masking them as dangling; ``resolve(strict=True)`` distinguishes a truly
        missing target, which raises ``FileNotFoundError``, from those.)

    Raises:
        ConfigError: the discovered ``.crossby.yml`` is a direct non-regular file
            (a directory or fifo/socket), a symlink to an existing non-file target,
            a symlink loop, or an otherwise unresolvable target — or its contents
            fail to parse (see :func:`parse_config_file`).
    """
    entry = find_config_entry(start)
    if entry is None:
        return CrossbyConfig()

    if entry.is_file():
        return parse_config_file(entry)

    if not entry.is_symlink():
        # A *direct* (non-symlink) non-regular entry that ``find_config_entry``
        # stopped at: a plain directory (or fifo/socket) named ``.crossby.yml``.
        # It is not a readable config and not a not-yet-populated identity — and
        # walking past it was the very bug (silently loading/editing an ancestor
        # config). Reject it here so callers exit cleanly instead.
        kind = "directory" if entry.is_dir() else "non-regular file"
        raise ConfigError(
            f"Config {entry} is not a regular file (it is a {kind}); "
            f"expected a regular file or a dangling symlink"
        )

    # A symlink entry. Classify it precisely: a genuinely dangling link (target
    # missing -> FileNotFoundError) is a legitimate not-yet-populated config
    # identity surfaced as an empty config rooted here; anything else — a link to
    # an existing non-regular target, a loop, an unreadable target — is rejected
    # rather than masked as empty.
    try:
        resolved = entry.resolve(strict=True)
    except FileNotFoundError:
        # A broken .crossby.yml symlink: a config identity we must not walk
        # past, but cannot read. Treat it as an empty config rooted here.
        return CrossbyConfig(
            config_path=str(entry),
            project_root=str(entry.parent),
        )
    except (OSError, RuntimeError) as exc:
        # A symlink loop raises OSError(ELOOP) on Python 3.13+ but RuntimeError
        # on 3.11/3.12; other errors (permission, over-long name) come through
        # as OSError. All mean "not a resolvable config identity" -> reject.
        raise ConfigError(f"Config {entry} is a symlink that cannot be resolved: {exc}") from exc

    # Resolved, but ``is_file()`` was False above: an existing non-regular
    # target (directory, socket, ...).
    raise ConfigError(
        f"Config {entry} is a symlink to a non-file target ({resolved}); "
        f"expected a regular file or a dangling symlink"
    )


def parse_config_file(config_path: Path) -> CrossbyConfig:
    """Parse a .crossby.yml file into a CrossbyConfig.

    Raises:
        ConfigError: the file cannot be read (e.g. unreadable/permission-denied,
            or non-UTF-8 bytes) or its contents are not valid YAML / a valid
            config structure. ``load_config`` classifies the *entry* (symlink /
            non-file) before calling this; here the entry is already a regular
            file, so read failures are surfaced as ``ConfigError`` too rather
            than leaking a raw ``OSError``.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"Could not read config {config_path}: {e}") from e
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}") from e

    validated = ensure_yaml_mapping(raw)
    if validated is None:
        # Empty file — treated as defaults
        return CrossbyConfig(
            config_path=str(config_path),
            project_root=str(config_path.parent),
        )

    try:
        return _build_config(validated, config_path)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ConfigError(f"Invalid config structure in {config_path}: {e}") from e


def _build_config(raw: dict[str, Any], config_path: Path) -> CrossbyConfig:
    """Build a CrossbyConfig from raw YAML dict."""
    version = raw.get("version", 1)

    # Warn about deprecated sections (from pre-stateless-sync configs)
    for section in _DEPRECATED_SECTIONS:
        if section in raw and raw[section] is not None:
            warnings.warn(
                f"'{section}' section in .crossby.yml is deprecated and ignored. "
                "Sync now reads directly from tool configs.",
                DeprecationWarning,
                stacklevel=4,
            )

    # Parse ai section
    ai_raw = raw.get("ai")
    if ai_raw is None:
        ai_raw = {}
    if not isinstance(ai_raw, dict):
        raise ConfigError("'ai' must be a mapping")

    commands: dict[str, CommandConfig] = {}
    commands_raw = ai_raw.get("commands")
    if commands_raw is None:
        commands_raw = {}
    if not isinstance(commands_raw, dict):
        raise ConfigError("'ai.commands' must be a mapping")
    for cmd_name, cmd_raw in commands_raw.items():
        if not isinstance(cmd_raw, dict):
            raise ConfigError(f"'ai.commands.{cmd_name}' must be a mapping")
        commands[cmd_name] = _parse_command_config(cmd_raw)
    ai = AIConfig(
        default_tool=ai_raw.get("default_tool"),
        default_model=ai_raw.get("default_model"),
        effort=ai_raw.get("effort"),
        yolo=ai_raw.get("yolo"),
        accept_edits=ai_raw.get("accept_edits"),
        auto=ai_raw.get("auto"),
        commands=commands,
    )

    # Parse models section (nested: tool -> complexity -> model)
    models_raw = raw.get("models")
    if models_raw is None:
        models_raw = {}
    if not isinstance(models_raw, dict):
        raise ConfigError("'models' must be a mapping")
    models: dict[str, ComplexityModelMapping] = {}
    for tool_name, mapping_raw in models_raw.items():
        if not isinstance(mapping_raw, dict):
            raise ConfigError(f"'models.{tool_name}' must be a mapping")
        models[tool_name] = ComplexityModelMapping(
            easy=mapping_raw.get("easy"),
            medium=mapping_raw.get("medium"),
            complex=mapping_raw.get("complex"),
            very_complex=mapping_raw.get("very_complex"),
        )

    # Parse profiles section
    profiles_raw = raw.get("profiles")
    if profiles_raw is None:
        profiles_raw = {}
    if not isinstance(profiles_raw, dict):
        raise ConfigError("'profiles' must be a mapping")
    profiles: dict[str, ProfileConfig] = {}
    for name, profile_raw in profiles_raw.items():
        if not isinstance(profile_raw, dict):
            raise ConfigError(f"'profiles.{name}' must be a mapping")
        # An explicit null (``allow_tools:`` with no value) parses as None; treat
        # it as absent — matching how ai/commands/models/profiles are normalized.
        allow_tools_raw = profile_raw.get("allow_tools")
        if allow_tools_raw is None:
            allow_tools_raw = []
        if not isinstance(allow_tools_raw, list) or not all(
            isinstance(t, str) for t in allow_tools_raw
        ):
            raise ConfigError(f"'profiles.{name}.allow_tools' must be a list of strings")
        profiles[name] = ProfileConfig(
            tool=profile_raw.get("tool"),
            model=profile_raw.get("model"),
            effort=profile_raw.get("effort"),
            yolo=profile_raw.get("yolo"),
            accept_edits=profile_raw.get("accept_edits"),
            auto=profile_raw.get("auto"),
            allow_tools=allow_tools_raw,
        )

    # Parse scenes section (after profiles — validates scene.profile against them)
    scenes = _parse_scenes(raw.get("scenes"), profiles)

    # Parse sync_defaults / handoff_defaults sections
    sync_defaults = _parse_sync_defaults(raw.get("sync_defaults"))
    handoff_defaults = _parse_handoff_defaults(raw.get("handoff_defaults"))

    return CrossbyConfig(
        version=version,
        ai=ai,
        models=models,
        profiles=profiles,
        scenes=scenes,
        sync_defaults=sync_defaults,
        handoff_defaults=handoff_defaults,
        config_path=str(config_path),
        project_root=str(config_path.parent),
    )


def _parse_scenes(raw: Any, profiles: dict[str, ProfileConfig]) -> dict[str, SceneConfig]:
    """Parse the ``scenes:`` section, mirroring the ``profiles:`` block.

    Uses the same separate ``is None`` / ``isinstance`` checks so a falsy scalar
    (``scenes: 0``) still raises, and the same path-qualified message form
    (``'scenes.<name>' must be a mapping``). Each scene's own declared
    ``profile:`` is validated against *profiles* **eagerly here** so a typo
    fails at load time rather than resolving to ``None`` at launch. ``extends``
    flattening and its cycle / undefined-parent checks are deferred to
    :meth:`CrossbyConfig.get_scene`.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("'scenes' must be a mapping")

    scenes: dict[str, SceneConfig] = {}
    for name, scene_raw in raw.items():
        if not isinstance(scene_raw, dict):
            raise ConfigError(f"'scenes.{name}' must be a mapping")
        try:
            scene = SceneConfig.model_validate(scene_raw)
        except ValidationError as exc:
            raise ConfigError(f"Invalid 'scenes.{name}': {exc}") from exc
        if scene.profile is not None and scene.profile not in profiles:
            valid = ", ".join(sorted(profiles)) if profiles else "(none defined)"
            raise ConfigError(
                f"scene '{name}' references undefined profile '{scene.profile}'. "
                f"Defined profiles: {valid}."
            )
        scenes[name] = scene
    return scenes


def _parse_sync_defaults(raw: Any) -> SyncDefaults:
    """Parse the ``sync_defaults`` section. Accepts ``from:`` alias."""
    if raw is None:
        return SyncDefaults()
    if not isinstance(raw, dict):
        raise ConfigError("'sync_defaults' must be a mapping")
    try:
        return SyncDefaults.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid 'sync_defaults': {exc}") from exc


def _parse_handoff_defaults(raw: Any) -> HandoffDefaults:
    """Parse the ``handoff_defaults`` section and validate ``prompt_preset``."""
    if raw is None:
        return HandoffDefaults()
    if not isinstance(raw, dict):
        raise ConfigError("'handoff_defaults' must be a mapping")
    try:
        defaults = HandoffDefaults.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid 'handoff_defaults': {exc}") from exc

    # Validate prompt_preset against the handoff preset registry here (not on
    # the model) to avoid a circular import between models/config.py and
    # handoff/prompts.py.
    if defaults.prompt_preset is not None:
        from crossby.handoff.prompts import PRESETS

        if defaults.prompt_preset not in PRESETS:
            valid = ", ".join(sorted(PRESETS))
            raise ConfigError(
                f"Invalid 'handoff_defaults.prompt_preset': "
                f"{defaults.prompt_preset!r}. Valid presets: {valid}."
            )
    return defaults


def _parse_command_config(raw: dict[str, Any]) -> CommandConfig:
    """Parse a per-command AI config section.

    The caller enforces that ``raw`` is a mapping; an empty dict is allowed
    and yields a ``CommandConfig`` with all-default fields.
    """
    if not raw:
        return CommandConfig()
    return CommandConfig(
        tool=raw.get("tool"),
        model=raw.get("model") or None,  # Treat empty string as None
        effort=raw.get("effort"),
        yolo=raw.get("yolo"),
        accept_edits=raw.get("accept_edits"),
        auto=raw.get("auto"),
    )
