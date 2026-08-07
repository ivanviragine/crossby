"""Scene authoring — edit a single ``scenes.<name>`` entry in ``.crossby.yml``.

The two jobs here are kept deliberately separate:

**Selector edits** (:func:`add_selectors` / :func:`remove_selectors`) work on the
:class:`~crossby.models.config.SceneConfig` model. Adding a pattern to one
channel (``include`` / ``exclude``) always removes it from the other, so the two
can never contradict each other; that move is reported so the CLI can tell the
user about it.

**Scoped splicing** (:func:`splice_scene_text` / :func:`remove_scene_text`) works
on the *text* of ``.crossby.yml``. It rewrites only the byte span of the single
scene entry being edited — located via :func:`yaml.compose` node offsets, never
line-scanning — so every byte outside that span (comments in ``ai:`` /
``profiles:`` / ``models:`` and in sibling scenes) survives untouched. Comments
*inside* the edited entry are lost, which is acceptable: that is the region the
user is actively editing. A brand-new ``scenes:`` key is appended deterministically
after the last top-level key so diffs stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from crossby.models.config import SCENE_CONCERNS, SceneConfig, SceneSelector

# ---------------------------------------------------------------------------
# Selector edits (model level) — the cross-channel rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorEdit:
    """One requested change to a concern's selector.

    ``concern`` is a member of :data:`SCENE_CONCERNS`; ``patterns`` are the globs
    to add/remove; ``exclude`` selects the channel (``True`` → the exclude list).
    """

    concern: str
    patterns: tuple[str, ...]
    exclude: bool


@dataclass(frozen=True)
class CrossChannelMove:
    """A pattern relocated from one channel to the other by an add.

    ``to_exclude`` is ``True`` when the pattern moved *into* the exclude list
    (i.e. an ``--exclude-*`` add displaced a matching include) and ``False`` when
    it moved into the include list.
    """

    concern: str
    pattern: str
    to_exclude: bool

    def describe(self) -> str:
        src, dst = ("include", "exclude") if self.to_exclude else ("exclude", "include")
        return f"{self.concern} {self.pattern!r} moved from {src} to {dst}"


def _channels(sel: SceneSelector | None) -> tuple[list[str] | None, list[str]]:
    """Return mutable copies of a selector's (include, exclude) channels."""
    if sel is None:
        return None, []
    include = None if sel.include is None else list(sel.include)
    return include, list(sel.exclude)


def _normalize(include: list[str] | None, exclude: list[str]) -> SceneSelector | None:
    """Collapse an all-default selector to ``None``; otherwise build one.

    ``include is None and exclude == []`` means "select everything", which is
    exactly what an *absent* concern means — so it is dropped to keep the written
    entry minimal. An explicit ``include: []`` (select nothing) is preserved.
    """
    if include is None and not exclude:
        return None
    return SceneSelector(include=include, exclude=exclude)


def _apply_add(
    concern: str,
    sel: SceneSelector | None,
    includes: list[str],
    excludes: list[str],
) -> tuple[SceneSelector | None, list[CrossChannelMove]]:
    include, exclude = _channels(sel)
    moves: list[CrossChannelMove] = []

    for pattern in includes:
        if pattern in exclude:
            exclude.remove(pattern)
            moves.append(CrossChannelMove(concern, pattern, to_exclude=False))
        # include is None ("everything") → make the scene select this explicitly.
        if include is None:
            include = [pattern]
        elif pattern not in include:
            include.append(pattern)

    for pattern in excludes:
        if include is not None and pattern in include:
            include.remove(pattern)
            moves.append(CrossChannelMove(concern, pattern, to_exclude=True))
        if pattern not in exclude:
            exclude.append(pattern)

    return _normalize(include, exclude), moves


def _apply_remove(
    sel: SceneSelector | None,
    includes: list[str],
    excludes: list[str],
) -> tuple[SceneSelector | None, list[str]]:
    include, exclude = _channels(sel)
    missing: list[str] = []

    for pattern in includes:
        if include is not None and pattern in include:
            include.remove(pattern)
        else:
            missing.append(pattern)
    for pattern in excludes:
        if pattern in exclude:
            exclude.remove(pattern)
        else:
            missing.append(pattern)

    return _normalize(include, exclude), missing


def _group_edits(edits: list[SelectorEdit]) -> dict[str, tuple[list[str], list[str]]]:
    """Collapse edits into ``{concern: (include_patterns, exclude_patterns)}``."""
    grouped: dict[str, tuple[list[str], list[str]]] = {}
    for edit in edits:
        includes, excludes = grouped.setdefault(edit.concern, ([], []))
        (excludes if edit.exclude else includes).extend(edit.patterns)
    return grouped


def add_selectors(
    scene: SceneConfig, edits: list[SelectorEdit]
) -> tuple[SceneConfig, list[CrossChannelMove]]:
    """Return a copy of *scene* with *edits* added, honoring the cross-channel rule.

    Adding a pattern to one channel removes it from the other (reported as a
    :class:`CrossChannelMove`). Idempotent: a pattern already present in its
    target channel leaves the selector unchanged.
    """
    grouped = _group_edits(edits)
    updates: dict[str, SceneSelector | None] = {}
    moves: list[CrossChannelMove] = []
    for concern, (includes, excludes) in grouped.items():
        new_sel, concern_moves = _apply_add(concern, getattr(scene, concern), includes, excludes)
        updates[concern] = new_sel
        moves.extend(concern_moves)
    return scene.model_copy(update=updates), moves


def remove_selectors(
    scene: SceneConfig, edits: list[SelectorEdit]
) -> tuple[SceneConfig, list[str]]:
    """Return a copy of *scene* with *edits* removed from their channels.

    Returns the new scene and the patterns that were not present (no-ops), so the
    CLI can report them. Idempotent: removing an absent pattern changes nothing.
    """
    grouped = _group_edits(edits)
    updates: dict[str, SceneSelector | None] = {}
    missing: list[str] = []
    for concern, (includes, excludes) in grouped.items():
        new_sel, concern_missing = _apply_remove(getattr(scene, concern), includes, excludes)
        updates[concern] = new_sel
        missing.extend(concern_missing)
    return scene.model_copy(update=updates), missing


# ---------------------------------------------------------------------------
# Entry rendering
# ---------------------------------------------------------------------------


def _selector_body(sel: SceneSelector) -> dict[str, list[str]]:
    body: dict[str, list[str]] = {}
    if sel.include is not None:
        body["include"] = list(sel.include)
    if sel.exclude:
        body["exclude"] = list(sel.exclude)
    return body


def render_scene_entry(name: str, scene: SceneConfig, *, indent: int = 2) -> str:
    """Render *scene* as a YAML ``<name>:`` entry indented by *indent* spaces.

    Field order is fixed (description, extends, profile, then the concern
    selectors in :data:`SCENE_CONCERNS` order) so re-renders diff cleanly. Leaf
    glob lists use flow style (``[a, b]``) to match the schema examples. Values
    are serialized by PyYAML, so any scalar needing quoting is quoted correctly.
    """
    body: dict[str, object] = {}
    if scene.description is not None:
        body["description"] = scene.description
    if scene.extends is not None:
        body["extends"] = scene.extends
    if scene.profile is not None:
        body["profile"] = scene.profile
    for concern in SCENE_CONCERNS:
        sel: SceneSelector | None = getattr(scene, concern)
        if sel is None:
            continue
        sel_body = _selector_body(sel)
        if sel_body:
            body[concern] = sel_body

    # default_flow_style=None → block for mappings, flow for scalar-only lists.
    dumped = yaml.safe_dump(
        {name: body}, sort_keys=False, default_flow_style=None, allow_unicode=True
    )
    pad = " " * indent
    lines = [pad + line if line.strip() else line for line in dumped.splitlines()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Scoped text splicing
# ---------------------------------------------------------------------------


def _line_start(text: str, index: int) -> int:
    return text.rfind("\n", 0, index) + 1


def _walk_back_trivia(text: str, end: int) -> int:
    """Move *end* back over trailing blank and comment-only lines.

    A comment directly above the next scene documents *that* scene, so excluding
    it from the current entry's span keeps it untouched when the current entry is
    replaced.
    """
    while end > 0:
        prev = _line_start(text, end - 1)
        stripped = text[prev:end].strip()
        if stripped == "" or stripped.startswith("#"):
            end = prev
        else:
            break
    return end


def _scenes_nodes(
    text: str,
) -> tuple[yaml.nodes.Node, yaml.nodes.Node] | None:
    """Return the ``(key_node, value_node)`` for the top-level ``scenes`` key.

    ``None`` when the document is empty, not a mapping, or has no ``scenes`` key.
    Uses :func:`yaml.compose` so a scalar value containing a line that *looks*
    like a top-level key never fools the search.
    """
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return None
    if not isinstance(root, yaml.nodes.MappingNode):
        return None
    for key_node, value_node in root.value:
        if isinstance(key_node, yaml.nodes.ScalarNode) and key_node.value == "scenes":
            return key_node, value_node
    return None


def _has_entries(value_node: yaml.nodes.Node) -> bool:
    return isinstance(value_node, yaml.nodes.MappingNode) and len(value_node.value) > 0


def _entry_span(text: str, scenes_val: yaml.nodes.Node, name: str) -> tuple[int, int] | None:
    """Byte span ``[start, end)`` of scene *name*'s entry, or ``None`` if absent.

    ``start`` is the beginning of the key's line (indentation included); ``end``
    is the start of the following sibling's line — or, for the last entry, the
    end of the ``scenes:`` block — walked back over trailing blank/comment lines
    so nothing belonging to a neighbour is caught. Anchoring the end on the
    *next key* (not the current value's ``end_mark``) is what keeps a flow-style
    entry (``x: {…}`` whose value ends on the key's own line) from collapsing to
    a zero-length span.
    """
    entries = scenes_val.value
    for i, (key_node, _value_node) in enumerate(entries):
        if key_node.value != name:
            continue
        start = _line_start(text, key_node.start_mark.index)
        if i + 1 < len(entries):
            end = _line_start(text, entries[i + 1][0].start_mark.index)
        else:
            end_index = scenes_val.end_mark.index
            end = _line_start(text, end_index) if end_index < len(text) else len(text)
        return start, _walk_back_trivia(text, end)
    return None


def _existing_indent(scenes_val: yaml.nodes.Node) -> int:
    """The column existing scene keys sit at, so inserts match the file's style."""
    if _has_entries(scenes_val):
        return int(scenes_val.value[0][0].start_mark.column)
    return 2


def _append_scenes_block(text: str, entry: str) -> str:
    """Append a brand-new ``scenes:`` block after the last top-level key."""
    base = text if text == "" or text.endswith("\n") else text + "\n"
    separator = "\n" if base and not base.endswith("\n\n") else ""
    return f"{base}{separator}scenes:\n{entry}"


def _replace_empty_scenes(text: str, key_node: yaml.nodes.Node, entry: str) -> str:
    """Turn an empty ``scenes:`` / ``scenes: {}`` into a block holding *entry*."""
    key_start = _line_start(text, key_node.start_mark.index)
    newline = text.find("\n", key_node.start_mark.index)
    line_end = len(text) if newline == -1 else newline + 1
    return f"{text[:key_start]}scenes:\n{entry}{text[line_end:]}"


def _insert_entry(text: str, scenes_val: yaml.nodes.Node, entry: str) -> str:
    """Insert *entry* after the last existing scene entry.

    The insertion point is the end of the ``scenes:`` block (walked back over
    trailing trivia). A newline is prepended when the preceding byte is not one,
    so a config whose last line lacks a trailing newline does not fuse the new
    key onto it.
    """
    end_index = scenes_val.end_mark.index
    end = _line_start(text, end_index) if end_index < len(text) else len(text)
    insert_at = _walk_back_trivia(text, end)
    prefix = "" if insert_at == 0 or text[insert_at - 1] == "\n" else "\n"
    return text[:insert_at] + prefix + entry + text[insert_at:]


def splice_scene_text(text: str, name: str, scene: SceneConfig) -> str:
    """Return *text* with scene *name* set to *scene*, editing only its entry.

    Replaces the existing ``scenes.<name>`` entry in place, or inserts a new one
    (into the existing block, an empty ``scenes:`` key, or a freshly appended
    block) — always at the granularity of the single entry, never the whole
    ``scenes:`` block, so sibling scenes and their comments are preserved.
    """
    nodes = _scenes_nodes(text)
    if nodes is None:
        return _append_scenes_block(text, render_scene_entry(name, scene))

    key_node, scenes_val = nodes
    if not _has_entries(scenes_val):
        return _replace_empty_scenes(text, key_node, render_scene_entry(name, scene))

    indent = _existing_indent(scenes_val)
    entry = render_scene_entry(name, scene, indent=indent)
    span = _entry_span(text, scenes_val, name)
    if span is not None:
        start, end = span
        return text[:start] + entry + text[end:]
    return _insert_entry(text, scenes_val, entry)


def remove_scene_text(text: str, name: str) -> tuple[str, bool]:
    """Return (*text* with scene *name* removed, whether it was present).

    Only the named entry's byte span is cut; every other byte is preserved. If
    *name* was the sole entry the ``scenes:`` key is left present but empty, which
    parses as "no scenes".
    """
    nodes = _scenes_nodes(text)
    if nodes is None or not _has_entries(nodes[1]):
        return text, False
    span = _entry_span(text, nodes[1], name)
    if span is None:
        return text, False
    start, end = span
    return text[:start] + text[end:], True
