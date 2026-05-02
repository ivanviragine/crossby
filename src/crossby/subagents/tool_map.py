"""Translation tables for subagent tool/capability names.

Each tool spells the same capability differently:

    Claude   Read         Edit        Bash         Grep      Glob       WebSearch
    Copilot  read         edit        shell        search    glob       web_search
    Gemini   read_file    edit        run_shell_command  grep_search  glob  google_web_search
    Cursor   (no allowlist — uses readonly: bool)
    Codex    (no allowlist — uses sandbox_mode)

The IR stores tools as canonical lowercase snake_case keys (the column on the
left below).  Parsers translate inbound names *to* canonical via the reverse
map; emitters translate canonical names *to* the target tool's spelling.

Unknown names pass through untouched — preserving fidelity for MCP-namespaced
tools (e.g. ``some-server/some-tool``) and tool-private names this table
doesn't know about.
"""

from __future__ import annotations

# canonical → per-tool name
CANONICAL_TOOLS: dict[str, dict[str, str]] = {
    "read_file": {"claude": "Read", "copilot": "read", "gemini": "read_file"},
    "write_file": {"claude": "Write", "copilot": "write", "gemini": "write_file"},
    "edit_file": {"claude": "Edit", "copilot": "edit", "gemini": "edit"},
    "bash": {"claude": "Bash", "copilot": "shell", "gemini": "run_shell_command"},
    "grep": {"claude": "Grep", "copilot": "search", "gemini": "grep_search"},
    "glob": {"claude": "Glob", "copilot": "glob", "gemini": "glob"},
    "web_search": {
        "claude": "WebSearch",
        "copilot": "web_search",
        "gemini": "google_web_search",
    },
    "web_fetch": {"claude": "WebFetch", "copilot": "web_fetch", "gemini": "web_fetch"},
    "agent": {"claude": "Agent", "copilot": "agent", "gemini": "agent"},
    "todo_write": {"claude": "TodoWrite", "copilot": "todo_write", "gemini": "todo_write"},
    "notebook_edit": {
        "claude": "NotebookEdit",
        "copilot": "notebook_edit",
        "gemini": "notebook_edit",
    },
}

# Tools that have a documented `tools:` allowlist field at all.
TOOLS_WITH_ALLOWLIST: frozenset[str] = frozenset({"claude", "copilot", "gemini"})


def to_canonical(name: str, tool: str) -> str:
    """Translate a tool-specific tool name to canonical form.

    Unknown names pass through unchanged — including MCP-namespaced tools
    like ``mcp_server/tool`` and any future tool the table doesn't list.
    """
    for canonical, per_tool in CANONICAL_TOOLS.items():
        if per_tool.get(tool) == name:
            return canonical
    return name


def from_canonical(name: str, tool: str) -> str:
    """Translate a canonical tool name to a target tool's spelling.

    If ``name`` isn't a known canonical key, return it unchanged so that
    user-defined or MCP-namespaced names are preserved verbatim.
    """
    entry = CANONICAL_TOOLS.get(name)
    if entry is None:
        return name
    return entry.get(tool, name)


def all_read_only(tools: list[str]) -> bool:
    """Heuristic for the Cursor ``readonly`` mapping.

    Returns True iff every canonical tool in ``tools`` is read-only — i.e.
    none of them write to the workspace or invoke the shell.  Empty lists
    return True (no tools = nothing can mutate state).
    """
    write_capable = {"write_file", "edit_file", "bash", "notebook_edit"}
    return not any(t in write_capable for t in tools)
