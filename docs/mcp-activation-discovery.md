# MCP activation discovery audit

Crossby discovers server declarations separately from effective activation and
portable connection configuration. A valid explicit source disable prevents new
provisioning and revokes only Crossby-owned target names. Native-only definitions
remain in scene inventory. Unknown or malformed activation cannot authorize
writes or ownership changes.

Scene selection also respects native activation: disabled names remain in the
restriction inventory, but strict launch files omit their connection entries
and native launch restrictions keep them disabled. Selecting a name in a scene
does not enable a source the user disabled. A malformed activation decision
cannot be rendered as a strict launch configuration.

## Audited surfaces

| Source | Activation evidence | Discovery behavior |
| --- | --- | --- |
| Antigravity CLI | `mcpServers.<name>.disabled`, boolean | Invert to canonical `enabled`; require a real boolean. |
| Codex | `mcp_servers.<name>.enabled`, boolean | Keep native boolean separately from portability checks. |
| Claude project definitions | `disabledMcpjsonServers` in user/project/local settings | Union rejection lists; project rejection wins over approval lists and `enableAllProjectMcpServers`. |
| Claude per-project toggles | `projects[<project path>].disabledMcpServers` in `~/.claude.json` | Apply to discovered regular servers for that project; other projects do not affect the scan. |
| Cursor CLI | Per-project `mcp-disabled.json`, array of server identifiers | Read names from the native data directory; never modify it. |
| VS Code/Copilot `.vscode/mcp.json` | VS Code stores enable/disable state outside the shared server file | The private VS Code profile state is not imported; native UI disablement cannot be inferred from this file. |

Claude user settings use `~/.claude/settings.json` and the project toggle record
uses `~/.claude.json`. `CLAUDE_CONFIG_DIR` relocates those to
`<dir>/settings.json` and `<dir>/.claude.json`; the native legacy
`<config home>/.config.json` takes precedence when present. Project toggle
lookup uses the Git project root, or the resolved working directory outside
Git. Project and local settings scanned here belong to the requested project
directory. Crossby does not reconstruct native enterprise/trust workspace
overrides or search ancestor settings layers outside that directory.

## Evidence

- [Antigravity MCP configuration](https://www.antigravity.google/docs/mcp?tab=cli)
  documents the boolean `disabled` server field.
- [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
  documents boolean `mcp_servers.<id>.enabled`.
- [Claude MCP documentation](https://code.claude.com/docs/en/mcp) distinguishes
  project approvals/rejections from per-project toggles in `~/.claude.json`;
  rejection in any settings file applies even in an untrusted folder.
- Claude Code 2.1.296's distributed native bundle confirms that
  `disabledMcpjsonServers` is a list of strings, project approval checks reject
  before considering enable lists, and the regular-server toggle reads the
  current project entry's `disabledMcpServers`. Built-in default-off servers use
  a separate `enabledMcpServers` opt-in list; Crossby does not discover them.
  The same bundle confirms the `CLAUDE_CONFIG_DIR` path override, legacy
  `.config.json` preference, and Git-root project-key lookup.
- [Cursor CLI documentation](https://prod.cursor.com/docs/cli/mcp) documents
  native enable/disable commands. The installed official Cursor CLI build
  `2026.09.10-fd3934a` exposes `src/mcp/disabled-store.ts`,
  `src/mcp/project-paths.ts`, `cursor-config/dist/paths.js`, and
  `utils/dist/workspace-paths.js` in its distributed JavaScript bundle:
  it stores a JSON string array in
  `${CURSOR_DATA_DIR:-~/.cursor}/projects/<sanitized-project-root>/mcp-disabled.json`.
  The project root is the nearest ancestor containing `.git`, or the supplied
  working directory; sanitization replaces non-ASCII-alphanumeric runs with
  hyphens and trims leading/trailing hyphens. `CURSOR_CONFIG_DIR` relocates
  configuration, while `CURSOR_DATA_DIR` relocates this state.
- [VS Code MCP documentation](https://code.visualstudio.com/docs/agent-customization/mcp-servers)
  explicitly separates enable/disable state from `mcp.json`.

## Limits

Crossby does not infer full native runtime activation from enterprise policy,
command-line overrides, trust, connector/plugin/built-in inventories, transient
session controls, or private IDE storage. Approval absence is not a source
disable. Only audited fields and supported files have activation semantics;
inline `enabled`/`disabled` fields on other adapters are not assumed to be
native toggles. Claude local-scope server definitions in `~/.claude.json`
and managed settings are not imported by this change. User-scope definitions
still require `--include-user-scope`; project activation metadata is read without
importing those definitions. No environment references or credentials are resolved.
