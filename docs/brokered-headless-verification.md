# Brokered headless transport verification

`HeadlessInteractionMode.BROKERED` selects a native interactive transport for an
ordinary session. It requires an interaction handler. The flat
`HeadlessCapability` fields continue to describe unattended CLI runs; the
`brokered` profile describes the alternate transport and controls preflight,
runtime terminal checks, and cleanup.

| Adapter | Profile floor | Transport and normal turn | Accepted output | Interaction binding | Cancellation |
| --- | --- | --- | --- | --- | --- |
| Codex CLI | 0.162.1 | `codex app-server --stdio`, `thread/start` or exact `thread/resume`, then `turn/start` without `collaborationMode` | text | exact JSON-RPC request ID plus thread, turn, item, and question IDs | `turn/interrupt` for the active thread and turn, then process cleanup |
| Cursor | 2026.09.10-fd3934a | ACP `initialize`, authentication, `session/new` or exact `session/load`, then `session/prompt` without plan mode | text | exact request and session IDs, question IDs, selected option IDs, and message chunk IDs | `session/cancel` for the active session, then process cleanup |
| OpenCode | 1.19.0 | authenticated loopback `POST /session` or validated exact session, then ordinary `prompt_async` | text | exact session ID, pending question/permission ID, new completed assistant message ID | `POST /session/{id}/abort`, then process cleanup |
| Antigravity CLI | 1.2.6 | `agy --print` JSON or stream-JSON without `--mode plan`; resume `WAITING` with the same conversation ID | JSON, JSONL | exact conversation and question IDs; at most nine invocations | owned process-group cleanup; **no native abort** |

Codex's 0.162.1 app-server schema was generated with
`codex app-server generate-json-schema --experimental`. It defines
`item/tool/requestUserInput`, command and file approval requests,
`item/permissions/requestApproval`, and `mcpServer/elicitation/request`.
Command approval requests may omit `availableDecisions`, and file-change
requests have no such field; the response schemas define the exact `accept`,
`acceptForSession`, `decline`, and `cancel` decisions. The brokered adapter
offers those native decisions when the request omits the list. Built-in
permission requests carry a structured permission profile; an explicit
approval grants that exact profile for the turn. MCP form content is checked
against the request schema before a response is sent. A missing or invalid
selection cannot become a native response.
Ordinary Codex threads are persisted so their returned thread IDs can be
resumed; collected-plan threads keep their prior ephemeral behavior.
A Codex 0.162.1 loopback probe completed one minimal turn in a temporary
directory, reopened the app-server process, and resumed the exact thread ID.
The generated server-request union does not identify a connector-specific
permission request, so connector permissions remain unsupported until a
versioned native request shape is captured.

Cursor ACP frames carry assistant message chunks, tool updates, usage updates,
and a `session/prompt` stop reason. The ordinary driver requires `end_turn`
and a clean process exit. It populates executable operation fields only from
`rawInput`; a tool's display title cannot authorize a command. Cursor's
brokered profile does not offer command policy or response schemas. It checks
`agentCapabilities.loadSession` before loading a saved session, validates
replayed session IDs, and rejects model or effort overrides on load.
The installed 2026.09.10 build advertised `loadSession` during an ACP
initialize probe; the local CLI required a Cursor login before a live
`session/new`/`session/load` smoke test could run. Deterministic protocol
fixtures cover the exact load request, replay binding, and new prompt result.

OpenCode keeps its random Basic Auth credential inside the owned loopback
client. It polls only the bound session's pending question and permission APIs
and accepts a final assistant message only when the exact session and message
IDs match and the native finish reason is `stop`. A bounded incremental SSE
reader normalizes matching assistant, tool, and usage progress when the server
delivers it. Polling remains the terminal authority because [some 1.18.x builds](https://github.com/anomalyco/opencode/issues/46733)
have an `/event` endpoint that stays quiet during a live session. The current
ordinary driver does not expose response schemas or command-policy changes on
resume. It records historical message IDs before prompting a resumed session
so an old assistant answer cannot become the new result. The HTTP request and
body read are bounded by the shared deadline and fixed size
limit. Permission reply IDs use the native pending request ID; no command is
inferred from display patterns.
On the installed OpenCode 1.19.0 build, a loopback smoke probe created a
session, fetched that exact session by ID, and observed empty initial history;
deterministic fixtures cover the resumed prompt and new-message binding.

Antigravity's JSON and stream-JSON result envelopes carry the conversation ID,
terminal status, and `WAITING` question. Each continuation uses the same
conversation and the handler's selected native option, mapped to the native
label required by the CLI. JSON schema output is accepted only when the native
result echoes the requested schema and the final structured output validates.
Antigravity does not expose an authoritative permission request or native
abort, so its brokered profile advertises questions only.

The deterministic wire tests are in
`tests/unit/test_ai_tools/test_brokered_sessions.py`; existing collected-plan
tests cover their independent plan artifacts. Run those tests and
`scripts/check.sh` after changing a profile. Increase a profile's version floor
when a newly advertised native request shape was verified only on a later
build. Unsupported combinations must fail during preflight before starting a
native process.
