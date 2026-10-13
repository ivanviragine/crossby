# Brokered headless transport verification

`HeadlessInteractionMode.BROKERED` selects a native interactive transport for an
ordinary session. It requires an interaction handler. The flat
`HeadlessCapability` fields continue to describe unattended CLI runs; the
`brokered` profile describes the alternate transport and controls preflight,
runtime terminal checks, and cleanup.

Brokered start, progress, and interaction events use their event kind without
fixed diagnostic messages. This avoids treating generic labels as prompt leaks
when a short prompt happens to match them; prompt-bearing event messages and
payloads still fail the shared runtime guard.
Optional tool-event identifiers are omitted when their key or value overlaps
the prompt, preserving activity events without exposing protected content.

Shared stdio processes use owned POSIX process groups or Windows Job Objects;
Windows children are attached before they resume. Cleanup kills the entire
owned tree even when the direct parent has already exited. Process lifetime
uses the overall deadline, while each brokered read, write, reply, and shutdown
uses the runtime's current idle/overall budget. Native I/O expiry is reconciled
with runtime timeout and cancellation before reporting a transport failure.
The stdio queue retains at most 256 frames and 8 MiB of UTF-8 data, applying
backpressure without dropping frames. Its 1 MiB frame limit counts bytes.

| Adapter | Profile floor | Transport and normal turn | Accepted output | Interaction binding | Cancellation |
| --- | --- | --- | --- | --- | --- |
| Claude Code | CLI 2.1.263 and optional SDK 0.2.165 | SDK `ClaudeSDKClient` with owned custom transport, ordinary query or exact resume | text, JSON, JSONL and response schemas | exact wire request and tool-use IDs plus initialized session; question text and labels are native answer keys | SDK `interrupt()`, then owned process cleanup |
| Codex CLI | 0.162.1 | `codex app-server --stdio`, `thread/start` or exact `thread/resume`, then `turn/start` without `collaborationMode` | text | exact JSON-RPC request ID plus thread, turn, item, and question IDs | `turn/interrupt` for the active thread and turn, then process cleanup |
| Cursor | 2026.09.10-fd3934a | ACP `initialize`, authentication, `session/new` or exact `session/load`, then `session/prompt` without plan mode | text | exact request and session IDs, question IDs, selected option IDs, and message chunk IDs | `session/cancel` notification for the active session, then process cleanup |
| OpenCode | 1.19.0 | authenticated loopback `POST /session` or validated exact session, then ordinary `prompt_async` | text | exact session ID, submitted user message ID and assistant `parentID` for final answers and pending tool calls, pending question/permission ID | session abort only for a newly created session, then process cleanup |
| Antigravity CLI | 1.2.6 | `agy --print` JSON or stream-JSON without `--mode plan`; resume `WAITING` with the same conversation ID | JSON, JSONL | exact conversation and question IDs; at most nine invocations | owned process-group cleanup; **no native abort** |

Claude SDK verification uses the published Python SDK `0.2.165` and its actual
control dispatcher with synthetic child processes; no paid/live model turn is
claimed. The independent CLI floor retains the existing Claude native streaming
contract. The custom transport and subprocess command builder are upstream
unstable APIs, so the optional extra is exactly pinned and imports/API are
checked before spawning. The owned process reads raw JSONL through Crossby's
byte-bounded queue and deadlines, while the SDK owns initialization, permission
dispatch and native response serialization. The installed CLI is used, never a
different bundled executable. Its SDK entrypoint/version and working-directory
environment match the pinned SDK; its inherited nested-session marker is cleared
as in upstream SDK startup.

The [official user-input contract](https://code.claude.com/docs/en/agent-sdk/user-input)
uses question text as the answer-map key and option labels as the values.
Crossby preserves them exactly, supports descriptions, multi-select arrays and
free text, and rejects ambiguous repeated keys/labels. The SDK callback omits
wire request and session IDs; the custom transport binds them before dispatch.
Duplicate IDs, missing session binding, cancelled requests, unresolved requests
at terminal completion and late answers fail closed. Bash commands come only
from native `input.command`; filesystem targets and detached `native_input`
preserve other operation details. Permission suggestions never become persistent
grants. Denied/skipped clarification ends the session without invented answers;
denied permissions send native denial. Handler errors remain managed transport
errors even though the SDK normally converts them into control errors.
Execution directory remains unset: Claude's Bash working directory can persist
across commands, and launch cwd is not authoritative execution evidence.
Native cancellation signals the bounded interaction-handler and related
event-observer waits so executor shutdown does not delay a successful result
until its idle deadline. Observer callbacks remain isolated; their late return
cannot revive withdrawn requests. Withdrawal and closure
share a lock with response commitment inside the physical writer; a response
queued behind a send lock or executor cannot revive a withdrawn request.
After committing response rights, the writer releases the registry lock before
pipe write/flush so backpressure cannot block native interruption or withdrawal.

Startup, callbacks, I/O and teardown share the Crossby deadline. Pending reads
recheck the current idle budget when sequential question callbacks refresh
semantic progress, and poll transport closure so executor shutdown does not rely
on an EOF record being queued after process closure begins. Physical writes use
the owned process's overall deadline,
while the reader and runtime monitor enforce the current idle deadline as native
progress arrives. Partially written frames are never retried; neither read nor
write progress extends the overall deadline. Cleanup attempts SDK interruption
once before process-group termination, killing descendants
even after the parent exits. The pinned SDK distinguishes a turn result from
run completion. The bridge retains the submitted prompt's native result,
including output/schema and usage, while consuming background turns and keeping
their permission/question callbacks live. Only `local_agent` and
`local_workflow` tasks defer input closure: `task_started` adds them;
`task_notification` or terminal `task_updated` removes them. Background snapshots
do not replace this ledger, and long-lived shells/monitors do not defer closure.

A result or idle state settles the run only when tracked work has finished and
no main-thread turn remains in progress. Native running/requires-action states
reopen it. Defined task/state lifecycle frames remain valid after a result;
response/control frames require an active continuation (interrupt replies are
also valid during cleanup). Unbound trailing content, conflicting session IDs,
and duplicate human results fail. Successful runs close stdin and drain stdout
through EOF, checking process exit; active state at EOF cannot become success.

Observed native failure/abort uses bounded interrupt and cleanup rather than
waiting for idle, preserving terminal status, actual exit code and prompt usage.
Native failure state and provenance are recorded before invoking usage observers.
An observer/dispatch exception takes precedence over native completion, even if
the cleanup task finishes before the SDK message consumer receives the error.
Native permission denials from every result, including background turns, are
recorded as safe generic diagnostics before any usage observer can stop dispatch;
background accounting still cannot replace the submitted prompt's usage.
Error teardown reserves remaining work time to deliver the exception rather
than allowing an unacknowledged interrupt wait to turn it into an idle timeout.
Numeric cost payloads are checked against Pydantic event-export JSON and
standard/compact JSON serializations, including exponent notation. A matching
protected prompt suppresses cost while native token accounting remains available.
Prompt usage/cost is recorded immediately when its result arrives, so later
background timeout/cancellation retains observed accounting. An absent result
origin means human; a second such result is rejected rather than attributed to
a background task. A background failure fails the run without replacing the
submitted prompt's output/schema (failed results suppress output as usual).
A reader-first startup stop completes the bounded SDK interruption attempt before SDK auto-disconnect can close its control channel; concurrent
cleanup callers share that single attempt. Native-supplied AskUserQuestion
`response` is removed before sending validated handler answers because Claude
otherwise gives that field precedence over the answer map.
Native result
subtype/error state, exact session ID, token usage, cost and permission-denial
presence are retained. JSON/JSONL without a schema preserve the final native
envelope (including native cost/metadata); schemas select `structured_output`
and use the shared validator. Cost is also available in a bounded usage-event
payload for text/schema results (omitted when its key overlaps the protected
prompt). Event/diagnostic content excludes prompts,
answers and raw tool arguments.

Claude remains tool-managed for sandbox policy. Unsupported network grants,
sandbox disabling and approval policies fail preflight. Shared requests have
no scene context or arbitrary tool-list selector; these are API capability gaps,
not advertised support. Existing interactive, plan and unattended paths remain
independent. See `test_claude_sdk.py` for real SDK dispatch, dependency failures,
native-confirmed resume identity (requested IDs alone never enter result
provenance), SDK dispatch-error propagation, structured output, native
cancellation, startup/callback timeouts, terminal failures and owned-process tests.

Codex's 0.162.1 app-server schema was generated with
`codex app-server generate-json-schema --experimental`. It defines
`item/tool/requestUserInput`, command and file approval requests,
`item/permissions/requestApproval`, and `mcpServer/elicitation/request`.
Command approval requests may omit `availableDecisions`, and file-change
requests have no such field; the response schemas define the exact `accept`,
`acceptForSession`, `decline`, and `cancel` decisions. The brokered adapter
offers those native decisions when the request omits the list. Built-in
permission requests carry a structured permission profile; an explicit
approval grants that exact profile for the turn. When `approval_policy=never` or
a brokered handler denies or skips the optionless built-in request,
the adapter responds with an empty permission profile scoped to the turn, so
the native turn can continue without granting the requested permissions. MCP
form content is checked
against the request schema before an accepted response is sent; denied or
skipped elicitation sends `decline`, and cancellation sends `cancel`. A missing
or invalid selection cannot become a native response.
Ordinary Codex threads are persisted so their returned thread IDs can be
resumed; collected-plan threads keep their prior ephemeral behavior.
Bounded agent-message, reasoning, plan, command, file-change, and MCP progress
notifications refresh the idle deadline only for the active thread and turn;
completed agent messages supply the final text in native order under an 8 MiB
aggregate UTF-8 limit.
Notifications buffered while awaiting initialize, thread, and turn responses
share an 8 MiB aggregate serialized UTF-8 budget and a 4,096-frame cap. The
driver checks both limits before retaining each notification, including unknown
methods, and rejects an oversized queue through managed transport cleanup.
A Codex 0.162.1 loopback probe completed one minimal turn in a temporary
directory, reopened the app-server process, and resumed the exact thread ID.
The generated server-request union does not identify a connector-specific
permission request, so connector permissions remain unsupported until a
versioned native request shape is captured.

Cursor ACP frames carry assistant message chunks, tool updates, usage updates,
and a `session/prompt` stop reason. The ordinary driver requires `end_turn`
and a clean process exit. It populates executable operation fields only from
`rawInput`; a tool's display title cannot authorize a command. At most 512
message chunk, usage, and fallback progress updates produce visible events;
tool updates have a separate 512-event cap. ACP `usage_update.used` describes
current context occupancy, so the brokered Cursor result leaves turn token
usage unset. Later updates still refresh the idle deadline, and assistant
messages assemble in native order under an 8 MiB aggregate UTF-8 limit. Each
assistant message uses a compact text buffer rather than retaining every native
chunk, including empty chunks. Message IDs have a separate 1 MiB aggregate
budget and a 4,096-message cap. An `end_turn` without non-blank assistant
text yields `INVALID_OUTPUT`. A denied permission response cannot forward a selected
native option, even if the handler supplies one. Cursor's shared permission
helper uses ACP's `allow_once` and `reject_once`/`reject_always`
option kinds for implicit permission decisions, never display labels. Without
a typed rejection option, denial returns native `cancelled`; an implicit
approval requires `allow_once`. An explicit offered option ID is preserved.
Cursor's brokered profile does not offer command policy or response schemas. It applies
model-encoded effort and rejects malformed or conflicting effort encodings in
preflight, even without a separate effort request. It rechecks model, effort,
and variants after all configuration updates. It checks
`agentCapabilities.loadSession` before loading a saved session, validates
replayed session IDs, and rejects model or effort overrides on load.
The installed 2026.09.10 build advertised `loadSession` during an ACP
initialize probe; the local CLI required a Cursor login before a live
`session/new`/`session/load` smoke test could run. Deterministic protocol
fixtures cover the exact load request, replay binding, and new prompt result.

OpenCode keeps its random Basic Auth credential inside the owned loopback
client. It polls the bound session's pending question and permission APIs,
answering only requests whose tool message belongs to the submitted prompt.
It accepts a final assistant message only when its native `parentID` matches
the submitted user message ID, the exact session and message IDs match, and
the native finish reason is `stop`. A bound assistant error or non-`stop`
terminal finish returns a failed session result with the native error name or
finish reason (or `error` when no name exists); missing successful output does
not turn that failure into a transport exception. A bounded incremental SSE
reader normalizes matching assistant, tool, and usage progress when the server
delivers it. Native part deltas from prompt-bound assistant messages refresh
the idle deadline without adding an event for each streamed fragment. The
message binding can be established by SSE, message polling, or a verified
pending interaction, so later part updates still count when the initial SSE
message update was missed. SSE and polling share a 512-event progress cap;
later updates continue to refresh the idle deadline. SSE progress is dropped
if its 256-event or 8 MiB queue fills; polling remains
the terminal authority because [some 1.18.x builds](https://github.com/anomalyco/opencode/issues/46733)
have an `/event` endpoint that stays quiet during a live session. The current
ordinary driver does not expose response schemas or command-policy changes on
resume. It records historical message IDs before prompting a resumed session
so an old assistant answer cannot become the new result; a concurrent reply
to another prompt cannot satisfy this binding. The SSE read and HTTP request
share the session deadline, and HTTP bodies have a fixed size
limit. HTTP startup and request waits also respect the current runtime idle
budget. Closing SSE shuts down the active socket before closing its connection
so a blocked buffered reader wakes during cleanup. Each SSE event is limited
to 1 MiB of data including inter-line separators and 4,096 `data:` lines;
line validation is independent of network chunk boundaries.
Native message, tool, observed-history, and answered-request bookkeeping has
a 4,096-entry cap per collection, a 4 KiB limit per retained identifier, and
a 64-byte limit for tool status tokens; excess state fails the run through managed cleanup. Permission interactions
carry the request's working directory as execution context. Reply IDs use the native pending request ID; no command is
inferred from display patterns. A resumed session may also be in use by another
client, so cleanup does not send its session-wide abort; it terminates this
run's loopback server instead.
On the installed OpenCode 1.19.0 build, a loopback smoke probe created a
session, fetched that exact session by ID, and observed empty initial history;
deterministic fixtures cover the resumed prompt and new-message binding.

Antigravity's JSON and stream-JSON result envelopes carry the conversation ID,
terminal status, and `WAITING` question. Each continuation uses the same
conversation and the handler's selected native option, mapped to the native
label required by the CLI. JSON schema output is accepted only when the native
result echoes the requested schema and the final structured output validates.
Antigravity does not expose an authoritative permission request or native
abort, so its brokered profile advertises questions only. Streaming progress
uses one 512-event limit across all continuations while still refreshing the
idle deadline for later frames. A handler response that cannot map to an
offered native answer ends with a failed terminal event.
Before each continuation, the complete command is revalidated with its mapped
answer and conversation ID. An answer or native option label that exceeds the
POSIX argument/argv-and-environment limits or Windows UTF-16 command-line limit
ends with a failed terminal event without starting another process or exposing
the answer in diagnostics. Deterministic tests cover free-form answers and
selected labels for all three limits.

The deterministic wire tests are in
`tests/unit/test_ai_tools/test_brokered_sessions.py`; existing collected-plan
tests cover their independent plan artifacts. Run those tests and
`tests/unit/test_ai_tools/test_brokered_transport_safety.py` for real-pipe idle
progress, operation budgets, exited-parent cleanup, simulated Windows job
ownership, blocked socket cleanup, and cross-path permission decisions. Run
`scripts/check.sh` after changing a profile. Increase a profile's version floor
when a newly advertised native request shape was verified only on a later
build. Unsupported combinations must fail during preflight before starting a
native process.
