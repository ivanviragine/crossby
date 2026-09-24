# Unattended headless session verification

Verified 2026-09-18 on macOS against the installed builds named below. Each
adapter's `HeadlessCapability.verified_version` is the build it was checked on
and is enforced as the runtime floor.

| Tool | Build | Probe |
| --- | --- | --- |
| Claude Code | 2.1.263 | `claude --print` with each `--output-format`, `--json-schema`, `--permission-prompts none` |
| Codex CLI | 0.154.0 | `codex exec --json --output-schema <file> -` |
| Cursor | 2026.09.10-fd3934a | `agent --print --output-format json\|stream-json --trust` |
| GitHub Copilot CLI | 1.0.83 | `copilot --help`; `copilot --prompt -s --no-ask-user` |
| OpenCode | 1.18.31 | `opencode run --format json` |
| Antigravity CLI | 1.2.6 | `agy --print --output-format json\|stream-json --json-schema --print-timeout` |

## What live probing established

**Flag contracts that would otherwise be guessed.** `claude --print` rejects
`--output-format stream-json` without `--verbose` ("Error: When using --print,
--output-format=stream-json requires --verbose"), so the adapter always emits
it. `codex exec --output-schema` takes a *file path*, not inline JSON, so the
adapter writes a temporary schema file outside the workspace.
`opencode run --format json` consumes a non-TTY stdin stream when no positional
message is supplied, so the managed adapter delivers its prompt on stdin and
closes the stream at EOF. Antigravity's verified `--print` text and JSON modes
take their prompt from argv; its stdin stream-json protocol requires a paired
stream-json output redesign, so the adapter does not substitute it. Every
rendered native argv is validated before a version probe or child spawn: POSIX
arguments are limited to 120,000 UTF-8 bytes and the
complete argv plus inherited child environment is kept below `SC_ARG_MAX` with
an 8 KiB safety margin; Windows validates the complete rendered command line
against its 32,767 UTF-16-unit limit. Native stderr and error-envelope text are
replaced with fixed or count-only warnings rather than being copied into
caller-visible diagnostics.
`agy --print-timeout` takes a duration string (`60s`) and bounds the *whole*
run, so the adapter derives it from the session's overall deadline rather than
from the shorter idle budget. `agent --print` stops on
an interactive workspace-trust gate — "Pass --trust, --yolo, or -f if you trust
this directory" — which is exactly the class of stall an unattended session must
never hit, so the adapter passes `--trust` while still withholding
`--force`/`--yolo`.

**Terminal status is not the same as exit status.** Claude's JSON envelope keeps
`"subtype": "success"` even on a failed turn; only `is_error` and
`terminal_reason` describe the real outcome. That is why the adapter reads
`is_error` and records `terminal_reason` as the native status, and why
`successful_native_statuses` is *not* declared for Claude: its text wire carries
no status at all, so a blanket declaration would be false.

**Envelopes and event names.** Recorded shapes now live as fixtures in
`tests/unit/test_ai_tools/test_headless_adapters.py`:

- Claude — `{"type":"result","subtype":…,"is_error":…,"terminal_reason":…,
  "result":…,"structured_output":…,"session_id":…,"permission_denials":[],
  "usage":{…}}`; `stream-json` prefixes it with `system/init` and `assistant`
  frames.
- Codex — `thread.started` (with `thread_id`), `turn.started`,
  `item.completed` (with a nested `item`), `error`, `turn.failed`
  (with `error.message`).
- Cursor — `{"type":"result","subtype":"success","is_error":false,"result":…,
  "session_id":…,"usage":{"inputTokens":…}}`; `stream-json` prefixes it with
  `system/init`, `user`, `thinking`, and `assistant` frames.
- OpenCode — `step_start`, `text` (response text under `part.text`), and
  `step_finish` (with `part.reason` and `part.tokens`), each carrying
  `sessionID`.
- Antigravity — `{"conversation_id":…,"status":"SUCCESS","response":…,
  "structured_output":…,"json_schema":…,"usage":{…}}`; `stream-json` wraps the
  same object as `{"event":"result","result":{…}}` after `init` and
  `step_update` frames.

Two of these streams echo caller input back — Cursor's `user` frame and agy's
`text_delta` — which is why normalized events carry only a frame's *kind*.

**Retry behavior justifies the outer deadline.** With invalid credentials both
`claude --print` and `codex exec` retried for far longer than a caller would
expect, Codex emitting five WebSocket and five HTTPS reconnect attempts before
`turn.failed`. A native timeout alone is not enough; Crossby's deadline and
process-group teardown are what bound the run.

## Not verified live

- **GitHub Copilot CLI stdout envelope.** The account available for probing is
  blocked by an org policy ("Access denied by policy settings"), so only
  `--help` flag contracts and the failure path (exit 1, diagnostics on stderr,
  empty stdout) were observed. Copilot therefore declares **text only**; its
  1.0.83 `--output-format json` (JSONL) flag exists but its event schema is
  unverified and is deliberately not advertised.
- **Codex success-path event names.** The account hit its usage limit
  mid-verification, so `turn.completed` and `item.completed` with
  `item.type == "agent_message"` come from Codex's documented exec JSONL schema
  rather than a captured success run. `thread.started`, `turn.started`,
  `error`, `item.completed`, and `turn.failed` were captured live. The parser
  accepts both `item.text` and `item.message`.
- **Antigravity waiting state.** The `AGY_WAITING` fixture is derived from agy's
  collected-plan waiting contract in `_agy_waiting` / `_agy_interaction`, not
  from a live capture. It exists to pin the policy: an unattended run routes the
  question through `context.interact()` and fails rather than stalling.

## Deterministic checks

`tests/unit/test_ai_tools/test_headless_adapters.py` replays every recorded
payload through a **real** child process, so process ownership, stdin delivery,
and cleanup are exercised rather than mocked. It covers per-adapter envelope
parsing, native failure on a zero exit, malformed output, schema enforcement and
pre-spawn schema rejection, prompt privacy in events and diagnostics, stdin
never being inherited, and a timeout that kills a grandchild in the owned
process group while keeping the streamed `thread_id` in the partial result.
