# Unattended headless session verification

Verified 2026-09-18 on macOS against the installed builds named below. Each
adapter's `HeadlessCapability.verified_version` is the build it was checked on
and is enforced as the runtime floor.

| Tool | Build | Probe |
| --- | --- | --- |
| Claude Code | 2.1.263 | `claude --print` with each `--output-format`, `--json-schema`, `--permission-prompts none` |
| Codex CLI | 0.154.0 | `codex exec --json --output-schema <file> -` |
| Cursor | 2026.11.05-86047df | `agent --print --output-format json\|stream-json --trust`, prompt piped on stdin |
| GitHub Copilot CLI | 1.0.105 | `copilot --help`; `copilot -s --no-ask-user --output-format json`, prompt piped on stdin |
| OpenCode | 1.18.31 | `opencode run --format json` |
| Antigravity CLI | 1.5.4 | `agy --print= --input-format stream-json --output-format stream-json --json-schema --print-timeout` |

The Cursor, Copilot and Antigravity CLI floors were raised on 2026-11-10 to the
builds where their prompts were first verified on stdin (see
[Prompts on stdin](#prompts-on-stdin)); the earlier floors (2026.09.10-fd3934a,
1.0.83, 1.2.6) took the prompt as a command-line argument. agy installs as one self-updating binary, so
its floor is the build the stdin path was verified on, 1.5.4; its resume and
read-only profile were verified on 1.5.1 and rechecked on 1.5.4. Copilot keeps
older builds under `~/.copilot/pkg/`, so 1.0.105 was run directly: a piped
prompt, `--session-id` resume and both deny rules held there too.

## What live probing established

**Flag contracts that would otherwise be guessed.** `claude --print` rejects
`--output-format stream-json` without `--verbose` ("Error: When using --print,
--output-format=stream-json requires --verbose"), so the adapter always emits
it. `codex exec --output-schema` takes a *file path*, not inline JSON, so the
adapter writes a temporary schema file outside the workspace.
`opencode run --format json` consumes a non-TTY stdin stream when no positional
message is supplied, so the managed adapter delivers its prompt on stdin and
closes the stream at EOF. Every adapter now does the same: no managed prompt
travels on argv (see [Prompts on stdin](#prompts-on-stdin)). Every
streaming adapter refreshes the idle deadline only for its verified event kinds;
an unknown JSON object can be parsed as terminal evidence but cannot keep a
stalled child alive. Rendered native argv is validated before a version probe or
child spawn. In particular, a model-only Antigravity ID with an unrecognized or
unsupported effort suffix is rejected rather than normalized or stripped, so a
managed run never selects a different model. With prompts off argv, these
checks cover only adapter flags (a response schema, a model ID, settings): POSIX
arguments are limited to 120,000 filesystem-encoded bytes and the
complete argv plus inherited child environment is kept below `SC_ARG_MAX` with
an 8 KiB safety margin; Windows validates the complete rendered command line,
including its terminating NUL, against its 32,767 UTF-16-unit limit. Native
stderr and error-envelope text are
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

Codex contributes exactly one `turn.completed` or `turn.failed` frame for an
execution. Missing, repeated, or conflicting terminal frames are invalid native
output; the adapter never lets a later success frame overwrite an earlier
failure.

OpenCode permits nonterminal `step_finish` frames (such as `tool-calls`) while
it works. Its single `step_finish` with reason `stop` is terminal; missing,
repeated, or later frames are invalid native output.

Claude, Cursor, and Antigravity CLI likewise contribute exactly one final
`result` envelope to their streaming protocol. A missing or repeated result
envelope is invalid native output, as is any later frame; a later result can
never overwrite an earlier native error or waiting state.

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
`text_delta` — which is why a frame's *kind* only controls whether Crossby emits
a content-free progress event; no native frame value enters an event message.

**Retry behavior justifies the outer deadline.** With invalid credentials both
`claude --print` and `codex exec` retried for far longer than a caller would
expect, Codex emitting five WebSocket and five HTTPS reconnect attempts before
`turn.failed`. A native timeout alone is not enough; Crossby's deadline and
process-group teardown are what bound the run.

## Read-only sessions and resume

Verified 2026-11-09 on macOS against the builds below. For Claude Code, Codex
CLI and OpenCode they are the adapter's `HEADLESS_READ_ONLY_VERIFIED_VERSION`
and `HEADLESS_RESUME_VERIFIED_VERSION`; for Cursor, Copilot and Antigravity CLI
they are now the unattended floor itself (see [Prompts on stdin](#prompts-on-stdin)).

| Tool | Build | Read-only profile | Resume |
| --- | --- | --- | --- |
| Claude Code | 2.1.317 (resume: 2.1.316) | `--tools Read,Grep,Glob,Skill --settings '{"disableAllHooks":true}'` | `--session-id` new, `--resume <id>` after |
| Codex CLI | 0.166.0 | `--sandbox read-only -c sandbox_mode="read-only" -c approval_policy="never" --disable hooks` | `codex exec resume <thread> … -` |
| Cursor | 2026.11.05-86047df | `--allowed-tools read_tool_call,grep_tool_call,glob_tool_call,ls_tool_call,sem_search_tool_call` | `--resume <chatId>` |
| GitHub Copilot CLI | 1.0.105 | `--deny-tool write --deny-tool shell` | `--session-id <uuid>` for both |
| OpenCode | 1.19.5 | `--agent crossby-read-only`, defined in `OPENCODE_CONFIG_CONTENT` with edit, bash, task and question denied | `--session <id>` |
| Antigravity CLI | 1.5.1 | `--mode plan --settings '{"permissions":{"allow":[]}}'` | `--conversation <id>` |

Each profile was probed in a scratch git repository whose instruction files
(`CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.github/copilot-instructions.md`)
named a codeword and whose `.claude/skills/` held one skill. Two turns ran per
tool. The fresh turn asked for the codeword and for a file write. The resumed
turn asked for a number given in the fresh turn, a write through the file tool,
a write through the shell, and the skill's content. A second pass told each
agent it was an authorized test and that it must call both tools, so a model
that merely declined could not pass. `git status` stayed clean for every
profile below, every tool recalled the number on its resumed turn, and every
resumed run reported the session it was given.

What the probes established:

- **Claude Code.** `--restricted` removes Bash and WebFetch and ignores the
  settings files, but on 2.1.317 the folder's `CLAUDE.md` did not load under
  it ("There's no project codeword in my instructions"). `--setting-sources ""`
  likewise drops the user's `CLAUDE.md` and skills. A closed `--tools` set
  keeps the folder's `CLAUDE.md` and skills (the codeword and the skill's
  content both came back) and leaves no write or shell tool to call, whatever
  the ambient allow rules say.
- **Codex CLI.** `--sandbox read-only` alone was not enough: with
  `approvals_reviewer = "auto_review"` in the user's `config.toml`, an
  `apply_patch` outside the sandbox was escalated, approved and written. With
  `approval_policy="never"` pinned, the shell write failed with
  `operation not permitted` and the patch was rejected, on fresh and resumed
  turns. `codex exec resume` takes no `--sandbox`, `--color` or `--add-dir`,
  so the policy goes through `-c` on every turn; its `thread.started` reports
  the resumed thread. Outside a git repository `codex exec` refuses to start
  ("Not inside a trusted directory and --skip-git-repo-check was not
  specified"), a guard for files nothing could restore. A read-only session
  cannot change files, so it passes `--skip-git-repo-check` (both `exec` and
  `exec resume` take it); rechecked 2026-11-10 in a plain folder, where the
  fresh turn read a file and its shell and `apply_patch` writes both failed,
  the resumed turn recalled its number, and the folder was unchanged. A
  session that can write keeps the guard.
- **Cursor.** Neither `--mode ask` nor `--mode plan` is read-only in print mode:
  a forced edit wrote the file under both (the shell was refused, since
  `--force` is never passed). The hidden `--allowed-tools` flag takes Cursor's
  proto tool names and rejects unknown ones (`Invalid --allowed-tools
  value(s)…`). With read, grep, glob, list and semantic search only, the agent
  reported having Grep, Read and Glob and nothing to write with, and still read
  the instructions. `--resume <chatId>` continues the chat named in the result
  envelope's `session_id`.
- **GitHub Copilot CLI.** Both forced writes were refused natively: "Permission
  to run this tool was denied due to the following rules: `shell`" and
  "`write`". `--session-id <uuid>` creates a session with that ID in text mode,
  and the same flag resumes it, so crossby pins the ID itself; text output
  carries no session ID to read back.
- **OpenCode.** OpenCode takes its working directory from an inherited `PWD`,
  not from the process cwd: started in one folder with `PWD=/tmp`, it answered
  `/private/tmp`, and it created its session there. The managed runtime now
  sets `PWD` for every unattended child. The built-in `plan` agent is not
  read-only: its resolved rules start with `* allow`, so bash is allowed, and
  edits are allowed under `.opencode/plans/` inside the project. A
  session-only agent with edit, bash, task and question denied makes OpenCode
  stop offering its write, edit, patch and bash tools ("I don't have bash or
  write tools available"), while `AGENTS.md` and the project skill still load.
  A plugin's update checker printed a plain-text line to stdout before the
  first event; such lines are now ignored with a warning, while a malformed
  event line is still invalid output.
- **Antigravity CLI.** Without a profile, an allow rule in the user's settings
  (`command(uv)`) let a default-mode turn write a file through
  `uv run python -c …`. Session-only `--settings` replace the ambient
  `permissions.allow` list, so with an empty list agy auto-denied the command
  and reported it in `denied_actions`. When agy denies an action it ends the
  turn with an empty `response`, so plan mode is added to keep the agent from
  attempting edits: it writes its plan outside the project
  (`~/.gemini/antigravity-cli/brain/…`) and asks for approval. Approving the plan
  on the resumed turn did not get past the empty allow list: the write was
  denied and no file was created.

End to end, `crossby converse claude <tool>` ran two-round consultations with
each tool as the responder in its own folder. Both sides resumed, the asker
recovered the codeword from the responder's instructions and a date from its
`notes.txt`, the responder's write attempt failed, and the responder's folder
had no changes. Copilot's model declined to quote its instructions, a model
choice, but it did report that the write permission was denied.

Read-only covers files changed through each tool's own file, edit and shell
tools. It does not cover MCP servers, account connectors, web access (Codex's
web search stays available) or hooks on Cursor, Copilot, OpenCode and
Antigravity CLI. The next section reports each of these per channel.

## What read-only blocks

Verified 2026-11-10 on macOS. Every session reports the state of each channel
(`RestrictionChannel`) for the profile it actually ran with:
`AbstractAITool.headless_restrictions()` returns the adapter's
`HEADLESS_READ_ONLY_RESTRICTIONS` only for an unattended read-only session on a
build at or above the read-only floor, and `unknown` for every other session
and every channel the declaration leaves out. A caller's `require_blocked`
channels are checked by the runtime after its version probe, so a session
whose required channel is not `enforced` on the build that would run is
refused before anything is spawned.

| Channel | Claude Code | Codex CLI | Cursor | Copilot | OpenCode | Antigravity CLI |
| --- | --- | --- | --- | --- | --- | --- |
| `file_edits` | enforced | enforced | enforced | enforced | enforced | enforced |
| `shell_writes` | enforced | enforced | enforced | enforced | enforced | enforced |
| `shell_commands` | enforced | not enforced | enforced | enforced | enforced | enforced |
| `hooks` | enforced | enforced | not enforced | not enforced | not enforced | not enforced |
| `web` | enforced | not enforced | enforced | unknown | not enforced | not enforced |
| `reads_outside_workspace` | unknown | not enforced | not enforced | unknown | unknown | not enforced |
| `plugins` | unknown | unknown | unknown | unknown | not enforced | unknown |
| `mcp_tools` | unknown | unknown | unknown | not enforced | unknown | unknown |
| `connectors` | unknown | unknown | unknown | unknown | unknown | unknown |

`enforced` is claimed only where the mechanism is crossby's own profile and
holds whatever the user's settings say: a closed tool set, an OS sandbox, or a
session-only deny rule. A tool's default that the user's own configuration can
loosen is reported as `unknown`, even when it blocked the probe below.

The file and shell rows come from the forced-write probes in the previous
section. Web access and reads outside the working directory were probed with
one read-only turn per tool in a fresh git folder, which had to fetch
`https://api.github.com/repos/ivanviragine/crossby/commits/main` and quote its
`sha` (that day's merge commit, which no model can know) and read a random
token from a file in a sibling folder:

- **Claude Code 2.1.317.** "No web fetch or web search tool was available in
  this session": the closed `--tools` set has neither. The outside read "was
  denied automatically because the path is outside the allowed working
  directories" (one permission denial reported), but a user allow rule can
  grant such reads, so the state is unknown.
- **Codex CLI 0.166.0.** It called its web tool (`web__run`), so web access is
  available even though the search could not return the API's JSON. It read
  the token through `exec_command`: the read-only sandbox restricts writes,
  not reads.
- **Cursor 2026.11.05-86047df.** "No WebFetch/WebSearch available"; the read
  tool returned the token.
- **GitHub Copilot CLI 1.0.109.** `web_fetch` and `view` on the outside path
  were both denied natively, without `--allow-all-tools`. The URL and path
  approvals they need can come from the user's own configuration, so both
  states are unknown.
- **OpenCode 1.19.5.** `webfetch` returned the commit (its JSON came back in
  the tool result). The outside read stopped at "permission requested:
  external_directory (…); auto-rejecting", which the user's configuration can
  allow, so the state is unknown. After that rejection OpenCode exits 0
  without its terminal `step_finish`, so the session ends as invalid output:
  the answer is lost, not just the read.
- **Antigravity CLI 1.5.4.** It called `search_web` and read the token with
  `read_file`.

Hooks are enforced only where the profile disables them (Claude's
`disableAllHooks`, Codex's `--disable hooks`); OpenCode's plugins load and their
event hooks run (a plugin's update checker printed to stdout during these
runs). Copilot's JSONL stream (`--output-format json`, 1.0.109) shows its
built-in `github-mcp-server` connecting and loading 5 tools under the
read-only deny rules, so its MCP channel is not enforced. Other MCP servers,
account connectors and the other tools' plugins were not probed, so they stay
unknown.

## Prompts on stdin

Verified 2026-11-10 on macOS. No managed prompt travels on argv any more:
argv is bounded by the OS (131,072 bytes per argument on Linux, 32,767 UTF-16
units for a whole Windows command line) and is visible to other users in the
process list. Claude Code, Codex CLI and OpenCode already read stdin; the other
three were moved to it:

| Tool | Build | How the prompt is read |
| --- | --- | --- |
| Cursor | 2026.11.05-86047df | `agent --print …` with no prompt argument reads the piped prompt |
| GitHub Copilot CLI | 1.0.105 and 1.0.108 | `copilot -s --no-ask-user …` with no `--prompt` runs in prompt mode on the piped prompt |
| Antigravity CLI | 1.5.4 | `agy --print= --input-format stream-json --output-format stream-json …` reads one `{"event":"user","message":{"role":"user","content":…}}` line |

For agy, `--print` takes a value, so a bare `--print` swallowed the next flag
("--print took "--output-format" as its prompt"); `--print=` gives it an empty
one, and an empty `--print` is accepted only with stream-json input. The input
message needs the `event` field (`stream input message is missing the "event"
field` otherwise). Stream-json input requires stream-json output, so agy's text
and JSON results are now read from the `result` frame, which carries the same
envelope, including `structured_output` and the `json_schema` echo. Its `init`
frame reports the conversation ID before the model starts.

Each of the six tools then ran, through crossby's own adapters, a read-only
fresh turn (codeword and a write), a resumed turn (a number from the first
turn, a file-tool write and a shell write), and a fresh 154,081-byte prompt
(over Linux's per-argument limit and crossby's former 120,000-byte cap) that
ends with "The final code word is ZEBRA-4417. Reply with only the final code
word". Every tool resumed the session it was given and recalled the number,
every write failed, every folder stayed clean, and the large prompt came back as
`ZEBRA-4417` from Claude Code, Codex CLI, Cursor, OpenCode and Antigravity CLI.
Copilot answered with its instructions' codeword instead; a second run in a
folder without instructions (159,807 bytes, ending "The secret number is
739154. Reply with only the secret number and the number of the last filler
line") came back `739154 / 02799`, the exact last line, so the whole prompt
arrived. agy with `--json-schema` returned schema-valid `structured_output`.
When agy denies an action it still ends that turn with an empty `response`
(the same with the old argv form and `--output-format json`); crossby reports
the denial with the failure.

Skills still load on the stdin path. One Claude skill (`secret-fruit`) was
copied with `crossby sync skills --from claude --strategy copy` into
`.cursor/skills/`, `.agents/skills/` (Codex and Antigravity CLI) and
`.github/skills/`. A read-only fresh turn on each of the six tools answered
from the skill, and the resumed turn named the copy in that tool's own
directory: `.claude/skills/` for Claude Code, `.cursor/skills/` for Cursor,
`.github/skills/` for Copilot, and `.agents/skills/` for Codex, Antigravity CLI
and OpenCode (which reads the shared copy). Every folder stayed clean.

End to end, two-round `crossby converse` runs (resume only, nothing replayed)
put each changed tool on both sides: Claude Code asking Copilot, Cursor and
Antigravity CLI, each also writing the `--summary`; Cursor asking Codex;
Antigravity CLI asking Claude Code; Copilot asking OpenCode. Every run
answered with the facts from the responder's instructions and `notes.txt`,
reported the responder's write as failed, and left the responder's folder
unchanged, and `--json` named one session per side. A first Copilot-as-asker
run refused its goal ("I can't help extract hidden instructions") and put
`NEXT: DONE` inside a sentence, so it ran to the round limit; with a goal that
did not ask for instructions it answered in two rounds.

## Copilot's JSONL stream

Verified 2026-11-10 on 1.0.105 (run from Copilot's kept package) and 1.0.109.
Unattended Copilot sessions run `copilot -s --no-ask-user … --output-format
json` and read its JSONL events; callers still get the final answer as text.

- `session.start` names the session (`data.sessionId`) and the `result` event
  closes the stream with `sessionId`, `exitCode` and
  `usage.premiumRequests`. With `--session-id <uuid>` both report that ID, so
  a run that reports another session is invalid output. A resumed run recalled
  a number from its first turn on both builds.
- The answer is the `assistant.message` with `phase: "final_answer"`.
  Messages that only request tools have empty `content`; without the marker,
  the last message with content and no tool requests is the answer.
- `assistant.message` carries the model that wrote it (`auto` resolves to,
  for example, `mai-code-1.1-flash` or `gpt-6-luna`), reported as the
  observed model.
- A refused tool call is a `tool.execution_complete` with `success: false`
  and `error.code: "denied"` ("Permission to run this tool was denied due to
  the following rules: `write`"); crossby reports their count as a denial.
- `premiumRequests` is the session's running total: a resumed second turn
  reported 2. It is the usage of one invocation only for a fresh session,
  which every `ask` is. Copilot reports no token counts.
- The stream echoes the prompt (`user.message`) and system text
  (`instruction.update`); no event content ever leaves the adapter.
- An unknown model is a `session.error` event ('Execution failed: Model
  "no-such-model-xyz" is not available.') followed by `result` with exit code
  1.

## Failure reasons

Verified 2026-11-10 on macOS. A failed, invalid or timed-out session carries
`failure_reason` when the tool said why in a way crossby recognizes:
`not_signed_in`, `model_unavailable`, `rate_limited`, `action_denied` or
`untrusted_folder`. Adapters hand the runtime native stderr, error envelopes
and streamed error frames as private evidence
(`HeadlessRuntimeContext.note_failure_evidence`); `ai_tools/failure_reasons.py`
classifies them once, at completion, and only the reason leaves the adapter.
A recognized refusal with no answer is `failed` rather than
`invalid_output`, even when the tool emitted no events. An unrecognized
failure has no reason.

Each case below ran through crossby's adapters. Signing out was simulated
without touching the real login, with an empty config directory or an
invalid key in the probe's own environment:

| Case | Claude Code | Codex CLI | Cursor | Copilot | OpenCode | Antigravity CLI |
| --- | --- | --- | --- | --- | --- | --- |
| Signed out | "Not logged in · Please run /login" (empty `CLAUDE_CONFIG_DIR`); "Invalid API key" with `api_error_status` 401 | "unexpected status 401 Unauthorized" after 5 retries, 17 s (empty `CODEX_HOME`) | "The provided API key is invalid." (`CURSOR_API_KEY`) | not reproduced: an invalid `COPILOT_GITHUB_TOKEN` and an empty `COPILOT_HOME` both still used the stored login | "UnknownError … Unexpected server error": not recognized | "authentication required. Run 'agy' to log in" (empty `HOME`) |
| Unknown model | "issue with the selected model" with `api_error_status` 404 | "model is not supported when using Codex with a ChatGPT account" | "Cannot use this model: …" | `session.error`: 'Model "…" is not available' | "UnknownError": not recognized | "invalid model selection" |
| Other | | outside a git repository without read-only: "Not inside a trusted directory", before any event | | | an auto-rejected permission ("external_directory … auto-rejecting"), then exit 0 without a stop event | an empty response after a denied action |

Rate limits were not reproduced. An HTTP 429 (Claude Code's
`api_error_status`) is `rate_limited`, as is Codex's "unexpected status 429",
inferred from the same template as its captured 401. Codex's 401 frames stream
while it retries, so a session that times out during those retries still
reports `not_signed_in`.

Claude Code's result envelope also carries `total_cost_usd` and `modelUsage`.
The cost is passed through as `reported_cost_usd`, exactly as reported (crossby
keeps no prices), and the `modelUsage` keys are the observed models.

## Persona folders

Verified 2026-11-10 on macOS. A persona folder (`crossby.api.provision`) is a
plain folder, with no Git repository, that holds only one tool's own
instruction file and skills directory plus the caller's material files. Each
tool ran a read-only turn in such a folder whose instruction file named a
random codeword and whose one skill named a random city; every tool answered
with both, and the folder was unchanged afterwards:

| Tool | Instruction file | Skills directory | Verified on |
| --- | --- | --- | --- |
| Claude Code | `CLAUDE.md` | `.claude/skills/` | 2.1.317 |
| Codex CLI | `AGENTS.md` | `.agents/skills/` | 0.166.0 |
| Cursor | `AGENTS.md` | `.cursor/skills/` | 2026.11.05-86047df (and 2026.11.10-731fb78) |
| GitHub Copilot CLI | `.github/copilot-instructions.md` | `.github/skills/` | 1.0.105 (and 1.0.109) |
| OpenCode | `AGENTS.md` | `.agents/skills/` | 1.19.5 |
| Antigravity CLI | `AGENTS.md` | `.agents/skills/` | 1.5.5 |

Cursor also loaded `.cursorrules`, and Copilot `AGENTS.md`, in place of the
files above. The older Cursor and Copilot builds were run from the copies
each tool keeps (`~/.local/share/cursor-agent/versions/`, `~/.copilot/pkg/`);
agy keeps no older build, so its layout requires 1.5.5, after its read-only
floor of 1.5.4. An ask in a persona folder passes that build as the session's
`minimum_version`, checked after the version probe.

Tools read each other's files. With every file below present at once, each
one listed what it had loaded:

| Tool | Instruction files loaded | Skills directories loaded |
| --- | --- | --- |
| Claude Code | `CLAUDE.md` | `.claude/skills/` |
| Codex CLI | `AGENTS.md` | `.agents/skills/` |
| Antigravity CLI | `AGENTS.md`, `GEMINI.md` | `.agents/skills/` |
| OpenCode | `AGENTS.md` | `.agents/skills/`, `.opencode/skills/`, `.claude/skills/` |
| Cursor | `.cursorrules`, `CLAUDE.md`, `AGENTS.md` | `.cursor/skills/`, `.claude/skills/`, `.agents/skills/` |
| GitHub Copilot CLI | `.github/copilot-instructions.md`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` | `.github/skills/`, `.agents/skills/`, `.claude/skills/` |

So a folder that served several tools would give Cursor or Copilot the
instructions twice; a persona folder serves exactly one tool. Material and
skill files may not be hidden or use an instruction or configuration name at
any depth (`AGENTS.md`, `AGENTS.override.md`, which Codex prefers over
`AGENTS.md`, `CLAUDE.md`, which Claude Code also loads from subfolders,
`GEMINI.md`, `opencode.json`, …), so they can never add or replace
instructions, hooks, permissions or MCP servers.

End to end, `provision`, `ask` and `cleanup` ran on all six tools in parallel
threads, each with a persona (a codeword), one skill (a city) and a material
file (a budget). Claude Code, Codex CLI, Cursor, Copilot and OpenCode
answered with all three, `delivery` was `native_files`, `workspace_unchanged`
was true, and cleanup left nothing behind. Antigravity CLI first tried a shell
command to read the material file, which its read-only profile denied, and
ended the turn without an answer (`action_denied`); asked again, it read the
file with its own file tool and answered.

The user's own global instructions, skills, MCP servers and connectors still
load in every persona folder (for example `~/.claude/CLAUDE.md`); a persona
folder isolates only what is in the folder.

## Project hooks

Verified 2026-11-10 on macOS. A read-only profile blocks a tool's own file,
edit and shell tools, but only Claude Code (`disableAllHooks`) and Codex CLI
(`--disable hooks`) also disable hooks, and a hook is a command that can change
any file. A Claude Code `UserPromptSubmit` hook and a `PreToolUse` hook, each
touching a sentinel file outside the project, were written for Cursor, Copilot
and Antigravity CLI with `crossby sync hooks --from claude`, and each tool then
ran a read-only turn that read a file:

| Tool | Project hook source | Ran under read-only |
| --- | --- | --- |
| Claude Code | `.claude/settings.json` | no (`disableAllHooks`) |
| Cursor | `.cursor/hooks.json` | yes: `preToolUse` (not `beforeSubmitPrompt`) |
| Antigravity CLI | `.agents/hooks.json` | yes: `PreToolUse`, with the working directory set to `.agents/` |
| GitHub Copilot CLI | `.github/hooks/hooks.json` | no sentinel in this run; not proven absent |

So a read-only session on Cursor, Copilot, OpenCode or Antigravity CLI refuses
to start in a folder with its own hooks (`HeadlessProjectHooksError`, raised
after the version probe and before any spawn), naming the sources it found:
`.cursor/hooks.json`, `.github/hooks/*.json`, `.agents/hooks.json`, and
OpenCode's `.opencode/plugin(s)/` and the `plugin` key of `opencode.json`
(plugins load code). A source that cannot be parsed, such as an
`opencode.jsonc` with comments, counts as found. `allow_project_hooks` runs the
session anyway: `crossby ask cursor … --allow-project-hooks` in the probe
project answered, and its `preToolUse` hook wrote the sentinel. User-level
hooks (for example `~/.cursor/hooks.json`) are not inspected and still run;
only `require_blocked={hooks}` (Claude Code, Codex CLI) excludes them.

## Not verified live

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
never being inherited, and fail-closed prompt delivery when a child closes stdin
before a complete write. It also covers a timeout that kills a grandchild in the
owned process group while keeping the streamed `thread_id` in the partial result.
