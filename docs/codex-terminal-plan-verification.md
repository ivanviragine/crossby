# Codex terminal Plan startup verification

Verified 2026-09-15 on macOS with Codex CLI 0.154.0. Startup and first-turn
submission were rechecked 2026-09-25 with Codex CLI 0.157.0.
Native planning through a final proposed plan was rechecked 2026-10-14 with
Codex CLI 0.162.1, model `gpt-6.1-sol`, and high effort.

## Why the terminal adapter exists

Codex has a native `/plan` command but no public CLI launch flag that selects
Plan before a positional prompt. Live experiments showed:

- A positional argument beginning `/plan` remains ordinary user text.
- Pasting `/plan` and a long task together triggers a large-paste placeholder;
  the slash command is then submitted as text.
- Submitting `/plan` alone, observing the Plan indicator, and then pasting the
  task starts its first turn in native Plan mode.
- An app-server thread with Plan settings and no completed turn could not be
  resumed by the native TUI (`no rollout found`).

The implementation follows the third path. The terminal display observations
are Crossby startup signals, not a public Codex event stream. Replace this
adapter when Codex exposes a verified native startup selector.

## Live implementation check

Used `CodexAdapter.launch(plan_mode=True, on_event=...)`, model `gpt-5.6-sol`,
low effort, workspace-write sandbox, and on-request approvals in a disposable
fixture. The callback submitted the task only on `PLAN_READY`.

- The 17,761-character task arrived byte-for-byte as one native user message.
- Native session `01a0a7f0-01d8-7763-a019-3174ab599113` recorded its first task
  turn with collaboration mode `plan`, effort `low`, and the requested sandbox
  and approval policy.
- Crossby emitted `PLAN_READY`, then `MESSAGE_SUBMITTED`, once each.
- Codex read `greeting.py` with its shell tool and asked a native
  `request_user_input` question. Keyboard input selected an answer in Codex.
- Codex rendered its proposed plan and native implementation decision menu.
  The probe stayed in Plan mode and exited normally with `/exit` (status 0).
- `greeting.py` remained unchanged.

This does not establish a plan-file destination or artifact collection contract;
those belong to the consumer's handoff or Crossby's existing collected API.

## Codex CLI 0.157.0 check

The 0.157.0 command menu places a selected `/plan` suggestion above the actual
`› /plan` composer. The earlier adapter read the first `›` line and timed out
before sending the task. Reading the bottommost composer line allowed Crossby to
observe the Plan indicator, emit `PLAN_READY`, submit a short task, and emit
`MESSAGE_SUBMITTED` after the first native Plan turn began. This live check used
an isolated temporary directory and exited normally. It did not exercise the
full plan artifact handoff or every approval/sandbox combination.

## Codex CLI 0.162.1 check

The installed `/opt/homebrew/bin/codex` reported `codex-cli 0.162.1`.
Direct screen inspection used a disposable fixture and this native command:

```sh
/opt/homebrew/bin/codex --no-alt-screen -m gpt-6.1-sol \
  --sandbox workspace-write -a on-request \
  -c 'model_reasoning_effort="high"' \
  -c 'plan_mode_reasoning_effort="high"'
```

Observed screens, with directory/session details sanitized in
`tests/fixtures/plan_sessions/codex_0_162_1/`:

- Startup shows `>_ OpenAI Codex (v0.162.1)` and a directory, with no `model:`
  header. The empty composer remains `› Ask Codex to do anything`.
- The status footer starts `GPT-6.1-Sol high · /…`. At 140 columns it also
  includes `Context 0% used`, usage/session information, and a warning. At 80
  columns the directory is truncated and Context is omitted; at 100 columns
  Context itself is truncated. Readiness requires the native banner, bottommost
  empty composer, and model/effort/directory footer, rather than Context text.
  Independent native captures with the default layout and an explicit minimal
  `tui.status_line=["model-with-reasoning","current-dir"]` also omit Context;
  both layouts are included in the regression fixtures.
- Typing `/plan` shows a selected `› /plan  switch to Plan mode` suggestion
  above the actual `› /plan` composer. Enter activates the footer's `Plan mode`
  label, followed by the warning. The empty composer text is unchanged.
- Short task text wraps in the composer. A real 1,855-character bracketed paste
  displays `[Pasted Content 1855 chars]`. The working screen retains the Plan
  footer and includes `Working (… • esc to interrupt)`.
- Native `request_user_input` replaces the composer/footer with a question and
  numbered choices. The final `Proposed Plan` opens the native `Implement this
  plan?` menu. Neither screen is treated as a ready composer by regression tests.

The candidate-code check used the public `CodexAdapter.launch()` with
`plan_mode=True`, `model="gpt-6.1-sol"`, `effort=EffortLevel.HIGH`,
`sandbox=True`, and an `on_event` callback. Candidate `src/` was supplied through
`PYTHONPATH`; the installed Crossby/Wade packages were not changed. The adapter
built the terminal wrapper around:

```sh
codex --model gpt-6.1-sol --no-alt-screen \
  -c 'model_reasoning_effort="high"' \
  -c 'plan_mode_reasoning_effort="high"'
```

This disposable non-worktree launch used default approvals (no yolo, auto, or
accept-edits overrides) and normal existing authentication without copying or
printing credentials or editing user config. The native rollout recorded
`workspace-write` with network disabled, and an effective granular approval
policy: sandbox approval, rules, and skill approval false; permission requests
and MCP elicitations true. No trust or command-approval prompt appeared, so this
run does not verify interactive approval handling.

At 80 columns and 40 rows, Crossby emitted `PLAN_READY` once, sent one task, then
emitted `MESSAGE_SUBMITTED` once. Codex read only the fixture's `greeting.py` and
asked which greeting to use. Enter selected `Hi (Recommended)` through the native
question UI. Codex rendered a final proposed plan to replace `print("Hello")`
with `print("Hi")` and verify its output. The probe explicitly selected `No,
stay in Plan mode`, then exited with `/exit` (status 0); the fixture remained
unchanged.

The run's own rollout confirmed native collaboration mode `plan`, model
`gpt-6.1-sol`, and effort `high`, with exactly one task. CLI 0.162.1 records the
user task as a `response_item` message, rather than the older `event_msg`
`user_message`, and trims its final newline: 1,855 characters sent, 1,854 recorded.
The initial evidence postprocessor assumed the older event type and failed after
the successful native exit; checking the current message representation confirmed
the task exactly after that native newline normalization.

A subsequent supervisor probe of Wade in a fresh detached worktree exposed a
second startup boundary: Codex displays `Folder access`, `Trust this folder?`,
`1. Trust and continue`, `2. Quit`, and `enter continue · esc quit`. The old
legacy-only trust matcher rejected the caller's Enter key. The adapter now
recognizes the complete observed modal and relays actual caller keystrokes while
waiting for readiness. It never generates trust approval; partial or unknown
dialogs remain blocked under the same startup deadline. Unit tests cover the
Trust and Quit selections, incomplete/unknown screens, and the legacy prompt.
Nested PTY coverage verifies no task/event before an explicit caller Enter, then
normal Plan startup. The sanitized native trust capture is `trust.txt` beside
the current-screen fixtures.

The supervisor then completed a separate actual Wade consumer probe with
candidate Crossby and Wade source on `PYTHONPATH`: `run_interactive_planning_session()`
in a freshly bootstrapped detached Git worktree, `gpt-6.1-sol`/high, sandbox
workspace-write, network disabled, and on-request approvals. The caller
explicitly pressed Enter at the observed trust modal. Native Plan activated,
one full Wade task arrived, and the planner reached `wade plan-session check`
(`IN_WORKTREE`). The caller answered the native question with `Finalize`.
The native planner produced the final greeting repair plan, imported it through
`wade plan-session done --from-stdin`, and obtained an independent review using:

```sh
wade review plan .wade/plans/PLAN.md --ai codex --model gpt-6.1-sol \
  --effort high --mode headless --sandbox
```

The first review launch failed under the host sandbox; retrying with approved
host execution kept the reviewer itself sandboxed and returned no material
findings. No self-review acknowledgement substituted for the external review.
`wade plan-session done` then validated the reviewed plan; implementation was
declined, `/exit` returned 0, and the actual parent collector received one
completed, reviewed plan. Independent recollection confirmed the handoff, one
initial task, native Plan/model/high context, and unchanged `greeting.py`.
The supervisor retained `215-wade-evidence.json`, native session
`01a13a03-226d-7872-8764-82175565cd42`, and completed Wade handoff
`3f17a75a-1970-42a9-b10d-a314bd0059ae` under `/tmp/crossby-owned`.
This consumer probe exercised native launch/import/review/done/collection; it
created no provider issues or PRs and did not update installed packages.

The bootstrap repair plan and independent headless reviews are separate from
the two real native planning proofs. The current model was not run on CLI
0.154.0, which rejects it; legacy screen support is retained through
deterministic tests and the earlier live evidence. Other custom status layouts
that omit the observed model/effort/directory structure remain fail-closed.
Directory text such as `/tmp/loading` is excluded from loading-status checks.

## Deterministic checks

`./scripts/test.sh tests/unit/test_ai_tools/test_codex_terminal.py
 tests/integration/test_codex_terminal.py` exercises the state machine and real
nested PTYs with a fake native CLI. Coverage includes deferred input, duplicate
submission rejection, long/unicode messages, native keyboard handoff, window
resize, terminal query replies, trust prompts, timeout, early exit, cancellation,
exit-code propagation, and terminal restoration. Approval/sandbox combinations
are checked independently without requesting unrestricted live execution.

## Scope

Interactive startup requires Codex 0.154.0 or newer on POSIX and must run on the
main thread with terminal stdin/stdout. Newer versions are not pinned:
unrecognized screens fail closed within the startup deadline without
submitting the task. The real provider run was on
macOS; Linux PTY behavior is covered by the deterministic CI suite. Windows is
unsupported for this adapter. App-server collection remains independent and
retains its 0.153.4 floor.
