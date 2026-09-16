# Codex terminal Plan startup verification

Verified 2026-09-15 on macOS with Codex CLI 0.154.0.

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

## Deterministic checks

`./scripts/test.sh tests/unit/test_ai_tools/test_codex_terminal.py
 tests/integration/test_codex_terminal.py` exercises the state machine and real
nested PTYs with a fake native CLI. Coverage includes deferred input, duplicate
submission rejection, long/unicode messages, native keyboard handoff, window
resize, terminal query replies, trust prompts, timeout, early exit, cancellation,
exit-code propagation, and terminal restoration. Approval/sandbox combinations
are checked independently without requesting unrestricted live execution.

## Scope

Interactive startup is limited to Codex 0.154.x on POSIX and must run on the
main thread with terminal stdin/stdout. Unknown versions and unrecognized
screens fail closed within the startup deadline. The real provider run was on
macOS; Linux PTY behavior is covered by the deterministic CI suite. Windows is
unsupported for this adapter. App-server collection remains independent and
retains its 0.153.4 floor.
