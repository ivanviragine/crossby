# Native interactive Plan launch verification

Checked on 2026-09-15. This change covers ordinary interactive `launch()` and
`crossby launch --plan`. Codex, collected-session transports, and WADE integration
are outside this change.

## Native approval combinations

| Tool | Version exercised | Plan + skip approvals | Plan + classifier approval |
| --- | --- | --- | --- |
| Claude Code | 2.1.263 | `--permission-mode plan --allow-dangerously-skip-permissions` | `--permission-mode plan --settings '{"useAutoModeDuringPlan":true}'` |
| Cursor CLI | 2026.09.02-c22c1a3 | `--mode plan --force` | `--mode plan --auto-review` |
| GitHub Copilot | 1.0.83 | `--plan --yolo` | No mapping declared |
| OpenCode | 1.18.29 | `--agent plan --auto` | No mapping declared |
| Antigravity CLI | 1.2.3 | `--mode plan --dangerously-skip-permissions` | No mapping declared |

Copilot can additionally compose Plan with `--allow-tool write`. Other
accept-edits combinations are rejected: a different execution selector must not
replace Plan, and ordinary Agent-mode defaults do not establish Plan behavior.
No permission setting enables Copilot's autopilot or approves implementation.

## Evidence and limitations

- All five actual native terminal UIs were started with Plan and their native
  skip-approval option. No slash-command injection, protocol bridge, or WADE
  question renderer was used.
- Claude, Cursor, OpenCode, and Antigravity executed a local Python probe and
  returned a plan while remaining in native Plan mode. No shell approval was
  entered. Folder trust was confirmed where required; this is distinct from
  tool-call approval. The public Crossby `launch()` path was exercised with
  saved argv, terminal transcripts, and result files.
- Copilot reached its native Plan UI, but its model request failed with an
  authorization error. The user requested continuing without the authenticated
  Copilot test. Its live command execution is therefore **unverified**.
- Cursor initially rejected a named model because the account allows only Auto.
  Retrying with its native `--model auto` completed the probe.
- Claude and Cursor also executed the read-only probe with classifier approval.
  Claude reported "Allowed by auto mode classifier" with Plan still selected;
  Cursor showed both Plan and Auto-review and the successful shell output.
- Claude's ordinary `--dangerously-skip-permissions` selected bypass mode even
  when `--permission-mode plan` was present. The documented interactive
  `--allow-dangerously-skip-permissions` combination preserved Plan instead.
- An initial probe explicitly wrote a marker file. Claude declined it under
  its planning instructions before making a shell call. The reusable probe now
  only prints a unique token. This distinction matters: native permission
  skipping does not force a model to perform mutations in Plan mode.
- Explicit native deny/ask rules, organization policy, classifier eligibility,
  workspace trust, and plan/implementation confirmation retain their native
  meanings. These are supported launch requests, not guarantees that every
  command is unconditionally executable on every account.
- `allowed_commands` still uses the existing adapter mechanisms. This change
  does not establish portable per-command preauthorization for every CLI or
  change the stricter collected-session command-policy contract.

Run the repeatable real-CLI probe with normal native authentication:

```bash
uv run python scripts/probe_native_plan.py --tool claude --approval yolo
uv run python scripts/probe_native_plan.py --tool cursor --model auto --approval yolo
uv run python scripts/probe_native_plan.py --tool opencode --approval yolo
uv run python scripts/probe_native_plan.py --tool antigravity-cli --approval yolo
uv run python scripts/probe_native_plan.py --tool copilot --approval yolo
```

For Claude or Cursor, `--approval auto` exercises the classifier setting.
The script prints its temporary artifact directory. Inspect the actual native
mode indicators and shell call in `terminal.log`; a token in a transcript alone
does not prove command execution. Expand collapsed shell results before exiting
(Antigravity: Ctrl+O), or output detection can miss a successful command.
Exit normally without approving implementation.
The script does not collect a canonical plan or infer completion from a native
"ready to build" menu.

## Automated verification

`./scripts/check-all.sh`: 3,546 passed, 25 opt-in tests skipped; lint, formatting,
and strict type checking passed. Tests cover all five native Plan/YOLO argv
combinations, public interactive launch, classifier composition, incompatible
mode rejection, precedence, sandbox composition, CLI reporting, and Claude's
merged classifier/output-directory settings. Collected-session tests remain
unchanged.

## Plan retrieval: next boundary

These probes establish launch behavior, not a complete artifact-return contract.
Claude and Cursor displayed native plan-file paths during the probes. Cursor's
Markdown file was read after exit and contained the plan plus its exact session
UUID, including when implementation was declined. OpenCode's native
`opencode export <session-id>` successfully exported the exact completed probe
session after exit: it contained the completed shell call and a final assistant
text part tagged `agent: plan` and `finish: stop`. These observations do not establish
reliable unattended discovery, final revision selection, or custom output
routing across all tools. A one-sentence plan can also remain only in the
conversation, as the read-only Claude probe demonstrated.

| CLI | Current retrieval evidence | Still to prove for ordinary interactive launch |
| --- | --- | --- |
| Claude | Native plan file observed; documented `plansDirectory` is already exposed by Crossby | Required file creation and final revision in a per-launch directory with the new approval combinations |
| Cursor | Native `.plan.md` survives exit and contains the exact session UUID | Supported unattended discovery and final-revision selection; custom destination not established |
| OpenCode | Exact-session export survives exit and includes the final Plan response | Discovering the session without reading terminal output, and selecting a completed plan across review/revision turns |
| Antigravity CLI | Native UI provides a resumable conversation ID | A supported plan reference/export after interactive exit; headless structured output is a separate transport |
| Copilot | Native Plan startup only | Authenticated artifact creation and retrieval; intentionally untested here |

The next investigation should verify, independently for each CLI:

1. Whether native startup can select a plan directory or exact filename.
2. Whether a supported hook/export/session API supplies the plan reference for
   the exact launched session, and whether that reference survives exit.
3. Whether the final revised content can be retrieved without approving coding.
4. Whether ordinary review/validation commands can run while planning and return
   findings for revision, including commands that create their own local state.

A tool-native file path or supported session reference is sufficient; a hosted
URL is not required. A consumer can materialize its own copy after retrieval.
Do not guess from the newest global plan file or treat terminal output as a
canonical plan artifact. WADE's eventual completion command can write retrieved
or explicitly supplied plan content, but still needs a reliable content source.

## Primary references

- [Claude permission modes](https://code.claude.com/docs/en/permission-modes)
- [Claude Plan classifier setting](https://code.claude.com/docs/en/settings-reference#useautomodeduringplan)
- [Cursor CLI parameters](https://cursor.com/docs/cli/reference/parameters), plus
  the installed CLI's `--help` for `--auto-review`
- [Copilot CLI reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [OpenCode CLI](https://opencode.ai/docs/cli/)
- [Antigravity execution modes](https://www.antigravity.google/docs/cli/modes/)
