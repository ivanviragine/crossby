# Project Knowledge

Shared learnings from AI planning and implementation sessions.
Read this at the start of every session. Add new entries via `wade knowledge add`.

---

## 071e3567f6c7 | 2026-10-13 | implementation | tags: handoff, codex | Issue #165

Codex serializes function/custom-tool output as a plain string or an array of content items, even though its Rust FunctionCallOutputPayload contains body and success fields. The success field is internal metadata and is not serialized; do not expect a {body, success} object or infer success from the presence of a result. Protocol reference: https://github.com/openai/codex/blob/main/codex-rs/protocol/src/models.rs (FunctionCallOutputPayload serializer).

---

## e5e4d7f953ac | 2026-10-13 | implementation | tags: claude-sdk, headless | Issue #189

ClaudeSDKClient configures can_use_tool by replacing its options with permission_prompt_tool_name=stdio during connect; that replacement does not reach a preconstructed custom transport. A separately rendered SDK subprocess command must explicitly carry the stdio permission-prompt selector or native callbacks will not be routed to the broker.
## 06a038aa1225 | 2026-10-13 | implementation | tags: testing, macos | Issue #161

On macOS, sandboxed full checks can fail process-state probes (ps), controlling-terminal acquisition, and setgid preservation even when untouched base passes those tests with normal approved execution permissions. Compare failures under identical permissions before attributing them to a change; the PTY naturally-exited-session assertion still reports gone rather than Linux /proc state Z on untouched macOS base.

---

## b488bca45a62 | 2026-10-13 | implementation | tags: codex, ai-tools | Issue #153

Native Codex exec enables CODEX_API_KEY authentication while standalone app-server disables it, so identical child environments can yield different model catalogs/defaults and are not proof of launch equivalence. At rust-v0.162.1, see exec/src/lib.rs and app-server/src/lib.rs AuthManager construction, plus models-manager catalog filtering by authentication mode. Do not copy or translate credentials to repair a readiness probe; return Unknown when the effective authentication context cannot be established.

---

## f1261b3941d1 | 2026-10-14 | implementation | tags: copilot-sdk, headless | Issue #190

Copilot SDK 1.0.21 generates a local session ID before a resumable transcript exists; cold resume of an untouched session reports session-not-found. A persisted native turn is required to verify resume, so retaining an interrupted session ID must not imply guaranteed transcript recovery.

---

## 0f15f6bf27db | 2026-10-14 | implementation | tags: terminal, testing | Issue #217

Load raw terminal captures with read_bytes() and feed bytes to pyte, or explicitly preserve newlines when decoding. Path.read_text() normalizes carriage returns, which changes cursor behavior and can make a real native startup sequence appear unrecognized even though its raw bytes replay correctly.

---
