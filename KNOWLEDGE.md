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
