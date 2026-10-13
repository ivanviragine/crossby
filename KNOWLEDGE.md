# Project Knowledge

Shared learnings from AI planning and implementation sessions.
Read this at the start of every session. Add new entries via `wade knowledge add`.

---

## 071e3567f6c7 | 2026-10-13 | implementation | tags: handoff, codex | Issue #165

Codex serializes function/custom-tool output as a plain string or an array of content items, even though its Rust FunctionCallOutputPayload contains body and success fields. The success field is internal metadata and is not serialized; do not expect a {body, success} object or infer success from the presence of a result. Protocol reference: https://github.com/openai/codex/blob/main/codex-rs/protocol/src/models.rs (FunctionCallOutputPayload serializer).

---
