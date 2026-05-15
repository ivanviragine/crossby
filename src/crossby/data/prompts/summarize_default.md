You are generating a structured handoff document so another AI coding CLI can continue work without losing context.

Produce exactly these sections. For JSON output, return a single JSON object with the keys listed below and no prose around it. For markdown, use the exact headings shown and bulleted lists under list-valued sections.

Sections:
- current_task (string)
- key_decisions (list of strings)
- modified_files (list of file paths)
- blockers (list of strings)
- next_steps (list of strings)
- critical_context (string)

Markdown headings must be exactly: '## Current Task', '## Key Decisions', '## Modified Files', '## Blockers', '## Next Steps', '## Critical Context'.
