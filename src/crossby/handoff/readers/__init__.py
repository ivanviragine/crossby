"""Per-tool session readers.

Each module implements two free functions — ``locate_sessions(project_path)``
and ``read_session(ref)`` — that the adapter's methods delegate to. Keeping
readers out of the adapter files keeps the adapter classes focused on
launch/capabilities concerns.

Readers return only user messages and visible assistant responses. Filter
reasoning, tool activity, source instructions, and empty turns at the source
before summarization applies the transcript token budget.
"""
