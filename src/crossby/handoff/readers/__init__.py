"""Per-tool session readers.

Each module implements two free functions — ``locate_sessions(project_path)``
and ``read_session(ref)`` — that the adapter's methods delegate to. Keeping
readers out of the adapter files keeps the adapter classes focused on
launch/capabilities concerns.
"""
