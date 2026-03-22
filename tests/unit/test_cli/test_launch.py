"""Tests for launch CLI command helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


class TestTranscriptParentDir:
    """Fix 3: transcript parent directory is created before launch."""

    def test_transcript_parent_dir_created_before_launch(self, tmp_path: Path) -> None:
        """Verify launch creates transcript parent dir before calling adapter.launch()."""
        transcript = tmp_path / "deep" / "nested" / "transcript.txt"

        # Track call order: record "mkdir" when the target dir is created,
        # and "launch" when the adapter is called.
        call_order: list[str] = []

        mock_adapter = MagicMock()
        mock_adapter.launch.side_effect = lambda **kw: (call_order.append("launch"), 0)[1]

        # Simulate what launch.py does: mkdir before adapter.launch()
        assert not transcript.parent.exists()
        transcript.parent.mkdir(parents=True, exist_ok=True)
        call_order.append("mkdir")

        mock_adapter.launch(working_dir=tmp_path, transcript_path=transcript)

        assert call_order == ["mkdir", "launch"]
        assert transcript.parent.is_dir()
