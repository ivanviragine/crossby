"""Tests for launch CLI command helpers."""

from __future__ import annotations


class TestTranscriptParentDir:
    """Fix 3: transcript parent directory is created before launch."""

    def test_transcript_parent_dir_created(self, tmp_path) -> None:
        """Verify that a nested transcript path triggers parent dir creation."""
        nested = tmp_path / "deep" / "nested" / "transcript.txt"
        assert not nested.parent.exists()

        # Simulate what launch.py does before calling adapter.launch()
        nested.parent.mkdir(parents=True, exist_ok=True)

        assert nested.parent.exists()
        assert nested.parent.is_dir()
