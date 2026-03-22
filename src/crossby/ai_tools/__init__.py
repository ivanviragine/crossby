"""AI tool adapters — ABC with self-registering concrete implementations.

Import all adapters here to trigger __init_subclass__ registration.
"""

# Import adapters to trigger registration
import crossby.ai_tools.antigravity
import crossby.ai_tools.claude
import crossby.ai_tools.codex
import crossby.ai_tools.copilot
import crossby.ai_tools.cursor
import crossby.ai_tools.gemini
import crossby.ai_tools.opencode
import crossby.ai_tools.vscode  # noqa: F401
from crossby.ai_tools.base import AbstractAITool, pick_best_model

__all__ = ["AbstractAITool", "pick_best_model"]
