"""Compatibility-neutral import path for managed session process primitives.

The implementation still lives at :mod:`crossby.ai_tools.plan_process` so old
collectors, downstream imports, and monkeypatch targets observe exactly the
same module state.  Replacing this module entry (rather than copying symbols)
also keeps safety-limit patches effective through either path.
"""

from __future__ import annotations

import sys

from crossby.ai_tools import plan_process as _shared_process
from crossby.ai_tools.plan_process import *  # noqa: F403

sys.modules[__name__] = _shared_process
