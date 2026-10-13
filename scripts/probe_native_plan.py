#!/usr/bin/env python3
"""Run a small real interactive Plan-mode probe through Crossby's public API.

Uses the selected CLI's normal authentication and model. No model is simulated.
Inspect the native mode/approval indicators, then exit without implementing.
Artifacts stay in the printed temporary directory for inspection.
"""

from __future__ import annotations

import argparse
import json
import shlex
import tempfile
import uuid
from pathlib import Path

from crossby.ai_tools import AbstractAITool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        required=True,
        choices=["claude", "cursor", "copilot", "opencode", "antigravity-cli"],
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--approval", choices=["default", "yolo", "auto", "accept_edits"], default="yolo"
    )
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix=f"crossby-plan-{args.tool}-")).resolve()
    marker = "CROSSBY_NATIVE_PLAN_" + uuid.uuid4().hex
    probe = root / "probe.py"
    probe.write_text(
        f"print({marker!r})\n",
        encoding="utf-8",
    )
    command = f"python3 {shlex.quote(str(probe))}"
    prompt = (
        "This is a bounded native CLI integration test. Stay in native Plan mode. "
        f"First run this exact shell command: {command}\n"
        "This is a read-only probe that only prints a token; it writes no files. "
        "Then propose a one-sentence plan to "
        "change a hypothetical greeting from Hello to Hi. Do not explore other "
        "directories, delegate, implement changes, or leave Plan mode. "
        "Stop after presenting the plan and wait for the user."
    )
    adapter = AbstractAITool.get(args.tool)
    approval = {
        "yolo": args.approval == "yolo",
        "auto": args.approval == "auto",
        "accept_edits": args.approval == "accept_edits",
    }
    argv = adapter.build_launch_command(
        initial_message=prompt, model=args.model, plan_mode=True, working_dir=root, **approval
    )
    (root / "launch.json").write_text(json.dumps(argv, indent=2) + "\n", encoding="utf-8")
    print(f"Probe directory: {root}", flush=True)
    print(
        "Check the native Plan/approval indicators and expand the shell result "
        "(Antigravity: Ctrl+O). Exit normally after the reply; do not implement.",
        flush=True,
    )
    exit_code = adapter.launch(
        root,
        model=args.model,
        prompt=prompt,
        plan_mode=True,
        transcript_path=root / "terminal.log",
        **approval,
    )
    result = {
        "tool": args.tool,
        "approval": args.approval,
        "exit_code": exit_code,
        "command_output_seen": marker in (root / "terminal.log").read_text(errors="replace"),
        "native_ui_plan_and_tool_call": "inspect terminal.log; not inferred from token alone",
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if exit_code == 0 and result["command_output_seen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
