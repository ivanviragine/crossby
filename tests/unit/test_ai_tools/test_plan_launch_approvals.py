"""Interactive Plan-mode selectors survive native approval composition."""

from pathlib import Path
from unittest.mock import patch

import pytest

from crossby.ai_tools.base import AbstractAITool
from crossby.ai_tools.plan_mode import PlanModeConflictError
from crossby.models.ai import PlanLaunchApprovalMode


@pytest.fixture(autouse=True)
def installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crossby.utils.versioning.detect_binary_version", lambda _: (9999, 0, 0))


@pytest.mark.parametrize(
    ("tool", "selector", "approval"),
    [
        ("claude", ["--permission-mode", "plan"], "--allow-dangerously-skip-permissions"),
        ("cursor", ["--mode", "plan"], "--force"),
        ("copilot", ["--plan"], "--yolo"),
        ("opencode", ["--agent", "plan"], "--auto"),
        ("antigravity-cli", ["--mode", "plan"], "--dangerously-skip-permissions"),
    ],
)
def test_interactive_launch_retains_plan_and_approvals(
    tool: str, selector: list[str], approval: str, tmp_path: Path
) -> None:
    adapter = AbstractAITool.get(tool)
    supported = adapter.capabilities().plan_mode.supported_launch_approval_modes
    assert PlanLaunchApprovalMode.YOLO in supported
    with patch("crossby.utils.process.run_with_transcript", return_value=0) as run:
        assert adapter.launch(tmp_path, prompt="Plan this", plan_mode=True, yolo=True) == 0
    argv = run.call_args.args[0]
    index = argv.index(selector[0])
    assert argv[index : index + len(selector)] == selector
    assert approval in argv
    assert run.call_args.kwargs["cwd"] == tmp_path
    assert "--print" not in argv
    assert "app-server" not in argv
    assert "--mode=autopilot" not in argv
    if tool == "claude":
        assert "--dangerously-skip-permissions" not in argv
        assert argv.count("--permission-mode") == 1


@pytest.mark.parametrize("tool", ["claude", "cursor"])
def test_classifier_approval_keeps_plan(tool: str, tmp_path: Path) -> None:
    argv = AbstractAITool.get(tool).build_launch_command(
        plan_mode=True, auto=True, working_dir=tmp_path
    )
    assert "plan" in argv
    if tool == "claude":
        assert argv[argv.index("--permission-mode") + 1] == "plan"
        assert argv[argv.index("--settings") + 1] == '{"useAutoModeDuringPlan":true}'
    else:
        assert "--auto-review" in argv


def test_claude_settings_preserve_auto_and_output_directory(tmp_path: Path) -> None:
    import json

    argv = AbstractAITool.get("claude").build_launch_command(
        plan_mode=True, auto=True, working_dir=tmp_path, plan_output_dir=tmp_path / "plans"
    )
    assert argv.count("--settings") == 1
    assert json.loads(argv[argv.index("--settings") + 1]) == {
        "useAutoModeDuringPlan": True,
        "plansDirectory": "./plans",
    }


@pytest.mark.parametrize(
    ("tool", "flag"),
    [
        ("claude", "accept_edits"),
        ("cursor", "accept_edits"),
        ("antigravity-cli", "accept_edits"),
        ("antigravity-cli", "auto"),
        ("copilot", "auto"),
        ("opencode", "auto"),
        ("opencode", "accept_edits"),
    ],
)
def test_unsupported_combination_cannot_remove_plan_or_downgrade(tool: str, flag: str) -> None:
    with pytest.raises(PlanModeConflictError, match=flag.replace("_", "-")):
        AbstractAITool.get(tool).build_launch_command(plan_mode=True, **{flag: True})


def test_copilot_write_permissions_do_not_select_autopilot() -> None:
    assert AbstractAITool.get("copilot").build_launch_command(
        plan_mode=True, accept_edits=True
    ) == ["copilot", "--plan", "--allow-tool", "write"]


def test_highest_approval_tier_wins_without_changing_plan() -> None:
    assert AbstractAITool.get("claude").build_launch_command(
        plan_mode=True, yolo=True, auto=True, accept_edits=True
    ) == ["claude", "--permission-mode", "plan", "--allow-dangerously-skip-permissions"]


@pytest.mark.parametrize("tool", ["claude", "cursor", "copilot", "opencode", "antigravity-cli"])
def test_interactive_approval_contract_is_not_assumed_for_headless(tool: str) -> None:
    with pytest.raises(PlanModeConflictError):
        AbstractAITool.get(tool).build_launch_command(prompt="Plan this", plan_mode=True, yolo=True)


@pytest.mark.parametrize("tool", ["claude", "copilot"])
def test_scoped_shell_preauthorization_composes_with_native_plan(tool: str) -> None:
    argv = AbstractAITool.get(tool).build_launch_command(
        plan_mode=True, allowed_commands=["python3:probe.py"]
    )
    assert "plan" in argv or "--plan" in argv
    assert (
        "Bash(python3:probe.py)" in argv if tool == "claude" else "shell(python3:probe.py)" in argv
    )
