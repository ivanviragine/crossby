"""Portable collected-session command-policy matching."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crossby.ai_tools import (
    PlanCommandPolicy,
    PlanOperation,
    PlanOperationKind,
    PlanPermissionTarget,
    PlanPermissionTargetKind,
    PlanSessionRequest,
)
from crossby.ai_tools.plan_policy import (
    command_pattern_to_shell_pattern,
    operation_matches_command_policy,
)


@pytest.mark.parametrize(
    ("pattern", "native"),
    [
        ("git status", "git status"),
        ("git diff:*", "git diff *"),
        ("./scripts/check.sh *", "./scripts/check.sh *"),
        ("python:-m pytest", "python -m pytest"),
        ("curl:https://example.invalid", "curl https://example.invalid"),
        ("echo scheme:https", "echo scheme:https"),
    ],
)
def test_supported_canonical_patterns_have_shell_equivalents(pattern: str, native: str) -> None:
    policy = PlanCommandPolicy(allowed_commands=(pattern,))

    assert policy.allowed_commands == (pattern,)
    assert command_pattern_to_shell_pattern(pattern) == native


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        " git status",
        "git status ",
        "git * status",
        "git status && echo widened",
        "git status | cat",
        "echo $TOKEN",
        "echo `whoami`",
        "git:",
        ":status",
    ],
)
def test_ambiguous_or_compound_patterns_are_rejected(pattern: str) -> None:
    with pytest.raises(ValidationError):
        PlanCommandPolicy(allowed_commands=(pattern,))


def test_empty_and_duplicate_policies_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PlanCommandPolicy(allowed_commands=())
    with pytest.raises(ValidationError, match="unique"):
        PlanCommandPolicy(allowed_commands=("git status", "git status"))


def test_nested_request_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlanSessionRequest.model_validate(
            {
                "prompt": "Plan the change",
                "working_dir": "/future",
                "command_policy": {
                    "allowed_commands": ["git status"],
                    "allow_everything_else": True,
                },
            }
        )


@pytest.mark.parametrize(
    "operation",
    [
        PlanOperation(
            kind=PlanOperationKind.COMMAND,
            argv=("git", "diff", "--stat"),
            execution_dir=Path("/workspace"),
        ),
        PlanOperation(
            kind=PlanOperationKind.COMMAND,
            shell_expression="git diff --stat",
            execution_dir=Path("/workspace/subdirectory"),
        ),
    ],
)
def test_authoritative_argv_and_simple_shell_expression_match(operation: PlanOperation) -> None:
    assert operation_matches_command_policy(
        PlanCommandPolicy(allowed_commands=("git diff:*",)),
        operation,
        allowed_execution_roots=(Path("/workspace"),),
    )


@pytest.mark.parametrize(
    "operation",
    [
        None,
        PlanOperation(kind=PlanOperationKind.OTHER),
        PlanOperation(kind=PlanOperationKind.COMMAND, shell_expression="git diff --stat"),
        PlanOperation(
            kind=PlanOperationKind.COMMAND,
            shell_expression="git diff --stat && curl example.invalid",
            execution_dir=Path("/workspace"),
        ),
        PlanOperation(
            kind=PlanOperationKind.COMMAND,
            shell_expression="git diff --stat",
            execution_dir=Path("relative/workspace"),
        ),
        PlanOperation(
            kind=PlanOperationKind.COMMAND,
            shell_expression="git diff --stat",
            execution_dir=Path("/outside"),
        ),
        PlanOperation(
            kind=PlanOperationKind.COMMAND,
            shell_expression="git diff --stat",
            execution_dir=Path("/workspace"),
            permission_targets=(
                PlanPermissionTarget(
                    kind=PlanPermissionTargetKind.NETWORK_HOST,
                    value="example.invalid",
                ),
            ),
        ),
    ],
)
def test_missing_ambiguous_or_extra_authority_never_matches(
    operation: PlanOperation | None,
) -> None:
    assert not operation_matches_command_policy(
        PlanCommandPolicy(allowed_commands=("git diff:*",)),
        operation,
        allowed_execution_roots=(Path("/workspace"),),
    )


def test_operation_rejects_two_command_representations() -> None:
    with pytest.raises(ValidationError, match="both argv and a shell expression"):
        PlanOperation(
            kind=PlanOperationKind.COMMAND,
            argv=("git", "status"),
            shell_expression="git status",
        )
