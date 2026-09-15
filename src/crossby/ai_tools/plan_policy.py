"""Portable, fail-closed matching for collected-session command policy."""

from __future__ import annotations

import shlex
from pathlib import Path

from crossby.models.ai import PlanCommandPolicy, PlanOperation, PlanOperationKind

_SHELL_PUNCTUATION = frozenset("();<>|&")
_UNSUPPORTED_EXPANSION = frozenset("*?[]{}$`")


def _pattern_parts(pattern: str) -> tuple[str, bool]:
    """Return the executable text and whether trailing arguments are allowed."""
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("command policy patterns must be non-blank strings")
    if pattern != pattern.strip() or "\n" in pattern or "\r" in pattern or "\0" in pattern:
        raise ValueError("command policy patterns must be single-line without outer whitespace")

    wildcard = False
    command = pattern
    if pattern.endswith((":*", " *")):
        command = pattern[:-2]
        wildcard = True
    if "*" in command or "?" in command or "[" in command or "]" in command:
        raise ValueError("command policy supports only one trailing :* or space-* wildcard")

    # The long-standing canonical form is ``command:arguments``. Translate its
    # first separator to a shell-space form for token comparison and OpenCode's
    # native shell resource grammar. A trailing ``:*`` was handled above.
    colon_index = command.find(":")
    whitespace_indexes = [index for index, character in enumerate(command) if character.isspace()]
    first_whitespace = min(whitespace_indexes, default=-1)
    if colon_index >= 0 and (first_whitespace < 0 or colon_index < first_whitespace):
        executable, arguments = command.split(":", 1)
        if not executable or not arguments:
            raise ValueError("canonical command:arguments patterns require both sides")
        command = f"{executable} {arguments}"
    if not command.strip():
        raise ValueError("command policy patterns require an executable")
    return command, wildcard


def _shell_tokens(expression: str) -> tuple[str, ...] | None:
    """Tokenize only simple shell expressions; return ``None`` for unsafe forms."""
    if not expression.strip() or any(char in expression for char in "\n\r\0"):
        return None
    if any(char in expression for char in _UNSUPPORTED_EXPANSION):
        return None
    try:
        lexer = shlex.shlex(expression, posix=True, punctuation_chars="".join(_SHELL_PUNCTUATION))
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = tuple(lexer)
    except ValueError:
        return None
    if not tokens or any(token and set(token) <= _SHELL_PUNCTUATION for token in tokens):
        return None
    return tokens


def validate_plan_command_pattern(pattern: str) -> None:
    """Reject canonical patterns without portable, simple-command semantics."""
    command, _wildcard = _pattern_parts(pattern)
    if _shell_tokens(command) is None:
        raise ValueError(
            "command policy patterns cannot contain compound shell syntax or expansions"
        )


def command_pattern_to_shell_pattern(pattern: str) -> str:
    """Render the supported canonical subset as an OpenCode shell resource."""
    command, wildcard = _pattern_parts(pattern)
    return f"{command} *" if wildcard else command


def operation_matches_command_policy(
    policy: PlanCommandPolicy,
    operation: PlanOperation | None,
    *,
    allowed_execution_roots: tuple[Path, ...],
) -> bool:
    """Match authoritative command evidence without parsing display prose.

    Shell expressions are accepted only when they tokenize as one simple
    command. Missing cwd, compound expressions, extra permission targets, and
    unknown operation kinds all remain unapproved.
    """
    if operation is None or operation.kind is not PlanOperationKind.COMMAND:
        return False
    if operation.permission_targets or operation.execution_dir is None:
        return False
    if not operation.execution_dir.is_absolute():
        return False
    try:
        execution_dir = operation.execution_dir.resolve()
        roots = tuple(path.resolve() for path in allowed_execution_roots)
    except (OSError, RuntimeError):
        return False
    if not any(execution_dir == root or execution_dir.is_relative_to(root) for root in roots):
        return False

    if operation.argv is not None:
        tokens = operation.argv
    elif operation.shell_expression is not None:
        parsed = _shell_tokens(operation.shell_expression)
        if parsed is None:
            return False
        tokens = parsed
    else:
        return False

    for pattern in policy.allowed_commands:
        command, wildcard = _pattern_parts(pattern)
        expected = _shell_tokens(command)
        assert expected is not None  # PlanCommandPolicy validated the pattern.
        if tokens == expected or (wildcard and tokens[: len(expected)] == expected):
            return True
    return False


__all__ = [
    "command_pattern_to_shell_pattern",
    "operation_matches_command_policy",
    "validate_plan_command_pattern",
]
