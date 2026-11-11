"""Why a session failed: recognized from native evidence that stays private.

Every payload here was recorded live on 2026-11-10 (see
``docs/unattended-headless-verification.md#failure-reasons``): an unknown
model, credentials removed by pointing the tool at an empty config directory,
or an invalid key. They run through real child processes, so the evidence
arrives the way it does in production: on stderr, in a result envelope, or in
streamed error frames.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from crossby.ai_tools.failure_reasons import classify_failure
from crossby.models.ai import (
    AIToolID,
    HeadlessFailureReason,
    HeadlessNativeOutput,
    HeadlessSessionResult,
    HeadlessTerminalStatus,
)
from tests.unit.test_ai_tools.test_headless_adapters import (
    AGY_RESULT,
    CLAUDE_RESULT,
    CODEX_EVENTS,
    OPENCODE_EVENTS,
    PROMPT,
    _adapter,
    _agy_result,
    _fake_cli,
    _jsonl,
    _request,
)

R = HeadlessFailureReason


def _claude_error(result: str, status: int | None) -> str:
    return json.dumps(
        {
            **CLAUDE_RESULT,
            "is_error": True,
            "terminal_reason": "api_error",
            "api_error_status": status,
            "result": result,
            "structured_output": None,
        }
    )


CLAUDE_NOT_LOGGED_IN = _claude_error("Not logged in · Please run /login", None)
CLAUDE_INVALID_KEY = _claude_error("Invalid API key · Fix external API key", 401)
CLAUDE_UNKNOWN_MODEL = _claude_error(
    "There's an issue with the selected model (no-such-model-xyz). It may not exist or you "
    "may not have access to it. Run --model to pick a different model.",
    404,
)
CODEX_401 = (
    "unexpected status 401 Unauthorized: Missing bearer or basic authentication in header, "
    "url: https://api.openai.com/v1/responses"
)
CODEX_UNKNOWN_MODEL = (
    "The 'no-such-model-xyz' model is not supported when using Codex with a ChatGPT account."
)


def _codex_failed(message: str) -> str:
    return _jsonl(
        (
            CODEX_EVENTS[0],
            {"type": "turn.started"},
            {"type": "error", "message": f"Reconnecting... 2/5 ({message})"},
            {"type": "turn.failed", "error": {"message": message}},
        )
    )


def _agy_error(error: str) -> str:
    return _agy_result(
        {
            "conversation_id": "",
            "status": "ERROR",
            "response": "",
            "error": error,
            "duration_seconds": 0,
            "num_turns": 0,
            "finished_reason": "error",
        }
    )


CASES: list[tuple[str, AIToolID, dict[str, Any], HeadlessFailureReason]] = [
    (
        "claude-not-logged-in",
        AIToolID.CLAUDE,
        {"stdout": CLAUDE_NOT_LOGGED_IN, "exit_code": 1},
        R.NOT_SIGNED_IN,
    ),
    (
        "claude-invalid-key",
        AIToolID.CLAUDE,
        {"stdout": CLAUDE_INVALID_KEY, "exit_code": 1},
        R.NOT_SIGNED_IN,
    ),
    (
        "claude-unknown-model",
        AIToolID.CLAUDE,
        {
            "stdout": CLAUDE_UNKNOWN_MODEL,
            "stderr": '[claude-code:unrecognized_model] {"model":"no-such-model-xyz"}\n',
            "exit_code": 1,
        },
        R.MODEL_UNAVAILABLE,
    ),
    (
        "claude-rate-limited",
        AIToolID.CLAUDE,
        {"stdout": _claude_error("API Error: 429", 429), "exit_code": 1},
        R.RATE_LIMITED,
    ),
    (
        "codex-401",
        AIToolID.CODEX,
        {"stdout": _codex_failed(CODEX_401), "exit_code": 1},
        R.NOT_SIGNED_IN,
    ),
    (
        "codex-unknown-model",
        AIToolID.CODEX,
        {"stdout": _codex_failed(CODEX_UNKNOWN_MODEL), "exit_code": 1},
        R.MODEL_UNAVAILABLE,
    ),
    (
        # Before any event: stdout is empty and the reason is only on stderr.
        "codex-untrusted-folder",
        AIToolID.CODEX,
        {
            "stderr": (
                "Not inside a trusted directory and --skip-git-repo-check was not specified.\n"
            ),
            "exit_code": 1,
        },
        R.UNTRUSTED_FOLDER,
    ),
    (
        "cursor-invalid-key",
        AIToolID.CURSOR,
        {
            "stderr": "\x1b[33m⚠ Warning: The provided API key is invalid.\x1b[0m\n"
            "The API key was loaded from the CURSOR_API_KEY environment variable.\n",
            "exit_code": 1,
        },
        R.NOT_SIGNED_IN,
    ),
    (
        "cursor-unknown-model",
        AIToolID.CURSOR,
        {
            "stderr": "Cannot use this model: no-such-model-xyz. Available models: auto, gpt-5.2\n",
            "exit_code": 1,
        },
        R.MODEL_UNAVAILABLE,
    ),
    (
        "agy-not-signed-in",
        AIToolID.ANTIGRAVITY_CLI,
        {
            "stdout": _agy_error("authentication failed or timed out"),
            "stderr": "Error: authentication required. Run 'agy' to log in, then retry.\n",
            "exit_code": 1,
        },
        R.NOT_SIGNED_IN,
    ),
    (
        "agy-unknown-model",
        AIToolID.ANTIGRAVITY_CLI,
        {
            "stdout": _agy_error(
                'invalid model selection (--model "no-such-model-xyz" --effort ""): model '
                "no-such-model-xyz is not recognized as a known model or custom model in settings"
            ),
            "exit_code": 1,
        },
        R.MODEL_UNAVAILABLE,
    ),
]


def _assert_evidence_is_private(result: HeadlessSessionResult, *fragments: str) -> None:
    """The native wording that explained the failure never leaves the adapter."""
    exposed = json.dumps(
        {
            "warnings": result.warnings,
            "denials": result.denials,
            "events": [event.model_dump(mode="json") for event in result.events],
        }
    )
    for fragment in (*fragments, PROMPT):
        assert fragment not in exposed


@pytest.mark.parametrize(
    ("tool", "cli", "reason"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_a_recognized_failure_names_its_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: AIToolID,
    cli: dict[str, Any],
    reason: HeadlessFailureReason,
) -> None:
    adapter = _adapter(monkeypatch, tool, _fake_cli(**cli))

    result = adapter.run_headless_session(_request(tmp_path))

    # A recognized refusal is a failure, never "invalid output", even when
    # the tool emitted no events at all.
    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.failure_reason is reason
    _assert_evidence_is_private(
        result, "Not logged in", "Invalid API key", "unexpected status", "Cannot use this model"
    )


def test_a_timeout_while_retrying_a_refused_login_keeps_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex retries a 401 for a long time; the error frames stream live.
    program = "\n".join(
        (
            "import json,sys,time",
            "sys.stdin.read()",
            f"print(json.dumps({CODEX_EVENTS[0]!r}), flush=True)",
            f"frame = {{'type': 'error', 'message': 'Reconnecting... 1/5 ({CODEX_401})'}}",
            "print(json.dumps(frame), flush=True)",
            "time.sleep(30)",
        )
    )
    adapter = _adapter(monkeypatch, AIToolID.CODEX, [sys.executable, "-c", program])

    result = adapter.run_headless_session(_request(tmp_path, timeout_seconds=2.0))

    assert result.status is HeadlessTerminalStatus.TIMED_OUT
    assert result.failure_reason is R.NOT_SIGNED_IN


def test_opencode_names_an_auto_rejected_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Recorded on 1.19.5: after auto-rejecting a permission, `opencode run`
    # exits 0 with no terminal stop event.
    events = _jsonl(OPENCODE_EVENTS[:2])
    stderr = (
        "\x1b[93m\x1b[1m! \x1b[0mpermission requested: external_directory "
        "(/private/tmp/outside/*); auto-rejecting\n"
    )
    adapter = _adapter(monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=events, stderr=stderr))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.failure_reason is R.ACTION_DENIED
    assert result.denials == ("OpenCode auto-rejected a external_directory permission request",)
    _assert_evidence_is_private(result, "/private/tmp/outside")


def test_agy_ending_a_turn_silently_after_a_denial_is_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silent = {
        **AGY_RESULT,
        "response": "",
        "structured_output": None,
        "json_schema": None,
        "denied_actions": [{"action": "command(echo hello > note.txt)"}],
    }
    adapter = _adapter(monkeypatch, AIToolID.ANTIGRAVITY_CLI, _fake_cli(stdout=_agy_result(silent)))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.failure_reason is R.ACTION_DENIED
    assert len(result.denials) == 1
    _assert_evidence_is_private(result, "echo hello")


def test_successes_and_unrecognized_failures_carry_no_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok = _adapter(monkeypatch, AIToolID.CODEX, _fake_cli(stdout=_jsonl(CODEX_EVENTS)))
    assert ok.run_headless_session(_request(tmp_path)).failure_reason is None
    # OpenCode reports a missing login and an unknown model alike as an
    # "Unexpected server error", so nothing is claimed.
    unknown = _jsonl(
        (
            {
                "type": "error",
                "sessionID": "ses_x",
                "error": {
                    "name": "UnknownError",
                    "data": {"message": "Unexpected server error. Check server logs for details."},
                },
            },
        )
    )
    failed = _adapter(monkeypatch, AIToolID.OPENCODE, _fake_cli(stdout=unknown, exit_code=1))
    result = failed.run_headless_session(_request(tmp_path))
    assert result.status is not HeadlessTerminalStatus.SUCCEEDED
    assert result.failure_reason is None


def test_only_failures_may_carry_a_reason() -> None:
    from crossby.models.ai import HeadlessEvent, HeadlessEventKind

    for status in (HeadlessTerminalStatus.SUCCEEDED, HeadlessTerminalStatus.CANCELLED):
        with pytest.raises(ValueError, match="failure reason"):
            HeadlessSessionResult(
                tool=AIToolID.CODEX,
                version="0.166.0",
                status=status,
                events=(
                    HeadlessEvent(
                        sequence=1, kind=HeadlessEventKind.TERMINAL, terminal_status=status
                    ),
                ),
                final_text="OK" if status is HeadlessTerminalStatus.SUCCEEDED else None,
                duration_seconds=1.0,
                failure_reason=R.NOT_SIGNED_IN,
            )


@pytest.mark.parametrize(
    ("tool", "text", "reason"),
    [
        (
            AIToolID.COPILOT,
            'Execution failed: Model "no-such-model-xyz" is not available.',
            R.MODEL_UNAVAILABLE,
        ),
        (AIToolID.CODEX, "unexpected status 429 Too Many Requests", R.RATE_LIMITED),
        (AIToolID.CLAUDE, "Not logged in · Please run /login", R.NOT_SIGNED_IN),
        # Wording is matched only against the tool that uses it.
        (AIToolID.CURSOR, "Not logged in · Please run /login", None),
        (AIToolID.CLAUDE, "an answer that mentions rate limits", None),
    ],
)
def test_wording_is_matched_per_tool(
    tool: AIToolID, text: str, reason: HeadlessFailureReason | None
) -> None:
    assert classify_failure(tool, texts=(text,)) is reason


def test_structural_reasons_win_over_status_codes_and_wording() -> None:
    assert (
        classify_failure(
            AIToolID.CLAUDE,
            texts=("Not logged in",),
            http_statuses=(429,),
            reasons=(R.ACTION_DENIED,),
        )
        is R.ACTION_DENIED
    )
    assert classify_failure(AIToolID.CLAUDE, texts=("Not logged in",), http_statuses=(429,)) is (
        R.RATE_LIMITED
    )


def test_claude_passes_through_its_reported_cost_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = {
        **CLAUDE_RESULT,
        "structured_output": None,
        "result": "OK",
        "total_cost_usd": 0.0123,
        "modelUsage": {"claude-opus-5-5": {"inputTokens": 2, "outputTokens": 63}},
    }
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, _fake_cli(stdout=json.dumps(envelope)))

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.usage is not None
    assert result.usage.reported_cost_usd == 0.0123
    assert [item.model for item in result.usage.model_breakdown] == ["claude-opus-5-5"]
    assert (result.usage.input_tokens, result.usage.output_tokens) == (2, 63)


def test_a_run_that_looked_successful_but_left_no_answer_keeps_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not an error envelope, but no answer either, and the login refusal is on
    # stderr: the missing output is a recognized refusal, not malformed output.
    envelope = {**CLAUDE_RESULT, "result": "", "structured_output": None}
    cli = _fake_cli(stdout=json.dumps(envelope), stderr="Not logged in · Please run /login\n")
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, cli)

    result = adapter.run_headless_session(_request(tmp_path))

    assert result.status is HeadlessTerminalStatus.FAILED
    assert result.failure_reason is R.NOT_SIGNED_IN


def test_an_answer_that_fails_validation_gets_no_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_ai_tools.test_headless_adapters import SCHEMA

    envelope = {**CLAUDE_RESULT, "structured_output": {"unexpected": 1}}
    cli = _fake_cli(stdout=json.dumps(envelope), stderr="Not logged in · Please run /login\n")
    adapter = _adapter(monkeypatch, AIToolID.CLAUDE, cli)

    result = adapter.run_headless_session(
        _request(tmp_path, response_schema=SCHEMA, native_output=HeadlessNativeOutput.JSON)
    )

    assert result.status is HeadlessTerminalStatus.INVALID_OUTPUT
    assert result.failure_reason is None
