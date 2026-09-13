"""Fixture-backed contracts for deterministic native plan collection."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from crossby.ai_tools import (
    AbstractAITool,
    PlanApprovalPolicy,
    PlanArtifactAmbiguousError,
    PlanArtifactLocationError,
    PlanArtifactMalformedError,
    PlanArtifactMissingError,
    PlanArtifactSource,
    PlanBindingMismatchError,
    PlanInteraction,
    PlanInteractionOutcome,
    PlanInteractionRequiredError,
    PlanInteractionResponse,
    PlanSessionBinding,
    PlanSessionError,
    PlanSessionRequest,
    PlanSessionResult,
    PlanSessionUnsupportedError,
    PlanTransportError,
    terminal_interaction_handler,
)
from crossby.ai_tools.plan_mode import safe_error_excerpt
from crossby.ai_tools.plan_process import (
    _CAPTURE_CLEANUP_GRACE_SECONDS,
    _JSON_RPC_FRAME_LIMIT,
    _STDERR_TAIL_LIMIT,
    _STDOUT_QUEUE_LIMIT,
    CapturedOutputDecodeError,
    CapturedOutputLimitError,
    CapturedProcess,
    JsonRpcFrameLimitError,
    JsonRpcProcess,
    PlanArtifactSizeError,
    parse_jsonl,
    read_text_bounded,
    run_captured,
    run_interactive,
)
from crossby.models.ai import AIToolID, EffortLevel, PlanInteractionKind
from crossby.utils.versioning import BinaryVersion

FIXTURES = Path(__file__).parents[2] / "fixtures" / "plan_sessions"
EXACT_VERSION = "fixture-tool 9999.0.0+exact"


@pytest.fixture(autouse=True)
def _exact_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda _binary, **_kwargs: BinaryVersion((9999, 0, 0), EXACT_VERSION),
    )


def _request(tmp_path: Path, **updates: Any) -> PlanSessionRequest:
    values: dict[str, Any] = {
        "prompt": "Create a native implementation plan",
        "working_dir": tmp_path,
    }
    values.update(updates)
    return PlanSessionRequest(**values)


def _antigravity_request(tmp_path: Path, **updates: Any) -> PlanSessionRequest:
    values = {"model": "gemini-3.8-flash", "effort": EffortLevel.MEDIUM, **updates}
    return _request(tmp_path, **values)


def _cursor_request(tmp_path: Path, **updates: Any) -> PlanSessionRequest:
    values = {"model": "gpt-5.3-codex", "effort": EffortLevel.MEDIUM, **updates}
    return _request(tmp_path, **values)


def _run_claude_session(request: PlanSessionRequest) -> PlanSessionResult:
    return AbstractAITool.get(AIToolID.CLAUDE).run_plan_session(
        request, terminal_interaction_handler
    )


def _assert_result(
    result: PlanSessionResult,
    *,
    tool: AIToolID,
    source: PlanArtifactSource,
    binding: PlanSessionBinding,
) -> None:
    assert result.tool is tool
    assert result.version == EXACT_VERSION
    assert result.plan.strip()
    assert result.native_mode
    assert result.session_id
    assert result.exit_code == 0
    assert result.artifact_source is source
    assert result.binding is binding


class FakeRpc:
    """Deterministic JSON-RPC peer backed by sanitized wire fixtures."""

    scripts: ClassVar[list[list[dict[str, Any]]]] = []
    instances: ClassVar[list[FakeRpc]] = []

    def __init__(self, command: list[str], *, cwd: Path, timeout: float = 600.0) -> None:
        assert timeout > 0
        self.command = command
        self.cwd = cwd
        self.messages = list(self.scripts.pop(0))
        self.sent: list[tuple[str, Any]] = []
        self.stderr = ""
        self.closed = False
        self.instances.append(self)

    def request(self, request_id: int, method: str, params: dict[str, Any]) -> None:
        self.sent.append(("request", {"id": request_id, "method": method, "params": params}))

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.sent.append(("notify", {"method": method, "params": params}))

    def respond(self, request_id: object, result: Any) -> None:
        self.sent.append(("respond", {"id": request_id, "result": result}))

    def read(self, *, timeout: float) -> dict[str, Any]:
        assert timeout > 0
        if not self.messages:
            raise EOFError("fixture exhausted")
        return self.messages.pop(0)

    def close(self) -> int:
        self.closed = True
        return 0


def _rpc_fixture(name: str) -> list[dict[str, Any]]:
    return parse_jsonl((FIXTURES / name).read_text(encoding="utf-8"))


def _install_rpc(
    monkeypatch: pytest.MonkeyPatch, fixture: str, *, headerless: bool = False
) -> None:
    FakeRpc.scripts = [_rpc_fixture(fixture)]
    FakeRpc.instances = []
    transport = "HeaderlessJsonRpcProcess" if headerless else "JsonRpcProcess"
    monkeypatch.setattr(f"crossby.ai_tools.plan_process.{transport}", FakeRpc)


class TestNormalizedContract:
    @pytest.mark.parametrize("failure", [RuntimeError("callback failed"), KeyboardInterrupt()])
    def test_callback_exceptions_propagate_to_collector_cleanup(
        self, failure: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AbstractAITool.get(AIToolID.CODEX)
        closed: list[bool] = []

        def collect(_request: Any, _version: str, handler: Any) -> Any:
            try:
                handler(
                    PlanInteraction(
                        kind=PlanInteractionKind.QUESTION,
                        question_id="question",
                        prompt="Choose a scope",
                        session_id="session",
                    )
                )
            finally:
                closed.append(True)

        def answer(_interaction: Any) -> Any:
            raise failure

        monkeypatch.setattr(adapter, "_run_plan_session", collect)
        with pytest.raises(type(failure)) as raised:
            adapter.run_plan_session(_request(tmp_path), answer)
        assert raised.value is failure
        assert closed == [True]

    def test_request_rejects_unknown_fields(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="approvalPolicy"):
            PlanSessionRequest.model_validate(
                {
                    "prompt": "Create a plan",
                    "working_dir": tmp_path,
                    "approvalPolicy": "never",
                }
            )

    def test_result_rejects_blank_or_inconsistent_success(self, tmp_path: Path) -> None:
        common = {
            "tool": AIToolID.CLAUDE,
            "version": "2.1.263 (Claude Code)",
            "session_id": "session-1",
            "native_mode": "--permission-mode plan",
            "artifact_source": PlanArtifactSource.REQUESTED_PATH,
            "binding": PlanSessionBinding.ISOLATED_RUN_PATH,
            "exit_code": 0,
            "artifact_path": tmp_path / "plan.md",
        }
        with pytest.raises(ValidationError, match="non-blank"):
            PlanSessionResult(plan=" ", **common)
        with pytest.raises(ValidationError, match="exit_code=0"):
            PlanSessionResult(plan="# Plan", **{**common, "exit_code": 1})
        with pytest.raises(ValidationError, match="both thread_id and turn_id"):
            PlanSessionResult(
                plan="# Plan",
                **{
                    **common,
                    "artifact_source": PlanArtifactSource.PROTOCOL_EVENT,
                    "binding": PlanSessionBinding.THREAD_TURN_IDS,
                    "artifact_id": "item-1",
                    "artifact_path": None,
                },
            )
        with pytest.raises(ValidationError, match="provenance IDs must be non-blank"):
            PlanSessionResult(
                plan="# Plan",
                **{
                    **common,
                    "artifact_source": PlanArtifactSource.PROTOCOL_EVENT,
                    "binding": PlanSessionBinding.SESSION_ID,
                    "artifact_id": " ",
                    "artifact_path": None,
                },
            )

    def test_gui_adapters_fail_before_collection(self, tmp_path: Path) -> None:
        for tool_id in (AIToolID.VSCODE, AIToolID.ANTIGRAVITY):
            adapter = AbstractAITool.get(tool_id)
            with pytest.raises(PlanSessionUnsupportedError):
                adapter.run_plan_session(_request(tmp_path))

    @pytest.mark.parametrize(
        "tool_id",
        [
            AIToolID.CLAUDE,
            AIToolID.CODEX,
            AIToolID.CURSOR,
            AIToolID.OPENCODE,
            AIToolID.ANTIGRAVITY_CLI,
        ],
    )
    def test_unknown_version_fails_before_adapter_process(
        self, tool_id: AIToolID, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AbstractAITool.get(tool_id)
        collector = patch.object(adapter, "_run_plan_session")
        monkeypatch.setattr(
            "crossby.utils.versioning.detect_binary_version_info", lambda _b, **_kwargs: None
        )
        with collector as run, pytest.raises(PlanSessionUnsupportedError, match="version unknown"):
            adapter.run_plan_session(_request(tmp_path))
        run.assert_not_called()

    def test_below_floor_version_fails_before_adapter_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AbstractAITool.get(AIToolID.CODEX)
        monkeypatch.setattr(
            "crossby.utils.versioning.detect_binary_version_info",
            lambda _b, **_kwargs: BinaryVersion((0, 1, 0), "codex-cli 0.1.0"),
        )
        with (
            patch.object(adapter, "_run_plan_session") as run,
            pytest.raises(PlanSessionUnsupportedError, match=r"0\.1\.0"),
        ):
            adapter.run_plan_session(_request(tmp_path))
        run.assert_not_called()

    def test_version_probe_and_collector_share_request_deadline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AbstractAITool.get(AIToolID.CODEX)
        clock = iter((100.0, 102.0, 102.5))
        probe_timeouts: list[float] = []

        def detect(_binary: str, *, timeout_seconds: float) -> BinaryVersion:
            probe_timeouts.append(timeout_seconds)
            return BinaryVersion((9999, 0, 0), EXACT_VERSION)

        result = PlanSessionResult(
            tool=AIToolID.CODEX,
            version=EXACT_VERSION,
            plan="# Plan",
            session_id="thread-1",
            native_mode='collaborationMode.mode="plan"',
            artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
            binding=PlanSessionBinding.THREAD_TURN_IDS,
            exit_code=0,
            thread_id="thread-1",
            turn_id="turn-1",
            artifact_id="plan-1",
        )
        monkeypatch.setattr("crossby.ai_tools.base.monotonic", lambda: next(clock))
        monkeypatch.setattr("crossby.utils.versioning.detect_binary_version_info", detect)

        with patch.object(adapter, "_run_plan_session", return_value=result) as run:
            adapter.run_plan_session(_request(tmp_path, timeout_seconds=10))

        assert probe_timeouts == [8.0]
        assert run.call_args.args[0].timeout_seconds == 7.5

    def test_tty_does_not_install_an_implicit_interaction_handler(self, tmp_path: Path) -> None:
        adapter = AbstractAITool.get(AIToolID.CODEX)
        result = PlanSessionResult(
            tool=AIToolID.CODEX,
            version=EXACT_VERSION,
            plan="# Plan",
            session_id="thread-1",
            native_mode='collaborationMode.mode="plan"',
            artifact_source=PlanArtifactSource.PROTOCOL_EVENT,
            binding=PlanSessionBinding.THREAD_TURN_IDS,
            exit_code=0,
            thread_id="thread-1",
            turn_id="turn-1",
            artifact_id="plan-1",
        )

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch.object(adapter, "_run_plan_session", return_value=result) as run,
        ):
            adapter.run_plan_session(_request(tmp_path))

        assert run.call_args.args[2] is None

    @pytest.mark.parametrize(
        "tool_id", [AIToolID.CLAUDE, AIToolID.OPENCODE, AIToolID.ANTIGRAVITY_CLI]
    )
    def test_tool_managed_policy_is_rejected_before_collection(
        self, tool_id: AIToolID, tmp_path: Path
    ) -> None:
        adapter = AbstractAITool.get(tool_id)
        with (
            patch.object(adapter, "_run_plan_session") as run,
            pytest.raises(PlanSessionUnsupportedError, match="approval_policy"),
        ):
            adapter.run_plan_session(_request(tmp_path, approval_policy=PlanApprovalPolicy.NEVER))
        run.assert_not_called()

    @pytest.mark.parametrize("tool_id", [AIToolID.CODEX, AIToolID.CURSOR])
    def test_untrusted_policy_is_rejected_before_collection(
        self, tool_id: AIToolID, tmp_path: Path
    ) -> None:
        adapter = AbstractAITool.get(tool_id)
        with (
            patch.object(adapter, "_run_plan_session") as run,
            pytest.raises(PlanSessionUnsupportedError, match="untrusted"),
        ):
            adapter.run_plan_session(
                _request(tmp_path, approval_policy=PlanApprovalPolicy.UNTRUSTED)
            )
        run.assert_not_called()

    @pytest.mark.parametrize("policy", list(PlanApprovalPolicy))
    def test_copilot_collection_is_unsupported_before_any_process(
        self, policy: PlanApprovalPolicy, tmp_path: Path
    ) -> None:
        adapter = AbstractAITool.get(AIToolID.COPILOT)
        assert adapter.capabilities().supports_plan_mode
        assert not adapter.capabilities().supports_plan_session
        with (
            patch.object(adapter, "_run_plan_session") as run,
            patch("crossby.utils.versioning.detect_binary_version_info") as probe,
            pytest.raises(PlanSessionUnsupportedError, match="--prompt omits ask_user"),
        ):
            adapter.run_plan_session(_request(tmp_path, approval_policy=policy))

        run.assert_not_called()
        probe.assert_not_called()

    def test_unsupported_effort_is_rejected_before_collection(self, tmp_path: Path) -> None:
        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        caps = adapter.capabilities().model_copy(update={"supports_effort": False})
        with (
            patch.object(adapter, "capabilities", return_value=caps),
            patch.object(adapter, "_run_plan_session") as run,
            pytest.raises(PlanSessionUnsupportedError, match="effort='high'"),
        ):
            adapter.run_plan_session(_request(tmp_path, effort=EffortLevel.HIGH))
        run.assert_not_called()

    @pytest.mark.parametrize("effort", [EffortLevel.XHIGH, EffortLevel.MAX])
    def test_unsupported_effort_tier_is_rejected_before_collection(
        self, effort: EffortLevel, tmp_path: Path
    ) -> None:
        adapter = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI)
        with (
            patch.object(adapter, "_run_plan_session") as run,
            pytest.raises(PlanSessionUnsupportedError, match=rf"effort='{effort.value}'"),
        ):
            adapter.run_plan_session(_request(tmp_path, effort=effort))
        run.assert_not_called()

    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            ("claude-sonnet-4-6", EffortLevel.HIGH),
            ("gemini-3.1-pro", EffortLevel.MEDIUM),
            ("gemini-3.8-flash-low", EffortLevel.HIGH),
        ],
    )
    def test_antigravity_rejects_effort_its_model_cannot_encode(
        self,
        model: str | None,
        effort: EffortLevel,
        tmp_path: Path,
    ) -> None:
        adapter = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI)
        with (
            patch("crossby.ai_tools.plan_process.run_captured") as run,
            pytest.raises(PlanSessionUnsupportedError, match="cannot preserve effort"),
        ):
            adapter.run_plan_session(_request(tmp_path, model=model, effort=effort))
        run.assert_not_called()

    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            (None, None),
            (None, EffortLevel.HIGH),
            ("gemini-3.8-flash", None),
            (" ", EffortLevel.HIGH),
        ],
        ids=("both-missing", "model-missing", "effort-missing", "model-blank"),
    )
    def test_antigravity_rejects_incomplete_request_before_collection(
        self,
        model: str | None,
        effort: EffortLevel | None,
        tmp_path: Path,
    ) -> None:
        with (
            patch("crossby.ai_tools.plan_process.run_captured") as run,
            pytest.raises(PlanSessionUnsupportedError, match="explicit model and effort"),
        ):
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
                _request(tmp_path, model=model, effort=effort)
            )

        run.assert_not_called()

    def test_error_excerpt_redacts_and_bounds_secrets(self) -> None:
        excerpt = safe_error_excerpt("API_KEY=super-secret " + ("x" * 1000))
        assert excerpt is not None
        assert "super-secret" not in excerpt
        assert len(excerpt) == 500

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("OPENAI_API_KEY=sk-example", "OPENAI_API_KEY=<redacted>"),
            ("GITHUB_TOKEN=ghp_example", "GITHUB_TOKEN=<redacted>"),
            ("AWS_SECRET_ACCESS_KEY=abc123", "AWS_SECRET_ACCESS_KEY=<redacted>"),
            ("PRIVATE_KEY=private-material", "PRIVATE_KEY=<redacted>"),
        ],
    )
    def test_error_excerpt_redacts_provider_prefixed_credentials(
        self, text: str, expected: str
    ) -> None:
        assert safe_error_excerpt(text) == expected

    def test_error_excerpt_redacts_authorization_scheme_and_credential(self) -> None:
        excerpt = safe_error_excerpt(
            "request failed: Authorization: Bearer ghp_abcdef; retry denied"
        )

        assert excerpt == "request failed: Authorization=<redacted> retry denied"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('password="two words"', 'password="<redacted>"'),
            ('{"api_key":"two words"}', '{"api_key":"<redacted>"}'),
        ],
    )
    def test_error_excerpt_redacts_complete_quoted_secrets(self, text: str, expected: str) -> None:
        assert safe_error_excerpt(text) == expected

    @pytest.mark.parametrize(
        ("tool_id", "outside_workspace"),
        [
            (AIToolID.OPENCODE, False),
            (AIToolID.CLAUDE, True),
        ],
    )
    def test_plan_output_location_errors_are_session_errors(
        self,
        tool_id: AIToolID,
        outside_workspace: bool,
        tmp_path: Path,
    ) -> None:
        adapter = AbstractAITool.get(tool_id)
        output_dir = tmp_path.parent / "external-plans" if outside_workspace else tmp_path / "plans"

        with (
            patch.object(adapter, "_run_plan_session") as run,
            pytest.raises(PlanSessionError) as raised,
        ):
            adapter.run_plan_session(_request(tmp_path, plan_output_dir=output_dir))

        assert isinstance(raised.value, PlanArtifactLocationError)
        run.assert_not_called()


class TestSubprocessTimeoutRedaction:
    @pytest.mark.parametrize(
        ("tool_id", "runner"),
        [
            (AIToolID.ANTIGRAVITY_CLI, "run_captured"),
            (AIToolID.CLAUDE, "run_interactive"),
        ],
    )
    def test_initial_prompt_is_not_exposed_by_timeout_error(
        self,
        tool_id: AIToolID,
        runner: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        prompt = "Plan with private-value-that-must-not-leak"

        def time_out(command: list[str], **kwargs: Any) -> Any:
            assert prompt in command
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        monkeypatch.setattr(f"crossby.ai_tools.plan_process.{runner}", time_out)

        updates: dict[str, Any] = {}
        if tool_id is AIToolID.ANTIGRAVITY_CLI:
            updates.update(model="gemini-3.8-flash", effort=EffortLevel.MEDIUM)
        with pytest.raises(PlanTransportError, match="timed out") as raised:
            AbstractAITool.get(tool_id).run_plan_session(
                _request(tmp_path, prompt=prompt, **updates),
                terminal_interaction_handler if tool_id is AIToolID.CLAUDE else None,
            )

        message = str(raised.value)
        assert prompt not in message
        assert "Command '" not in message


class TestPlanProcess:
    @pytest.mark.parametrize(
        ("stream", "limit_name"),
        [
            ("stdout", "_CAPTURED_STDOUT_LIMIT"),
            ("stderr", "_CAPTURED_STDERR_LIMIT"),
        ],
    )
    def test_captured_process_output_is_hard_limited(
        self,
        stream: str,
        limit_name: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(f"crossby.ai_tools.plan_process.{limit_name}", 64)
        script = f"import sys; sys.{stream}.write('x' * 4096)"

        with pytest.raises(CapturedOutputLimitError) as raised:
            run_captured([sys.executable, "-c", script], cwd=tmp_path, timeout=5)

        assert raised.value.stream == stream
        assert raised.value.limit == 64

    def test_captured_process_preserves_bounded_text_and_input(self, tmp_path: Path) -> None:
        result = run_captured(
            [
                sys.executable,
                "-c",
                "import sys; print(sys.stdin.read()); print('diagnostic', file=sys.stderr)",
            ],
            cwd=tmp_path,
            timeout=5,
            input_text="hello",
        )

        assert result.returncode == 0
        assert result.stdout == "hello\n"
        assert result.stderr == "diagnostic\n"

    def test_captured_process_interrupt_kills_and_reaps_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = None
                self.stdout = BytesIO()
                self.stderr = BytesIO()
                self.returncode: int | None = None
                self.wait_calls = 0

            def wait(self, *, timeout: float) -> int:
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise KeyboardInterrupt
                assert timeout == _CAPTURE_CLEANUP_GRACE_SECONDS
                self.returncode = -9
                return self.returncode

        process = FakeProcess()
        killed: list[FakeProcess] = []
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.subprocess.Popen", lambda *_a, **_k: process
        )
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process._kill_process_group", lambda proc: killed.append(proc)
        )

        with pytest.raises(KeyboardInterrupt):
            run_captured(["interrupted"], cwd=tmp_path, timeout=5)

        assert killed == [process]
        assert process.wait_calls == 2
        assert process.returncode == -9

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
    def test_captured_process_timeout_kills_descendants_holding_pipes(self, tmp_path: Path) -> None:
        script = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
        )
        started = time.monotonic()

        with pytest.raises(subprocess.TimeoutExpired):
            run_captured([sys.executable, "-c", script], cwd=tmp_path, timeout=0.2)

        assert time.monotonic() - started < 2

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
    def test_captured_process_success_kills_descendants_with_redirected_pipes(
        self, tmp_path: Path
    ) -> None:
        started = tmp_path / "descendant-started"
        survived = tmp_path / "descendant-survived"
        child = (
            "import pathlib,time; "
            f"pathlib.Path({str(started)!r}).write_text('started'); "
            "time.sleep(0.5); "
            f"pathlib.Path({str(survived)!r}).write_text('survived')"
        )
        parent = (
            "import pathlib,subprocess,sys,time\n"
            f"started=pathlib.Path({str(started)!r})\n"
            "subprocess.Popen("
            f"[sys.executable,'-c',{child!r}],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            "while not started.exists():\n"
            "    time.sleep(0.01)\n"
        )

        result = run_captured([sys.executable, "-c", parent], cwd=tmp_path, timeout=5)

        assert result.returncode == 0
        assert started.is_file()
        time.sleep(0.8)
        assert not survived.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
    def test_interactive_process_timeout_kills_descendants(self, tmp_path: Path) -> None:
        started = tmp_path / "descendant-started"
        survived = tmp_path / "descendant-survived"
        child = (
            "import pathlib,time; "
            f"pathlib.Path({str(started)!r}).write_text('started'); "
            "time.sleep(1.2); "
            f"pathlib.Path({str(survived)!r}).write_text('survived')"
        )
        parent = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
            "time.sleep(30)"
        )

        with pytest.raises(subprocess.TimeoutExpired):
            run_interactive([sys.executable, "-c", parent], cwd=tmp_path, timeout=0.8)

        assert started.is_file()
        time.sleep(0.8)
        assert not survived.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group contract")
    def test_interactive_process_success_kills_descendants(self, tmp_path: Path) -> None:
        started = tmp_path / "descendant-started"
        survived = tmp_path / "descendant-survived"
        child = (
            "import pathlib,time; "
            f"pathlib.Path({str(started)!r}).write_text('started'); "
            "time.sleep(0.5); "
            f"pathlib.Path({str(survived)!r}).write_text('survived')"
        )
        parent = (
            "import pathlib,subprocess,sys,time\n"
            f"started=pathlib.Path({str(started)!r})\n"
            f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
            "while not started.exists():\n"
            "    time.sleep(0.01)\n"
        )

        assert run_interactive([sys.executable, "-c", parent], cwd=tmp_path, timeout=5) == 0

        assert started.is_file()
        time.sleep(0.8)
        assert not survived.exists()

    def test_interactive_process_interrupt_kills_and_reaps_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.returncode: int | None = None
                self.wait_calls = 0

            def wait(self, *, timeout: float) -> int:
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise KeyboardInterrupt
                assert timeout == _CAPTURE_CLEANUP_GRACE_SECONDS
                self.returncode = -9
                return self.returncode

        process = FakeProcess()
        killed: list[FakeProcess] = []
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.subprocess.Popen", lambda *_a, **_k: process
        )
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process._kill_process_group", lambda proc: killed.append(proc)
        )

        with pytest.raises(KeyboardInterrupt):
            run_interactive(["interrupted"], cwd=tmp_path, timeout=5)

        assert killed == [process]
        assert process.wait_calls == 2
        assert process.returncode == -9

    def test_captured_process_uses_utf8_independent_of_host_locale(self, tmp_path: Path) -> None:
        script = (
            "import sys; "
            "received=sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write('café'.encode('utf-8') + b':' + received.hex().encode())"
        )

        with patch("locale.getpreferredencoding", return_value="cp1252"):
            result = run_captured(
                [sys.executable, "-c", script],
                cwd=tmp_path,
                timeout=5,
                input_text="crème",
            )

        assert result.stdout == "café:6372c3a86d65"

    @pytest.mark.parametrize("stream", ["stdout", "stderr"])
    def test_captured_process_translates_invalid_utf8(self, stream: str, tmp_path: Path) -> None:
        script = f"import sys; sys.{stream}.buffer.write(bytes([255]))"

        with pytest.raises(CapturedOutputDecodeError) as raised:
            run_captured([sys.executable, "-c", script], cwd=tmp_path, timeout=5)

        assert raised.value.stream == stream
        assert raised.value.encoding == "utf-8"

    def test_json_rpc_process_configures_strict_utf8_stdio(self, tmp_path: Path) -> None:
        with (
            patch(
                "crossby.ai_tools.plan_process.subprocess.Popen",
                side_effect=OSError("stop after capturing arguments"),
            ) as popen,
            pytest.raises(OSError, match="stop after capturing arguments"),
        ):
            JsonRpcProcess(["unused"], cwd=tmp_path)

        assert popen.call_args.kwargs["encoding"] == "utf-8"
        assert popen.call_args.kwargs["errors"] == "strict"

    def test_stdout_reader_applies_bounded_backpressure_until_closed(self) -> None:
        blocked = threading.Event()

        class SignalingQueue(queue.Queue[str | None]):
            def put(
                self,
                item: str | None,
                block: bool = True,
                timeout: float | None = None,
            ) -> None:
                if self.full():
                    blocked.set()
                super().put(item, block=block, timeout=timeout)

        rpc = object.__new__(JsonRpcProcess)
        rpc._stdout_queue = SignalingQueue(maxsize=1)
        rpc._closing = threading.Event()
        rpc._discard_stdout = threading.Event()
        reader = threading.Thread(target=rpc._read_stdout, args=(StringIO("one\ntwo\n"),))

        reader.start()
        assert blocked.wait(timeout=1.0)
        assert rpc._stdout_queue.qsize() == 1
        rpc._closing.set()
        reader.join(timeout=1.0)

        assert not reader.is_alive()

    def test_stdout_reader_rejects_an_oversized_json_rpc_frame_and_kills_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.killed = False

            def poll(self) -> None:
                return None

            def kill(self) -> None:
                self.killed = True

        rpc = object.__new__(JsonRpcProcess)
        rpc._stdout_queue = queue.Queue(maxsize=2)
        rpc._closing = threading.Event()
        rpc._discard_stdout = threading.Event()
        rpc._proc = FakeProcess()  # type: ignore[assignment]
        monkeypatch.setattr("crossby.ai_tools.plan_process._kill_process_group", lambda p: p.kill())
        private_payload = "private-protocol-payload"
        oversized = private_payload * (_JSON_RPC_FRAME_LIMIT // len(private_payload) + 1)

        rpc._read_stdout(StringIO(oversized))

        with pytest.raises(JsonRpcFrameLimitError, match="frame exceeded") as raised:
            rpc.read(timeout=0.1)
        assert raised.value.limit == _JSON_RPC_FRAME_LIMIT
        assert private_payload not in str(raised.value)
        assert rpc._proc.killed
        assert rpc._stdout_queue.get_nowait() is None

    @pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
    @pytest.mark.parametrize("failure", ["timeout", "parent-exited", "frame-overflow"])
    def test_protocol_cleanup_stops_descendants_holding_pipes(
        self, failure: str, tmp_path: Path
    ) -> None:
        script = (
            "import json,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(json.dumps({'jsonrpc':'2.0','method':'ready','pid':child.pid}),flush=True); "
        )
        if failure == "frame-overflow":
            script += f"sys.stdout.write('x'*{_JSON_RPC_FRAME_LIMIT + 1}); sys.stdout.flush(); "
        if failure != "parent-exited":
            script += "time.sleep(30)"
        rpc = JsonRpcProcess([sys.executable, "-c", script], cwd=tmp_path)
        child_pid = rpc.read(timeout=5)["pid"]
        closer = threading.Thread(target=rpc.close, daemon=True)
        try:
            if failure == "frame-overflow":
                with pytest.raises(JsonRpcFrameLimitError):
                    rpc.read(timeout=5)
            elif failure == "timeout":
                with pytest.raises(TimeoutError):
                    rpc.read(timeout=0.05)
            else:
                rpc._proc.wait(timeout=5)
            closer.start()
            closer.join(timeout=2)
            assert not closer.is_alive(), "protocol cleanup blocked on a descendant's pipe"
            assert not rpc._stdout_thread.is_alive()
            assert not rpc._stderr_thread.is_alive()
        finally:
            with suppress(ProcessLookupError):
                os.kill(child_pid, 9)
            if rpc._proc.poll() is None:
                rpc._proc.kill()
                rpc._proc.wait(timeout=2)
            if closer.ident is not None:
                closer.join(timeout=2)
            else:
                rpc.close()

    def test_file_artifact_read_is_hard_limited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifact = tmp_path / "plan.md"
        artifact.write_bytes(b"private-payload" * 5)
        monkeypatch.setattr("crossby.ai_tools.plan_process._PLAN_ARTIFACT_TEXT_LIMIT", 64)

        with pytest.raises(PlanArtifactSizeError) as raised:
            read_text_bounded(artifact)

        assert raised.value.limit == 64
        assert "private-payload" not in str(raised.value)

    def test_stdout_queue_has_a_fixed_message_limit(self, tmp_path: Path) -> None:
        with (
            patch("crossby.ai_tools.plan_process.subprocess.Popen") as popen,
            patch("crossby.ai_tools.plan_process.threading.Thread"),
        ):
            popen.return_value.stdin = StringIO()
            popen.return_value.stdout = StringIO()
            popen.return_value.stderr = StringIO()
            rpc = JsonRpcProcess(["fake-json-rpc"], cwd=tmp_path)

        assert rpc._stdout_queue.maxsize == _STDOUT_QUEUE_LIMIT

    def test_stdout_discard_mode_drains_queued_and_future_output(self) -> None:
        rpc = object.__new__(JsonRpcProcess)
        rpc._stdout_queue = queue.Queue(maxsize=1)
        rpc._stdout_queue.put_nowait("already queued\n")
        rpc._closing = threading.Event()
        rpc._discard_stdout = threading.Event()

        rpc.discard_stdout()
        rpc._read_stdout(StringIO("later output\n"))

        assert rpc._stdout_queue.empty()

    def test_stderr_reader_retains_only_a_bounded_tail(self) -> None:
        rpc = object.__new__(JsonRpcProcess)
        rpc._stderr_tail = ""
        rpc._stderr_lock = threading.Lock()
        rpc._closing = threading.Event()
        latest = "latest diagnostic\n"

        rpc._read_stderr(StringIO(("old diagnostic\n" * _STDERR_TAIL_LIMIT) + latest))

        assert len(rpc.stderr) == _STDERR_TAIL_LIMIT
        assert rpc.stderr.endswith(latest)

    def test_close_joins_reader_threads_before_closing_streams(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []

        class FakeStream:
            def __init__(self, name: str) -> None:
                self.name = name
                self.closed = False

            def close(self) -> None:
                events.append(f"close:{self.name}")
                self.closed = True

        class FakeThread:
            def __init__(self, name: str) -> None:
                self.name = name

            def join(self, *, timeout: float) -> None:
                assert timeout > 0
                events.append(f"join:{self.name}")

            def is_alive(self) -> bool:
                return False

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakeStream("stdin")
                self.stdout = FakeStream("stdout")
                self.stderr = FakeStream("stderr")
                self.returncode = 0

            def poll(self) -> int:
                return 0

        rpc = object.__new__(JsonRpcProcess)
        rpc._proc = FakeProcess()  # type: ignore[assignment]
        rpc._stdin = rpc._proc.stdin  # type: ignore[assignment]
        rpc._write_thread = None
        rpc._deadline = time.monotonic() + 1
        rpc._closing = threading.Event()
        rpc._stdout_thread = FakeThread("stdout")  # type: ignore[assignment]
        rpc._stderr_thread = FakeThread("stderr")  # type: ignore[assignment]
        monkeypatch.setattr("crossby.ai_tools.plan_process._kill_process_group", lambda _p: None)

        assert rpc.close() == 0
        assert events == [
            "close:stdin",
            "join:stdout",
            "join:stderr",
            "close:stdout",
            "close:stderr",
        ]

    def test_protocol_cleanup_reaps_after_session_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []

        class FakeStream:
            def __init__(self, name: str) -> None:
                self.name = name
                self.closed = False

            def close(self) -> None:
                events.append(f"close:{self.name}")
                self.closed = True

        class FakeThread:
            def join(self, *, timeout: float) -> None:
                raise AssertionError(f"cleanup waited {timeout}s after its deadline")

            def is_alive(self) -> bool:
                return False

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakeStream("stdin")
                self.stdout = FakeStream("stdout")
                self.stderr = FakeStream("stderr")
                self.returncode: int | None = None

            def poll(self) -> int | None:
                return self.returncode

            def wait(self, *, timeout: float) -> int:
                events.append(f"wait:{timeout}")
                self.returncode = -9
                return self.returncode

        rpc = object.__new__(JsonRpcProcess)
        rpc._proc = FakeProcess()  # type: ignore[assignment]
        rpc._stdin = rpc._proc.stdin  # type: ignore[assignment]
        rpc._write_thread = None
        rpc._deadline = 0.0
        rpc._closing = threading.Event()
        rpc._stdout_thread = FakeThread()  # type: ignore[assignment]
        rpc._stderr_thread = FakeThread()  # type: ignore[assignment]

        def kill(process: FakeProcess) -> None:
            events.append("kill")

        monkeypatch.setattr("crossby.ai_tools.plan_process.time.monotonic", lambda: 1.0)
        monkeypatch.setattr("crossby.ai_tools.plan_process._kill_process_group", kill)

        assert rpc.close() == -9
        assert events == [
            "close:stdin",
            "kill",
            f"wait:{_CAPTURE_CLEANUP_GRACE_SECONDS}",
            "close:stdout",
            "close:stderr",
        ]

    def test_protocol_cleanup_reaps_real_child_after_session_deadline(self, tmp_path: Path) -> None:
        rpc = JsonRpcProcess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
        )
        process = rpc._proc
        rpc._deadline = 0.0

        returncode = rpc.close()

        assert returncode != 0
        assert process.returncode == returncode

    def test_protocol_cleanup_never_reports_success_while_child_is_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeStream:
            closed = False

            def close(self) -> None:
                self.closed = True

        class FakeThread:
            def is_alive(self) -> bool:
                return False

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakeStream()
                self.stdout = FakeStream()
                self.stderr = FakeStream()

            def poll(self) -> None:
                return None

            def wait(self, *, timeout: float) -> None:
                assert timeout == _CAPTURE_CLEANUP_GRACE_SECONDS
                raise subprocess.TimeoutExpired("unused", timeout)

        rpc = object.__new__(JsonRpcProcess)
        rpc._proc = FakeProcess()  # type: ignore[assignment]
        rpc._stdin = rpc._proc.stdin  # type: ignore[assignment]
        rpc._write_thread = None
        rpc._deadline = 0.0
        rpc._closing = threading.Event()
        rpc._stdout_thread = FakeThread()  # type: ignore[assignment]
        rpc._stderr_thread = FakeThread()  # type: ignore[assignment]
        monkeypatch.setattr("crossby.ai_tools.plan_process._kill_process_group", lambda _p: None)
        monkeypatch.setattr("crossby.ai_tools.plan_process.time.monotonic", lambda: 1.0)

        assert rpc.close() == -getattr(signal, "SIGKILL", 9)

    def test_protocol_write_cannot_outlive_deadline(self, tmp_path: Path) -> None:
        script = "import time; print('ready', flush=True); time.sleep(30)"
        rpc = JsonRpcProcess([sys.executable, "-c", script], cwd=tmp_path, timeout=0.5)
        assert rpc.read_line(timeout=5).strip() == "ready"
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="writing a JSON-RPC message"):
                rpc.notify("turn/start", {"text": "x" * (2 * 1024 * 1024)})
        finally:
            rpc.close()
        assert time.monotonic() - started < 2
        assert rpc._write_thread is not None and not rpc._write_thread.is_alive()


class TestClaudeCollector:
    @pytest.mark.parametrize("handler_kind", ["missing", "callback"])
    def test_requires_explicit_terminal_handler_before_attaching(
        self,
        handler_kind: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        attached = False

        def fake_run(*_args: Any, **_kwargs: Any) -> int:
            nonlocal attached
            attached = True
            return 0

        handler = (
            None
            if handler_kind == "missing"
            else lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.SKIPPED
            )
        )
        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", fake_run)

        with pytest.raises(PlanInteractionRequiredError, match="terminal_interaction_handler"):
            AbstractAITool.get(AIToolID.CLAUDE).run_plan_session(_request(tmp_path), handler)

        assert not attached
        assert not (tmp_path / ".crossby").exists()

    def test_setup_time_is_subtracted_from_process_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = iter((100.0, 103.0))
        now = [100.0]
        seen_timeouts: list[float] = []

        def monotonic() -> float:
            now[0] = next(clock, now[0])
            return now[0]

        def write_plan(command: list[str], *, cwd: Path, timeout: float) -> int:
            seen_timeouts.append(timeout)
            settings = json.loads(command[command.index("--settings") + 1])
            (cwd / settings["plansDirectory"] / "plan.md").write_text("# Plan")
            return 0

        monkeypatch.setattr("crossby.ai_tools.base.monotonic", lambda: 50.0)
        monkeypatch.setattr("crossby.ai_tools.claude.time.monotonic", monotonic)
        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", write_plan)

        _run_claude_session(_request(tmp_path, timeout_seconds=10))

        assert seen_timeouts == [7.0]

    @pytest.mark.parametrize("stage", ["enumeration", "read"])
    def test_artifact_collection_cannot_outlive_request_deadline(
        self,
        stage: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = [100.0]
        real_listdir = os.listdir
        real_read = read_text_bounded

        def write_plan(command: list[str], *, cwd: Path, timeout: float) -> int:
            assert timeout == 10.0
            settings = json.loads(command[command.index("--settings") + 1])
            (cwd / settings["plansDirectory"] / "plan.md").write_text("# Plan")
            return 0

        def expire_during_listdir(path: Any) -> list[str]:
            entries = real_listdir(path)
            now[0] = 111.0
            return entries

        def expire_during_read(*args: Any, **kwargs: Any) -> str:
            plan = real_read(*args, **kwargs)
            now[0] = 111.0
            return plan

        monkeypatch.setattr("crossby.ai_tools.base.monotonic", lambda: 50.0)
        monkeypatch.setattr("crossby.ai_tools.claude.time.monotonic", lambda: now[0])
        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", write_plan)
        if stage == "enumeration":
            monkeypatch.setattr(os, "listdir", expire_during_listdir)
        else:
            monkeypatch.setattr(
                "crossby.ai_tools.plan_process.read_text_bounded", expire_during_read
            )

        with pytest.raises(PlanTransportError, match="exceeded its timeout"):
            _run_claude_session(_request(tmp_path, timeout_seconds=10))

    @pytest.mark.skipif(os.name != "posix", reason="directory-descriptor contract")
    def test_plan_root_creation_is_anchored_to_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        working_dir = tmp_path / "workspace"
        working_dir.mkdir()
        crossby_dir = working_dir / ".crossby"
        original_crossby = working_dir / "original-crossby"
        outside = tmp_path / "outside"
        outside.mkdir()
        real_mkdir = os.mkdir
        raced = False

        def replace_intermediate_before_plan_root_mkdir(
            path: Any,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> None:
            nonlocal raced
            if not raced and Path(path).name == "plan-sessions" and crossby_dir.is_dir():
                raced = True
                crossby_dir.rename(original_crossby)
                crossby_dir.symlink_to(outside, target_is_directory=True)
            real_mkdir(path, mode, dir_fd=dir_fd)

        def write_plan(command: list[str], *, cwd: Path, **_kwargs: Any) -> int:
            settings = json.loads(command[command.index("--settings") + 1])
            run_dir = cwd / settings["plansDirectory"]
            (run_dir / "plan.md").write_text("# Escaped plan", encoding="utf-8")
            return 0

        monkeypatch.setattr(os, "mkdir", replace_intermediate_before_plan_root_mkdir)
        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", write_plan)

        with pytest.raises(PlanTransportError, match="plan root could not be created or opened"):
            _run_claude_session(_request(working_dir))

        assert raced
        assert not (outside / "plan-sessions").exists()
        assert (original_crossby / "plan-sessions").is_dir()

    @pytest.mark.skipif(os.name != "posix", reason="directory-descriptor contract")
    def test_isolated_directory_creation_is_anchored_to_validated_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        working_dir = tmp_path / "workspace"
        working_dir.mkdir()
        root = working_dir / ".crossby" / "plan-sessions"
        original_root = root.with_name("original-plan-sessions")
        outside = tmp_path / "outside"
        outside.mkdir()
        session_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        real_mkdir = os.mkdir
        raced = False

        def replace_root_before_session_mkdir(
            path: Any,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> None:
            nonlocal raced
            if not raced and Path(path).name == str(session_id):
                raced = True
                root.rename(original_root)
                root.symlink_to(outside, target_is_directory=True)
            real_mkdir(path, mode, dir_fd=dir_fd)

        monkeypatch.setattr("crossby.ai_tools.claude.uuid.uuid4", lambda: session_id)
        monkeypatch.setattr(os, "mkdir", replace_root_before_session_mkdir)

        with pytest.raises(PlanBindingMismatchError, match=r"plan root.*replaced"):
            _run_claude_session(_request(working_dir))

        assert raced
        assert not (outside / str(session_id)).exists()
        assert list(original_root.iterdir()) == []

    @pytest.mark.parametrize(
        ("failure", "error"),
        [
            ("timeout", PlanTransportError),
            ("subprocess", PlanTransportError),
            ("nonzero", PlanTransportError),
            ("symlink", PlanArtifactMalformedError),
            ("missing", PlanArtifactMissingError),
            ("ambiguous", PlanArtifactAmbiguousError),
            ("unreadable", PlanArtifactMalformedError),
            ("oversized", PlanArtifactMalformedError),
            ("blank", PlanArtifactMalformedError),
        ],
    )
    def test_failed_collection_preserves_partial_artifacts(
        self,
        failure: str,
        error: type[PlanSessionError],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        outside = tmp_path / "outside.md"
        outside.write_text("# Keep", encoding="utf-8")

        def fake_run(command: list[str], *, cwd: Path, timeout: float) -> int:
            settings = json.loads(command[command.index("--settings") + 1])
            run_dir = cwd / settings["plansDirectory"]
            if failure in {"timeout", "subprocess", "nonzero"}:
                (run_dir / "partial.tmp").write_text("partial", encoding="utf-8")
            elif failure == "symlink":
                (run_dir / "plan.md").symlink_to(outside)
            elif failure == "missing":
                (run_dir / "notes.txt").write_text("not a plan", encoding="utf-8")
            elif failure == "ambiguous":
                (run_dir / "one.md").write_text("# One", encoding="utf-8")
                (run_dir / "two.md").write_text("# Two", encoding="utf-8")
            elif failure == "unreadable":
                (run_dir / "plan.md").write_bytes(b"\xff")
            elif failure == "oversized":
                (run_dir / "plan.md").write_bytes(b"x" * 65)
            elif failure == "blank":
                (run_dir / "plan.md").write_text("   ", encoding="utf-8")

            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, timeout)
            if failure == "subprocess":
                raise OSError("collector failed")
            return 1 if failure == "nonzero" else 0

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", fake_run)
        if failure == "oversized":
            monkeypatch.setattr("crossby.ai_tools.plan_process._PLAN_ARTIFACT_TEXT_LIMIT", 64)

        with pytest.raises(error):
            _run_claude_session(_request(tmp_path))

        root = tmp_path / ".crossby" / "plan-sessions"
        assert root.is_dir()
        assert len(list(root.iterdir())) == 1
        assert outside.read_text(encoding="utf-8") == "# Keep"

    @pytest.mark.parametrize("stage", ["command", "enumeration", "result"])
    def test_post_directory_failures_are_wrapped_and_cleaned(
        self, stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        root = tmp_path / "plans"
        run_dir = root / str(session_id)
        adapter = AbstractAITool.get(AIToolID.CLAUDE)
        monkeypatch.setattr("crossby.ai_tools.claude.uuid.uuid4", lambda: session_id)

        if stage == "command":

            def fail_build(**_kwargs: Any) -> list[str]:
                raise ValueError("invalid launch command")

            monkeypatch.setattr(adapter, "build_launch_command", fail_build)
        elif stage == "enumeration":
            original_listdir = os.listdir
            monkeypatch.setattr(
                "crossby.ai_tools.plan_process.run_interactive",
                lambda *_args, **_kwargs: 0,
            )

            def fail_run_directory(path: Any):
                if isinstance(path, int) or Path(path) == run_dir:
                    raise OSError("plan directory became unreadable")
                return original_listdir(path)

            monkeypatch.setattr(os, "listdir", fail_run_directory)
        else:

            def write_plan(*_args: Any, **_kwargs: Any) -> int:
                (run_dir / "plan.md").write_text("# Plan", encoding="utf-8")
                return 0

            monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", write_plan)

            def fail_result(**_kwargs: Any) -> None:
                raise ValueError("result validation failed")

            monkeypatch.setattr("crossby.ai_tools.claude.PlanSessionResult", fail_result)

        with pytest.raises(PlanTransportError, match="after creating its isolated directory"):
            adapter.run_plan_session(
                _request(tmp_path, plan_output_dir=root), terminal_interaction_handler
            )

        assert root.is_dir()
        if stage == "result":
            assert (run_dir / "plan.md").read_text() == "# Plan"
        else:
            assert list(root.iterdir()) == []

    @pytest.mark.parametrize("replacement", ["symlink", "directory"])
    def test_replaced_run_directory_is_never_collected_or_deleted(
        self, replacement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decoy = tmp_path / "another-session"
        decoy.mkdir()
        (decoy / "plan.md").write_text("# Another run")
        replaced: list[Path] = []

        def fake_run(command: list[str], *, cwd: Path, **_kwargs: Any) -> int:
            settings = json.loads(command[command.index("--settings") + 1])
            run_dir = cwd / settings["plansDirectory"]
            run_dir.rename(run_dir.with_name("original-run"))
            if replacement == "symlink":
                run_dir.symlink_to(decoy, target_is_directory=True)
            else:
                decoy.rename(run_dir)
            replaced.append(run_dir)
            return 0

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", fake_run)
        with pytest.raises(PlanBindingMismatchError, match="directory was replaced"):
            _run_claude_session(_request(tmp_path))

        assert (replaced[0] / "plan.md").read_text() == "# Another run"

    @pytest.mark.parametrize("replacement", ["symlink", "file"])
    def test_artifact_replaced_between_validation_and_open_is_rejected(
        self, replacement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path / "outside.md"
        outside.write_text("# Another run")
        original_open = os.open
        artifacts: list[Path] = []

        def fake_run(command: list[str], *, cwd: Path, **_kwargs: Any) -> int:
            settings = json.loads(command[command.index("--settings") + 1])
            artifact = cwd / settings["plansDirectory"] / "plan.md"
            artifact.write_text("# This run")
            artifacts.append(artifact)
            return 0

        def replace_before_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if Path(path).name == "plan.md":
                artifact = artifacts[0]
                artifact.rename(artifact.with_suffix(".original"))
                if replacement == "symlink":
                    artifact.symlink_to(outside)
                else:
                    artifact.write_text("# Replacement")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", fake_run)
        monkeypatch.setattr(os, "open", replace_before_open)
        with pytest.raises(PlanArtifactMalformedError, match="bounded UTF-8 contract"):
            _run_claude_session(_request(tmp_path))

        assert outside.read_text() == "# Another run"

    def test_isolated_directory_creation_errors_are_session_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output_dir = tmp_path / "plans"
        output_dir.mkdir()
        session_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        run_dir = output_dir / str(session_id)
        original_mkdir = os.mkdir

        def fake_mkdir(
            path: Any,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> None:
            if Path(path).name == str(session_id):
                raise PermissionError("read-only plan root")
            original_mkdir(path, mode, dir_fd=dir_fd)

        monkeypatch.setattr("crossby.ai_tools.claude.uuid.uuid4", lambda: session_id)
        monkeypatch.setattr(os, "mkdir", fake_mkdir)

        with pytest.raises(PlanTransportError, match="isolated plan directory") as raised:
            _run_claude_session(_request(tmp_path, plan_output_dir=output_dir))

        assert raised.value.session_id == str(session_id)
        assert raised.value.paths == (run_dir,)

    def test_exact_isolated_directory_ignores_decoy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        decoy = tmp_path / ".crossby" / "plan-sessions" / "newer-decoy.md"
        decoy.parent.mkdir(parents=True)
        decoy.write_text("# Wrong plan", encoding="utf-8")
        fixture = (FIXTURES / "claude_plan.md").read_text(encoding="utf-8")

        def fake_run(command: list[str], *, cwd: Path, timeout: float) -> int:
            assert cwd == tmp_path
            assert timeout > 0
            settings = json.loads(command[command.index("--settings") + 1])
            run_dir = cwd / settings["plansDirectory"]
            (run_dir / "native.md").write_text(fixture, encoding="utf-8")
            return 0

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", fake_run)
        result = _run_claude_session(_request(tmp_path))
        _assert_result(
            result,
            tool=AIToolID.CLAUDE,
            source=PlanArtifactSource.REQUESTED_PATH,
            binding=PlanSessionBinding.ISOLATED_RUN_PATH,
        )
        assert result.artifact_path is not None and result.artifact_path.is_file()
        assert result.artifact_path.parent.name == result.session_id
        assert decoy.read_text(encoding="utf-8") == "# Wrong plan"

    def test_multiple_run_artifacts_fail_as_ambiguous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(command: list[str], *, cwd: Path, timeout: float) -> int:
            del timeout
            settings = json.loads(command[command.index("--settings") + 1])
            run_dir = cwd / settings["plansDirectory"]
            (run_dir / "one.md").write_text("# One", encoding="utf-8")
            (run_dir / "two.md").write_text("# Two", encoding="utf-8")
            return 0

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_interactive", fake_run)
        with pytest.raises(PlanArtifactAmbiguousError):
            _run_claude_session(_request(tmp_path))


class TestExactSessionCliCollectors:
    def test_antigravity_requires_exact_schema_echo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = (FIXTURES / "antigravity_success.json").read_text(encoding="utf-8")
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: CapturedProcess(0, payload, ""),
        )
        result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
            _antigravity_request(tmp_path)
        )
        _assert_result(
            result,
            tool=AIToolID.ANTIGRAVITY_CLI,
            source=PlanArtifactSource.STRUCTURED_OUTPUT,
            binding=PlanSessionBinding.CONVERSATION_ID,
        )
        assert result.session_id == "conv-exact-123"

        malformed = json.loads(payload)
        malformed["schema"]["additionalProperties"] = True
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: CapturedProcess(0, json.dumps(malformed), ""),
        )
        with pytest.raises(PlanArtifactMalformedError, match="schema"):
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
                _antigravity_request(tmp_path)
            )

    def test_antigravity_normalizes_blank_artifact_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = json.loads((FIXTURES / "antigravity_success.json").read_text())
        payload["artifact_id"] = "   "
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: CapturedProcess(0, json.dumps(payload), ""),
        )

        result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
            _antigravity_request(tmp_path)
        )

        assert result.artifact_id is None

    def test_antigravity_continues_questions_by_exact_conversation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waiting = json.loads((FIXTURES / "antigravity_waiting.json").read_text())
        waiting_again = {**waiting, "question": {**waiting["question"], "id": "tests"}}
        success = (FIXTURES / "antigravity_success.json").read_text()
        runs = iter(
            [
                CapturedProcess(0, json.dumps(waiting), ""),
                CapturedProcess(0, json.dumps(waiting_again), ""),
                CapturedProcess(0, success, ""),
            ]
        )
        commands: list[list[str]] = []
        timeouts: list[float] = []
        questions: list[str] = []

        def fake_run(command: list[str], **kwargs: Any) -> CapturedProcess:
            commands.append(command)
            timeouts.append(kwargs["timeout"])
            return next(runs)

        def answer(interaction: Any) -> PlanInteractionResponse:
            questions.append(interaction.question_id)
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                option_id="yes",
            )

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", fake_run)
        monkeypatch.setattr("crossby.ai_tools.base.monotonic", lambda: 100.0)
        clock = iter((100.0, 101.0, 103.0, 105.0))
        monkeypatch.setattr("crossby.ai_tools.antigravity_cli.time.monotonic", lambda: next(clock))
        result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
            _antigravity_request(
                tmp_path,
                trusted_dirs=(tmp_path / "reference",),
                timeout_seconds=10,
            ),
            answer,
        )
        assert result.session_id == "conv-exact-123"
        assert questions == ["scope", "tests"]
        assert timeouts == [9.0, 7.0, 5.0]
        for command in commands[1:]:
            assert command[command.index("--conversation") + 1] == "conv-exact-123"
            assert command[command.index("--print") + 1] == "Yes"
            assert command[command.index("--mode") + 1] == "plan"
            assert command[command.index("--model") + 1] == "gemini-3.8-flash-medium"
            assert command[command.index("--add-dir") + 1] == str(tmp_path / "reference")

    def test_antigravity_exposes_open_ended_questions_as_free_form(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waiting = json.loads((FIXTURES / "antigravity_waiting.json").read_text())
        waiting["question"]["options"] = []
        success = (FIXTURES / "antigravity_success.json").read_text()
        runs = iter((CapturedProcess(0, json.dumps(waiting), ""), CapturedProcess(0, success, "")))
        commands: list[list[str]] = []
        seen: list[Any] = []

        def fake_run(command: list[str], **_kwargs: Any) -> CapturedProcess:
            commands.append(command)
            return next(runs)

        def answer(interaction: Any) -> PlanInteractionResponse:
            seen.append(interaction)
            if not interaction.allow_other:
                return PlanInteractionResponse(outcome=PlanInteractionOutcome.SKIPPED)
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                answer="Keep compatibility",
            )

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", fake_run)

        AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
            _antigravity_request(tmp_path), answer
        )

        assert seen[0].allow_other is True
        assert commands[1][commands[1].index("--print") + 1] == "Keep compatibility"

    @pytest.mark.parametrize(
        "response",
        [
            PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                answer="Maybe",
            ),
            PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                answer="Maybe",
                option_id="yes",
            ),
        ],
        ids=("text-only", "mixed"),
    )
    def test_antigravity_rejects_free_text_for_option_only_questions(
        self,
        response: PlanInteractionResponse,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        waiting = (FIXTURES / "antigravity_waiting.json").read_text(encoding="utf-8")
        runs = iter((CapturedProcess(0, waiting, ""),))
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: next(runs),
        )

        with pytest.raises(PlanInteractionRequiredError, match="free-form"):
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
                _antigravity_request(tmp_path), lambda _interaction: response
            )

    def test_antigravity_denial_discards_stale_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waiting = (FIXTURES / "antigravity_waiting.json").read_text(encoding="utf-8")
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: Any) -> CapturedProcess:
            commands.append(command)
            return CapturedProcess(0, waiting, "")

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", fake_run)

        with pytest.raises(PlanInteractionRequiredError, match="left unanswered"):
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
                _antigravity_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    answer="stale answer",
                    option_id="yes",
                ),
            )

        assert len(commands) == 1

    @pytest.mark.parametrize(
        "question",
        [
            {"prompt": "Choose a scope"},
            {"id": "scope"},
            {"id": " ", "prompt": "Choose a scope"},
            {"id": "scope", "prompt": " "},
        ],
        ids=("missing-question-id", "missing-prompt", "blank-question-id", "blank-prompt"),
    )
    def test_antigravity_rejects_blank_interaction_fields_as_malformed_artifacts(
        self,
        question: dict[str, str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        waiting = json.loads((FIXTURES / "antigravity_waiting.json").read_text())
        waiting["question"] = question
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: CapturedProcess(0, json.dumps(waiting), ""),
        )

        with pytest.raises(PlanArtifactMalformedError, match="malformed interaction data"):
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
                _antigravity_request(tmp_path)
            )

    def test_antigravity_rejects_malformed_native_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waiting = json.loads((FIXTURES / "antigravity_waiting.json").read_text())
        waiting["question"]["options"].append({"id": "no-label"})
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: CapturedProcess(0, json.dumps(waiting), ""),
        )

        with pytest.raises(PlanArtifactMalformedError, match="malformed interaction data"):
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
                _antigravity_request(tmp_path)
            )

    def test_antigravity_rejects_unknown_native_option_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waiting = (FIXTURES / "antigravity_waiting.json").read_text(encoding="utf-8")
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: CapturedProcess(0, waiting, ""),
        )

        with pytest.raises(PlanInteractionRequiredError, match="valid native option IDs"):
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
                _antigravity_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    option_id="fabricated",
                ),
            )


class TestProtocolCollectors:
    def test_codex_clears_unrequested_writable_roots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_success.jsonl", headerless=True)

        AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))

        thread_start = next(
            payload
            for kind, payload in FakeRpc.instances[0].sent
            if kind == "request" and payload["id"] == 3
        )
        assert thread_start["params"]["config"]["sandbox_workspace_write"] == {
            "network_access": False,
            "writable_roots": [],
        }

    def test_cursor_normalizes_frame_overflow_to_a_session_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class OverflowRpc(FakeRpc):
            def read(self, *, timeout: float) -> dict[str, Any]:
                assert timeout > 0
                raise JsonRpcFrameLimitError(64)

        OverflowRpc.scripts = [[]]
        OverflowRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", OverflowRpc)

        with pytest.raises(PlanTransportError, match="frame exceeded") as raised:
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path))

        assert raised.value.tool_id is AIToolID.CURSOR
        assert OverflowRpc.instances[0].closed

    @pytest.mark.parametrize(
        ("model", "effort"),
        [
            (None, None),
            (None, EffortLevel.HIGH),
            ("sonnet-4.6", None),
            (" ", EffortLevel.HIGH),
        ],
        ids=("both-missing", "model-missing", "effort-missing", "model-blank"),
    )
    def test_cursor_rejects_incomplete_request_before_protocol_process(
        self,
        model: str | None,
        effort: EffortLevel | None,
        tmp_path: Path,
    ) -> None:
        with (
            patch("crossby.ai_tools.plan_process.JsonRpcProcess") as process,
            pytest.raises(PlanSessionUnsupportedError, match="explicit model and effort"),
        ):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _request(tmp_path, model=model, effort=effort)
            )

        process.assert_not_called()

    @pytest.mark.parametrize(
        ("model", "effort", "message"),
        [
            ("gpt-5.4-low", EffortLevel.HIGH, "encodes effort='low'"),
            (
                "sonnet-4.6[effort=medium]",
                EffortLevel.LOW,
                "encodes effort='medium'",
            ),
            (
                "sonnet-4.6[effort=invalid]",
                EffortLevel.HIGH,
                "invalid effort overrides",
            ),
            (
                "gpt-5.4-low[effort=high]",
                EffortLevel.HIGH,
                "conflicting effort encodings",
            ),
            (
                "gpt-5.3-codex[context=300k]",
                EffortLevel.MEDIUM,
                "unsupported bracket override 'context'",
            ),
            ("auto", EffortLevel.HIGH, "selected model is not known"),
        ],
    )
    def test_cursor_rejects_incompatible_effort_model_before_protocol_process(
        self,
        model: str,
        effort: EffortLevel,
        message: str,
        tmp_path: Path,
    ) -> None:
        with (
            patch("crossby.ai_tools.plan_process.JsonRpcProcess") as process,
            pytest.raises(PlanSessionUnsupportedError, match=message),
        ):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _request(tmp_path, model=model, effort=effort)
            )

        process.assert_not_called()

    def test_cursor_configures_low_and_medium_as_distinct_acp_efforts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configured: dict[EffortLevel, dict[str, Any]] = {}
        for effort in (EffortLevel.LOW, EffortLevel.MEDIUM):
            messages = _rpc_fixture("cursor_acp_success.jsonl")
            for index in (3, 4):
                messages[index]["result"]["configOptions"][1]["currentValue"] = effort.value
            FakeRpc.scripts = [messages]
            FakeRpc.instances = []
            monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _request(tmp_path, model="gpt-5.3-codex", effort=effort),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="rejected",
                ),
            )
            configured[effort] = next(
                payload
                for kind, payload in FakeRpc.instances[0].sent
                if kind == "request" and payload["id"] == 5
            )

        assert configured[EffortLevel.LOW]["params"]["value"] == "low"
        assert configured[EffortLevel.MEDIUM]["params"]["value"] == "medium"

    def test_cursor_accepts_tier_specific_effort_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        for index in (2, 3, 4):
            messages[index]["result"]["configOptions"][0]["currentValue"] = "gpt-5.4"
        for index in (3, 4):
            messages[index]["result"]["configOptions"][1]["currentValue"] = "extra-high"
        messages[4]["result"]["configOptions"][2]["currentValue"] = "true"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        result = AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _request(tmp_path, model="gpt-5.4-xhigh-fast", effort=EffortLevel.XHIGH),
            lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            ),
        )

        assert result.artifact_id == "plan-tool-exact-123"
        assert FakeRpc.instances[0].command == ["agent", "--sandbox", "enabled", "acp"]
        requests = [payload for kind, payload in FakeRpc.instances[0].sent if kind == "request"]
        assert requests[2]["params"] == {
            "sessionId": "cursor-exact-123",
            "configId": "model",
            "value": "gpt-5.4",
        }
        assert requests[3]["params"] == {
            "sessionId": "cursor-exact-123",
            "configId": "reasoning",
            "value": "extra-high",
        }
        assert requests[4]["params"] == {
            "sessionId": "cursor-exact-123",
            "configId": "fast",
            "value": "true",
        }
        assert requests[5]["params"] == {
            "sessionId": "cursor-exact-123",
            "modeId": "plan",
        }

    def test_cursor_rejects_configuration_drift_before_plan_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        messages[4]["result"]["configOptions"][1]["currentValue"] = "low"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        with pytest.raises(PlanTransportError, match="reset the requested reasoning effort"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="rejected",
                ),
            )

        assert not any(
            kind == "request" and payload["method"] == "session/set_mode"
            for kind, payload in FakeRpc.instances[0].sent
        )
        assert FakeRpc.instances[0].closed

    def test_cursor_requires_configured_variants_in_final_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        thinking_option = {
            "id": "thinking",
            "category": "model_config",
            "currentValue": "false",
            "options": [{"value": "false"}, {"value": "true"}],
        }
        for index in (1, 2, 3, 4):
            options = messages[index]["result"]["configOptions"]
            options[0]["options"].append({"value": "claude-sonnet-4-6"})
            if index > 1:
                options[0]["currentValue"] = "claude-sonnet-4-6"
            options.insert(2, {**thinking_option})
        messages[3]["result"]["configOptions"][1]["currentValue"] = "high"
        messages[4]["result"]["configOptions"][1]["currentValue"] = "high"
        messages[4]["result"]["configOptions"][2]["currentValue"] = "true"
        messages[4]["result"]["configOptions"][3]["currentValue"] = "true"
        final_state = {
            **messages[4],
            "id": 7,
            "result": {
                "configOptions": [
                    option
                    for option in messages[4]["result"]["configOptions"]
                    if option["id"] != "thinking"
                ]
            },
        }
        messages.insert(5, final_state)
        messages[6]["id"] = 8
        messages[-1]["id"] = 9
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        with pytest.raises(PlanTransportError, match="thinking"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _request(
                    tmp_path,
                    model="claude-sonnet-4-6-thinking-fast",
                    effort=EffortLevel.HIGH,
                ),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="rejected",
                ),
            )

        assert not any(
            kind == "request" and payload["method"] == "session/set_mode"
            for kind, payload in FakeRpc.instances[0].sent
        )
        assert FakeRpc.instances[0].closed

    def test_cursor_accepts_a_model_advertised_only_by_acp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        for index in (1, 2, 3, 4):
            config_options = messages[index]["result"]["configOptions"]
            model_option = config_options[0]
            model_option["options"].append({"value": "claude-sonnet-4-6"})
            if index > 1:
                model_option["currentValue"] = "claude-sonnet-4-6"
            config_options.insert(
                1,
                {
                    "id": "thinking",
                    "category": "thought_level",
                    "currentValue": "true",
                    "options": [{"value": "false"}, {"value": "true"}],
                },
            )
            config_options[:] = [item for item in config_options if item["id"] != "fast"]
            if index == 4:
                config_options[1]["currentValue"] = "false"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        result = AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _request(tmp_path, model="claude-sonnet-4-6", effort=EffortLevel.MEDIUM),
            lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            ),
        )

        assert result.artifact_id == "plan-tool-exact-123"
        requests = [payload for kind, payload in FakeRpc.instances[0].sent if kind == "request"]
        assert requests[2]["params"]["value"] == "claude-sonnet-4-6"
        assert requests[3]["params"] == {
            "sessionId": "cursor-exact-123",
            "configId": "reasoning",
            "value": "medium",
        }
        assert requests[4]["params"] == {
            "sessionId": "cursor-exact-123",
            "configId": "thinking",
            "value": "false",
        }
        assert requests[5]["params"] == {
            "sessionId": "cursor-exact-123",
            "modeId": "plan",
        }

    def test_codex_collects_completed_plan_item_after_successful_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_success.jsonl", headerless=True)
        request = _request(
            tmp_path,
            network_access=True,
            trusted_dirs=(tmp_path / "reference",),
            approval_policy=PlanApprovalPolicy.NEVER,
        )
        result = AbstractAITool.get(AIToolID.CODEX).run_plan_session(request)
        _assert_result(
            result,
            tool=AIToolID.CODEX,
            source=PlanArtifactSource.PROTOCOL_EVENT,
            binding=PlanSessionBinding.THREAD_TURN_IDS,
        )
        assert result.thread_id == "thr-exact-123"
        assert result.turn_id == "turn-exact-123"
        assert result.artifact_id == "plan-exact-123"
        rpc = FakeRpc.instances[0]
        thread_start = next(
            payload for kind, payload in rpc.sent if kind == "request" and payload["id"] == 3
        )
        assert thread_start["params"]["sandbox"] == "workspace-write"
        assert thread_start["params"]["approvalPolicy"] == "never"
        assert thread_start["params"]["config"]["sandbox_workspace_write"] == {
            "writable_roots": [str(tmp_path / "reference")],
            "network_access": True,
        }
        assert not any(
            kind == "request" and payload["method"] == "turn/interrupt"
            for kind, payload in rpc.sent
        )
        cleanup = next(
            payload for kind, payload in rpc.sent if kind == "request" and payload["id"] == 5
        )
        assert cleanup == {
            "id": 5,
            "method": "thread/backgroundTerminals/clean",
            "params": {"threadId": "thr-exact-123"},
        }
        assert all(
            "jsonrpc" not in message for message in _rpc_fixture("codex_app_server_success.jsonl")
        )
        assert rpc.closed

    def test_codex_collects_completed_plan_item_while_waiting_for_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        completed_plan = messages.pop(-3)
        messages.insert(-1, completed_plan)
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        result = AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))

        assert result.artifact_id == "plan-exact-123"
        assert result.plan.startswith("# Native plan")
        assert FakeRpc.instances[0].closed

    def test_codex_rejects_nonzero_protocol_exit_after_successful_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class NonzeroCloseRpc(FakeRpc):
            def close(self) -> int:
                self.closed = True
                return 17

        NonzeroCloseRpc.scripts = [_rpc_fixture("codex_app_server_success.jsonl")]
        NonzeroCloseRpc.instances = []
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", NonzeroCloseRpc
        )

        with pytest.raises(PlanTransportError, match="exited with status 17") as raised:
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))

        assert raised.value.exit_code == 17
        assert NonzeroCloseRpc.instances[0].closed

    def test_codex_rejects_multiple_completed_plan_items_for_bound_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_success.jsonl", headerless=True)
        FakeRpc.scripts[0].insert(
            -2,
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-exact-123",
                    "turnId": "turn-exact-123",
                    "item": {
                        "type": "plan",
                        "id": "plan-conflicting-456",
                        "text": "# Conflicting plan\n\n1. Do something else.",
                    },
                },
            },
        )

        with pytest.raises(PlanArtifactAmbiguousError, match="multiple authoritative"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))

    def test_codex_rejects_completed_plan_item_while_waiting_for_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages.insert(
            -1,
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thr-exact-123",
                    "turnId": "turn-exact-123",
                    "item": {
                        "type": "plan",
                        "id": "plan-late-conflict-789",
                        "text": "# Late conflicting plan\n\n1. Replace the prior plan.",
                    },
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        with pytest.raises(PlanArtifactAmbiguousError, match="multiple authoritative"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))

    def test_codex_maps_max_effort_in_plan_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_success.jsonl", headerless=True)

        AbstractAITool.get(AIToolID.CODEX).run_plan_session(
            _request(tmp_path, effort=EffortLevel.MAX)
        )

        turn_start = next(
            payload
            for kind, payload in FakeRpc.instances[0].sent
            if kind == "request" and payload["id"] == 4
        )
        assert turn_start["params"]["collaborationMode"]["settings"]["reasoning_effort"] == "xhigh"

    def test_codex_handles_inflight_requests_before_successful_turn_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages.insert(
            -2,
            {
                "id": 72,
                "method": "item/tool/requestUserInput",
                "params": {
                    "threadId": "thr-exact-123",
                    "turnId": "turn-exact-123",
                    "itemId": "question-item-late",
                    "questions": [
                        {
                            "id": "late",
                            "question": "One more choice?",
                            "options": [],
                            "isOther": True,
                        }
                    ],
                },
            },
        )
        messages.insert(
            -2,
            {
                "id": 73,
                "method": "item/commandExecution/requestApproval",
                "params": {
                    "threadId": "thr-exact-123",
                    "turnId": "turn-exact-123",
                    "approvalId": "approval-late",
                    "itemId": "command-late",
                    "reason": "Run a command?",
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        def answer(interaction: Any) -> PlanInteractionResponse:
            if interaction.kind is PlanInteractionKind.QUESTION:
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    answer="No",
                )
            return PlanInteractionResponse(outcome=PlanInteractionOutcome.DENIED)

        result = AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path), answer)

        assert result.artifact_id == "plan-exact-123"
        assert (
            "respond",
            {"id": 72, "result": {"answers": {"late": {"answers": ["No"]}}}},
        ) in FakeRpc.instances[0].sent
        assert (
            "respond",
            {"id": 73, "result": {"decision": "decline"}},
        ) in FakeRpc.instances[0].sent

    def test_codex_rejects_failed_turn_after_completed_plan_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages[-2]["params"]["turn"]["status"] = "failed"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        with pytest.raises(PlanTransportError, match="status 'failed'"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))

        assert FakeRpc.instances[0].closed

    def test_codex_denial_overrides_stale_accept_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages.insert(
            4,
            {
                "id": 70,
                "method": "item/commandExecution/requestApproval",
                "params": {
                    "threadId": "thr-exact-123",
                    "turnId": "turn-exact-123",
                    "approvalId": "approval-1",
                    "itemId": "command-1",
                    "reason": "Run a command?",
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        AbstractAITool.get(AIToolID.CODEX).run_plan_session(
            _request(tmp_path),
            lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="accept",
            ),
        )

        assert (
            "respond",
            {"id": 70, "result": {"decision": "decline"}},
        ) in FakeRpc.instances[0].sent

    @pytest.mark.parametrize(
        "response",
        [
            PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                option_id="fabricated",
            ),
            PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                option_ids=("accept", "decline"),
            ),
        ],
        ids=("unknown-option", "multiple-options"),
    )
    def test_codex_rejects_invalid_answered_approval_options(
        self,
        response: PlanInteractionResponse,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages.insert(
            4,
            {
                "id": 70,
                "method": "item/commandExecution/requestApproval",
                "params": {
                    "threadId": "thr-exact-123",
                    "turnId": "turn-exact-123",
                    "approvalId": "approval-1",
                    "itemId": "command-1",
                    "reason": "Run a command?",
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        with pytest.raises(PlanInteractionRequiredError, match="exactly one valid native"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path), lambda _interaction: response
            )

        assert FakeRpc.instances[0].closed
        assert not any(
            kind == "respond" and payload["id"] == 70 for kind, payload in FakeRpc.instances[0].sent
        )

    def test_codex_forwards_native_question_ids_and_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_question.jsonl", headerless=True)
        seen: list[Any] = []

        def answer(interaction: Any) -> PlanInteractionResponse:
            seen.append(interaction)
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                option_id="adapter",
            )

        result = AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path), answer)
        assert result.artifact_id == "plan-exact-123"
        assert seen[0].question_id == "architecture"
        assert seen[0].allow_other is True
        assert [option.option_id for option in seen[0].options] == ["adapter", "service"]
        assert (
            "respond",
            {
                "id": 71,
                "result": {"answers": {"architecture": {"answers": ["Adapter"]}}},
            },
        ) in FakeRpc.instances[0].sent

    def test_codex_forwards_allowed_free_form_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_question.jsonl", headerless=True)

        AbstractAITool.get(AIToolID.CODEX).run_plan_session(
            _request(tmp_path),
            lambda interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                answer="Keep collection in the native adapter",
            ),
        )

        assert (
            "respond",
            {
                "id": 71,
                "result": {
                    "answers": {
                        "architecture": {"answers": ["Keep collection in the native adapter"]}
                    }
                },
            },
        ) in FakeRpc.instances[0].sent

    def test_codex_rejects_mixed_answer_for_single_select_question(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_question.jsonl", headerless=True)

        with pytest.raises(PlanInteractionRequiredError, match="exactly one answer"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    answer="Keep collection in the native adapter",
                    option_id="adapter",
                ),
            )

        assert not any(
            kind == "respond" and payload["id"] == 71 for kind, payload in FakeRpc.instances[0].sent
        )
        assert FakeRpc.instances[0].closed

    def test_codex_rejects_disallowed_free_form_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_question.jsonl")
        messages[4]["params"]["questions"][0]["isOther"] = False
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)
        seen: list[Any] = []

        with pytest.raises(PlanInteractionRequiredError, match="does not allow a free-form"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path),
                lambda interaction: (
                    seen.append(interaction)
                    or PlanInteractionResponse(
                        outcome=PlanInteractionOutcome.ANSWERED,
                        answer="Keep collection in the native adapter",
                    )
                ),
            )

        assert seen[0].allow_other is False
        assert not any(
            kind == "respond" and payload["id"] == 71 for kind, payload in FakeRpc.instances[0].sent
        )
        assert FakeRpc.instances[0].closed

    def test_codex_rejects_non_boolean_free_form_field_before_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_question.jsonl")
        messages[4]["params"]["questions"][0]["isOther"] = "true"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)
        seen: list[Any] = []

        with pytest.raises(PlanTransportError, match="non-boolean free-form"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path),
                lambda interaction: (
                    seen.append(interaction)
                    or PlanInteractionResponse(
                        outcome=PlanInteractionOutcome.ANSWERED,
                        option_id="adapter",
                    )
                ),
            )

        assert seen == []
        assert FakeRpc.instances[0].closed

    def test_codex_rejects_malformed_native_question_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_question.jsonl")
        messages[4]["params"]["questions"][0]["options"].append({"id": "missing-label"})
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        with pytest.raises(PlanTransportError, match="native question options"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))

        assert FakeRpc.instances[0].closed

    def test_codex_rejects_duplicate_question_ids_before_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_question.jsonl")
        duplicate = dict(messages[4]["params"]["questions"][0])
        duplicate["question"] = "Which boundary should own validation?"
        messages[4]["params"]["questions"].append(duplicate)
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)
        seen: list[Any] = []

        with pytest.raises(PlanTransportError, match="duplicate native ID"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path),
                lambda interaction: (
                    seen.append(interaction)
                    or PlanInteractionResponse(
                        outcome=PlanInteractionOutcome.ANSWERED,
                        option_id="adapter",
                    )
                ),
            )

        assert seen == []
        assert FakeRpc.instances[0].closed

    @pytest.mark.parametrize(
        ("field", "value"),
        [("allowMultiple", "false"), ("isMultiple", 1)],
    )
    def test_codex_rejects_non_boolean_question_cardinality_before_callback(
        self,
        field: str,
        value: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages = _rpc_fixture("codex_app_server_question.jsonl")
        messages[4]["params"]["questions"][0][field] = value
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)
        seen: list[Any] = []

        with pytest.raises(PlanTransportError, match="non-boolean multi-select"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path),
                lambda interaction: (
                    seen.append(interaction)
                    or PlanInteractionResponse(
                        outcome=PlanInteractionOutcome.ANSWERED,
                        option_id="adapter",
                    )
                ),
            )

        assert seen == []
        assert FakeRpc.instances[0].closed

    def test_codex_rejects_unknown_native_question_option_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_question.jsonl", headerless=True)

        with pytest.raises(PlanInteractionRequiredError, match="valid native option IDs"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    option_id="fabricated",
                ),
            )

        assert FakeRpc.instances[0].closed

    def test_codex_question_denial_discards_stale_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_question.jsonl", headerless=True)

        with pytest.raises(PlanInteractionRequiredError, match="left unanswered"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(
                _request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="adapter",
                ),
            )

        assert FakeRpc.instances[0].closed
        assert not any(
            kind == "respond" and payload["id"] == 71 for kind, payload in FakeRpc.instances[0].sent
        )

    def test_codex_rejects_cross_turn_plan_and_closes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages[4]["params"]["turnId"] = "turn-decoy"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)
        with pytest.raises(PlanBindingMismatchError):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))
        assert FakeRpc.instances[0].closed

    def test_codex_rejects_blank_completed_plan_item_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages[4]["params"]["item"]["id"] = " "
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.HeaderlessJsonRpcProcess", FakeRpc)

        with pytest.raises(PlanArtifactMalformedError, match="omitted its ID"):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))
        assert FakeRpc.instances[0].closed

    def test_cursor_requires_nonexecuting_plan_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_success.jsonl")
        with pytest.raises(PlanInteractionRequiredError) as raised:
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path))
        assert raised.value.interaction.session_id == "cursor-exact-123"
        assert FakeRpc.instances[0].closed

        _install_rpc(monkeypatch, "cursor_acp_success.jsonl")

        def keep_plan(_interaction: Any) -> PlanInteractionResponse:
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            )

        result = AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _cursor_request(tmp_path, sandbox=False, approval_policy=PlanApprovalPolicy.NEVER),
            keep_plan,
        )
        _assert_result(
            result,
            tool=AIToolID.CURSOR,
            source=PlanArtifactSource.PROTOCOL_EVENT,
            binding=PlanSessionBinding.SESSION_ID,
        )
        assert FakeRpc.instances[0].command[:3] == ["agent", "--sandbox", "disabled"]
        assert FakeRpc.instances[0].command[-1] == "acp"
        assert (
            "respond",
            {
                "id": 91,
                "result": {
                    "outcome": {
                        "outcome": "rejected",
                        "reason": "Plan collected without implementation.",
                    }
                },
            },
        ) in FakeRpc.instances[0].sent

    def test_cursor_rejects_nonzero_protocol_exit_after_successful_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class NonzeroCloseRpc(FakeRpc):
            def close(self) -> int:
                self.closed = True
                return 23

        NonzeroCloseRpc.scripts = [_rpc_fixture("cursor_acp_success.jsonl")]
        NonzeroCloseRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", NonzeroCloseRpc)

        with pytest.raises(PlanTransportError, match="exited with status 23") as raised:
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="rejected",
                ),
            )

        assert raised.value.exit_code == 23
        assert NonzeroCloseRpc.instances[0].closed

    def test_cursor_final_plan_denial_overrides_stale_cancel_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_success.jsonl")

        AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _cursor_request(tmp_path),
            lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="cancelled",
            ),
        )

        assert (
            "respond",
            {
                "id": 91,
                "result": {
                    "outcome": {
                        "outcome": "rejected",
                        "reason": "Plan collected without implementation.",
                    }
                },
            },
        ) in FakeRpc.instances[0].sent

    @pytest.mark.parametrize(
        "response",
        [
            {"option_id": "stale"},
            {"option_ids": ("stale",)},
            {"option_ids": ("rejected", "cancelled")},
            {"option_id": "rejected", "option_ids": ("rejected",)},
            {"answer": "free text is not a native selection"},
        ],
    )
    def test_cursor_rejects_invalid_final_plan_selection(
        self, response: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_success.jsonl")
        with pytest.raises(PlanInteractionRequiredError, match="valid native option ID"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED, **response
                ),
            )
        assert FakeRpc.instances[0].closed
        assert not any(kind == "respond" for kind, _payload in FakeRpc.instances[0].sent)

    @pytest.mark.parametrize("option_id", ["rejected", "cancelled"])
    @pytest.mark.parametrize("field", ["option_id", "option_ids"])
    def test_cursor_preserves_explicit_final_plan_selection(
        self, option_id: str, field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_success.jsonl")
        response = {field: option_id if field == "option_id" else (option_id,)}
        AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _cursor_request(tmp_path),
            lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED, **response
            ),
        )
        replies = [payload for kind, payload in FakeRpc.instances[0].sent if kind == "respond"]
        assert replies[0]["result"]["outcome"]["outcome"] == option_id

    @pytest.mark.parametrize(
        ("prompt_response", "message"),
        [
            ({"jsonrpc": "2.0", "id": 8}, "malformed session/prompt response"),
            (
                {"jsonrpc": "2.0", "id": 8, "result": {}},
                "malformed session/prompt stopReason",
            ),
            (
                {"jsonrpc": "2.0", "id": 8, "result": {"stopReason": 42}},
                "malformed session/prompt stopReason",
            ),
            (
                {"jsonrpc": "2.0", "id": 8, "result": {"stopReason": "cancelled"}},
                "stop reason 'cancelled'",
            ),
            (
                {"jsonrpc": "2.0", "id": 8, "result": {"stopReason": "refusal"}},
                "stop reason 'refusal'",
            ),
            (
                {"jsonrpc": "2.0", "id": 8, "result": {"stopReason": "max_tokens"}},
                "stop reason 'max_tokens'",
            ),
            (
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "result": {"stopReason": "max_turn_requests"},
                },
                "stop reason 'max_turn_requests'",
            ),
        ],
        ids=(
            "missing-result",
            "missing-stop-reason",
            "malformed-stop-reason",
            "cancelled",
            "refused",
            "max-tokens",
            "max-turn-requests",
        ),
    )
    def test_cursor_requires_successful_prompt_completion(
        self,
        prompt_response: dict[str, Any],
        message: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        messages[-1] = prompt_response
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        with pytest.raises(PlanTransportError, match=message):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="rejected",
                ),
            )

        assert FakeRpc.instances[0].closed

    def test_cursor_denial_overrides_stale_allow_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        messages.insert(
            6,
            {
                "jsonrpc": "2.0",
                "id": 90,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "cursor-exact-123",
                    "toolCall": {"toolCallId": "command-1", "title": "Run a command?"},
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once"},
                        {"optionId": "deny-once", "name": "Deny once"},
                    ],
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        def deny(interaction: Any) -> PlanInteractionResponse:
            if interaction.kind is PlanInteractionKind.PERMISSION:
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="allow-once",
                )
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            )

        AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path), deny)

        assert (
            "respond",
            {
                "id": 90,
                "result": {"outcome": {"outcome": "selected", "optionId": "deny-once"}},
            },
        ) in FakeRpc.instances[0].sent

    @pytest.mark.parametrize(
        ("options", "permission_response"),
        [
            (
                [
                    {"optionId": "allow-once", "name": "Allow once"},
                    {"optionId": "deny-once", "name": "Deny once"},
                ],
                PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.APPROVED,
                    option_id="fabricated-option",
                ),
            ),
            (
                [{"optionId": "proceed-once", "name": "Proceed once"}],
                PlanInteractionResponse(outcome=PlanInteractionOutcome.APPROVED),
            ),
            (
                [{"optionId": "allow-once", "name": "Allow once"}],
                PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.APPROVED,
                    option_id="fabricated-option",
                    option_ids=("allow-once",),
                ),
            ),
        ],
        ids=(
            "unknown-callback-option",
            "no-fabricated-approved-fallback",
            "validate-singular-and-plural-selections",
        ),
    )
    def test_cursor_permission_requires_a_native_option(
        self,
        options: list[dict[str, str]],
        permission_response: PlanInteractionResponse,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        messages.insert(
            6,
            {
                "jsonrpc": "2.0",
                "id": 90,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "cursor-exact-123",
                    "toolCall": {"toolCallId": "command-1", "title": "Run a command?"},
                    "options": options,
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        def answer(interaction: Any) -> PlanInteractionResponse:
            if interaction.kind is PlanInteractionKind.PERMISSION:
                return permission_response
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            )

        with pytest.raises(PlanInteractionRequiredError, match="valid native option ID"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path), answer)

        assert FakeRpc.instances[0].closed
        assert not any(
            kind == "respond" and payload["id"] == 90 for kind, payload in FakeRpc.instances[0].sent
        )

    def test_cursor_permission_rejects_native_option_without_an_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        messages.insert(
            6,
            {
                "jsonrpc": "2.0",
                "id": 90,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "cursor-exact-123",
                    "toolCall": {"toolCallId": "command-1", "title": "Run a command?"},
                    "options": [{"name": "Allow once"}],
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        with pytest.raises(PlanTransportError, match="native question options omitted"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.APPROVED
                ),
            )

        assert FakeRpc.instances[0].closed

    def test_cursor_permission_forwards_a_valid_native_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        messages.insert(
            6,
            {
                "jsonrpc": "2.0",
                "id": 90,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "cursor-exact-123",
                    "toolCall": {"toolCallId": "command-1", "title": "Run a command?"},
                    "options": [
                        {"optionId": "allow-once", "name": "Allow once"},
                        {"optionId": "deny-once", "name": "Deny once"},
                    ],
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        def answer(interaction: Any) -> PlanInteractionResponse:
            if interaction.kind is PlanInteractionKind.PERMISSION:
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.APPROVED,
                    option_id="allow-once",
                )
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            )

        AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path), answer)

        assert (
            "respond",
            {
                "id": 90,
                "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}},
            },
        ) in FakeRpc.instances[0].sent

    def test_cursor_permission_cancels_when_no_native_denial_option_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_success.jsonl")
        messages.insert(
            6,
            {
                "jsonrpc": "2.0",
                "id": 90,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "cursor-exact-123",
                    "toolCall": {"toolCallId": "command-1", "title": "Run a command?"},
                    "options": [{"optionId": "allow-once", "name": "Allow once"}],
                },
            },
        )
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _cursor_request(tmp_path, approval_policy=PlanApprovalPolicy.NEVER),
            lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            ),
        )

        assert (
            "respond",
            {"id": 90, "result": {"outcome": {"outcome": "cancelled"}}},
        ) in FakeRpc.instances[0].sent

    def test_cursor_forwards_multi_question_native_option_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_question.jsonl")
        seen: list[Any] = []

        def answer(interaction: Any) -> PlanInteractionResponse:
            seen.append(interaction)
            if interaction.question_id == "scope":
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    option_id="api",
                )
            if interaction.question_id == "tests":
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    option_ids=("unit", "integration"),
                )
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            )

        result = AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _cursor_request(tmp_path), answer
        )
        assert result.artifact_id == "plan-tool-exact-123"
        assert [(item.question_id, item.allow_multiple) for item in seen[:2]] == [
            ("scope", False),
            ("tests", True),
        ]
        assert (
            "respond",
            {
                "id": 90,
                "result": {
                    "outcome": {
                        "outcome": "answered",
                        "answers": [
                            {"questionId": "scope", "selectedOptionIds": ["api"]},
                            {
                                "questionId": "tests",
                                "selectedOptionIds": ["unit", "integration"],
                            },
                        ],
                    }
                },
            },
        ) in FakeRpc.instances[0].sent

    def test_cursor_rejects_malformed_native_question_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_question.jsonl")
        messages[6]["params"]["questions"][0]["options"].append({"id": "missing-label"})
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        with pytest.raises(PlanTransportError, match="native question options"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path))

        assert FakeRpc.instances[0].closed

    def test_cursor_rejects_ambiguous_textual_option_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_question.jsonl")
        for option in messages[6]["params"]["questions"][1]["options"]:
            option["label"] = "Required"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)

        def answer(interaction: Any) -> PlanInteractionResponse:
            if interaction.kind is PlanInteractionKind.PLAN_APPROVAL:
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.DENIED,
                    option_id="rejected",
                )
            if interaction.question_id == "scope":
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    option_id="api",
                )
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                answer="Required",
            )

        with pytest.raises(PlanInteractionRequiredError, match="valid native option IDs"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path), answer)

        assert not any(
            kind == "respond" and payload["id"] == 90 for kind, payload in FakeRpc.instances[0].sent
        )
        assert FakeRpc.instances[0].closed

    def test_cursor_rejects_text_with_explicit_question_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_question.jsonl")

        def answer(interaction: Any) -> PlanInteractionResponse:
            if interaction.question_id == "scope":
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    answer="Use the public API with compatibility shims",
                    option_id="api",
                )
            if interaction.question_id == "tests":
                return PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    option_ids=("unit", "integration"),
                )
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            )

        with pytest.raises(PlanInteractionRequiredError, match="cannot combine answer text"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_cursor_request(tmp_path), answer)

        assert not any(
            kind == "respond" and payload["id"] == 90 for kind, payload in FakeRpc.instances[0].sent
        )
        assert FakeRpc.instances[0].closed

    @pytest.mark.parametrize("value", ["false", 1, None])
    def test_cursor_rejects_non_boolean_question_cardinality_before_callback(
        self,
        value: object,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        messages = _rpc_fixture("cursor_acp_question.jsonl")
        messages[6]["params"]["questions"][1]["allowMultiple"] = value
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)
        seen: list[Any] = []

        with pytest.raises(PlanTransportError, match="non-boolean multi-select"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda interaction: (
                    seen.append(interaction)
                    or PlanInteractionResponse(
                        outcome=PlanInteractionOutcome.ANSWERED,
                        option_id="api",
                    )
                ),
            )

        assert seen == []
        assert FakeRpc.instances[0].closed

    def test_cursor_rejects_duplicate_question_ids_before_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("cursor_acp_question.jsonl")
        messages[6]["params"]["questions"][1]["id"] = "scope"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)
        seen: list[Any] = []

        with pytest.raises(PlanTransportError, match="duplicate native ID"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda interaction: (
                    seen.append(interaction)
                    or PlanInteractionResponse(
                        outcome=PlanInteractionOutcome.ANSWERED,
                        option_id="api",
                    )
                ),
            )

        assert seen == []
        assert FakeRpc.instances[0].closed

    def test_cursor_rejects_unknown_native_question_option_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_question.jsonl")

        with pytest.raises(PlanInteractionRequiredError, match="valid native option IDs"):
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
                _cursor_request(tmp_path),
                lambda _interaction: PlanInteractionResponse(
                    outcome=PlanInteractionOutcome.ANSWERED,
                    option_id="fabricated",
                ),
            )

        assert FakeRpc.instances[0].closed

    def test_cursor_question_denial_discards_stale_option(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_question.jsonl")

        result = AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _cursor_request(tmp_path),
            lambda _interaction: PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="api",
            ),
        )

        assert result.artifact_id == "plan-tool-exact-123"
        assert (
            "respond",
            {
                "id": 90,
                "result": {
                    "outcome": {
                        "outcome": "skipped",
                        "reason": "Question was not answered.",
                    }
                },
            },
        ) in FakeRpc.instances[0].sent
        assert not any(
            kind == "respond"
            and payload["id"] == 90
            and payload["result"]["outcome"]["outcome"] == "answered"
            for kind, payload in FakeRpc.instances[0].sent
        )
