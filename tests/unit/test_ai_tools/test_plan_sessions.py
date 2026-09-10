"""Fixture-backed contracts for deterministic native plan collection."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from crossby.ai_tools import (
    AbstractAITool,
    PlanApprovalPolicy,
    PlanArtifactAmbiguousError,
    PlanArtifactMalformedError,
    PlanArtifactSource,
    PlanBindingMismatchError,
    PlanInteractionOutcome,
    PlanInteractionRequiredError,
    PlanInteractionResponse,
    PlanSessionBinding,
    PlanSessionRequest,
    PlanSessionResult,
    PlanSessionUnsupportedError,
)
from crossby.ai_tools.plan_mode import safe_error_excerpt
from crossby.ai_tools.plan_process import CapturedProcess, parse_jsonl
from crossby.models.ai import AIToolID
from crossby.utils.versioning import BinaryVersion

FIXTURES = Path(__file__).parents[2] / "fixtures" / "plan_sessions"
EXACT_VERSION = "fixture-tool 9999.0.0+exact"


@pytest.fixture(autouse=True)
def _exact_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "crossby.utils.versioning.detect_binary_version_info",
        lambda _binary: BinaryVersion((9999, 0, 0), EXACT_VERSION),
    )


def _request(tmp_path: Path, **updates: Any) -> PlanSessionRequest:
    values: dict[str, Any] = {
        "prompt": "Create a native implementation plan",
        "working_dir": tmp_path,
    }
    values.update(updates)
    return PlanSessionRequest(**values)


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

    def __init__(self, command: list[str], *, cwd: Path) -> None:
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


def _install_rpc(monkeypatch: pytest.MonkeyPatch, fixture: str) -> None:
    FakeRpc.scripts = [_rpc_fixture(fixture)]
    FakeRpc.instances = []
    monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)


class TestNormalizedContract:
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
            AIToolID.COPILOT,
            AIToolID.OPENCODE,
            AIToolID.ANTIGRAVITY_CLI,
        ],
    )
    def test_unknown_version_fails_before_adapter_process(
        self, tool_id: AIToolID, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AbstractAITool.get(tool_id)
        collector = patch.object(adapter, "_run_plan_session")
        monkeypatch.setattr("crossby.utils.versioning.detect_binary_version_info", lambda _b: None)
        with collector as run, pytest.raises(PlanSessionUnsupportedError, match="version unknown"):
            adapter.run_plan_session(_request(tmp_path))
        run.assert_not_called()

    def test_below_floor_version_fails_before_adapter_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AbstractAITool.get(AIToolID.CODEX)
        monkeypatch.setattr(
            "crossby.utils.versioning.detect_binary_version_info",
            lambda _b: BinaryVersion((0, 1, 0), "codex-cli 0.1.0"),
        )
        with (
            patch.object(adapter, "_run_plan_session") as run,
            pytest.raises(PlanSessionUnsupportedError, match=r"0\.1\.0"),
        ):
            adapter.run_plan_session(_request(tmp_path))
        run.assert_not_called()

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

    def test_error_excerpt_redacts_and_bounds_secrets(self) -> None:
        excerpt = safe_error_excerpt("API_KEY=super-secret " + ("x" * 1000))
        assert excerpt is not None
        assert "super-secret" not in excerpt
        assert len(excerpt) == 500


class TestClaudeCollector:
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
        result = AbstractAITool.get(AIToolID.CLAUDE).run_plan_session(_request(tmp_path))
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
            AbstractAITool.get(AIToolID.CLAUDE).run_plan_session(_request(tmp_path))


class TestExactSessionCliCollectors:
    def test_opencode_exports_only_emitted_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runs = iter(
            [
                CapturedProcess(
                    0,
                    (FIXTURES / "opencode_events_success.jsonl").read_text(encoding="utf-8"),
                    "",
                ),
                CapturedProcess(
                    0,
                    (FIXTURES / "opencode_export_success.json").read_text(encoding="utf-8"),
                    "",
                ),
            ]
        )
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: Any) -> CapturedProcess:
            commands.append(command)
            return next(runs)

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", fake_run)
        result = AbstractAITool.get(AIToolID.OPENCODE).run_plan_session(_request(tmp_path))
        _assert_result(
            result,
            tool=AIToolID.OPENCODE,
            source=PlanArtifactSource.SESSION_EXPORT,
            binding=PlanSessionBinding.SESSION_ID,
        )
        assert result.session_id == "ses_exact_123"
        assert commands[-1] == ["opencode", "export", "ses_exact_123"]

    def test_opencode_rejects_mismatched_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exported = json.loads((FIXTURES / "opencode_export_success.json").read_text())
        exported["info"]["id"] = "ses-decoy"
        runs = iter(
            [
                CapturedProcess(
                    0,
                    (FIXTURES / "opencode_events_success.jsonl").read_text(),
                    "",
                ),
                CapturedProcess(0, json.dumps(exported), ""),
            ]
        )
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: next(runs),
        )
        with pytest.raises(PlanBindingMismatchError):
            AbstractAITool.get(AIToolID.OPENCODE).run_plan_session(_request(tmp_path))

    def test_antigravity_requires_exact_schema_echo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = (FIXTURES / "antigravity_success.json").read_text(encoding="utf-8")
        monkeypatch.setattr(
            "crossby.ai_tools.plan_process.run_captured",
            lambda *_args, **_kwargs: CapturedProcess(0, payload, ""),
        )
        result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(_request(tmp_path))
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
            AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(_request(tmp_path))

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
        questions: list[str] = []

        def fake_run(command: list[str], **_kwargs: Any) -> CapturedProcess:
            commands.append(command)
            return next(runs)

        def answer(interaction: Any) -> PlanInteractionResponse:
            questions.append(interaction.question_id)
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.ANSWERED,
                option_id="yes",
            )

        monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", fake_run)
        result = AbstractAITool.get(AIToolID.ANTIGRAVITY_CLI).run_plan_session(
            _request(tmp_path), answer
        )
        assert result.session_id == "conv-exact-123"
        assert questions == ["scope", "tests"]
        for command in commands[1:]:
            assert command[command.index("--conversation") + 1] == "conv-exact-123"

    def test_copilot_uuid_share_is_local_and_cleaned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exact_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        events = (FIXTURES / "copilot_events_success.jsonl").read_text(encoding="utf-8")
        share = FIXTURES / "copilot_share_success.md"
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: Any) -> CapturedProcess:
            commands.append(command)
            export_arg = next(value for value in command if value.startswith("--share="))
            shutil.copyfile(share, Path(export_arg.removeprefix("--share=")))
            return CapturedProcess(0, events, "")

        monkeypatch.setattr("crossby.ai_tools.copilot.uuid.uuid4", lambda: exact_uuid)
        monkeypatch.setattr("crossby.ai_tools.plan_process.run_captured", fake_run)
        result = AbstractAITool.get(AIToolID.COPILOT).run_plan_session(
            _request(tmp_path, approval_policy=PlanApprovalPolicy.NEVER)
        )
        _assert_result(
            result,
            tool=AIToolID.COPILOT,
            source=PlanArtifactSource.SESSION_EXPORT,
            binding=PlanSessionBinding.SESSION_ID,
        )
        assert result.session_id == str(exact_uuid)
        assert "--no-remote" in commands[0]
        assert "--no-remote-export" in commands[0]
        assert "--deny-tool=*" in commands[0]
        assert result.artifact_path is not None and not result.artifact_path.exists()


class TestProtocolCollectors:
    def test_codex_collects_completed_plan_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_success.jsonl")
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
        assert rpc.closed

    def test_codex_forwards_native_question_ids_and_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "codex_app_server_question.jsonl")
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
        assert [option.option_id for option in seen[0].options] == ["adapter", "service"]
        assert (
            "respond",
            {
                "id": 71,
                "result": {"answers": {"architecture": {"answers": ["Adapter"]}}},
            },
        ) in FakeRpc.instances[0].sent

    def test_codex_rejects_cross_turn_plan_and_closes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = _rpc_fixture("codex_app_server_success.jsonl")
        messages[4]["params"]["turnId"] = "turn-decoy"
        FakeRpc.scripts = [messages]
        FakeRpc.instances = []
        monkeypatch.setattr("crossby.ai_tools.plan_process.JsonRpcProcess", FakeRpc)
        with pytest.raises(PlanBindingMismatchError):
            AbstractAITool.get(AIToolID.CODEX).run_plan_session(_request(tmp_path))
        assert FakeRpc.instances[0].closed

    def test_cursor_requires_nonexecuting_plan_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_rpc(monkeypatch, "cursor_acp_success.jsonl")
        with pytest.raises(PlanInteractionRequiredError) as raised:
            AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_request(tmp_path))
        assert raised.value.interaction.session_id == "cursor-exact-123"
        assert FakeRpc.instances[0].closed

        _install_rpc(monkeypatch, "cursor_acp_success.jsonl")

        def keep_plan(_interaction: Any) -> PlanInteractionResponse:
            return PlanInteractionResponse(
                outcome=PlanInteractionOutcome.DENIED,
                option_id="rejected",
            )

        result = AbstractAITool.get(AIToolID.CURSOR).run_plan_session(
            _request(tmp_path, sandbox=False, approval_policy=PlanApprovalPolicy.NEVER),
            keep_plan,
        )
        _assert_result(
            result,
            tool=AIToolID.CURSOR,
            source=PlanArtifactSource.PROTOCOL_EVENT,
            binding=PlanSessionBinding.SESSION_ID,
        )
        assert FakeRpc.instances[0].command[:4] == ["agent", "--sandbox", "disabled", "acp"]
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

        result = AbstractAITool.get(AIToolID.CURSOR).run_plan_session(_request(tmp_path), answer)
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
