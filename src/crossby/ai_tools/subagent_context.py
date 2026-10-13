"""Private normalization for a prompt-free subagent-readiness boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from crossby.models.ai import (
    AIToolID,
    HeadlessSessionRequest,
    SubagentSpawnEvidence,
    SubagentSpawnReason,
    SubagentSpawnState,
    SubagentSpawnStatus,
)


@dataclass(frozen=True)
class SubagentLaunchContext:
    working_dir: Path
    headless: bool
    launch_args: tuple[str, ...]
    environment: dict[str, str]
    request: HeadlessSessionRequest | None
    fingerprint: str
    conflicting_inputs: bool

    @classmethod
    def normalize(
        cls,
        *,
        working_dir: Path | None,
        headless: bool,
        launch_args: Sequence[str],
        environment: Mapping[str, str] | None,
        request: HeadlessSessionRequest | None,
    ) -> SubagentLaunchContext:
        scope = (
            request.working_dir if request is not None else working_dir or Path.cwd()
        ).resolve()
        args = tuple(launch_args)
        env = dict(os.environ if environment is None else environment)
        request_data = request.model_dump(mode="json", exclude={"prompt"}) if request else None
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "cwd": str(scope),
                    "headless": headless,
                    "args": args,
                    "environment": env,
                    "request": request_data,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        conflicting = request is not None and (
            not headless
            or bool(args)
            or (working_dir is not None and working_dir.resolve() != scope)
        )
        return cls(scope, headless, args, env, request, fingerprint, conflicting)

    def binary_path(self, binary: str) -> str | None:
        # Relative PATH entries belong to the child cwd, not the parent's cwd.
        path = self.environment.get("PATH", os.defpath)
        scoped_path = os.pathsep.join(
            str((self.working_dir / entry).absolute()) if not Path(entry).is_absolute() else entry
            for entry in path.split(os.pathsep)
        )
        selected = shutil.which(binary, path=scoped_path)
        return str((self.working_dir / selected).absolute()) if selected else None

    def status(
        self,
        tool_id: AIToolID,
        state: SubagentSpawnState,
        reason_code: SubagentSpawnReason,
        reason: str,
        evidence: SubagentSpawnEvidence = SubagentSpawnEvidence.ADAPTER_DECLARATION,
        *,
        tool_version: str | None = None,
    ) -> SubagentSpawnStatus:
        return SubagentSpawnStatus(
            tool_id=tool_id,
            state=state,
            reason_code=reason_code,
            reason=reason,
            evidence=evidence,
            working_dir=self.working_dir,
            headless=self.headless,
            context_fingerprint=self.fingerprint,
            tool_version=tool_version,
        )
