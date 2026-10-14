"""Deterministic native peer for the real pinned Copilot Python SDK."""

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

scenario = os.environ.get("CROSSBY_COPILOT_SCENARIO", "success")
log = Path(os.environ["CROSSBY_COPILOT_LOG"])
sid = None
message_id = "submitted-message"


def write(frame):
    data = json.dumps({"jsonrpc": "2.0", **frame}).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.buffer.flush()


def event(kind, data, agent=None):
    write(
        {
            "method": "session.event",
            "params": {
                "sessionId": sid,
                "event": {
                    "id": str(uuid.uuid4()),
                    "timestamp": "2026-10-13T00:00:00Z",
                    "type": kind,
                    "agentId": agent,
                    "data": data,
                },
            },
        }
    )


def finish():
    if scenario == "usage":
        event(
            "assistant.usage",
            {"model": "synthetic", "inputTokens": 3, "outputTokens": 4, "apiCallId": "native-call"},
        )
    if scenario == "native-error":
        event("session.error", {"errorType": "synthetic", "message": "private diagnostic"})
    event(
        "assistant.message",
        {
            "messageId": "assistant-result",
            "content": "result ✓",
            "originatingMessageId": message_id,
            "turnId": "0",
        },
    )
    event(
        "assistant.message",
        {
            "messageId": "background",
            "content": "wrong background",
            "originatingMessageId": "other-message",
        },
        "child-agent",
    )
    event("session.idle", {"mode": "interactive", "aborted": scenario == "aborted"})
    if scenario == "malformed-after-idle":
        event("assistant.message", {"messageId": "broken"})


while True:
    header = sys.stdin.buffer.readline()
    if not header:
        break
    size = int(header.split(b":")[1])
    assert sys.stdin.buffer.readline() == b"\r\n"
    frame = json.loads(sys.stdin.buffer.read(size))
    with log.open("a") as stream:
        stream.write(json.dumps(frame) + "\n")
    method = frame.get("method")
    params = frame.get("params", {})
    if method is None:
        if frame.get("id") == 101:
            finish()
        continue
    result = {"success": True}
    if method == "connect":
        result = {"protocolVersion": 3, "ok": True, "version": "1.0.97"}
    elif method == "status.get":
        result = {"version": "1.0.97", "protocolVersion": 3}
    elif method in {"session.create", "session.resume"}:
        sid = params["sessionId"]
        result = {"sessionId": sid, "capabilities": {"ui": {"elicitation": True}}}
        if scenario == "resume-mismatch":
            result["sessionId"] = "different-native-session"
    elif method == "session.permissions.setMode":
        result = {"success": True, "mode": "manual"}
    elif method == "session.send":
        if scenario == "startup-error":
            write(
                {"id": frame["id"], "error": {"code": -32000, "message": "private native refusal"}}
            )
            continue
        result = {"messageId": message_id}
        if scenario == "send-delay":
            event(
                "user.message",
                {"content": params["prompt"], "messageId": message_id, "turnId": "0"},
            )
            for _ in range(12):
                time.sleep(0.05)
                event("assistant.turn_start", {"turnId": "0"})
    write({"id": frame["id"], "result": result})
    if method == "session.send":
        if scenario == "old-idle":
            event("session.idle", {"mode": "interactive", "aborted": True})
        event("user.message", {"content": params["prompt"], "messageId": message_id, "turnId": "0"})
        if scenario in {
            "question",
            "withdraw-question",
            "stall-question",
            "freeform",
            "backpressure",
        }:
            write(
                {
                    "id": 101,
                    "method": "userInput.request",
                    "params": {
                        "sessionId": sid,
                        "question": "Choose",
                        "choices": ["A", "B"],
                        "allowFreeform": scenario in {"freeform", "backpressure"},
                    },
                }
            )
            if scenario == "backpressure":
                for _ in range(12):
                    time.sleep(0.05)
                    event("assistant.message_delta", {"messageId": "inflight", "deltaContent": "p"})
            if scenario == "withdraw-question":
                time.sleep(0.05)
                write({"method": "$/cancelRequest", "params": {"id": 101}})
                finish()
        elif scenario in {"permission", "withdraw-permission", "denial"}:
            event(
                "permission.requested",
                {
                    "requestId": "permission-id",
                    "permissionRequest": {
                        "kind": "shell",
                        "canOfferSessionApproval": False,
                        "commands": [],
                        "fullCommandText": "git status",
                        "hasWriteFileRedirection": False,
                        "intention": "display text is not authority",
                        "possiblePaths": [],
                        "possibleUrls": [],
                        "resolvedWorkingDirectory": "/tmp/native-cwd",
                        "toolCallId": "native-tool",
                        "managedApprovalRequired": True,
                    },
                },
            )
            if scenario == "withdraw-permission":
                time.sleep(0.05)
                event(
                    "permission.completed",
                    {
                        "requestId": "permission-id",
                        "result": {"kind": "denied-interactively-by-user"},
                    },
                )
                finish()
        elif scenario in {"form", "url-form"}:
            event(
                "elicitation.requested",
                {
                    "requestId": "form-id",
                    "message": "A form",
                    "mode": "url" if scenario == "url-form" else "form",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
                    },
                },
            )
        elif scenario in {"idle-progress", "idle-progress-long"}:

            def progress():
                for _ in range(40 if scenario == "idle-progress-long" else 8):
                    time.sleep(0.03)
                    event("assistant.turn_start", {"turnId": "0"})
                finish()

            threading.Thread(target=progress, daemon=True).start()
        elif scenario == "eof":
            break
        elif scenario == "silent":
            pass
        elif scenario == "malformed":
            event("assistant.message", {"messageId": "broken"})
        else:
            finish()
    elif method == "session.permissions.handlePendingPermissionRequest":
        event(
            "permission.completed",
            {
                "requestId": "permission-id",
                "result": {
                    "kind": "denied-interactively-by-user" if scenario == "denial" else "approved"
                },
            },
        )
        finish()
    elif method == "session.ui.handlePendingElicitation":
        event("elicitation.completed", {"requestId": "form-id", "action": "accept"})
        finish()
    elif method == "runtime.shutdown":
        break
