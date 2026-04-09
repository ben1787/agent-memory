from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from agent_memory.config import MemoryConfig, init_project
from agent_memory.engine import open_memory_with_retry
from agent_memory.hooks.common import consolidation_status, hook_log_entries


def _python_module_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def test_claude_stop_capture_persists_latest_turn_and_schedules_consolidation(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    transcript_path = tmp_path / "claude.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": "Where is the billing webhook handler?",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "It lives in services/billing/webhooks.py."}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding='utf-8',
    )
    payload = {
        "hook_event_name": "Stop",
        "cwd": str(tmp_path),
        "session_id": "session-2",
        "transcript_path": str(transcript_path),
        "last_assistant_message": "It lives in services/billing/webhooks.py.",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}

    result = subprocess.run(
        _python_module_cmd("agent_memory.hooks.claude_stop_capture"),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    assert result.stdout == ""
    memory = open_memory_with_retry(tmp_path, exact=True, read_only=True)
    try:
        recall = memory.recall("billing webhook handler", limit=5).to_dict()
    finally:
        memory.close()
    texts = [node["text"] for node in recall["nodes"]]
    assert "User message: Where is the billing webhook handler?" in texts
    assert "Assistant reply: It lives in services/billing/webhooks.py." in texts

    entries = hook_log_entries(tmp_path)
    assert entries[0]["action"] == "start"
    assert entries[1]["action"] == "consolidation_scheduled"
    assert entries[-1]["action"] == "captured"
    status = consolidation_status(tmp_path)
    assert status["is_pending_today"] is True


def test_hook_dispatch_via_internal_subcommand_runs_claude_stop_handler(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "Stop",
        "cwd": str(tmp_path),
        "session_id": "session-4",
        "prompt": "hello",
        "last_assistant_message": "world",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "_hook", "claude-stop-capture"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    assert result.stdout == ""
    status = consolidation_status(tmp_path)
    assert status["is_pending_today"] is True
