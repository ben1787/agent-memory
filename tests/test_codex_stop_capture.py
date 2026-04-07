from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from agent_memory.config import MemoryConfig, init_project
from agent_memory.engine import open_memory_with_retry
from agent_memory.hooks.common import hook_log_entries


def _python_module_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def test_codex_stop_capture_persists_latest_turn_from_codex_transcript(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    transcript_path = tmp_path / "codex.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Where is the billing webhook handler?"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "agent_message",
                            "phase": "final_answer",
                            "message": "It lives in services/billing/webhooks.py.",
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    payload = {
        "hook_event_name": "Stop",
        "cwd": str(tmp_path),
        "session_id": "session-1",
        "turn_id": "turn-1",
        "transcript_path": str(transcript_path),
        "last_assistant_message": "It lives in services/billing/webhooks.py.",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}

    result = subprocess.run(
        _python_module_cmd("agent_memory.hooks.codex_stop_capture"),
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
    texts = [hit["text"] for hit in recall["hits"]]
    assert "User message: Where is the billing webhook handler?" in texts
    assert "Assistant reply: It lives in services/billing/webhooks.py." in texts

    entries = hook_log_entries(tmp_path)
    assert entries[0]["action"] == "start"
    assert entries[0]["payload"]["session_id"] == "session-1"
    assert entries[-1]["action"] == "captured"
    assert entries[-1]["payload"]["session_id"] == "session-1"
