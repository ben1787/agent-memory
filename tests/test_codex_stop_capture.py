from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from agent_memory.config import MemoryConfig, init_project
from agent_memory.hooks.common import consolidation_status, hook_log_entries


def _python_module_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def test_codex_stop_capture_only_schedules_consolidation(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "Stop",
        "cwd": str(tmp_path),
        "session_id": "session-1",
        "turn_id": "turn-1",
        "prompt": "Where is the billing webhook handler?",
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
    assert hook_log_entries(tmp_path) == []
    status = consolidation_status(tmp_path)
    assert status["last_consolidation_date"] is None
    assert status["is_due_today"] is True


def test_hook_dispatch_via_internal_subcommand_runs_codex_stop_handler(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "Stop",
        "cwd": str(tmp_path),
        "session_id": "session-3",
        "prompt": "hello",
        "last_assistant_message": "world",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "_hook", "codex-stop-capture"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    assert result.stdout == ""
    status = consolidation_status(tmp_path)
    assert status["last_consolidation_date"] is None
    assert status["is_due_today"] is True
