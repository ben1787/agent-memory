from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from agent_memory.config import MemoryConfig, init_project
from agent_memory.hooks.common import hook_log_entries


def _python_module_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def test_codex_user_prompt_submit_returns_hook_specific_context(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-1",
        "prompt": "where is the billing webhook handler",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}
    result = subprocess.run(
        _python_module_cmd("agent_memory.hooks.codex_user_prompt_submit"),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    output = json.loads(result.stdout)
    hook_output = output["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "UserPromptSubmit"
    context = hook_output["additionalContext"]
    assert "Agent Memory" in context
    assert "agent-memory recall" in context
    assert "agent-memory save" in context
    assert "save_memory" not in context
    assert "recall_memories" not in context

    entries = hook_log_entries(tmp_path)
    assert entries[0]["action"] == "start"
    assert entries[0]["payload"]["turn_id"] == "turn-1"
    assert entries[1]["action"] == "inject_context"
    assert entries[1]["payload"]["turn_id"] == "turn-1"
