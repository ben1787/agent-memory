from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.config import MemoryConfig, init_project


def _python_module_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def test_claude_user_prompt_submit_returns_additional_context(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "prompt": "where is the billing webhook handler",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}
    result = subprocess.run(
        _python_module_cmd("agent_memory.hooks.claude_user_prompt_submit"),
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
    # CLI-only — must not steer the agent at MCP tool calls.
    assert "save_memory" not in context
    assert "recall_memories" not in context


def test_hook_dispatch_via_internal_subcommand_runs_claude_handler(tmp_path: Path) -> None:
    """The portable hook command (`agent-memory _hook claude-user-prompt-submit`)
    should produce the same JSON output as invoking the Python module directly.
    Tests the dispatch path that real-world hook configs will use after the refactor."""
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "prompt": "anything",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "_hook", "claude-user-prompt-submit"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "Agent Memory" in output["hookSpecificOutput"]["additionalContext"]


def test_hook_subcommand_is_hidden_from_main_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Internal subcommand must not surface in user-facing help.
    assert "_hook" not in result.stdout
