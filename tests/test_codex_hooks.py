from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from agent_memory.config import MemoryConfig, default_instructions, init_project
from agent_memory.engine import open_memory_with_retry
from agent_memory.hooks.common import hook_log_entries, mark_consolidation_completed
from agent_memory.integration import INSTRUCTIONS_BEGIN_MARKER, INSTRUCTIONS_END_MARKER


def _python_module_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def test_codex_user_prompt_submit_returns_hook_specific_context(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    memory.save("Billing webhook handler lives in services/billing/webhooks.py.")
    memory.close()
    mark_consolidation_completed(tmp_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-1",
        "prompt": "billing webhook handler",
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
    assert "Storing:" in context
    assert "agent-memory recall" in context
    assert "agent-memory save" in context
    assert "--title" in context
    assert "--kind" in context
    assert "--subsystem" in context
    assert "--workstream" in context
    assert "--environment" in context
    assert "50-250 words" in context
    assert "*what*" in context
    assert "*why*" in context
    assert "*when*" in context
    assert "no conversational context" in context
    assert "agent-memory edit" in context
    assert "agent-memory delete" in context
    assert "Possibly related memories:" in context
    assert "Billing webhook handler lives in services/billing/webhooks.py." in context
    assert "[A] mem_" in context
    assert "agent-memory feedback evt_" in context
    assert "--stdin" in context
    assert "If you need more, call `agent-memory recall \"<more specific query>\"`." in context
    assert "save_memory" not in context
    assert "recall_memories" not in context

    entries = hook_log_entries(tmp_path)
    assert entries[0]["action"] == "start"
    assert entries[0]["payload"]["turn_id"] == "turn-1"
    assert entries[1]["action"] == "inject_context"
    assert entries[1]["payload"]["turn_id"] == "turn-1"
    assert str(entries[1]["payload"]["feedback_event_id"]).startswith("evt_")


def test_codex_user_prompt_submit_injects_consolidation_instruction_when_due(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-1",
        "prompt": "continue working",
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
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "Agent Memory daily consolidation is due." in context
    assert "Run the daily consolidation skill, ideally in a sub-agent" in context
    assert "Storing:" in context


def test_codex_user_prompt_submit_skips_non_interval_turns(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-2",
        "prompt": "continue working",
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

    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Agent Memory\n" not in context
    assert "Memory: save what's worth remembering later" in context
    assert "self-contained 50-250 words" in context
    assert "no conversational context" in context
    assert "--title" in context
    assert "--kind" in context
    assert "--subsystem" in context
    assert "--workstream" in context
    assert "--environment" in context

    entries = hook_log_entries(tmp_path)
    assert entries[0]["action"] == "start"
    assert entries[1]["action"] == "skip_context"
    assert entries[1]["payload"]["reason"] == "non_interval_turn"
    assert entries[2]["action"] == "inject_context"
    assert entries[2]["payload"]["inject_guidance"] is False
    assert entries[2]["payload"]["inject_auto_recall"] is False
    assert entries[2]["payload"]["inject_save_nudge"] is True


def test_codex_user_prompt_submit_injects_auto_recall_on_non_interval_turn(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    memory.save("Billing webhook handler lives in services/billing/webhooks.py.")
    memory.close()
    mark_consolidation_completed(tmp_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-2",
        "prompt": "billing webhook handler",
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

    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Agent Memory\n" not in context
    assert "Possibly related memories:" in context
    assert "Billing webhook handler lives in services/billing/webhooks.py." in context
    assert "Memory: save what's worth remembering later" in context
    assert "self-contained 50-250 words" in context
    assert "no conversational context" in context
    assert "--title" in context
    assert "--kind" in context
    assert "--subsystem" in context
    assert "--workstream" in context
    assert "--environment" in context
    assert "[A] mem_" in context
    assert "agent-memory feedback evt_" in context
    assert "--stdin" in context
    assert 'If you need more, call `agent-memory recall "<more specific query>"`.' in context

    entries = hook_log_entries(tmp_path)
    assert entries[0]["action"] == "start"
    assert entries[1]["action"] == "skip_context"
    assert entries[2]["action"] == "inject_context"
    assert entries[2]["payload"]["inject_guidance"] is False
    assert entries[2]["payload"]["inject_auto_recall"] is True
    assert entries[2]["payload"]["inject_save_nudge"] is True
    assert str(entries[2]["payload"]["feedback_event_id"]).startswith("evt_")


def test_codex_user_prompt_submit_drops_auto_recall_below_threshold(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    memory.save("Release flow requires tagging from main after tests pass.")
    memory.close()
    mark_consolidation_completed(tmp_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-2",
        "prompt": "continue working",
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

    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Possibly related memories:" not in context
    assert "Memory: save what's worth remembering later" in context


def test_codex_user_prompt_submit_refreshes_stale_prompt_artifacts(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    (tmp_path / ".agent-memory" / "instructions.md").write_text("stale instructions\n", encoding='utf-8')
    (tmp_path / "CLAUDE.md").write_text(
        "# Project\n\n"
        f"{INSTRUCTIONS_BEGIN_MARKER}\nold stale block\n{INSTRUCTIONS_END_MARKER}\n",
        encoding='utf-8',
    )
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-refresh",
        "prompt": "where is the billing webhook handler",
    }
    env = os.environ | {"AGENT_MEMORY_PROJECT_ROOT": str(tmp_path)}

    subprocess.run(
        _python_module_cmd("agent_memory.hooks.codex_user_prompt_submit"),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    instructions_text = (tmp_path / ".agent-memory" / "instructions.md").read_text(encoding='utf-8')
    assert instructions_text == default_instructions()
    assert "50-250 words" in instructions_text
    assert "no conversational context" in instructions_text
    claude_text = (tmp_path / "CLAUDE.md").read_text(encoding='utf-8')
    assert "old stale block" not in claude_text
    assert "Recall when useful" in claude_text
