from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.config import MemoryConfig, init_project
from agent_memory.engine import open_memory_with_retry
from agent_memory.hooks.common import mark_consolidation_completed


def _python_module_cmd(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def test_claude_user_prompt_submit_returns_additional_context(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    memory = open_memory_with_retry(tmp_path, exact=True)
    memory.save("Billing webhook handler lives in services/billing/webhooks.py.")
    memory.close()
    mark_consolidation_completed(tmp_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "prompt": "billing webhook handler",
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
    assert "Storing:" in context
    assert "agent-memory recall" in context
    assert "agent-memory save" in context
    assert "--title" in context
    assert "--kind" in context
    assert "--subsystem" in context
    assert "--workstream" in context
    assert "--environment" in context
    assert "30-250 words" in context
    assert "*what*" in context
    assert "*why*" in context
    assert "*when*" in context
    assert "agent-memory edit" in context
    assert "agent-memory delete" in context
    assert "Possibly related memories:" in context
    assert "Billing webhook handler lives in services/billing/webhooks.py." in context
    assert "[A] mem_" in context
    assert "agent-memory feedback evt_" in context
    assert "--stdin" in context
    assert "If you need more, call `agent-memory recall \"<more specific query>\"`." in context
    # CLI-only — must not steer the agent at MCP tool calls.
    assert "save_memory" not in context
    assert "recall_memories" not in context


def test_claude_user_prompt_submit_injects_consolidation_instruction_when_due(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "prompt": "continue working",
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
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "Agent Memory daily consolidation is due." in context
    assert "Run the daily consolidation skill, ideally in a sub-agent" in context
    assert "Storing:" in context


def test_claude_user_prompt_submit_skips_non_interval_turns(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(tmp_path),
        "turn_id": "turn-2",
        "prompt": "continue working",
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

    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Agent Memory\n" not in context
    assert "Memory: save what's worth remembering later" in context


def test_claude_user_prompt_submit_injects_auto_recall_on_non_interval_turn(tmp_path: Path) -> None:
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
        _python_module_cmd("agent_memory.hooks.claude_user_prompt_submit"),
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
    assert "[A] mem_" in context
    assert "agent-memory feedback evt_" in context
    assert "--stdin" in context
    assert 'If you need more, call `agent-memory recall "<more specific query>"`.' in context


def test_claude_user_prompt_submit_drops_auto_recall_below_threshold(tmp_path: Path) -> None:
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
        _python_module_cmd("agent_memory.hooks.claude_user_prompt_submit"),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Possibly related memories:" not in context
    assert "Memory: save what's worth remembering later" in context


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


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Hook command targets POSIX /bin/sh; Windows hooks are dispatched differently.",
)
def test_install_claude_hooks_command_runs_under_minimal_path_shell(tmp_path: Path, monkeypatch) -> None:
    """Regression test for the silent-failure bug where the hook command relied
    on `agent-memory` being on PATH but Claude Code's /bin/sh -c subprocess uses
    a stripped-down PATH=/usr/bin:/bin:/usr/sbin:/sbin that does NOT include
    ~/.local/bin. Without the PATH=$HOME/.local/bin:$PATH prefix in the hook
    command, the subprocess fails with `agent-memory: command not found` and
    no context gets injected into Claude — but the hook log doesn't record the
    failure either, so the bug is invisible without an end-to-end check.

    This test simulates the exact /bin/sh subprocess environment Claude Code
    uses and asserts that the hook command emitted by install_claude_hooks
    actually executes there.
    """
    from agent_memory.integration import install_claude_hooks

    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    install_claude_hooks(tmp_path)

    settings = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding='utf-8'))
    cmd = settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert "PATH=$HOME/.local/bin:$PATH" in cmd, (
        f"Hook command must prefix PATH so /bin/sh -c can resolve agent-memory. "
        f"Got: {cmd}"
    )

    # Drop a fake `agent-memory` shim into a $HOME/.local/bin we control,
    # then run the real hook command via /bin/sh with a stripped PATH (no
    # default ~/.local/bin entry). If the hook command's PATH prefix works,
    # the shim runs and prints a marker we can grep for. If it doesn't, the
    # shim is never reached and the assertion below fails.
    fake_home = tmp_path / "fake-home"
    (fake_home / ".local" / "bin").mkdir(parents=True)
    shim = fake_home / ".local" / "bin" / "agent-memory"
    shim.write_text(
        "#!/bin/sh\n"
        "echo HOOK_MARKER args=\"$*\" project=\"$AGENT_MEMORY_PROJECT_ROOT\" 1>&2\n"
        "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"UserPromptSubmit\",\"additionalContext\":\"shim ok\"}}'\n"
    , encoding='utf-8')
    shim.chmod(0o755)

    minimal_env = {
        "HOME": str(fake_home),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",  # exactly what /bin/sh -c gives Claude Code hooks
    }
    proc = subprocess.run(
        ["/bin/sh", "-c", cmd],
        env=minimal_env,
        capture_output=True,
        text=True,
    )
    # Shim was reached → hook command's PATH prefix works.
    assert "HOOK_MARKER" in proc.stderr, (
        f"Hook command did not reach `agent-memory` under stripped /bin/sh PATH. "
        f"This is the silent-failure regression. stderr={proc.stderr!r}"
    )
    # Hook output JSON is well-formed (would fail if /bin/sh dropped output).
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["additionalContext"] == "shim ok"
