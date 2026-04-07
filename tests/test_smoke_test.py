from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.smoke_test import SmokeTestError, SmokeTestResult, _parse_transcript, run_codex_smoke_test


def test_parse_transcript_extracts_user_messages_commands_and_final_answers() -> None:
    transcript = "\n".join(
        [
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "first prompt"}],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": 'agent-memory save "one"'}),
                    },
                }
            ),
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "phase": "final_answer",
                        "message": "saved",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "recall_memories",
                        "arguments": json.dumps({"query": "billing webhook handler"}),
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

    summary = _parse_transcript(transcript)

    assert summary.user_messages == ["first prompt"]
    assert summary.exec_commands == ['agent-memory save "one"']
    assert summary.tool_calls == ["exec_command", "recall_memories"]
    assert summary.final_answers == ["saved", "It lives in services/billing/webhooks.py."]


def test_run_codex_smoke_test_requires_destructive_for_explicit_project(tmp_path: Path) -> None:
    try:
        run_codex_smoke_test(project_root=tmp_path)
    except SmokeTestError as exc:
        assert "--destructive" in str(exc)
    else:
        raise AssertionError("Expected SmokeTestError")


def test_smoke_test_command_prints_json(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        "agent_memory.cli.run_codex_smoke_test",
        lambda **kwargs: SmokeTestResult(
            repo_root="/tmp/repo",
            first_session_file="/tmp/first-session.jsonl",
            second_session_file="/tmp/second-session.jsonl",
            used_temp_repo=True,
            uninstall_verified=True,
            baseline_memory_count=0,
            post_save_memory_count=2,
            hook_event_count=8,
            first_pre_submit_verified=True,
            second_pre_submit_verified=True,
            save_path_verified=True,
            read_path_verified=True,
            recall_top_hit="The billing webhook handler lives in services/billing/webhooks.py.",
            first_final_answer="saved",
            second_final_answer="services/billing/webhooks.py",
        ),
    )

    result = runner.invoke(app, ["smoke-test", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["post_save_memory_count"] == 2
    assert payload["read_path_verified"] is True
