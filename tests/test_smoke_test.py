from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# smoke_test uses pty/fcntl/termios which don't exist on Windows. Skip the
# entire module there so the CI matrix still builds Windows artifacts.
if sys.platform.startswith("win"):
    pytest.skip("smoke_test is POSIX-only", allow_module_level=True)

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

    # cli.py lazy-imports run_codex_smoke_test from agent_memory.smoke_test
    # so Windows can still load the CLI. Patch at the source module.
    monkeypatch.setattr(
        "agent_memory.smoke_test.run_codex_smoke_test",
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


def test_smoke_test_defaults_reinstall_from_source_checkout(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    (tmp_path / "src" / "agent_memory").mkdir(parents=True)
    (tmp_path / "src" / "agent_memory" / "cli.py").write_text("# marker\n", encoding='utf-8')
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "agent-memory"\nversion = "0.1.0"\n',
        encoding='utf-8',
    )
    monkeypatch.chdir(tmp_path)

    seen: dict[str, object] = {}

    def _fake_run_codex_smoke_test(**kwargs):
        seen.update(kwargs)
        return SmokeTestResult(
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
        )

    monkeypatch.setattr("agent_memory.smoke_test.run_codex_smoke_test", _fake_run_codex_smoke_test)

    result = runner.invoke(app, ["smoke-test", "--json"])

    assert result.exit_code == 0
    assert seen["reinstall_from"] == tmp_path.resolve()
