from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_memory.engine import AgentMemory, open_memory_with_retry as open_memory_with_retry_engine, word_count


HOOK_LOG_FILENAME = "hook-events.jsonl"
CONSOLIDATION_STATE_FILENAME = "consolidation-state.json"

_DEFAULT_CONSOLIDATION_STATE: dict[str, Any] = {
    "pending_for_date": None,
    "in_progress_for_date": None,
    "completed_for_date": None,
    "last_scheduled_at": None,
    "last_started_at": None,
    "last_completed_at": None,
}


def read_hook_input() -> dict[str, Any]:
    raw = sys.stdin.read()
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Hook input must be a JSON object")
    return payload


def _hook_log_path(project_root: Path) -> Path:
    return project_root / ".agent-memory" / HOOK_LOG_FILENAME


def _consolidation_state_path(project_root: Path) -> Path:
    return project_root / ".agent-memory" / CONSOLIDATION_STATE_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_today() -> str:
    return datetime.now().astimezone().date().isoformat()


def read_consolidation_state(project_root: Path) -> dict[str, Any]:
    path = _consolidation_state_path(project_root)
    if not path.exists():
        return dict(_DEFAULT_CONSOLIDATION_STATE)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return dict(_DEFAULT_CONSOLIDATION_STATE)
    if not isinstance(payload, dict):
        return dict(_DEFAULT_CONSOLIDATION_STATE)
    state = dict(_DEFAULT_CONSOLIDATION_STATE)
    for key in state:
        state[key] = payload.get(key)
    return state


def write_consolidation_state(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    path = _consolidation_state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(_DEFAULT_CONSOLIDATION_STATE)
    for key in payload:
        payload[key] = state.get(key)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    return payload


def consolidation_status(project_root: Path) -> dict[str, Any]:
    state = read_consolidation_state(project_root)
    today = _local_today()
    is_pending_today = state.get("pending_for_date") == today
    is_in_progress_today = state.get("in_progress_for_date") == today
    is_completed_today = state.get("completed_for_date") == today
    return {
        **state,
        "today": today,
        "is_pending_today": is_pending_today,
        "is_in_progress_today": is_in_progress_today,
        "is_completed_today": is_completed_today,
    }


def schedule_daily_consolidation(project_root: Path) -> dict[str, Any]:
    state = read_consolidation_state(project_root)
    today = _local_today()
    scheduled = False
    if (
        state.get("pending_for_date") != today
        and state.get("in_progress_for_date") != today
        and state.get("completed_for_date") != today
    ):
        state["pending_for_date"] = today
        state["last_scheduled_at"] = _utc_now()
        scheduled = True
        write_consolidation_state(project_root, state)
    return {
        "scheduled": scheduled,
        **consolidation_status(project_root),
    }


def mark_consolidation_started(project_root: Path) -> dict[str, Any]:
    state = read_consolidation_state(project_root)
    today = _local_today()
    if state.get("completed_for_date") == today:
        return {
            "status": "already_completed",
            **consolidation_status(project_root),
        }
    state["pending_for_date"] = today
    state["in_progress_for_date"] = today
    state["last_started_at"] = _utc_now()
    write_consolidation_state(project_root, state)
    return {
        "status": "started",
        **consolidation_status(project_root),
    }


def mark_consolidation_completed(project_root: Path) -> dict[str, Any]:
    state = read_consolidation_state(project_root)
    today = _local_today()
    state["pending_for_date"] = None
    state["in_progress_for_date"] = None
    state["completed_for_date"] = today
    state["last_completed_at"] = _utc_now()
    write_consolidation_state(project_root, state)
    return {
        "status": "completed",
        **consolidation_status(project_root),
    }


def pending_consolidation_instruction(project_root: Path) -> str | None:
    status = consolidation_status(project_root)
    if status["is_completed_today"]:
        return None
    if not status["is_pending_today"] and not status["is_in_progress_today"]:
        return None
    return (
        "Daily memory consolidation is due for this project.\n"
        "- If your client exposes the Agent Memory consolidation skill, use it now. "
        "In Claude plugin setups, that skill is `/agent-memory:consolidate`.\n"
        "- If your client supports delegation, you may run the consolidation skill in a subagent.\n"
        "- Start the workflow with `agent-memory consolidation-start`.\n"
        "- Inspect `agent-memory consolidate --json`, which reports overlapping similarity clusters at cosine "
        "similarity >= 0.92.\n"
        "- For each cluster, decide whether to keep it as-is or replace it with fewer, more orthogonal memories.\n"
        "- Apply the chosen `agent-memory delete`, `agent-memory edit`, and `agent-memory save` actions directly.\n"
        "- Do not do contradiction resolution or timestamp-based truth arbitration in this pass.\n"
        "- After the edits, run `agent-memory consolidation-complete`."
    )


def hook_log_entries(project_root: Path) -> list[dict[str, Any]]:
    path = _hook_log_path(project_root)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def log_hook_event(
    project_root: Path,
    *,
    hook_name: str,
    action: str,
    payload: dict[str, Any] | None = None,
) -> None:
    path = _hook_log_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hook": hook_name,
        "action": action,
        "payload": payload or {},
    }
    path.open("a").write(json.dumps(event) + "\n")


def project_root_from_env_or_cwd(cwd: str | None = None) -> Path:
    root = os.environ.get("AGENT_MEMORY_PROJECT_ROOT")
    if root:
        return Path(root).resolve()
    if cwd:
        return Path(cwd).resolve()
    return Path.cwd()


def sync_prompt_artifacts(project_root: Path) -> None:
    config_path = project_root / ".agent-memory" / "config.json"
    if not config_path.exists():
        return

    try:
        from agent_memory.config import default_instructions

        instructions_path = project_root / ".agent-memory" / "instructions.md"
        desired = default_instructions()
        if not instructions_path.exists() or instructions_path.read_text(encoding='utf-8') != desired:
            instructions_path.parent.mkdir(parents=True, exist_ok=True)
            instructions_path.write_text(desired, encoding='utf-8')
    except Exception:
        pass

    try:
        from agent_memory.integration import install_memory_instructions

        install_memory_instructions(project_root)
    except Exception:
        pass


def truncate_words(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= limit:
        return cleaned
    clipped = " ".join(words[: max(1, limit - 2)])
    return f"{clipped} [truncated]"


def open_memory_with_retry(project_root: Path, attempts: int = 5, delay_s: float = 0.15) -> AgentMemory:
    return open_memory_with_retry_engine(
        project_root,
        exact=True,
        attempts=attempts,
        delay_s=delay_s,
    )


def latest_transcript_turn(transcript_path: Path) -> tuple[str | None, str | None, str | None]:
    latest_user: tuple[str | None, str | None] = (None, None)
    latest_assistant: tuple[str | None, str | None] = (None, None)

    lines = transcript_path.read_text(encoding='utf-8').splitlines()
    for raw in reversed(lines):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue

        record_type = record.get("type")
        message = record.get("message")
        if record_type == "assistant" and latest_assistant[0] is None:
            text = _extract_assistant_text(message)
            if text:
                latest_assistant = (text, str(record.get("uuid") or record.get("timestamp") or ""))
        elif record_type == "user" and latest_user[0] is None:
            text = _extract_user_text(message)
            if text:
                latest_user = (text, str(record.get("uuid") or record.get("timestamp") or ""))
        elif record_type == "event_msg" and latest_assistant[0] is None:
            payload = record.get("payload")
            if isinstance(payload, dict):
                if payload.get("type") == "agent_message" and payload.get("phase") == "final_answer":
                    text = payload.get("message")
                    if isinstance(text, str) and text.strip():
                        latest_assistant = (
                            text.strip(),
                            str(record.get("timestamp") or record.get("uuid") or ""),
                        )
        elif record_type == "response_item" and latest_user[0] is None:
            payload = record.get("payload")
            if isinstance(payload, dict):
                if payload.get("type") == "message" and payload.get("role") == "user":
                    text = _extract_codex_user_text(payload.get("content"))
                    if text:
                        latest_user = (
                            text,
                            str(record.get("timestamp") or record.get("uuid") or ""),
                        )

        if latest_user[0] is not None and latest_assistant[0] is not None:
            break

    return latest_user[0], latest_assistant[0], latest_assistant[1]


def summarize_hook_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("hook_event_name", "cwd", "session_id", "turn_id", "transcript_path"):
        value = payload.get(key)
        if value:
            summary[key] = value
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        summary["prompt"] = truncate_words(prompt, 30)
    last_assistant = payload.get("last_assistant_message")
    if isinstance(last_assistant, str) and last_assistant.strip():
        summary["last_assistant_message"] = truncate_words(last_assistant, 40)
    return summary


def _extract_user_text(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    if message.get("role") != "user":
        return None
    content = message.get("content")
    if isinstance(content, str):
        cleaned = content.strip()
        return cleaned or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_result":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    return None


def _extract_assistant_text(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    if message.get("role") != "assistant":
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    if not parts:
        return None
    return "\n\n".join(parts)


def _extract_codex_user_text(content: Any) -> str | None:
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "input_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    if not parts:
        return None
    return "\n\n".join(parts)
