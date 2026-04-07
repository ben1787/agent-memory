from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_memory.engine import AgentMemory, open_memory_with_retry as open_memory_with_retry_engine, word_count


HOOK_LOG_FILENAME = "hook-events.jsonl"


def read_hook_input() -> dict[str, Any]:
    raw = sys.stdin.read()
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Hook input must be a JSON object")
    return payload


def _hook_log_path(project_root: Path) -> Path:
    return project_root / ".agent-memory" / HOOK_LOG_FILENAME


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
