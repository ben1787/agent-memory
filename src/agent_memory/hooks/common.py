from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_memory.engine import AgentMemory, open_memory_with_retry as open_memory_with_retry_engine


HOOK_LOG_FILENAME = "hook-events.jsonl"
CONSOLIDATION_STATE_FILENAME = "consolidation-state.json"
AUTO_RECALL_LIMIT = 3
AUTO_RECALL_MIN_QUERY_SIMILARITY = 0.4
AUTO_RECALL_MAX_WORDS_PER_MEMORY = 48
AUTO_RECALL_FALLBACK = 'If you need more, call `agent-memory recall "<more specific query>"`.'
STORING_SECTION = """Storing memories:
  Save only durable, repo-specific facts that will materially speed up future work or prevent likely mistakes.
  Save only if at least 2:
    - Likely to matter again
    - Hard to rediscover quickly
    - Changes tools/files/search path
    - Missing it wastes time or causes bad assumptions
    - Stable beyond this session
  Prefer: workflow rules, architecture map, search shortcuts, environment quirks, external system behavior, validation/release constraints, recurring customer/project facts.
  Do not save: temp branch/PR/test state, logs/transcripts, generic advice, grep-easy facts, speculation, soon-changing details.
  Format: Scope + Category; 1-sentence fact; 1-sentence why; optional exception; confidence high/med/low.
  Worthiness test: if it won’t save time, prevent a likely error, or narrow the search path, don’t save.
  After work, save memories if the criteria are met with `agent-memory save "<memory>" "<memory>"`. Use `--stdin` for quotes/newlines.
  If you saved something wrong: use `list --recent 5`, then `edit`/`delete`, or `undo`."""

_DEFAULT_CONSOLIDATION_STATE: dict[str, Any] = {
    "last_consolidation_date": None,
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
    last_consolidation_date = payload.get("last_consolidation_date")
    if not isinstance(last_consolidation_date, str) or not last_consolidation_date.strip():
        legacy_date = payload.get("completed_for_date")
        if isinstance(legacy_date, str) and legacy_date.strip():
            last_consolidation_date = legacy_date.strip()
        else:
            last_consolidation_date = None
    return {
        "last_consolidation_date": last_consolidation_date,
    }


def write_consolidation_state(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    path = _consolidation_state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = state.get("last_consolidation_date")
    payload = {
        "last_consolidation_date": value if isinstance(value, str) and value.strip() else None,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    return payload


def consolidation_status(project_root: Path) -> dict[str, Any]:
    state = read_consolidation_state(project_root)
    today = _local_today()
    is_completed_today = state.get("last_consolidation_date") == today
    return {
        **state,
        "today": today,
        "is_completed_today": is_completed_today,
        "is_due_today": not is_completed_today,
    }


def mark_consolidation_completed(project_root: Path) -> dict[str, Any]:
    today = _local_today()
    write_consolidation_state(project_root, {"last_consolidation_date": today})
    return {
        "status": "completed",
        **consolidation_status(project_root),
    }


def pending_consolidation_instruction(project_root: Path) -> str | None:
    status = consolidation_status(project_root)
    if not status["is_due_today"]:
        return None
    return (
        "Agent Memory daily consolidation is due.\n"
        "- Run the daily consolidation skill, ideally in a sub-agent so it does not block the main thread. "
        "If you cannot delegate it, run it in the current thread."
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


def load_memory_config(project_root: Path) -> "MemoryConfig | None":
    config_path = project_root / ".agent-memory" / "config.json"
    if not config_path.exists():
        return None
    try:
        from agent_memory.config import MemoryConfig

        raw = json.loads(config_path.read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            return None
        return MemoryConfig.from_dict(raw)
    except Exception:
        return None


def truncate_words(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= limit:
        return cleaned
    clipped = " ".join(words[: max(1, limit - 2)])
    return f"{clipped} [truncated]"


def open_memory_with_retry(
    project_root: Path,
    attempts: int = 5,
    delay_s: float = 0.15,
    *,
    read_only: bool = False,
) -> AgentMemory:
    return open_memory_with_retry_engine(
        project_root,
        exact=True,
        read_only=read_only,
        attempts=attempts,
        delay_s=delay_s,
    )


def auto_recall_matches(project_root: Path, query: str) -> tuple[list[str] | None, dict[str, Any]]:
    cleaned_query = " ".join(query.split())
    metadata: dict[str, Any] = {
        "query": cleaned_query,
        "limit": AUTO_RECALL_LIMIT,
        "threshold": AUTO_RECALL_MIN_QUERY_SIMILARITY,
    }
    if not cleaned_query:
        metadata["status"] = "empty_query"
        return None, metadata

    try:
        memory = open_memory_with_retry(project_root, read_only=True)
        try:
            recall = memory.recall(cleaned_query, limit=AUTO_RECALL_LIMIT)
        finally:
            memory.close()
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = str(exc)
        return None, metadata

    if not recall.hits:
        metadata["status"] = "no_hits"
        metadata["top_query_similarity"] = 0.0
        return None, metadata

    top_query_similarity = round(recall.hits[0].query_similarity, 4)
    metadata["top_query_similarity"] = top_query_similarity
    if top_query_similarity < AUTO_RECALL_MIN_QUERY_SIMILARITY:
        metadata["status"] = "below_threshold"
        return None, metadata

    included_hits = [
        hit
        for hit in recall.hits
        if hit.query_similarity >= AUTO_RECALL_MIN_QUERY_SIMILARITY
    ]
    if not included_hits:
        metadata["status"] = "below_threshold"
        return None, metadata

    metadata["status"] = "matched"
    metadata["matched_count"] = len(included_hits)
    return [truncate_words(hit.text, AUTO_RECALL_MAX_WORDS_PER_MEMORY) for hit in included_hits], metadata


def render_auto_recall_block(recalled_memories: list[str]) -> str:
    lines = ["Here is some context from Agent Memory that might be related:"]
    for text in recalled_memories:
        lines.append(f"- {text}")
    lines.append(AUTO_RECALL_FALLBACK)
    return "\n".join(lines)


def render_guidance_context(
    recalled_memories: list[str] | None,
    *,
    consolidation_instruction: str | None,
) -> str:
    lines = [
        "Agent Memory",
        "",
        "Reading memories:",
    ]
    if recalled_memories:
        lines.append("  - Here is some context from Agent Memory that might be related:")
        for text in recalled_memories:
            lines.append(f"    - {text}")
        lines.append(f"  - {AUTO_RECALL_FALLBACK}")
    else:
        lines.append("  - Related memory may be injected automatically from the current user prompt when there is a strong match.")
        lines.append("  - If the injected context is missing or incomplete, call `agent-memory recall <task-shaped query>`.")
    lines.extend(["", *STORING_SECTION.splitlines()])
    if consolidation_instruction:
        lines.extend(["", consolidation_instruction])
    return "\n".join(lines)


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
