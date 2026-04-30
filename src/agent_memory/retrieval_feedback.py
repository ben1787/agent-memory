from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid


RETRIEVAL_EVENT_LOG_FILENAME = "retrieval-events.jsonl"
RETRIEVAL_FEEDBACK_LOG_FILENAME = "retrieval-feedback.jsonl"

MEMORY_FEEDBACK_LABELS = frozenset(
    {
        "helpful",
        "partial",
        "irrelevant",
        "stale",
        "wrong",
    }
)
OVERALL_FEEDBACK_LABELS = frozenset({"helpful", "mixed", "irrelevant"})
POSITIVE_MEMORY_FEEDBACK_LABELS = frozenset({"helpful", "partial"})
NEGATIVE_MEMORY_FEEDBACK_LABELS = frozenset({"irrelevant", "stale", "wrong"})

_MEMORY_FEEDBACK_WEIGHTS = {
    "helpful": 1.0,
    "partial": 0.35,
    "irrelevant": -0.45,
    "stale": -0.75,
    "wrong": -1.0,
}
_MAX_ABSOLUTE_BIAS = 0.08


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_path(project_root: Path, filename: str) -> Path:
    return project_root / ".agent-memory" / filename


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    except OSError:
        return []
    return rows


def _truncate_words(text: str, limit: int = 24) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) <= limit:
        return cleaned
    clipped = " ".join(words[: max(1, limit - 2)])
    return f"{clipped} [truncated]"


def record_retrieval_event(
    project_root: Path,
    *,
    query: str,
    matches: list[dict[str, Any]],
    hook_payload: dict[str, Any] | None = None,
) -> str | None:
    """Persist one retrieval event so later feedback can resolve aliases to memory ids.

    Best-effort: returns None on write failure so prompt injection still proceeds.
    """
    event_id = f"evt_{uuid.uuid4().hex[:10]}"
    entry = {
        "ts": _utc_now_iso(),
        "event_id": event_id,
        "query": query,
        "matches": [
            {
                "alias": match.get("alias"),
                "memory_id": match.get("memory_id"),
                "query_similarity": match.get("query_similarity"),
                "score": match.get("score"),
                "feedback_bias": match.get("feedback_bias"),
                "preview": _truncate_words(str(match.get("text") or "")),
            }
            for match in matches
        ],
        "hook_payload": hook_payload or {},
    }
    path = _log_path(project_root, RETRIEVAL_EVENT_LOG_FILENAME)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        return None
    return event_id


def find_retrieval_event(project_root: Path, event_id: str) -> dict[str, Any] | None:
    if not event_id.strip():
        return None
    path = _log_path(project_root, RETRIEVAL_EVENT_LOG_FILENAME)
    for entry in reversed(_iter_jsonl(path)):
        if entry.get("event_id") == event_id:
            return entry
    return None


def parse_feedback_assignments(items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw in items:
        if "=" not in raw:
            raise ValueError(
                "Per-memory feedback must look like `A=helpful` or `mem_xxx=stale`."
            )
        ref, label = raw.split("=", 1)
        ref = ref.strip()
        label = label.strip().lower()
        if not ref:
            raise ValueError("Per-memory feedback reference cannot be empty.")
        if label not in MEMORY_FEEDBACK_LABELS:
            raise ValueError(
                "Unknown per-memory feedback label "
                f"{label!r}. Use one of: {', '.join(sorted(MEMORY_FEEDBACK_LABELS))}."
            )
        parsed.append((ref, label))
    return parsed


def _clean_optional_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolve_feedback_ref(event: dict[str, Any], ref: str) -> tuple[str, str]:
    matches = event.get("matches")
    if not isinstance(matches, list):
        raise ValueError(f"Retrieval event {event.get('event_id')!r} has no matches.")

    normalized_ref = ref.strip()
    alias_ref = normalized_ref.upper()
    for match in matches:
        if not isinstance(match, dict):
            continue
        memory_id = match.get("memory_id")
        alias = match.get("alias")
        if isinstance(memory_id, str) and memory_id == normalized_ref:
            return str(alias or memory_id), memory_id
        if isinstance(alias, str) and alias.upper() == alias_ref:
            if not isinstance(memory_id, str) or not memory_id:
                break
            return alias, memory_id

    known_refs: list[str] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        alias = match.get("alias")
        memory_id = match.get("memory_id")
        if isinstance(alias, str):
            known_refs.append(alias)
        if isinstance(memory_id, str):
            known_refs.append(memory_id)
    raise ValueError(
        f"Unknown memory reference {ref!r} for retrieval event {event.get('event_id')!r}. "
        f"Known refs: {', '.join(known_refs)}."
    )


def record_retrieval_feedback(
    project_root: Path,
    *,
    event_id: str,
    overall: str | None,
    memory_feedback: list[tuple[str, str]],
    why: str | None,
    better: str | None,
    missing: str | None,
    note: str | None,
) -> dict[str, Any]:
    event = find_retrieval_event(project_root, event_id)
    if event is None:
        raise ValueError(f"Retrieval event {event_id!r} was not found.")

    if overall is not None:
        overall = overall.strip().lower()
        if overall not in OVERALL_FEEDBACK_LABELS:
            raise ValueError(
                "Unknown overall feedback label "
                f"{overall!r}. Use one of: {', '.join(sorted(OVERALL_FEEDBACK_LABELS))}."
            )

    cleaned_why = _clean_optional_text(why)
    cleaned_better = _clean_optional_text(better)
    cleaned_missing = _clean_optional_text(missing)
    cleaned_note = _clean_optional_text(note)

    resolved_by_memory_id: dict[str, dict[str, str]] = {}
    for ref, label in memory_feedback:
        alias, memory_id = _resolve_feedback_ref(event, ref)
        resolved_by_memory_id[memory_id] = {
            "ref": ref,
            "alias": alias,
            "memory_id": memory_id,
            "label": label,
        }

    if (
        overall is None
        and not resolved_by_memory_id
        and cleaned_why is None
        and cleaned_better is None
        and cleaned_missing is None
        and cleaned_note is None
    ):
        raise ValueError(
            "Feedback must include at least one of: --overall, --memory, --why, --better, --missing, or --note."
        )

    payload = {
        "ts": _utc_now_iso(),
        "event_id": event_id,
        "query": event.get("query"),
        "overall": overall,
        "why": cleaned_why,
        "better": cleaned_better,
        "memory_feedback": list(resolved_by_memory_id.values()),
        "missing": cleaned_missing,
        "note": cleaned_note,
    }

    path = _log_path(project_root, RETRIEVAL_FEEDBACK_LOG_FILENAME)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise ValueError(f"Could not write retrieval feedback: {exc}") from exc
    return payload


def reset_memory_feedback(
    project_root: Path,
    memory_ids: list[str],
    *,
    reason: str,
) -> dict[str, Any] | None:
    cleaned_ids = sorted({memory_id for memory_id in memory_ids if memory_id.strip()})
    if not cleaned_ids:
        return None
    payload = {
        "ts": _utc_now_iso(),
        "type": "memory_feedback_reset",
        "memory_ids": cleaned_ids,
        "reason": reason,
    }
    path = _log_path(project_root, RETRIEVAL_FEEDBACK_LOG_FILENAME)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        return None
    return payload


def memory_feedback_label_counts(project_root: Path) -> dict[str, dict[str, int]]:
    path = _log_path(project_root, RETRIEVAL_FEEDBACK_LOG_FILENAME)
    counts: dict[str, dict[str, int]] = {}

    for entry in _iter_jsonl(path):
        if entry.get("type") == "memory_feedback_reset":
            memory_ids = entry.get("memory_ids")
            if isinstance(memory_ids, list):
                for memory_id in memory_ids:
                    if isinstance(memory_id, str):
                        counts.pop(memory_id, None)
            continue
        items = entry.get("memory_feedback")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("memory_id")
            label = item.get("label")
            if not isinstance(memory_id, str) or not isinstance(label, str):
                continue
            if label not in MEMORY_FEEDBACK_LABELS:
                continue
            label_counts = counts.setdefault(memory_id, {})
            label_counts[label] = label_counts.get(label, 0) + 1

    return counts


def feedback_bias_by_memory(project_root: Path) -> dict[str, float]:
    """Aggregate historical per-memory feedback into a small bounded ranking bias."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for memory_id, label_counts in memory_feedback_label_counts(project_root).items():
        for label, count in label_counts.items():
            weight = _MEMORY_FEEDBACK_WEIGHTS.get(label)
            if weight is None:
                continue
            totals[memory_id] = totals.get(memory_id, 0.0) + (weight * count)
            counts[memory_id] = counts.get(memory_id, 0) + count

    biases: dict[str, float] = {}
    for memory_id, total in totals.items():
        count = counts[memory_id]
        average = total / count
        # Shrink low-count signals heavily so one stray label does not jerk ranking around.
        confidence = min(count, 5) / 5
        bias = average * 0.06 * confidence
        biases[memory_id] = round(
            max(-_MAX_ABSOLUTE_BIAS, min(_MAX_ABSOLUTE_BIAS, bias)),
            4,
        )
    return biases
