"""Append-only log of mutating memory operations + undo support.

Each line in `.agent-memory/operations.log` is a JSON object describing one
operation: a `save`, `edit`, `delete`, or `undo`. The `before`/`after` fields
hold the full `MemoryRecord` payloads needed to reverse the operation.

Undo semantics:
- The "last undoable" operation is the highest-seq save/edit/delete entry that
  has not yet been pointed at by an `undo` entry's `reverts_seq`.
- Undoing a save → delete the memory.
- Undoing an edit → restore the prior text/embedding/timestamps.
- Undoing a delete → re-create the memory with its original id and payload.
- Each undo appends a new entry referencing the seq it reversed, so a future
  undo can find the next-most-recent undoable operation in O(N).

The log is intentionally append-only with no truncation: it doubles as an
audit trail. At ~3KB per entry (text + 384-float embedding), 10k operations
fit in ~30MB which is fine for a project store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPERATIONS_LOG_FILENAME = "operations.log"

OP_SAVE = "save"
OP_EDIT = "edit"
OP_DELETE = "delete"
OP_UNDO = "undo"

UNDOABLE_OPS = {OP_SAVE, OP_EDIT, OP_DELETE}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class LogEntry:
    seq: int
    timestamp: str
    op: str
    memory_id: str | None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    reverts_seq: int | None = None

    def to_json_line(self) -> str:
        payload: dict[str, Any] = {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "op": self.op,
        }
        if self.memory_id is not None:
            payload["memory_id"] = self.memory_id
        if self.before is not None:
            payload["before"] = self.before
        if self.after is not None:
            payload["after"] = self.after
        if self.reverts_seq is not None:
            payload["reverts_seq"] = self.reverts_seq
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "LogEntry":
        payload = json.loads(line)
        return cls(
            seq=int(payload["seq"]),
            timestamp=str(payload["timestamp"]),
            op=str(payload["op"]),
            memory_id=payload.get("memory_id"),
            before=payload.get("before"),
            after=payload.get("after"),
            reverts_seq=payload.get("reverts_seq"),
        )


class OperationsLog:
    """Append-only JSONL log scoped to a single project's `.agent-memory/`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[LogEntry]:
        if not self.path.exists():
            return []
        entries: list[LogEntry] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                entries.append(LogEntry.from_json_line(line))
        return entries

    def _next_seq(self) -> int:
        entries = self._read_all()
        if not entries:
            return 1
        return entries[-1].seq + 1

    def _append(self, entry: LogEntry) -> LogEntry:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json_line() + "\n")
        return entry

    def record_save(self, memory_id: str, after: dict[str, Any]) -> LogEntry:
        return self._append(
            LogEntry(
                seq=self._next_seq(),
                timestamp=_utc_now(),
                op=OP_SAVE,
                memory_id=memory_id,
                after=after,
            )
        )

    def record_edit(
        self,
        memory_id: str,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> LogEntry:
        return self._append(
            LogEntry(
                seq=self._next_seq(),
                timestamp=_utc_now(),
                op=OP_EDIT,
                memory_id=memory_id,
                before=before,
                after=after,
            )
        )

    def record_delete(self, memory_id: str, before: dict[str, Any]) -> LogEntry:
        return self._append(
            LogEntry(
                seq=self._next_seq(),
                timestamp=_utc_now(),
                op=OP_DELETE,
                memory_id=memory_id,
                before=before,
            )
        )

    def record_undo(self, reverts_seq: int, memory_id: str | None) -> LogEntry:
        return self._append(
            LogEntry(
                seq=self._next_seq(),
                timestamp=_utc_now(),
                op=OP_UNDO,
                memory_id=memory_id,
                reverts_seq=reverts_seq,
            )
        )

    def last_undoable(self) -> LogEntry | None:
        """Find the most recent save/edit/delete that has not yet been reverted."""
        entries = self._read_all()
        reverted_seqs: set[int] = {
            entry.reverts_seq
            for entry in entries
            if entry.op == OP_UNDO and entry.reverts_seq is not None
        }
        for entry in reversed(entries):
            if entry.op in UNDOABLE_OPS and entry.seq not in reverted_seqs:
                return entry
        return None

    def recent(self, limit: int = 20) -> list[LogEntry]:
        return self._read_all()[-limit:]
