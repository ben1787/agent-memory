from __future__ import annotations

from pathlib import Path

from agent_memory.operations_log import (
    OP_DELETE,
    OP_EDIT,
    OP_SAVE,
    OP_UNDO,
    LogEntry,
    OperationsLog,
)


def _payload(memory_id: str, text: str) -> dict:
    return {
        "id": memory_id,
        "text": text,
        "created_at": "2026-04-07T00:00:00+00:00",
        "embedding": [0.0, 1.0, 0.0],
        "importance": 0.5,
        "access_count": 0,
        "last_accessed": None,
    }


def test_log_appends_save_entry_with_increasing_seq(tmp_path: Path) -> None:
    log = OperationsLog(tmp_path / "operations.log")

    log.record_save("mem_a", _payload("mem_a", "first"))
    log.record_save("mem_b", _payload("mem_b", "second"))

    entries = log._read_all()
    assert [e.seq for e in entries] == [1, 2]
    assert [e.op for e in entries] == [OP_SAVE, OP_SAVE]


def test_last_undoable_returns_most_recent_unreverted(tmp_path: Path) -> None:
    log = OperationsLog(tmp_path / "operations.log")

    log.record_save("mem_a", _payload("mem_a", "first"))
    log.record_save("mem_b", _payload("mem_b", "second"))
    log.record_edit("mem_b", _payload("mem_b", "second"), _payload("mem_b", "second v2"))

    last = log.last_undoable()
    assert last is not None
    assert last.op == OP_EDIT
    assert last.memory_id == "mem_b"


def test_undo_chain_walks_backwards_skipping_already_reverted(tmp_path: Path) -> None:
    log = OperationsLog(tmp_path / "operations.log")

    log.record_save("mem_a", _payload("mem_a", "first"))                       # seq 1
    log.record_save("mem_b", _payload("mem_b", "second"))                      # seq 2
    log.record_delete("mem_a", _payload("mem_a", "first"))                     # seq 3

    # First undo reverts seq 3 (the delete).
    first_target = log.last_undoable()
    assert first_target is not None
    assert first_target.seq == 3
    log.record_undo(first_target.seq, "mem_a")                                 # seq 4

    # Second undo should skip seq 3 (reverted) and seq 4 (an undo entry),
    # landing on seq 2.
    second_target = log.last_undoable()
    assert second_target is not None
    assert second_target.seq == 2
    log.record_undo(second_target.seq, "mem_b")                                # seq 5

    # Third undo should land on seq 1.
    third_target = log.last_undoable()
    assert third_target is not None
    assert third_target.seq == 1


def test_last_undoable_returns_none_when_everything_reverted(tmp_path: Path) -> None:
    log = OperationsLog(tmp_path / "operations.log")
    log.record_save("mem_a", _payload("mem_a", "first"))
    target = log.last_undoable()
    assert target is not None
    log.record_undo(target.seq, "mem_a")

    assert log.last_undoable() is None


def test_log_round_trips_through_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "operations.log"
    log = OperationsLog(log_path)

    log.record_save("mem_a", _payload("mem_a", "alpha"))
    log.record_edit("mem_a", _payload("mem_a", "alpha"), _payload("mem_a", "alpha prime"))
    log.record_delete("mem_a", _payload("mem_a", "alpha prime"))

    # Re-open with a fresh OperationsLog and confirm we can still find undoables.
    fresh = OperationsLog(log_path)
    last = fresh.last_undoable()
    assert last is not None
    assert last.op == OP_DELETE
    assert last.seq == 3


def test_log_entry_serialization_omits_none_fields(tmp_path: Path) -> None:
    entry = LogEntry(
        seq=7,
        timestamp="2026-04-07T12:00:00+00:00",
        op=OP_SAVE,
        memory_id="mem_xyz",
        after=_payload("mem_xyz", "hello"),
    )
    line = entry.to_json_line()
    assert '"reverts_seq"' not in line
    assert '"before"' not in line

    parsed = LogEntry.from_json_line(line)
    assert parsed.seq == 7
    assert parsed.op == OP_SAVE
    assert parsed.memory_id == "mem_xyz"
    assert parsed.before is None
    assert parsed.reverts_seq is None
