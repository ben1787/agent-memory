from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory.config import MemoryConfig, init_project
from agent_memory.engine import AgentMemory, open_memory_with_retry
from agent_memory.query_log import QUERY_LOG_FILENAME


def make_memory(tmp_path: Path) -> AgentMemory:
    config = MemoryConfig(
        embedding_backend="hash",
    )
    project = init_project(tmp_path, config=config)
    return AgentMemory(project)


def test_save_and_recall_returns_ranked_hits(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        save_result = memory.save("Graph theory studies nodes, edges, and traversal.")
        memory.save("Graph databases store connected data with nodes and edges.")
        memory.save("Knowledge graphs describe entities and relationships in a graph.")
        memory.save("Sourdough bread uses a fermented starter and long proofing.")
        result = memory.recall("graph relationships and nodes", limit=3)
    finally:
        memory.close()

    assert len(save_result.saved) == 1
    assert save_result.saved[0].created_at
    assert result.seed_ids == []
    assert len(result.hits) == 3
    assert result.hits == sorted(result.hits, key=lambda hit: hit.score, reverse=True)
    all_text = " ".join(hit.text for hit in result.hits)
    assert "Graph theory" in all_text
    assert "Graph databases" in all_text or "Knowledge graphs" in all_text
    assert "Sourdough bread" not in all_text


def test_recall_cosine_returns_flat_top_n_by_similarity(tmp_path: Path) -> None:
    # Reason: verifies the flat-cosine recall path bypasses the graph
    # entirely and returns memories purely ordered by query cosine, so the
    # benchmark harness has a principled "no graph" baseline to compare
    # against `recall()`.
    memory = make_memory(tmp_path)
    try:
        memory.save("Graph theory studies nodes, edges, and traversal.")
        memory.save("Graph databases store connected data with nodes and edges.")
        memory.save("Knowledge graphs describe entities and relationships in a graph.")
        memory.save("Sourdough bread uses a fermented starter and long proofing.")
        result = memory.recall_cosine("graph relationships and nodes", limit=3)
    finally:
        memory.close()

    assert result.seed_ids == []
    assert len(result.hits) == 3
    # Hits must be strictly ordered by query_similarity descending — no PPV
    # amplification, no graph expansion, just cosine.
    sims = [hit.query_similarity for hit in result.hits]
    assert sims == sorted(sims, reverse=True)
    # For every returned hit, score == query_similarity (clamped via
    # _path_weight to [0, 1]) — the flat path performs no score mutation.
    for hit in result.hits:
        assert hit.score == pytest.approx(max(0.0, min(hit.query_similarity, 1.0)))
    returned_text = " ".join(hit.text for hit in result.hits)
    assert "Sourdough bread" not in returned_text


def test_recall_cosine_rejects_empty_query_and_bad_limit(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        memory.save("Graph databases store connected data.")
        with pytest.raises(ValueError):
            memory.recall_cosine("   ", limit=3)
        with pytest.raises(ValueError):
            memory.recall_cosine("graph", limit=0)
    finally:
        memory.close()


def test_consolidate_reports_duplicates_without_mutating_nodes(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        memory.save("Kuzu is an embedded graph database.")
        memory.save("Kuzu is an embedded graph database.")
        before = memory.stats().memory_count
        report = memory.consolidate()
        after = memory.stats().memory_count
    finally:
        memory.close()

    assert before == 2
    assert len(report.merged_groups) == 1
    assert after == 2


def test_save_rejects_overlong_memory(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        text = "word " * 1001
        try:
            memory.save(text)
        except ValueError as exc:
            assert "too long" in str(exc)
        else:
            raise AssertionError("Expected save() to reject an overlong memory.")
    finally:
        memory.close()


def test_capture_turn_batches_raw_turn_and_distilled_memories(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        result = memory.capture_turn(
            user_text="Where is the billing webhook handler?",
            assistant_text="It lives in services/billing/webhooks.py.",
            memories=[
                "Billing webhook handler lives in services/billing/webhooks.py.",
                "Use the billing webhook path when tracing payment event handling.",
            ],
        )
        stats = memory.stats()
    finally:
        memory.close()

    assert len(result.saved) == 4
    assert stats.memory_count == 4


def test_capture_turn_truncates_overlong_messages(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        overlong = "word " * 1105
        result = memory.capture_turn(
            user_text=overlong,
            assistant_text=overlong,
        )
        hits = memory.recall("word", limit=2).hits
    finally:
        memory.close()

    assert len(result.saved) == 2
    assert all("[truncated]" in hit.text for hit in hits)


def test_multiple_read_only_readers_can_open_same_store(tmp_path: Path) -> None:
    writer = make_memory(tmp_path)
    try:
        writer.save("The billing webhook handler lives in services/billing/webhooks.py.")
    finally:
        writer.close()

    left = AgentMemory.open(tmp_path, exact=True, read_only=True)
    right = AgentMemory.open(tmp_path, exact=True, read_only=True)
    try:
        assert left.stats().memory_count == 1
        assert right.recall("billing webhook handler").hits[0].text.endswith("webhooks.py.")
    finally:
        left.close()
        right.close()


def test_read_only_recall_does_not_touch_access_metadata(tmp_path: Path) -> None:
    writer = make_memory(tmp_path)
    try:
        result = writer.save("The billing webhook handler lives in services/billing/webhooks.py.")
        memory_id = result.saved[0].memory_id
    finally:
        writer.close()

    reader = AgentMemory.open(tmp_path, exact=True, read_only=True)
    try:
        hits = reader.recall("billing webhook handler").hits
    finally:
        reader.close()

    verifier = AgentMemory.open(tmp_path, exact=True)
    try:
        stored = verifier.store.list_memories()
    finally:
        verifier.close()

    assert hits[0].memory_id == memory_id
    saved = next(memory for memory in stored if memory.id == memory_id)
    assert saved.access_count == 0
    assert saved.last_accessed is None


def test_recall_logs_query_to_queries_jsonl(tmp_path: Path) -> None:
    # Reason: verifies that recall() captures the query into the
    # production query distribution log so future algorithm changes can
    # be replayed against real questions.
    memory = make_memory(tmp_path)
    try:
        memory.save("Graph databases store connected data.")
        memory.recall("graph data", limit=1)
        memory.recall_cosine("graph data again", limit=1)
        log_path = memory._query_log_path
    finally:
        memory.close()

    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["method"] == "recall"
    assert lines[0]["query"] == "graph data"
    assert "ts" in lines[0]
    assert lines[1]["method"] == "recall_cosine"
    assert lines[1]["query"] == "graph data again"


def test_recall_does_not_log_when_query_validation_fails(tmp_path: Path) -> None:
    # Reason: invalid queries raise before logging, so the log file must
    # not be created (or must remain empty) — junk queries shouldn't
    # pollute the replay corpus.
    memory = make_memory(tmp_path)
    try:
        memory.save("Graph databases store connected data.")
        with pytest.raises(ValueError):
            memory.recall_cosine("   ", limit=3)
        log_path = memory._query_log_path
    finally:
        memory.close()

    assert not log_path.exists() or log_path.read_text().strip() == ""


def test_open_memory_with_retry_retries_transient_lock_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    attempts = {"count": 0}

    def fake_open(
        start: Path | None = None,
        embedder: object | None = None,
        exact: bool = False,
        read_only: bool = False,
    ) -> object:
        del start, embedder, exact, read_only
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("IO exception: Could not set lock on file : /tmp/memory.kuzu")
        return sentinel

    monkeypatch.setattr(AgentMemory, "open", staticmethod(fake_open))

    memory = open_memory_with_retry(Path("/tmp/project"), exact=True, read_only=True, delay_s=0.0)

    assert memory is sentinel
    assert attempts["count"] == 3
