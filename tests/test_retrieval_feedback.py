from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agent_memory.cli import app
from agent_memory.config import MemoryConfig, init_project
from agent_memory.hooks.common import auto_recall_matches
from agent_memory.models import MemoryHit
from agent_memory.retrieval_feedback import (
    record_retrieval_event,
    record_retrieval_feedback,
)


class _FakeMemory:
    def __init__(self, hits: list[MemoryHit], *, seed_score: float | None = None) -> None:
        self._hits = hits
        self._seed_score = (
            seed_score
            if seed_score is not None
            else max((hit.query_similarity for hit in hits), default=0.0)
        )

    def recall(self, query: str, limit: int = 3):  # noqa: ANN001 - test double
        class _Result:
            def __init__(self, hits: list[MemoryHit], seed_score: float) -> None:
                self.hits = hits
                self.seed_score = seed_score

        return _Result(self._hits[:limit], self._seed_score)

    def close(self) -> None:
        return None


def test_feedback_command_records_resolved_alias_labels(tmp_path: Path) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    event_id = record_retrieval_event(
        tmp_path,
        query="billing webhook handler",
        matches=[
            {
                "alias": "A",
                "memory_id": "mem_alpha",
                "text": "Billing webhook handler lives in services/billing/webhooks.py.",
                "query_similarity": 0.81,
                "score": 0.81,
                "feedback_bias": 0.0,
                "adjusted_score": 0.81,
            }
        ],
        hook_payload={"turn_id": "turn-1"},
    )
    assert event_id is not None

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "feedback",
            event_id,
            "--cwd",
            str(tmp_path),
            "--overall",
            "helpful",
            "--why",
            "The recalled set identified the right code path quickly.",
            "--better",
            "A second memory about retry behavior would have made the set complete.",
            "--memory",
            "A=helpful",
            "--missing",
            "Should also have surfaced the retry behavior.",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"event_id":' in result.stdout
    assert '"overall": "helpful"' in result.stdout
    assert '"why": "The recalled set identified the right code path quickly."' in result.stdout
    assert '"better": "A second memory about retry behavior would have made the set complete."' in result.stdout
    assert '"memory_id": "mem_alpha"' in result.stdout
    assert '"label": "helpful"' in result.stdout


def test_auto_recall_matches_reranks_using_feedback_bias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))
    event_id = record_retrieval_event(
        tmp_path,
        query="old retrieval",
        matches=[
            {
                "alias": "A",
                "memory_id": "mem_beta",
                "text": "Billing queue retry semantics live in services/billing/queue.py.",
                "query_similarity": 0.72,
                "score": 0.72,
                "feedback_bias": 0.0,
                "adjusted_score": 0.72,
            }
        ],
        hook_payload={},
    )
    assert event_id is not None
    record_retrieval_feedback(
        tmp_path,
        event_id=event_id,
        overall="helpful",
        memory_feedback=[("A", "helpful")],
        why="The set was useful overall.",
        better="A slightly more specific queue memory would help.",
        missing=None,
        note=None,
    )

    fake_memory = _FakeMemory(
        [
            MemoryHit(
                memory_id="mem_alpha",
                text="Billing webhook handler lives in services/billing/webhooks.py.",
                score=0.757,
                query_similarity=0.757,
                created_at="2025-01-01T00:00:00+00:00",
            ),
            MemoryHit(
                memory_id="mem_beta",
                text="Billing queue retry semantics live in services/billing/queue.py.",
                score=0.75,
                query_similarity=0.75,
                created_at="2025-01-01T00:00:01+00:00",
            ),
        ]
    )
    monkeypatch.setattr("agent_memory.hooks.common.open_memory_with_retry", lambda *args, **kwargs: fake_memory)

    matches, metadata = auto_recall_matches(tmp_path, "billing queue retry behavior")

    assert matches is not None
    assert metadata["status"] == "matched"
    assert metadata["feedback_bias_applied"] is True
    assert matches[0]["memory_id"] == "mem_beta"
    assert matches[0]["alias"] == "A"
    assert matches[1]["memory_id"] == "mem_alpha"


def test_auto_recall_matches_filters_on_parent_score_not_query_similarity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    init_project(tmp_path, config=MemoryConfig(embedding_backend="hash"))

    fake_memory = _FakeMemory(
        [
            MemoryHit(
                memory_id="mem_secondary",
                text="Secondary node surfaced from a strong parent edge.",
                score=0.82,
                query_similarity=0.31,
                created_at="2025-01-01T00:00:00+00:00",
            ),
            MemoryHit(
                memory_id="mem_seed",
                text="Seed node directly similar to the query.",
                score=0.71,
                query_similarity=0.71,
                created_at="2025-01-01T00:00:01+00:00",
            ),
            MemoryHit(
                memory_id="mem_query_only",
                text="Direct query match but weak parent score.",
                score=0.33,
                query_similarity=0.9,
                created_at="2025-01-01T00:00:02+00:00",
            ),
        ],
        seed_score=0.9,
    )
    monkeypatch.setattr(
        "agent_memory.hooks.common.open_memory_with_retry",
        lambda *args, **kwargs: fake_memory,
    )

    matches, metadata = auto_recall_matches(tmp_path, "graph-expanded query")

    assert matches is not None
    assert metadata["status"] == "matched"
    assert metadata["threshold_basis"] == "parent_score"
    assert metadata["top_parent_score"] == 0.82
    assert metadata["top_query_similarity"] == 0.9
    assert [match["memory_id"] for match in matches] == [
        "mem_secondary",
        "mem_seed",
    ]
