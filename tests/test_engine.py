from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory.config import MemoryConfig, init_project, load_project
from agent_memory.embeddings import FastembedCachePruneResult
from agent_memory.engine import AgentMemory, open_memory_with_retry, reembed_project
from agent_memory.metadata_backfill import derive_metadata_from_text
from agent_memory.models import MemoryMetadata
from agent_memory.query_log import QUERY_LOG_FILENAME, log_query
from agent_memory.retrieval_feedback import (
    record_retrieval_event,
    record_retrieval_feedback,
)


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


def test_save_persists_explicit_metadata_fields(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        result = memory.save(
            "Billing webhook handler lives in services/billing/webhooks.py.\n"
            "This saves a repo search when debugging Stripe events.",
            metadata=MemoryMetadata(
                title="Billing webhook handler",
                kind="operational",
                subsystem="billing",
                workstream="webhooks",
                environment="prod",
            )
        )
        record = memory.get(result.saved[0].memory_id)
    finally:
        memory.close()

    assert record is not None
    assert record.text == (
        "Billing webhook handler lives in services/billing/webhooks.py.\n"
        "This saves a repo search when debugging Stripe events."
    )
    assert record.metadata.title == "Billing webhook handler"
    assert record.metadata.kind == "operational"
    assert record.metadata.subsystem == "billing"
    assert record.metadata.workstream == "webhooks"
    assert record.metadata.environment == "prod"


def test_derive_metadata_from_text_classifies_common_patterns() -> None:
    metadata = derive_metadata_from_text(
        "User preference (2026-04-07): when asked for a recommendation, give the best technical solution regardless of implementation effort."
    )

    assert metadata.title.startswith("when asked for a recommendation")
    assert metadata.kind == "preference"
    assert metadata.workstream == "general"
    assert metadata.environment == "unknown"


def test_recall_to_dict_returns_query_rooted_nodes(tmp_path: Path) -> None:
    config = MemoryConfig(embedding_backend="hash")
    project = init_project(tmp_path, config=config)

    class FixedEmbedder:
        dimensions = config.embedding_dimensions

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(text) for text in texts]

        def embed_text(self, text: str) -> list[float]:
            if text == "billing webhook handler":
                return _pad([1.0, 0.0])
            raise AssertionError(f"Unexpected embed_text call for {text!r}")

    def _pad(values: list[float]) -> list[float]:
        return values + [0.0] * (config.embedding_dimensions - len(values))

    memory = AgentMemory(project, embedder=FixedEmbedder())
    try:
        memory.save("Primary billing memory.", embedding=_pad([0.9, 0.435889894]))
        memory.save("Related webhook memory.", embedding=_pad([0.8, 0.6]))
        payload = memory.recall("billing webhook handler", limit=2).to_dict()
    finally:
        memory.close()

    assert payload["query"] == "billing webhook handler"
    assert [node["alias"] for node in payload["nodes"]] == ["A", "B"]
    assert payload["nodes"][0]["source"] == "QUERY"
    assert payload["nodes"][0]["text"] == "Primary billing memory."
    assert payload["nodes"][1]["source"] == "A"
    assert payload["nodes"][1]["text"] == "Related webhook memory."
    assert payload["nodes"][1]["source_similarity"] == pytest.approx(0.9815, abs=1e-4)


def test_load_project_migrates_legacy_default_config(tmp_path: Path) -> None:
    app_dir = tmp_path / ".agent-memory"
    app_dir.mkdir()
    config_path = app_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 4,
                "embedding_backend": "fastembed",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dimensions": 384,
                "max_memory_words": 1000,
                "duplicate_threshold": 0.97,
                "overlap_threshold": 0.9,
                "lexical_duplicate_threshold": 0.95,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (app_dir / "instructions.md").write_text("instructions\n", encoding="utf-8")

    project = load_project(tmp_path, exact=True)

    assert project.config.embedding_model == "snowflake/snowflake-arctic-embed-m"
    assert project.config.embedding_dimensions == 768
    assert project.config.max_memory_words == 250
    assert project.config.stored_embedding_model == "BAAI/bge-small-en-v1.5"
    assert project.config.stored_embedding_dimensions == 384

    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["version"] == 9
    assert persisted["embedding_model"] == "snowflake/snowflake-arctic-embed-m"
    assert persisted["stored_embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert persisted["consolidation_similarity_threshold"] == 0.85


def test_save_and_recall_use_document_and_query_embeddings_when_available(tmp_path: Path) -> None:
    config = MemoryConfig(embedding_backend="hash", embedding_dimensions=2)
    project = init_project(tmp_path, config=config)

    class RoutedEmbedder:
        dimensions = 2

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("generic embed_texts should not be used")

        def embed_text(self, text: str) -> list[float]:
            raise AssertionError("generic embed_text should not be used")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_document(text) for text in texts]

        def embed_document(self, text: str) -> list[float]:
            if text == "alpha memory":
                return [1.0, 0.0]
            if text == "beta memory":
                return [0.0, 1.0]
            raise AssertionError(f"Unexpected document {text!r}")

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            if text == "alpha":
                return [1.0, 0.0]
            raise AssertionError(f"Unexpected query {text!r}")

    memory = AgentMemory(project, embedder=RoutedEmbedder())
    try:
        memory.save("alpha memory")
        memory.save("beta memory")
        payload = memory.recall("alpha", limit=1).to_dict()
    finally:
        memory.close()

    assert payload["nodes"][0]["text"] == "alpha memory"


def test_open_memory_auto_reembeds_store_when_embedding_signature_changes(tmp_path: Path) -> None:
    legacy_config = MemoryConfig(
        version=4,
        embedding_backend="hash",
        embedding_model="hash-legacy",
        embedding_dimensions=2,
        max_memory_words=1000,
        stored_embedding_backend="hash",
        stored_embedding_model="hash-legacy",
        stored_embedding_dimensions=2,
    )
    project = init_project(tmp_path, config=legacy_config)
    memory = AgentMemory(project)
    try:
        memory.save("alpha memory")
        memory.save("beta memory")
    finally:
        memory.close()

    migrated_config = MemoryConfig(
        embedding_backend="hash",
        embedding_model="hash-v2",
        embedding_dimensions=8,
        stored_embedding_backend="hash",
        stored_embedding_model="hash-legacy",
        stored_embedding_dimensions=2,
        max_memory_words=250,
    )
    (tmp_path / ".agent-memory" / "config.json").write_text(
        json.dumps(migrated_config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    reopened = open_memory_with_retry(tmp_path)
    try:
        records = reopened.list_all()
        assert len(records) == 2
        assert all(len(record.embedding) == 8 for record in records)
        assert reopened.project.config.needs_reembed() is False
        assert reopened.project.config.stored_embedding_dimensions == 8
        result = reopened.recall("alpha", limit=1).to_dict()
    finally:
        reopened.close()

    assert result["nodes"][0]["text"] == "alpha memory"


def test_reembed_project_prunes_previous_fastembed_cache(monkeypatch, tmp_path: Path) -> None:
    legacy_config = MemoryConfig(
        version=4,
        embedding_backend="hash",
        embedding_model="hash-legacy",
        embedding_dimensions=2,
        max_memory_words=1000,
        stored_embedding_backend="hash",
        stored_embedding_model="hash-legacy",
        stored_embedding_dimensions=2,
    )
    project = init_project(tmp_path, config=legacy_config)
    memory = AgentMemory(project)
    try:
        memory.save("alpha memory")
    finally:
        memory.close()

    migrated_config = MemoryConfig(
        embedding_backend="fastembed",
        embedding_model="snowflake/snowflake-arctic-embed-m",
        embedding_dimensions=8,
        stored_embedding_backend="hash",
        stored_embedding_model="hash-legacy",
        stored_embedding_dimensions=2,
        max_memory_words=250,
    )
    (tmp_path / ".agent-memory" / "config.json").write_text(
        json.dumps(migrated_config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    def fake_prune(keep_model_names: list[str], *, cache_dir: str | None = None) -> FastembedCachePruneResult:
        seen["keep_model_names"] = keep_model_names
        return FastembedCachePruneResult(
            cache_dir=tmp_path / "cache",
            kept_models=keep_model_names,
            pruned=[],
        )

    monkeypatch.setattr("agent_memory.engine.prune_fastembed_model_cache", fake_prune)

    class ReembedEmbedder:
        dimensions = 8

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_document(text) for text in texts]

        def embed_text(self, text: str) -> list[float]:
            return self.embed_document(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_document(text) for text in texts]

        def embed_document(self, text: str) -> list[float]:
            if text == "alpha memory":
                return [1.0] + [0.0] * 7
            raise AssertionError(f"Unexpected document {text!r}")

        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            if text == "alpha":
                return [1.0] + [0.0] * 7
            raise AssertionError(f"Unexpected query {text!r}")

    result = reembed_project(tmp_path, embedder=ReembedEmbedder(), exact=True)

    assert result.reembedded is True
    assert seen["keep_model_names"] == ["snowflake/snowflake-arctic-embed-m"]
    assert result.cache_prune is not None
    assert result.cache_prune.kept_models == ["snowflake/snowflake-arctic-embed-m"]


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
    assert len(report.clusters) == 1
    assert len(report.clusters[0].member_ids) == 2
    assert after == 2


def test_consolidate_returns_overlapping_similarity_clusters(tmp_path: Path) -> None:
    config = MemoryConfig(
        embedding_backend="hash",
        consolidation_similarity_threshold=0.92,
    )
    project = init_project(tmp_path, config=config)
    memory = AgentMemory(project)
    try:
        def _pad(values: list[float]) -> list[float]:
            return values + [0.0] * (config.embedding_dimensions - len(values))

        memory.save("seed memory", embedding=_pad([1.0, 0.0]))
        memory.save("left neighbor", embedding=_pad([0.97, 0.2431049]))
        memory.save("right neighbor", embedding=_pad([0.97, -0.2431049]))
        memory.save("far away", embedding=_pad([0.0, 1.0]))
        report = memory.consolidate()
    finally:
        memory.close()

    member_sets = {tuple(cluster.member_ids) for cluster in report.clusters}
    assert len(report.clusters) == 3
    assert report.candidate_pair_count == 2
    assert report.clustered_memory_count == 3
    assert any(len(cluster.member_ids) == 3 for cluster in report.clusters)
    assert len(member_sets) == 3


def test_consolidate_reports_deterministic_cleanup_candidates(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        first = memory.save(
            "Use canonical nested params for execute calls.",
            metadata=MemoryMetadata(
                title="Execute dispatcher requires nested params",
                kind="operational",
                subsystem="porter-ai",
                workstream="porter ai ontology",
                environment="dev",
            ),
        ).saved[0].memory_id
        second = memory.save(
            "Use canonical nested params for execute calls.",
            metadata=MemoryMetadata(
                title="Execute dispatcher requires nested params",
                kind="operational",
                subsystem="porter_ai",
                workstream="porter-ai ontology",
                environment="dev",
            ),
        ).saved[0].memory_id
        for index in range(5):
            memory.save(
                f"Subscription import workflow fact {index} keeps API file handling centralized.",
                metadata=MemoryMetadata(
                    title=f"Subscription import workflow fact {index}",
                    kind="operational",
                    subsystem="subscriptions",
                    workstream="subscription import",
                    environment="dev",
                ),
            )
        report = memory.consolidate()
        payload = report.to_dict()
        summary_payload = report.to_summary_dict()
    finally:
        memory.close()

    assert payload["candidate_counts"]["clusters"] >= 1
    assert payload["candidate_counts"]["metadata_cleanup"] >= 1
    assert payload["instructions"]["section_actions"]["clusters"].startswith(
        "Review embedding-similar memories."
    )
    assert payload["instructions"]["commands"]["section"] == (
        "agent-memory consolidate --json --section <section>"
    )
    assert "duplicate_groups" not in payload
    assert "metadata_cohorts" not in payload
    assert "recent_bursts" not in payload
    assert "quality_flag_groups" not in payload
    assert any(
        {first, second}.issubset(set(group["member_ids"]))
        for group in payload["clusters"]
    )

    summary_variant = next(
        group
        for group in summary_payload["metadata_cleanup"]
        if group["field"] == "subsystem"
    )
    assert summary_variant["values"] == [
        {"value": "porter-ai", "count": 1},
        {"value": "porter_ai", "count": 1},
    ]
    variant_detail = report.group_detail_dict(summary_variant["group_id"])
    assert variant_detail is not None
    assert variant_detail["instructions"]["commands"]["complete"] == (
        "agent-memory consolidation-complete --json"
    )
    assert "member_ids" not in variant_detail
    assert "sample_members" not in variant_detail


def test_consolidate_reports_negative_feedback_and_edit_resets_it(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        memory_id = memory.save(
            "This stale fact should be reviewed after repeated bad feedback.",
            metadata=MemoryMetadata(
                title="Stale feedback candidate",
                kind="gotcha",
                subsystem="memory_cli",
                workstream="consolidation",
                environment="local",
            ),
        ).saved[0].memory_id
    finally:
        memory.close()

    for index in range(4):
        event_id = record_retrieval_event(
            tmp_path,
            query=f"query {index}",
            matches=[
                {
                    "alias": "A",
                    "memory_id": memory_id,
                    "query_similarity": 1.0,
                    "score": 1.0,
                    "text": "preview",
                }
            ],
        )
        assert event_id is not None
        record_retrieval_feedback(
            tmp_path,
            event_id=event_id,
            overall=None,
            memory_feedback=[("A", "wrong")],
            why=None,
            better=None,
            missing=None,
            note=None,
        )

    memory = open_memory_with_retry(tmp_path, exact=True)
    try:
        report = memory.consolidate()
        assert [candidate.memory_id for candidate in report.negative_feedback_memories] == [
            memory_id
        ]
        memory.edit(memory_id, "Rewritten durable fact after feedback.")
        report_after_edit = memory.consolidate()
    finally:
        memory.close()

    assert report_after_edit.negative_feedback_memories == []


def test_consolidate_reports_unretrieved_after_enough_later_queries(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        memory_id = memory.save(
            "Rarely useful local setup detail.",
            metadata=MemoryMetadata(
                title="Never retrieved candidate",
                kind="operational",
                subsystem="local-dev",
                workstream="consolidation",
                environment="local",
            ),
        ).saved[0].memory_id
        for index in range(1000):
            log_query(
                tmp_path / ".agent-memory" / QUERY_LOG_FILENAME,
                f"later query {index}",
                method="recall",
            )
        report = memory.consolidate()
    finally:
        memory.close()

    assert [candidate.memory_id for candidate in report.unretrieved_memories] == [
        memory_id
    ]
    assert report.unretrieved_memories[0].queries_since_created == 1000


def test_save_rejects_overlong_memory(tmp_path: Path) -> None:
    memory = make_memory(tmp_path)
    try:
        text = "word " * 251
        try:
            memory.save(text)
        except ValueError as exc:
            assert "too long" in str(exc)
            assert "stdin mode" in str(exc)
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
        overlong = "word " * 355
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
