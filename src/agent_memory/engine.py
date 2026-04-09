from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import heapq
import json
from pathlib import Path
import re
import time
from typing import overload
import uuid

import numpy as np

from agent_memory.config import (
    MemoryConfig,
    ProjectContext,
    init_project,
    load_project,
)
from agent_memory.embeddings import (
    Embedder,
    FastembedCachePruneResult,
    build_embedder,
    cosine_similarity,
    embed_document,
    embed_query,
    prune_fastembed_model_cache,
)
from agent_memory.models import (
    ConsolidationCluster,
    ConsolidationClusterEdge,
    ConsolidationClusterMember,
    ConsolidationReport,
    MemoryCluster,
    MemoryHit,
    MemoryRecord,
    SaveManyResult,
    MemoryStats,
    SaveResult,
)
from agent_memory.operations_log import (
    OP_DELETE,
    OP_EDIT,
    OP_SAVE,
    OPERATIONS_LOG_FILENAME,
    LogEntry,
    OperationsLog,
)
from agent_memory.query_log import QUERY_LOG_FILENAME, log_query
from agent_memory.store import GraphStore


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_to_payload(memory: MemoryRecord) -> dict[str, object]:
    """Snapshot a MemoryRecord for the operations log (used by undo)."""
    return {
        "id": memory.id,
        "text": memory.text,
        "created_at": memory.created_at,
        "embedding": list(memory.embedding),
        "importance": memory.importance,
        "access_count": memory.access_count,
        "last_accessed": memory.last_accessed,
    }


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def word_count(text: str) -> int:
    return len(text.split())


def truncate_to_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    clipped = " ".join(words[: max(1, limit - 2)])
    return f"{clipped} [truncated]"


def lexical_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=normalize_text(left), b=normalize_text(right)).ratio()


def split_sentences(text: str) -> list[str]:
    return [chunk.strip() for chunk in SENTENCE_SPLIT_PATTERN.split(text) if chunk.strip()]


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(slots=True)
class RecallResult:
    query: str
    seed_ids: list[str]
    hits: list[MemoryHit]
    sources: list[tuple[str, str | None, float]]
    seed_score: float = 0.0

    @staticmethod
    def _alias_for_index(index: int) -> str:
        label = ""
        value = index
        while True:
            value, remainder = divmod(value, 26)
            label = chr(ord("A") + remainder) + label
            if value == 0:
                return label
            value -= 1

    @property
    def clusters(self) -> list[MemoryCluster]:
        if not self.hits:
            return []
        return [
            MemoryCluster(
                cluster_id="ranked_1",
                score=round(self.seed_score or self.hits[0].score, 4),
                seed_ids=self.seed_ids,
                memory_ids=[hit.memory_id for hit in self.hits],
                hits=self.hits,
            )
        ]

    def to_dict(self) -> dict[str, object]:
        alias_by_memory_id = {
            hit.memory_id: self._alias_for_index(index)
            for index, hit in enumerate(self.hits)
        }
        source_by_memory_id = {
            memory_id: (source_id, source_similarity)
            for memory_id, source_id, source_similarity in self.sources
        }
        nodes: list[dict[str, object]] = []
        for index, hit in enumerate(self.hits):
            alias = alias_by_memory_id[hit.memory_id]
            source_id, source_similarity = source_by_memory_id.get(hit.memory_id, (None, 0.0))
            nodes.append(
                {
                    "alias": alias,
                    "source": "QUERY" if source_id is None else alias_by_memory_id.get(source_id, source_id),
                    "source_similarity": round(source_similarity, 4),
                    "created_at": hit.created_at,
                    "memory_id": hit.memory_id,
                    "text": hit.text,
                }
            )
        return {
            "query": self.query,
            "nodes": nodes,
        }


@dataclass(slots=True)
class ReembedResult:
    project_root: Path
    db_path: Path
    reembedded: bool
    memory_count: int
    previous_store_backend: str
    previous_store_model: str
    previous_store_dimensions: int
    current_store_backend: str
    current_store_model: str
    current_store_dimensions: int
    cache_prune: FastembedCachePruneResult | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "project_root": str(self.project_root),
            "db_path": str(self.db_path),
            "reembedded": self.reembedded,
            "memory_count": self.memory_count,
            "previous_store": {
                "embedding_backend": self.previous_store_backend,
                "embedding_model": self.previous_store_model,
                "embedding_dimensions": self.previous_store_dimensions,
            },
            "current_store": {
                "embedding_backend": self.current_store_backend,
                "embedding_model": self.current_store_model,
                "embedding_dimensions": self.current_store_dimensions,
            },
        }
        if self.cache_prune is not None:
            payload["cache_prune"] = self.cache_prune.to_dict()
        return payload


class AgentMemory:
    def __init__(
        self,
        project: ProjectContext,
        embedder: Embedder | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        self.project = project
        self.config = project.config
        self.read_only = read_only
        self._embedder = embedder
        self.store = GraphStore(
            project.db_path,
            self.config.embedding_dimensions,
            read_only=read_only,
        )
        self.operations_log = OperationsLog(project.db_path.parent / OPERATIONS_LOG_FILENAME)
        self._query_log_path = project.db_path.parent / QUERY_LOG_FILENAME
        self._memories: list[MemoryRecord] = []
        self._memory_by_id: dict[str, MemoryRecord] = {}
        self._memory_ids_in_order: list[str] = []
        self._embedding_matrix: np.ndarray | None = None
        self._strong_adjacency: dict[str, dict[str, float]] = {}
        self._sorted_neighbors: dict[str, list[tuple[str, float]]] | None = None
        self._reload_cache()

    @classmethod
    def open(
        cls,
        start: Path | None = None,
        embedder: Embedder | None = None,
        exact: bool = False,
        read_only: bool = False,
    ) -> "AgentMemory":
        project = load_project(start, exact=exact)
        if project.config.needs_reembed():
            reembed_project(project.root, embedder=embedder, exact=True)
            project = load_project(project.root, exact=True)
        refresh_project_integration(project, current_version=__display_version__)
        return cls(project, embedder=embedder, read_only=read_only)

    @classmethod
    def initialize(
        cls,
        root: Path,
        config: MemoryConfig | None = None,
        force: bool = False,
        embedder: Embedder | None = None,
    ) -> "AgentMemory":
        project = init_project(root, config=config, force=force)
        return cls(project, embedder=embedder)

    def close(self) -> None:
        self.store.close()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.config)
        return self._embedder

    def _save_one(
        self,
        text: str,
        embedding: list[float] | None = None,
        *,
        record_in_log: bool = True,
    ) -> SaveResult:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Memory text cannot be empty.")
        words = word_count(cleaned)
        if words > self.config.max_memory_words:
            raise ValueError(
                "Memory text is too long. "
                f"Got {words} words, max allowed is {self.config.max_memory_words}. "
                "Save a shorter summary instead."
            )

        existing = list(self._memories)
        resolved_embedding = embedding or embed_document(self.embedder, cleaned)
        timestamp = utc_now()
        memory = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            text=cleaned,
            created_at=timestamp,
            embedding=resolved_embedding,
        )
        self.store.add_memory(memory)
        self._memories.append(memory)
        self._memory_by_id[memory.id] = memory
        self._refresh_embedding_cache()

        neighbors = self._similarities_to_existing(resolved_embedding, existing)
        # Reason: SIMILAR edges are not persisted to Kuzu. PPV reads from the
        # in-memory `_strong_adjacency` dict, which `_reload_cache()` rebuilds
        # from embeddings via a single numpy matmul (O(n^2) flops in a BLAS
        # call). The persisted copy was a write-only duplicate that cost
        # O(n) transactional round-trips per save — ~1s at n=1000 vs ~10ms
        # for the matmul. We keep the adjacency update here so hot saves
        # don't have to round-trip through _reload_cache().
        for candidate, score in neighbors:
            self._strong_adjacency.setdefault(memory.id, {})[candidate.id] = score
            self._strong_adjacency.setdefault(candidate.id, {})[memory.id] = score
        self._sorted_neighbors = None

        if existing:
            previous = existing[-1]
            self.store.create_next_edge(previous.id, memory.id, 1.0, timestamp)

        if record_in_log:
            self.operations_log.record_save(memory.id, _record_to_payload(memory))

        return SaveResult(
            memory_id=memory.id,
            created_at=memory.created_at,
            connected_neighbors=[
                {"memory_id": candidate.id, "similarity": round(score, 4)}
                for candidate, score in neighbors
            ],
            total_memories=len(self._memories),
        )

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a single memory by id, or None if it doesn't exist."""
        cached = self._memory_by_id.get(memory_id)
        if cached is not None:
            return cached
        return self.store.get_memory(memory_id)

    def list_recent(self, limit: int = 10) -> list[MemoryRecord]:
        """Return the N most-recently-created memories, newest first."""
        if limit <= 0:
            return []
        # Reason: on Windows, datetime.now() has ~16 ms resolution, so two
        # saves in quick succession can produce identical created_at strings.
        # Break ties by insertion order (index in self._memories) so the
        # "newest first" contract holds reliably on every platform.
        indexed = list(enumerate(self._memories))
        indexed.sort(key=lambda pair: (pair[1].created_at, pair[0]), reverse=True)
        return [memory for _, memory in indexed[:limit]]

    def list_all(self) -> list[MemoryRecord]:
        return list(self._memories)

    def edit(
        self,
        memory_id: str,
        new_text: str,
        *,
        record_in_log: bool = True,
    ) -> MemoryRecord:
        """Replace the text of a memory and refresh its embedding + similarity edges."""
        existing = self.get(memory_id)
        if existing is None:
            raise KeyError(f"Memory {memory_id!r} does not exist.")

        cleaned = new_text.strip()
        if not cleaned:
            raise ValueError("Memory text cannot be empty.")
        words = word_count(cleaned)
        if words > self.config.max_memory_words:
            raise ValueError(
                "Memory text is too long. "
                f"Got {words} words, max allowed is {self.config.max_memory_words}."
            )

        before_payload = _record_to_payload(existing)
        new_embedding = embed_document(self.embedder, cleaned)
        timestamp = utc_now()
        updated = MemoryRecord(
            id=existing.id,
            text=cleaned,
            created_at=existing.created_at,
            embedding=new_embedding,
            importance=existing.importance,
            access_count=existing.access_count,
            last_accessed=timestamp,
        )

        self.store.update_memory(updated)
        # Reason: `_reload_cache()` rebuilds `_strong_adjacency` from the
        # current embedding matrix, so dropping the node's row and
        # recomputing neighbors happens implicitly — no need to touch Kuzu
        # SIMILAR rows (they are not read by recall anyway).
        self._reload_cache()

        if record_in_log:
            self.operations_log.record_edit(
                existing.id,
                before=before_payload,
                after=_record_to_payload(updated),
            )
        return updated

    def delete(self, memory_id: str, *, record_in_log: bool = True) -> MemoryRecord:
        """Delete a memory and its incident edges. Returns the deleted record."""
        existing = self.get(memory_id)
        if existing is None:
            raise KeyError(f"Memory {memory_id!r} does not exist.")
        before_payload = _record_to_payload(existing)
        self.store.delete_memory(existing.id)
        self._reload_cache()
        if record_in_log:
            self.operations_log.record_delete(existing.id, before=before_payload)
        return existing

    def undo(self) -> dict[str, object]:
        """Reverse the most recent undoable operation. Idempotent across crashes."""
        entry = self.operations_log.last_undoable()
        if entry is None:
            return {"reverted": None, "reason": "Nothing to undo."}

        if entry.op == OP_SAVE:
            assert entry.memory_id is not None
            try:
                self.delete(entry.memory_id, record_in_log=False)
            except KeyError:
                # Already gone — still log the undo so the seq is consumed.
                pass
            self.operations_log.record_undo(entry.seq, entry.memory_id)
            return {
                "reverted": "save",
                "memory_id": entry.memory_id,
                "seq": entry.seq,
                "details": f"Removed memory {entry.memory_id} that was added by seq {entry.seq}.",
            }

        if entry.op == OP_EDIT:
            assert entry.before is not None
            before = entry.before
            try:
                self.edit(
                    str(before["id"]),
                    str(before["text"]),
                    record_in_log=False,
                )
                # Restore the prior created_at/access metadata that edit() doesn't touch.
                self._restore_metadata(before)
            except KeyError:
                # Memory was deleted after the edit; restore it from the prior payload.
                self._restore_from_payload(before, record_in_log=False)
            self.operations_log.record_undo(entry.seq, entry.memory_id)
            return {
                "reverted": "edit",
                "memory_id": entry.memory_id,
                "seq": entry.seq,
                "details": f"Restored prior text of {entry.memory_id} from seq {entry.seq}.",
            }

        if entry.op == OP_DELETE:
            assert entry.before is not None
            self._restore_from_payload(entry.before, record_in_log=False)
            self.operations_log.record_undo(entry.seq, entry.memory_id)
            return {
                "reverted": "delete",
                "memory_id": entry.memory_id,
                "seq": entry.seq,
                "details": f"Re-created memory {entry.memory_id} that was deleted by seq {entry.seq}.",
            }

        return {"reverted": None, "reason": f"Unknown op {entry.op!r} at seq {entry.seq}."}

    def _restore_metadata(self, payload: dict[str, object]) -> None:
        """After an edit-undo, also restore created_at/last_accessed/importance/access_count."""
        memory_id = str(payload["id"])
        existing = self.get(memory_id)
        if existing is None:
            return
        restored = MemoryRecord(
            id=existing.id,
            text=existing.text,
            created_at=str(payload["created_at"]),
            embedding=existing.embedding,
            importance=float(payload.get("importance", existing.importance)),
            access_count=int(payload.get("access_count", existing.access_count)),
            last_accessed=payload.get("last_accessed"),  # type: ignore[arg-type]
        )
        self.store.update_memory(restored)
        self._reload_cache()

    def _restore_from_payload(self, payload: dict[str, object], *, record_in_log: bool) -> None:
        """Insert a memory record verbatim (preserving its original id)."""
        memory = MemoryRecord(
            id=str(payload["id"]),
            text=str(payload["text"]),
            created_at=str(payload["created_at"]),
            embedding=[float(value) for value in payload["embedding"]],  # type: ignore[arg-type]
            importance=float(payload.get("importance", 0.5)),
            access_count=int(payload.get("access_count", 0)),
            last_accessed=payload.get("last_accessed"),  # type: ignore[arg-type]
        )
        # If something else has since taken this id (extremely unlikely with uuid4)
        # bail out rather than corrupting the store.
        if self.store.get_memory(memory.id) is not None:
            raise ValueError(
                f"Cannot restore memory {memory.id!r}: id is already in use."
            )
        self.store.add_memory(memory)
        # Reason: `_reload_cache()` rebuilds `_strong_adjacency` from the
        # embedding matrix, which covers the restored node's neighbors for
        # free. SIMILAR edges are no longer persisted (see _save_one).
        self._reload_cache()
        if record_in_log:
            self.operations_log.record_save(memory.id, _record_to_payload(memory))

    @overload
    def save(
        self,
        text: str,
        embedding: list[float] | None = None,
    ) -> SaveManyResult: ...

    @overload
    def save(
        self,
        text: list[str],
        embedding: None = None,
    ) -> SaveManyResult: ...

    def save(
        self,
        text: str | list[str],
        embedding: list[float] | None = None,
    ) -> SaveManyResult:
        if isinstance(text, str):
            saved = [self._save_one(text, embedding=embedding)]
        else:
            if embedding is not None:
                raise ValueError("Batch save does not support a single shared embedding.")
            saved = [self._save_one(item) for item in text]
        return SaveManyResult(saved=saved, total_memories=len(self._memories))

    def capture_turn(
        self,
        user_text: str | None = None,
        assistant_text: str | None = None,
        memories: list[str] | None = None,
    ) -> SaveManyResult:
        payloads: list[str] = []
        budget = max(10, self.config.max_memory_words - 4)
        if user_text and user_text.strip():
            payloads.append(f"User message: {truncate_to_words(user_text.strip(), budget)}")
        if assistant_text and assistant_text.strip():
            payloads.append(
                f"Assistant reply: {truncate_to_words(assistant_text.strip(), budget)}"
            )
        for memory in memories or []:
            cleaned = memory.strip()
            if cleaned:
                payloads.append(truncate_to_words(cleaned, budget))
        if not payloads:
            raise ValueError("capture_turn requires at least one non-empty user, assistant, or memory text.")
        return self.save(payloads)

    def import_memories(self, records: list[dict[str, object]]) -> MemoryStats:
        for record in records:
            cleaned = str(record["text"]).strip()
            if not cleaned:
                continue
            resolved_embedding = record.get("embedding")
            if not isinstance(resolved_embedding, list):
                resolved_embedding = embed_document(self.embedder, cleaned)
            memory = MemoryRecord(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                text=cleaned,
                created_at=utc_now(),
                embedding=resolved_embedding,
            )
            self.store.add_memory(memory)
        self._reload_cache()
        return self.rewire()

    def recall(
        self,
        query: str,
        limit: int = 15,
        max_clusters: int | None = None,
    ) -> RecallResult:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty.")
        if limit < 1:
            raise ValueError("Recall limit must be at least 1.")

        # Reason: capture production query distribution to queries.jsonl so
        # future algorithm changes can be replayed against real questions
        # instead of synthetic benchmarks. Best-effort, never breaks recall.
        log_query(self._query_log_path, cleaned_query, method="recall")

        memories = self._memories
        if not memories:
            return RecallResult(query=cleaned_query, seed_ids=[], hits=[], sources=[], seed_score=0.0)

        query_scores = self._query_similarities(cleaned_query)
        top_query_similarity = max(
            (self._path_weight(score) for score in query_scores.values()),
            default=0.0,
        )
        ranked_scores = self._rank_memories(
            query_scores=query_scores,
            limit=limit,
        )
        ordered_hits = []
        sources: list[tuple[str, str | None, float]] = []
        for memory_id, score, source_id, source_similarity in ranked_scores[:limit]:
            ordered_hits.append(
                MemoryHit(
                    memory_id=memory_id,
                    text=self._memory_by_id[memory_id].text,
                    score=round(score, 4),
                    query_similarity=round(query_scores.get(memory_id, 0.0), 4),
                    created_at=self._memory_by_id[memory_id].created_at,
                )
            )
            sources.append((memory_id, source_id, source_similarity))
        touched = [hit.memory_id for hit in ordered_hits]
        if touched and not self.read_only:
            timestamp = utc_now()
            self.store.touch_memories(touched, timestamp)
            for memory_id in set(touched):
                cached = self._memory_by_id.get(memory_id)
                if cached is None:
                    continue
                cached.access_count += 1
                cached.last_accessed = timestamp
        return RecallResult(
            query=cleaned_query,
            seed_ids=[],
            hits=ordered_hits,
            sources=sources,
            seed_score=top_query_similarity,
        )

    def recall_cosine(
        self,
        query: str,
        limit: int = 15,
    ) -> RecallResult:
        # Reason: flat top-N cosine recall with no graph traversal or PPV
        # spreading. Exists as a baseline path so agents / benchmarks can
        # compare "raw embedding nearest neighbors" against the default
        # graph-expanded `recall()` on the same store, same embedder, same
        # query. Returns the same RecallResult shape as `recall()` so
        # downstream code can consume both paths uniformly.
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query cannot be empty.")
        if limit < 1:
            raise ValueError("Recall limit must be at least 1.")

        # Reason: same query-distribution capture as recall(), tagged with
        # method="recall_cosine" so the log distinguishes which path served
        # the query.
        log_query(self._query_log_path, cleaned_query, method="recall_cosine")

        memories = self._memories
        if not memories:
            return RecallResult(query=cleaned_query, seed_ids=[], hits=[], sources=[], seed_score=0.0)

        query_scores = self._query_similarities(cleaned_query)
        ranked_memories = sorted(
            self._memories,
            key=lambda memory: (
                self._path_weight(query_scores.get(memory.id, 0.0)),
                memory.created_at,
            ),
            reverse=True,
        )[:limit]

        ordered_hits = [
            MemoryHit(
                memory_id=memory.id,
                text=memory.text,
                score=round(self._path_weight(query_scores.get(memory.id, 0.0)), 4),
                query_similarity=round(query_scores.get(memory.id, 0.0), 4),
                created_at=memory.created_at,
            )
            for memory in ranked_memories
        ]
        sources = [
            (
                memory.memory_id,
                None,
                self._path_weight(query_scores.get(memory.memory_id, 0.0)),
            )
            for memory in ordered_hits
        ]
        top_query_similarity = ordered_hits[0].score if ordered_hits else 0.0

        # Reason: mirror the touch-on-read behavior of `recall()` so that
        # access stats stay consistent regardless of which retrieval path
        # produced the hit. Keeps read_only honored.
        touched = [hit.memory_id for hit in ordered_hits]
        if touched and not self.read_only:
            timestamp = utc_now()
            self.store.touch_memories(touched, timestamp)
            for memory_id in set(touched):
                cached = self._memory_by_id.get(memory_id)
                if cached is None:
                    continue
                cached.access_count += 1
                cached.last_accessed = timestamp

        return RecallResult(
            query=cleaned_query,
            seed_ids=[],
            hits=ordered_hits,
            sources=sources,
            seed_score=top_query_similarity,
        )

    def consolidate(self) -> ConsolidationReport:
        threshold = self.config.consolidation_similarity_threshold
        if len(self._memories) < 2:
            return ConsolidationReport(
                threshold=threshold,
                clusters=[],
                total_memories=len(self._memories),
                clustered_memory_count=0,
                candidate_pair_count=0,
                generated_at=utc_now(),
            )

        def member_sort_key(memory_id: str) -> tuple[str, str]:
            memory = self._memory_by_id[memory_id]
            return (memory.created_at, memory.id)

        clusters_by_key: dict[tuple[str, ...], ConsolidationCluster] = {}
        unique_pairs: set[tuple[str, str]] = set()

        for memory in self._memories:
            neighbors = [
                neighbor_id
                for neighbor_id, score in self._strong_adjacency.get(memory.id, {}).items()
                if score >= threshold
            ]
            if not neighbors:
                continue

            member_ids = sorted({memory.id, *neighbors}, key=member_sort_key)
            key = tuple(member_ids)
            if key in clusters_by_key:
                seed_ids = clusters_by_key[key].seed_memory_ids
                if memory.id not in seed_ids:
                    seed_ids.append(memory.id)
                    seed_ids.sort(key=member_sort_key)
                continue

            pair_edges: list[ConsolidationClusterEdge] = []
            similarities: list[float] = []
            for left_index, left_id in enumerate(member_ids):
                for right_id in member_ids[left_index + 1 :]:
                    similarity = self._strong_adjacency.get(left_id, {}).get(right_id, 0.0)
                    if similarity < threshold:
                        continue
                    rounded_similarity = round(similarity, 4)
                    pair_edges.append(
                        ConsolidationClusterEdge(
                            source_id=left_id,
                            target_id=right_id,
                            similarity=rounded_similarity,
                        )
                    )
                    similarities.append(rounded_similarity)
                    unique_pairs.add(tuple(sorted((left_id, right_id))))

            if not pair_edges:
                continue

            clusters_by_key[key] = ConsolidationCluster(
                cluster_id="",
                seed_memory_ids=[memory.id],
                member_ids=member_ids,
                members=[
                    ConsolidationClusterMember(
                        memory_id=member_id,
                        text=self._memory_by_id[member_id].text,
                        created_at=self._memory_by_id[member_id].created_at,
                    )
                    for member_id in member_ids
                ],
                pair_edges=pair_edges,
                average_similarity=round(sum(similarities) / len(similarities), 4),
                max_similarity=max(similarities),
            )

        clusters = sorted(
            clusters_by_key.values(),
            key=lambda cluster: (
                -len(cluster.member_ids),
                -cluster.max_similarity,
                -cluster.average_similarity,
                cluster.member_ids,
            ),
        )
        for index, cluster in enumerate(clusters, start=1):
            cluster.cluster_id = f"cluster_{index}"
            cluster.seed_memory_ids.sort(key=member_sort_key)
            cluster.pair_edges.sort(
                key=lambda edge: (edge.source_id, edge.target_id),
            )

        clustered_memory_ids = {
            member_id
            for cluster in clusters
            for member_id in cluster.member_ids
        }
        return ConsolidationReport(
            threshold=threshold,
            clusters=clusters,
            total_memories=len(self._memories),
            clustered_memory_count=len(clustered_memory_ids),
            candidate_pair_count=len(unique_pairs),
            generated_at=utc_now(),
        )

    def stats(self) -> MemoryStats:
        # Reason: SIMILAR edges live in the in-memory `_strong_adjacency`
        # dict (rebuilt from embeddings on every _reload_cache), not in
        # Kuzu. Report the undirected pair count to match the old shape
        # (each pair was written twice as directed rows).
        similarity_edge_count = sum(
            len(neighbors) for neighbors in self._strong_adjacency.values()
        )
        return MemoryStats(
            project_root=self.project.root,
            db_path=self.project.db_path,
            memory_count=len(self._memories),
            similarity_edge_count=similarity_edge_count,
            next_edge_count=self.store.count_relationships("NEXT"),
        )

    def rewire(self) -> MemoryStats:
        # Reason: SIMILAR edges are no longer persisted, so there is
        # nothing to "rewire" for similarity — _reload_cache rebuilds the
        # in-memory adjacency from the current embedding matrix. NEXT
        # edges are still persisted and can be rebuilt from the memory
        # ordering.
        self.store.clear_relationships("NEXT")
        self._rebuild_relationships()
        self._reload_cache()
        return self.stats()

    def _reload_cache(self) -> None:
        self._memories = self.store.list_memories()
        self._memory_by_id = {memory.id: memory for memory in self._memories}
        self._refresh_embedding_cache()
        self._strong_adjacency = self._build_dense_adjacency()
        self._sorted_neighbors = self._build_sorted_neighbors(self._strong_adjacency)

    def _refresh_embedding_cache(self) -> None:
        self._memory_ids_in_order = [memory.id for memory in self._memories]
        if not self._memories:
            self._embedding_matrix = None
            return
        matrix = np.asarray(
            [memory.embedding for memory in self._memories],
            dtype=np.float32,
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._embedding_matrix = matrix / norms

    def _normalize_embedding(self, embedding: list[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def _query_similarities(self, query: str) -> dict[str, float]:
        if self._embedding_matrix is None:
            return {}
        query_embedding = self._normalize_embedding(embed_query(self.embedder, query))
        scores = self._embedding_matrix @ query_embedding
        return {
            memory_id: float(score)
            for memory_id, score in zip(self._memory_ids_in_order, scores, strict=False)
        }

    def _similarities_to_existing(
        self,
        embedding: list[float],
        existing: list[MemoryRecord],
    ) -> list[tuple[MemoryRecord, float]]:
        if not existing:
            return []
        existing_ids = [memory.id for memory in existing]
        if self._embedding_matrix is None:
            return [
                (candidate, cosine_similarity(embedding, candidate.embedding))
                for candidate in existing
            ]
        normalized = self._normalize_embedding(embedding)
        score_by_id = {
            memory_id: float(score)
            for memory_id, score in zip(
                self._memory_ids_in_order,
                self._embedding_matrix @ normalized,
                strict=False,
            )
        }
        return sorted(
            (
                (candidate, score_by_id[candidate.id])
                for candidate in existing
                if candidate.id in score_by_id
            ),
            key=lambda item: item[1],
            reverse=True,
        )

    def _build_dense_adjacency(self) -> dict[str, dict[str, float]]:
        if self._embedding_matrix is None:
            return {}
        adjacency: dict[str, dict[str, float]] = {}
        similarity_matrix = self._embedding_matrix @ self._embedding_matrix.T
        np.fill_diagonal(similarity_matrix, 0.0)
        for row_index, source_id in enumerate(self._memory_ids_in_order):
            row = similarity_matrix[row_index]
            adjacency[source_id] = {
                target_id: float(row[col_index])
                for col_index, target_id in enumerate(self._memory_ids_in_order)
                if col_index != row_index
            }
        return adjacency

    def _build_sorted_neighbors(
        self,
        adjacency: dict[str, dict[str, float]],
    ) -> dict[str, list[tuple[str, float]]]:
        ordered: dict[str, list[tuple[str, float]]] = {}
        for source_id, neighbors in adjacency.items():
            ordered[source_id] = sorted(
                neighbors.items(),
                key=lambda item: (
                    item[1],
                    self._memory_by_id[item[0]].created_at,
                    item[0],
                ),
                reverse=True,
            )
        return ordered

    def _ensure_sorted_neighbors(self) -> dict[str, list[tuple[str, float]]]:
        if self._sorted_neighbors is None:
            self._sorted_neighbors = self._build_sorted_neighbors(self._strong_adjacency)
        return self._sorted_neighbors

    def _path_weight(self, value: float) -> float:
        return max(0.0, min(value, 1.0))

    def _rank_memories(
        self,
        query_scores: dict[str, float],
        limit: int,
    ) -> list[tuple[str, float, str | None, float]]:
        initial_scores = {
            memory_id: self._path_weight(score)
            for memory_id, score in query_scores.items()
        }
        if not any(initial_scores.values()):
            ranked = sorted(
                self._memories,
                key=lambda memory: (
                    query_scores.get(memory.id, 0.0),
                    memory.created_at,
                ),
                reverse=True,
            )
            return [
                (
                    memory.id,
                    self._path_weight(query_scores.get(memory.id, 0.0)),
                    None,
                    self._path_weight(query_scores.get(memory.id, 0.0)),
                )
                for memory in ranked[:limit]
            ]

        sorted_neighbors = self._ensure_sorted_neighbors()
        settled_scores: dict[str, float] = {}
        results: list[tuple[str, float, str | None, float]] = []
        serial = 0
        heap: list[tuple[float, float, int, str, str, int, float]] = []

        def push_candidate(
            score: float,
            node_id: str,
            parent_id: str | None,
            neighbor_index: int,
            source_similarity: float,
        ) -> int:
            nonlocal serial
            if score <= 0.0:
                return serial
            heapq.heappush(
                heap,
                (
                    -score,
                    -self._path_weight(query_scores.get(node_id, 0.0)),
                    serial,
                    node_id,
                    parent_id or "",
                    neighbor_index,
                    source_similarity,
                ),
            )
            serial += 1
            return serial

        def push_next_neighbor(parent_id: str, start_index: int) -> None:
            neighbors = sorted_neighbors.get(parent_id, [])
            index = start_index
            while index < len(neighbors):
                neighbor_id, edge_weight = neighbors[index]
                if neighbor_id not in settled_scores:
                    edge_similarity = self._path_weight(edge_weight)
                    push_candidate(
                        edge_similarity,
                        neighbor_id,
                        parent_id,
                        index,
                        edge_similarity,
                    )
                    return
                index += 1

        for memory_id, score in initial_scores.items():
            push_candidate(score, memory_id, None, -1, score)
        heapq.heapify(heap)

        while heap and len(results) < limit:
            negative_score, _, _, node_id, parent_id, neighbor_index, source_similarity = heapq.heappop(heap)
            score = -negative_score
            if parent_id:
                push_next_neighbor(parent_id, neighbor_index + 1)
            if node_id in settled_scores:
                continue
            settled_scores[node_id] = score
            results.append((node_id, score, parent_id or None, source_similarity))
            push_next_neighbor(node_id, 0)

        if len(results) < limit:
            fallback = sorted(
                self._memories,
                key=lambda memory: (
                    self._path_weight(query_scores.get(memory.id, 0.0)),
                    memory.created_at,
                ),
                reverse=True,
            )
            for memory in fallback:
                if memory.id in settled_scores:
                    continue
                similarity = self._path_weight(query_scores.get(memory.id, 0.0))
                results.append((memory.id, similarity, None, similarity))
                if len(results) >= limit:
                    break

        return results

    def _should_merge(
        self,
        left: MemoryRecord,
        right: MemoryRecord,
        semantic_similarity: float,
        lexical_sim: float,
    ) -> bool:
        if semantic_similarity >= self.config.duplicate_threshold:
            return True
        if lexical_sim >= self.config.lexical_duplicate_threshold:
            return True
        left_norm = normalize_text(left.text)
        right_norm = normalize_text(right.text)
        if left_norm in right_norm or right_norm in left_norm:
            return semantic_similarity >= self.config.overlap_threshold
        return False

    def _orthogonalize_texts(self, texts: list[str]) -> str:
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0].strip()
        sentences: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for sentence in split_sentences(text):
                normalized = normalize_text(sentence)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                sentences.append(sentence)
        if not sentences:
            return max(texts, key=len).strip()
        if len(sentences) == 1:
            return sentences[0]
        return "\n".join(f"- {sentence}" for sentence in sentences)

    def _rebuild_relationships(self) -> None:
        # Reason: SIMILAR edges are not persisted (recall reads from the
        # in-memory `_strong_adjacency` dict, which is rebuilt from
        # embeddings via numpy matmul on every _reload_cache). Only NEXT
        # edges need rewiring here.
        memories = self.store.list_memories()
        if not memories:
            return
        timestamp = utc_now()

        for previous, current in zip(memories, memories[1:]):
            self.store.create_next_edge(previous.id, current.id, 1.0, timestamp)


def is_lock_conflict(error: Exception) -> bool:
    message = str(error)
    return "Could not set lock on file" in message


def _write_project_config(project: ProjectContext, config: MemoryConfig) -> None:
    project.config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")


def reembed_project(
    start: Path | None = None,
    *,
    embedder: Embedder | None = None,
    exact: bool = False,
    force: bool = False,
) -> ReembedResult:
    project = load_project(start, exact=exact)
    config = project.config
    previous_backend, previous_model, previous_dimensions = config.stored_embedding_signature()
    current_backend, current_model, current_dimensions = config.desired_embedding_signature()

    if not force and not config.needs_reembed():
        memory_count = 0
        if project.db_path.exists():
            store = GraphStore(project.db_path, previous_dimensions, read_only=True)
            try:
                memory_count = store.count_memories()
            finally:
                store.close()
        return ReembedResult(
            project_root=project.root,
            db_path=project.db_path,
            reembedded=False,
            memory_count=memory_count,
            previous_store_backend=previous_backend,
            previous_store_model=previous_model,
            previous_store_dimensions=previous_dimensions,
            current_store_backend=current_backend,
            current_store_model=current_model,
            current_store_dimensions=current_dimensions,
        )

    if embedder is None:
        embedder = build_embedder(config)

    memories: list[MemoryRecord] = []
    next_edges = []
    if project.db_path.exists():
        source_store = GraphStore(
            project.db_path,
            previous_dimensions,
            read_only=True,
        )
        try:
            memories = source_store.list_memories()
            next_edges = source_store.list_next_edges()
        finally:
            source_store.close()

    temp_db_path = project.db_path.with_name(
        f"{project.db_path.name}.reembed-{uuid.uuid4().hex}.tmp"
    )
    temp_store = GraphStore(temp_db_path, current_dimensions)
    try:
        for memory in memories:
            temp_store.add_memory(
                MemoryRecord(
                    id=memory.id,
                    text=memory.text,
                    created_at=memory.created_at,
                    embedding=embed_document(embedder, memory.text),
                    importance=memory.importance,
                    access_count=memory.access_count,
                    last_accessed=memory.last_accessed,
                )
            )
        timestamp = utc_now()
        if next_edges:
            seen_edges: set[tuple[str, str]] = set()
            for edge in next_edges:
                key = (edge.source_id, edge.target_id)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                temp_store.create_next_edge(edge.source_id, edge.target_id, edge.weight, timestamp)
        else:
            for previous, current in zip(memories, memories[1:]):
                temp_store.create_next_edge(previous.id, current.id, 1.0, timestamp)
    finally:
        temp_store.close()

    backup_path = project.db_path.with_name(
        f"{project.db_path.name}.backup-{uuid.uuid4().hex}"
    )
    replaced_existing = False
    try:
        if project.db_path.exists():
            project.db_path.replace(backup_path)
            replaced_existing = True
        temp_db_path.replace(project.db_path)
    except Exception:
        if temp_db_path.exists():
            temp_db_path.unlink()
        if replaced_existing and backup_path.exists():
            backup_path.replace(project.db_path)
        raise
    else:
        if backup_path.exists():
            backup_path.unlink()

    updated_config = config.with_store_current()
    _write_project_config(project, updated_config)
    cache_prune: FastembedCachePruneResult | None = None
    if current_backend == "fastembed":
        try:
            cache_prune = prune_fastembed_model_cache([current_model])
        except Exception:
            # The store rewrite already succeeded. Cache pruning is best-effort
            # cleanup and should not fail the migration path after the DB has
            # been safely swapped.
            cache_prune = None
    return ReembedResult(
        project_root=project.root,
        db_path=project.db_path,
        reembedded=True,
        memory_count=len(memories),
        previous_store_backend=previous_backend,
        previous_store_model=previous_model,
        previous_store_dimensions=previous_dimensions,
        current_store_backend=current_backend,
        current_store_model=current_model,
        current_store_dimensions=current_dimensions,
        cache_prune=cache_prune,
    )


def open_memory_with_retry(
    start: Path | None = None,
    *,
    embedder: Embedder | None = None,
    exact: bool = False,
    read_only: bool = False,
    attempts: int = 5,
    delay_s: float = 0.15,
) -> AgentMemory:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return AgentMemory.open(
                start,
                embedder=embedder,
                exact=exact,
                read_only=read_only,
            )
        except Exception as exc:
            last_error = exc
            if not is_lock_conflict(exc) or attempt == attempts - 1:
                raise
            time.sleep(delay_s * (attempt + 1))
    assert last_error is not None
    raise last_error
from agent_memory import __display_version__
from agent_memory.integration import refresh_project_integration
