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
    embed_documents,
    embed_query,
    prune_fastembed_model_cache,
)
from agent_memory.models import (
    ConsolidationCandidateGroup,
    ConsolidationCluster,
    ConsolidationClusterEdge,
    ConsolidationFeedbackCandidate,
    ConsolidationClusterMember,
    ConsolidationReport,
    ConsolidationUnretrievedCandidate,
    MemoryCluster,
    MemoryHit,
    MemoryMetadata,
    MemoryRecord,
    SaveManyResult,
    MemoryStats,
    SaveResult,
)
from agent_memory.memory_metadata import (
    compose_embedding_text,
    merge_metadata,
)
from agent_memory.metadata_store import METADATA_FILENAME, MemoryMetadataStore
from agent_memory.operations_log import (
    OP_DELETE,
    OP_EDIT,
    OP_SAVE,
    OPERATIONS_LOG_FILENAME,
    LogEntry,
    OperationsLog,
)
from agent_memory.query_log import QUERY_LOG_FILENAME, log_query
from agent_memory.retrieval_feedback import (
    NEGATIVE_MEMORY_FEEDBACK_LABELS,
    POSITIVE_MEMORY_FEEDBACK_LABELS,
    memory_feedback_label_counts,
    reset_memory_feedback,
)
from agent_memory.store import GraphStore
from agent_memory.write_lock import ProjectWriteLock


@dataclass
class EditOutcome:
    """Per-row result for `AgentMemory.edit_many`."""

    memory_id: str
    status: str  # "changed" | "unchanged" | "failed"
    record: "MemoryRecord | None" = None
    error: str | None = None


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
STDIN_SAVE_HINT = (
    " If this text came through a shell or agent command, prefer piping it to the CLI "
    "`save`/`edit` stdin mode with a quoted heredoc so quotes, backticks, and newlines "
    "cannot be rewritten before Agent Memory sees them."
)
CONSOLIDATION_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "agent",
        "agents",
        "already",
        "before",
        "branch",
        "current",
        "data",
        "debug",
        "dev",
        "does",
        "done",
        "field",
        "fields",
        "file",
        "from",
        "local",
        "memory",
        "needs",
        "path",
        "paths",
        "porter",
        "repo",
        "route",
        "server",
        "shared",
        "should",
        "status",
        "still",
        "tool",
        "tools",
        "uses",
        "with",
        "workflow",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_to_payload(memory: MemoryRecord) -> dict[str, object]:
    """Snapshot a MemoryRecord for the operations log (used by undo)."""
    return {
        "id": memory.id,
        "text": memory.text,
        "metadata": memory.metadata.to_dict(),
        "created_at": memory.created_at,
        "embedding": list(memory.embedding),
        "importance": memory.importance,
        "access_count": memory.access_count,
        "last_accessed": memory.last_accessed,
    }


def _metadata_from_payload(
    payload: dict[str, object],
    *,
    fallback: MemoryMetadata | None = None,
) -> MemoryMetadata:
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        return fallback or MemoryMetadata()
    return merge_metadata(
        MemoryMetadata(
            title=str(metadata_payload["title"]).strip() or None
            if isinstance(metadata_payload.get("title"), str)
            else None,
            kind=str(metadata_payload["kind"]).strip() or None
            if isinstance(metadata_payload.get("kind"), str)
            else None,
            subsystem=str(metadata_payload["subsystem"]).strip() or None
            if isinstance(metadata_payload.get("subsystem"), str)
            else None,
            workstream=str(metadata_payload["workstream"]).strip() or None
            if isinstance(metadata_payload.get("workstream"), str)
            else None,
            environment=str(metadata_payload["environment"]).strip() or None
            if isinstance(metadata_payload.get("environment"), str)
            else None,
        ),
        fallback,
    )


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def normalize_metadata_value(value: str | None) -> str:
    if not value:
        return ""
    return normalize_text(value)


def compact_metadata_value(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def metadata_variant_key(value: str | None) -> str:
    compact = compact_metadata_value(value)
    if len(compact) > 4 and compact.endswith("s"):
        return compact[:-1]
    return compact


def created_at_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


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
                    "metadata": hit.metadata.to_dict(),
                    "display_text": hit.display_text(),
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
        self._write_lock: ProjectWriteLock | None = None
        if not read_only:
            self._write_lock = ProjectWriteLock(project.root)
            self._write_lock.acquire()
        try:
            self.store = GraphStore(
                project.db_path,
                self.config.embedding_dimensions,
                read_only=read_only,
            )
        except Exception:
            if self._write_lock is not None:
                self._write_lock.release()
            raise
        self.metadata_store = MemoryMetadataStore(
            project.db_path.parent / METADATA_FILENAME,
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
        try:
            self.store.close()
        finally:
            if self._write_lock is not None:
                self._write_lock.release()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.config)
        return self._embedder

    def _save_one(
        self,
        text: str,
        metadata: MemoryMetadata | None = None,
        embedding: list[float] | None = None,
        *,
        record_in_log: bool = True,
    ) -> SaveResult:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Memory text cannot be empty.")
        resolved_metadata = merge_metadata(metadata, None)
        normalized_text = compose_embedding_text(cleaned, resolved_metadata)
        words = word_count(normalized_text)
        if words > self.config.max_memory_words:
            raise ValueError(
                "Memory text is too long. "
                f"Got {words} words, max allowed is {self.config.max_memory_words}. "
                "Save a shorter summary instead."
                + STDIN_SAVE_HINT
            )

        existing = list(self._memories)
        resolved_embedding = embedding or embed_document(self.embedder, normalized_text)
        timestamp = utc_now()
        memory = MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            text=cleaned,
            created_at=timestamp,
            embedding=resolved_embedding,
            metadata=resolved_metadata,
        )
        self.store.add_memory(memory)
        self.metadata_store.upsert(memory.id, resolved_metadata)
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
            metadata=memory.metadata,
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
        metadata: MemoryMetadata | None = None,
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
        resolved_metadata = merge_metadata(metadata, existing.metadata)
        normalized_text = compose_embedding_text(cleaned, resolved_metadata)
        words = word_count(normalized_text)
        if words > self.config.max_memory_words:
            raise ValueError(
                "Memory text is too long. "
                f"Got {words} words, max allowed is {self.config.max_memory_words}."
                + STDIN_SAVE_HINT
            )

        before_payload = _record_to_payload(existing)
        new_embedding = embed_document(self.embedder, normalized_text)
        timestamp = utc_now()
        updated = MemoryRecord(
            id=existing.id,
            text=cleaned,
            created_at=existing.created_at,
            embedding=new_embedding,
            metadata=resolved_metadata,
            importance=existing.importance,
            access_count=existing.access_count,
            last_accessed=timestamp,
        )

        self.store.update_memory(updated)
        self.metadata_store.upsert(existing.id, resolved_metadata)
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
            reset_memory_feedback(
                self.project.root,
                [existing.id],
                reason="memory_edited",
            )
        return updated

    def edit_many(
        self,
        items: list[dict],
        *,
        record_in_log: bool = True,
    ) -> list[EditOutcome]:
        """Apply many edits with one trailing cache reload and one batched embed call.

        Each item: {"id": str, "text": str | None, "metadata": MemoryMetadata | None}.
        `text=None` keeps the existing body; `metadata=None` keeps existing metadata
        (a partial MemoryMetadata merges field-by-field via `merge_metadata`).
        """
        outcomes: list[EditOutcome | None] = [None] * len(items)
        plans: list[tuple[int, MemoryRecord, str, str, MemoryMetadata, dict]] = []

        for index, item in enumerate(items):
            try:
                memory_id = item["id"]
                existing = self.get(memory_id)
                if existing is None:
                    outcomes[index] = EditOutcome(
                        memory_id=memory_id,
                        status="failed",
                        error=f"Memory {memory_id!r} does not exist.",
                    )
                    continue
                provided_text = item.get("text")
                if provided_text is None:
                    cleaned = existing.text
                else:
                    cleaned = provided_text.strip()
                    if not cleaned:
                        outcomes[index] = EditOutcome(
                            memory_id=memory_id,
                            status="failed",
                            error="Memory text cannot be empty.",
                        )
                        continue
                resolved_metadata = merge_metadata(item.get("metadata"), existing.metadata)
                normalized_text = compose_embedding_text(cleaned, resolved_metadata)
                words = word_count(normalized_text)
                if words > self.config.max_memory_words:
                    outcomes[index] = EditOutcome(
                        memory_id=memory_id,
                        status="failed",
                        error=(
                            f"Memory text is too long. Got {words} words, "
                            f"max allowed is {self.config.max_memory_words}."
                        ),
                    )
                    continue
                if (
                    cleaned.strip() == existing.text.strip()
                    and resolved_metadata.to_dict() == existing.metadata.to_dict()
                ):
                    outcomes[index] = EditOutcome(
                        memory_id=memory_id,
                        status="unchanged",
                        record=existing,
                    )
                    continue
                before_payload = _record_to_payload(existing)
                plans.append(
                    (index, existing, cleaned, normalized_text, resolved_metadata, before_payload)
                )
            except Exception as exc:  # pragma: no cover - defensive
                outcomes[index] = EditOutcome(
                    memory_id=str(item.get("id", "")),
                    status="failed",
                    error=str(exc),
                )

        if plans:
            embeddings = embed_documents(
                self.embedder, [normalized for _, _, _, normalized, _, _ in plans]
            )
            timestamp = utc_now()
            for (index, existing, cleaned, _normalized, resolved_metadata, before_payload), embedding in zip(
                plans, embeddings, strict=True
            ):
                updated = MemoryRecord(
                    id=existing.id,
                    text=cleaned,
                    created_at=existing.created_at,
                    embedding=embedding,
                    metadata=resolved_metadata,
                    importance=existing.importance,
                    access_count=existing.access_count,
                    last_accessed=timestamp,
                )
                self.store.update_memory(updated)
                self.metadata_store.upsert(existing.id, resolved_metadata)
                if record_in_log:
                    self.operations_log.record_edit(
                        existing.id,
                        before=before_payload,
                        after=_record_to_payload(updated),
                    )
                outcomes[index] = EditOutcome(
                    memory_id=existing.id, status="changed", record=updated
                )
            self._reload_cache()
            if record_in_log:
                reset_memory_feedback(
                    self.project.root,
                    [existing.id for _, existing, *_ in plans],
                    reason="memory_edited",
                )

        return [outcome for outcome in outcomes if outcome is not None]

    def delete(self, memory_id: str, *, record_in_log: bool = True) -> MemoryRecord:
        """Delete a memory and its incident edges. Returns the deleted record."""
        existing = self.get(memory_id)
        if existing is None:
            raise KeyError(f"Memory {memory_id!r} does not exist.")
        before_payload = _record_to_payload(existing)
        self.store.delete_memory(existing.id)
        self.metadata_store.delete(existing.id)
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
            metadata=_metadata_from_payload(payload, fallback=existing.metadata),
            importance=float(payload.get("importance", existing.importance)),
            access_count=int(payload.get("access_count", existing.access_count)),
            last_accessed=payload.get("last_accessed"),  # type: ignore[arg-type]
        )
        self.store.update_memory(restored)
        self.metadata_store.upsert(existing.id, restored.metadata)
        self._reload_cache()

    def _restore_from_payload(self, payload: dict[str, object], *, record_in_log: bool) -> None:
        """Insert a memory record verbatim (preserving its original id)."""
        memory = MemoryRecord(
            id=str(payload["id"]),
            text=str(payload["text"]),
            created_at=str(payload["created_at"]),
            embedding=[float(value) for value in payload["embedding"]],  # type: ignore[arg-type]
            metadata=_metadata_from_payload(payload),
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
        self.metadata_store.upsert(memory.id, memory.metadata)
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
        metadata: MemoryMetadata | None = None,
        embedding: list[float] | None = None,
    ) -> SaveManyResult: ...

    @overload
    def save(
        self,
        text: list[str],
        metadata: None = None,
        embedding: None = None,
    ) -> SaveManyResult: ...

    def save(
        self,
        text: str | list[str],
        metadata: MemoryMetadata | None = None,
        embedding: list[float] | None = None,
    ) -> SaveManyResult:
        if isinstance(text, str):
            saved = [self._save_one(text, metadata=metadata, embedding=embedding)]
        else:
            if embedding is not None or metadata is not None:
                raise ValueError("Batch save does not support shared metadata or embeddings.")
            saved = [self._save_one(item) for item in text]
        return SaveManyResult(saved=saved, total_memories=len(self._memories))

    def save_many(self, items: list[dict]) -> SaveManyResult:
        """Save many memories in one engine session with batched embedding.

        Each item: {"text": str, "metadata": MemoryMetadata | None}.
        """
        prepared: list[tuple[str, MemoryMetadata, str]] = []
        for index, item in enumerate(items):
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"save_many item #{index} is missing non-empty `text`.")
            metadata = item.get("metadata")
            resolved = merge_metadata(metadata, None)
            cleaned = text.strip()
            normalized = compose_embedding_text(cleaned, resolved)
            words = word_count(normalized)
            if words > self.config.max_memory_words:
                raise ValueError(
                    f"save_many item #{index} text is too long. "
                    f"Got {words} words, max allowed is {self.config.max_memory_words}."
                )
            prepared.append((cleaned, resolved, normalized))

        embeddings = (
            embed_documents(self.embedder, [normalized for _, _, normalized in prepared])
            if prepared
            else []
        )
        saved: list[SaveResult] = []
        for (cleaned, resolved, _normalized), embedding in zip(prepared, embeddings, strict=True):
            saved.append(self._save_one(cleaned, metadata=resolved, embedding=embedding))
        return SaveManyResult(saved=saved, total_memories=len(self._memories))

    def recall_many(
        self,
        queries: list[str],
        limit: int = 15,
    ) -> list["RecallResult"]:
        """Run recall for each query against the same in-memory engine state."""
        return [self.recall(query, limit=limit) for query in queries]

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
            metadata = _metadata_from_payload(record)
            normalized_text = compose_embedding_text(cleaned, metadata)
            resolved_embedding = record.get("embedding")
            if not isinstance(resolved_embedding, list):
                resolved_embedding = embed_document(self.embedder, normalized_text)
            memory = MemoryRecord(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                text=cleaned,
                created_at=utc_now(),
                embedding=resolved_embedding,
                metadata=metadata,
            )
            self.store.add_memory(memory)
            self.metadata_store.upsert(memory.id, metadata)
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
                    metadata=self._memory_by_id[memory_id].metadata,
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
                metadata=memory.metadata,
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

    def _consolidation_member_sort_key(self, memory_id: str) -> tuple[str, str]:
        memory = self._memory_by_id[memory_id]
        return (memory.created_at, memory.id)

    def _consolidation_candidate_group(
        self,
        group_id: str,
        reason: str,
        memory_ids: list[str],
        *,
        recommended_action: str,
        signals: dict[str, object] | None = None,
    ) -> ConsolidationCandidateGroup:
        member_ids = sorted(dict.fromkeys(memory_ids), key=self._consolidation_member_sort_key)
        return ConsolidationCandidateGroup(
            group_id=group_id,
            reason=reason,
            member_ids=member_ids,
            members=[
                ConsolidationClusterMember(
                    memory_id=member_id,
                    text=self._memory_by_id[member_id].text,
                    created_at=self._memory_by_id[member_id].created_at,
                    metadata=self._memory_by_id[member_id].metadata,
                )
                for member_id in member_ids
            ],
            recommended_action=recommended_action,
            signals=signals or {},
        )

    def _metadata_field_value(self, memory: MemoryRecord, field_name: str) -> str:
        value = getattr(memory.metadata, field_name)
        return value.strip() if isinstance(value, str) else ""

    def _metadata_key(self, memory: MemoryRecord) -> tuple[str, str, str, str]:
        return (
            compact_metadata_value(memory.metadata.kind),
            compact_metadata_value(memory.metadata.subsystem),
            compact_metadata_value(memory.metadata.workstream),
            compact_metadata_value(memory.metadata.environment),
        )

    def _title_tokens(self, memory: MemoryRecord) -> set[str]:
        source = " ".join(
            [
                self._metadata_field_value(memory, "title"),
                self._metadata_field_value(memory, "subsystem"),
                self._metadata_field_value(memory, "workstream"),
            ]
        )
        return {
            token
            for token in re.findall(r"[a-z0-9]+", source.lower())
            if len(token) >= 4 and token not in CONSOLIDATION_STOPWORDS
        }

    def _duplicate_candidate_groups(self) -> list[ConsolidationCandidateGroup]:
        exact_text: dict[str, list[str]] = {}
        same_title: dict[str, list[str]] = {}
        title_by_id: dict[str, str] = {}

        for memory in self._memories:
            text_key = normalize_text(memory.text)
            if text_key:
                exact_text.setdefault(text_key, []).append(memory.id)
            title = normalize_metadata_value(memory.metadata.title)
            if title:
                same_title.setdefault(title, []).append(memory.id)
                title_by_id[memory.id] = title

        groups: list[ConsolidationCandidateGroup] = []
        seen_member_sets: set[tuple[str, ...]] = set()

        for text_key, memory_ids in exact_text.items():
            if len(memory_ids) < 2:
                continue
            member_set = tuple(sorted(memory_ids))
            seen_member_sets.add(member_set)
            groups.append(
                self._consolidation_candidate_group(
                    "",
                    "exact_text_duplicate",
                    memory_ids,
                    recommended_action="delete_or_merge_duplicates",
                    signals={
                        "normalized_text_length": len(text_key),
                        "metadata_keys": [
                            list(self._metadata_key(self._memory_by_id[memory_id]))
                            for memory_id in sorted(memory_ids)
                        ],
                    },
                )
            )

        for title_key, memory_ids in same_title.items():
            if len(memory_ids) < 2:
                continue
            member_set = tuple(sorted(memory_ids))
            if member_set in seen_member_sets:
                continue
            seen_member_sets.add(member_set)
            groups.append(
                self._consolidation_candidate_group(
                    "",
                    "same_title",
                    memory_ids,
                    recommended_action="review_for_merge_or_metadata_split",
                    signals={"normalized_title": title_key},
                )
            )

        title_ids = sorted(title_by_id)
        if len(title_ids) >= 2:
            union_find = UnionFind(title_ids)
            pair_scores: dict[tuple[str, str], float] = {}
            for left_index, left_id in enumerate(title_ids):
                left_title = title_by_id[left_id]
                for right_id in title_ids[left_index + 1 :]:
                    right_title = title_by_id[right_id]
                    if left_title == right_title:
                        continue
                    score = lexical_similarity(left_title, right_title)
                    if score >= 0.84:
                        union_find.union(left_id, right_id)
                        pair_scores[(left_id, right_id)] = round(score, 4)

            similar_title_groups: dict[str, list[str]] = {}
            for memory_id in title_ids:
                similar_title_groups.setdefault(union_find.find(memory_id), []).append(memory_id)

            for memory_ids in similar_title_groups.values():
                if len(memory_ids) < 2:
                    continue
                member_set = tuple(sorted(memory_ids))
                if member_set in seen_member_sets:
                    continue
                scores = [
                    score
                    for (left_id, right_id), score in pair_scores.items()
                    if left_id in member_set and right_id in member_set
                ]
                if not scores:
                    continue
                seen_member_sets.add(member_set)
                groups.append(
                    self._consolidation_candidate_group(
                        "",
                        "similar_title",
                        memory_ids,
                        recommended_action="review_for_merge_or_rewrite",
                        signals={
                            "minimum_title_similarity": min(scores),
                            "maximum_title_similarity": max(scores),
                            "titles": {
                                memory_id: self._metadata_field_value(
                                    self._memory_by_id[memory_id], "title"
                                )
                                for memory_id in sorted(memory_ids)
                            },
                        },
                    )
                )

        groups.sort(key=lambda group: (-len(group.member_ids), group.reason, group.member_ids))
        for index, group in enumerate(groups[:25], start=1):
            group.group_id = f"duplicate_{index}"
        return groups[:25]

    def _metadata_variant_candidate_groups(self) -> list[ConsolidationCandidateGroup]:
        groups: list[ConsolidationCandidateGroup] = []
        for field_name in ("kind", "subsystem", "workstream", "environment"):
            values_by_key: dict[str, dict[str, list[str]]] = {}
            for memory in self._memories:
                value = self._metadata_field_value(memory, field_name)
                if value:
                    key = metadata_variant_key(value)
                    if key:
                        values_by_key.setdefault(key, {}).setdefault(value, []).append(memory.id)

            for variant_map in values_by_key.values():
                if len(variant_map) < 2:
                    continue
                variants = sorted(variant_map)
                memory_ids = [
                    memory_id
                    for variant in variants
                    for memory_id in variant_map[variant]
                ]
                groups.append(
                    self._consolidation_candidate_group(
                        "",
                        "metadata_value_variant",
                        memory_ids,
                        recommended_action="normalize_metadata_values",
                        signals={
                            "field": field_name,
                            "normalized_tag": metadata_variant_key(variants[0]),
                            "variants": [
                                {
                                    "value": variant,
                                    "count": len(variant_map[variant]),
                                    "memory_ids": sorted(variant_map[variant]),
                                }
                                for variant in variants
                            ],
                        },
                    )
                )

        groups.sort(
            key=lambda group: (
                str(group.signals.get("field", "")),
                -len(group.member_ids),
                group.member_ids,
            )
        )
        for index, group in enumerate(groups[:50], start=1):
            group.group_id = f"metadata_variant_{index}"
        return groups[:50]

    def _metadata_cohort_candidate_groups(self) -> list[ConsolidationCandidateGroup]:
        cohorts: dict[tuple[str, str, str, str], list[str]] = {}
        for memory in self._memories:
            key = self._metadata_key(memory)
            if not any(key):
                continue
            cohorts.setdefault(key, []).append(memory.id)

        groups: list[ConsolidationCandidateGroup] = []
        for key, memory_ids in cohorts.items():
            if len(memory_ids) < 5:
                continue
            example = self._memory_by_id[memory_ids[0]].metadata
            groups.append(
                self._consolidation_candidate_group(
                    "",
                    "same_metadata_cohort",
                    memory_ids,
                    recommended_action="review_cohort_for_overlap",
                    signals={
                        "normalized_key": list(key),
                        "example_metadata": {
                            "kind": example.kind,
                            "subsystem": example.subsystem,
                            "workstream": example.workstream,
                            "environment": example.environment,
                        },
                    },
                )
            )

        groups.sort(key=lambda group: (-len(group.member_ids), group.member_ids))
        for index, group in enumerate(groups[:25], start=1):
            group.group_id = f"metadata_cohort_{index}"
        return groups[:25]

    def _recent_burst_candidate_groups(self) -> list[ConsolidationCandidateGroup]:
        token_buckets: dict[tuple[str, str], list[str]] = {}
        for memory in self._memories:
            date_key = memory.created_at[:10]
            if len(date_key) != 10:
                continue
            for token in self._title_tokens(memory):
                token_buckets.setdefault((date_key, token), []).append(memory.id)

        burst_sets: dict[tuple[str, ...], dict[str, object]] = {}
        for (date_key, token), memory_ids in token_buckets.items():
            if len(memory_ids) < 4:
                continue
            member_ids = sorted(dict.fromkeys(memory_ids), key=self._consolidation_member_sort_key)
            member_set = tuple(member_ids)
            entry = burst_sets.setdefault(
                member_set,
                {
                    "date": date_key,
                    "tokens": [],
                    "member_ids": member_ids,
                },
            )
            tokens = entry["tokens"]
            assert isinstance(tokens, list)
            tokens.append(token)

        groups: list[ConsolidationCandidateGroup] = []
        for entry in burst_sets.values():
            member_ids = entry["member_ids"]
            tokens = sorted(str(token) for token in entry["tokens"])
            date_key = str(entry["date"])
            assert isinstance(member_ids, list)
            start_at = self._memory_by_id[member_ids[0]].created_at
            end_at = self._memory_by_id[member_ids[-1]].created_at
            duration_hours = None
            start_dt = created_at_datetime(start_at)
            end_dt = created_at_datetime(end_at)
            if start_dt is not None and end_dt is not None:
                duration_hours = round((end_dt - start_dt).total_seconds() / 3600, 2)
            groups.append(
                self._consolidation_candidate_group(
                    "",
                    "same_day_title_token_burst",
                    member_ids,
                    recommended_action="review_burst_for_merge_or_stale_episode_notes",
                    signals={
                        "date": date_key,
                        "primary_token": tokens[0],
                        "tokens": tokens,
                        "start_at": start_at,
                        "end_at": end_at,
                        "duration_hours": duration_hours,
                    },
                )
            )

        groups.sort(
            key=lambda group: (
                -len(group.member_ids),
                str(group.signals.get("date", "")),
                str(group.signals.get("primary_token", "")),
            )
        )
        for index, group in enumerate(groups[:25], start=1):
            group.group_id = f"recent_burst_{index}"
        return groups[:25]

    def _quality_flag_candidate_groups(self) -> list[ConsolidationCandidateGroup]:
        flags: dict[str, list[str]] = {}
        for memory in self._memories:
            text = memory.text
            lowered = text.lower()
            words = word_count(text)
            if words < 12:
                flags.setdefault("very_short", []).append(memory.id)
            if re.search(r"\b(user message|assistant reply|task-notification)\b", lowered):
                flags.setdefault("raw_transcript_marker", []).append(memory.id)
            if "this session is being continued from a previous conversation" in lowered:
                flags.setdefault("session_handoff_blob", []).append(memory.id)
            if re.search(r"https?://github\.com/[^ \n]+/pull/\d+", text):
                flags.setdefault("contains_pr_url", []).append(memory.id)
            if re.search(r"(?<![-\w])[0-9a-f]{7,40}(?![-\w])", lowered):
                flags.setdefault("contains_commit_sha_like_token", []).append(memory.id)
            if re.search(r"\b(?:codex|fix|feat|chore|bugfix)/[A-Za-z0-9._#/-]+", text):
                flags.setdefault("contains_branch_name", []).append(memory.id)
            if re.search(
                r"\b(commit(?:ted)? and push|push and merge|use this worktree|leave .* in ai - reviewing)\b",
                lowered,
            ):
                flags.setdefault("one_off_process_directive", []).append(memory.id)
            has_date = re.search(r"\b(?:20\d\d-\d\d-\d\d|\d{1,2}/\d{1,2}/20\d\d)\b", text)
            has_status_word = re.search(
                r"\b(still|today|yesterday|now|leave|status|verified|reproduces|blocks)\b",
                lowered,
            )
            if has_date and has_status_word:
                flags.setdefault("dated_status_note", []).append(memory.id)

        descriptions = {
            "very_short": "Memory body has fewer than 12 words.",
            "raw_transcript_marker": "Memory body looks like raw transcript or task notification content.",
            "session_handoff_blob": "Memory body looks like a session handoff blob.",
            "contains_pr_url": "Memory body contains a pull-request URL that may be one-off release state.",
            "contains_commit_sha_like_token": "Memory body contains a commit-SHA-like token.",
            "contains_branch_name": "Memory body contains a branch-name-like token.",
            "one_off_process_directive": "Memory body contains a one-off process directive.",
            "dated_status_note": "Memory body combines a date with transient status wording.",
        }

        groups = [
            self._consolidation_candidate_group(
                "",
                flag,
                memory_ids,
                recommended_action="review_flagged_memories",
                signals={"description": descriptions.get(flag, flag), "count": len(memory_ids)},
            )
            for flag, memory_ids in sorted(flags.items())
        ]
        groups.sort(key=lambda group: (group.reason, group.member_ids))
        for index, group in enumerate(groups, start=1):
            group.group_id = f"quality_flag_{index}"
        return groups

    def _negative_feedback_candidates(self) -> list[ConsolidationFeedbackCandidate]:
        candidates: list[ConsolidationFeedbackCandidate] = []
        for memory_id, label_counts in memory_feedback_label_counts(self.project.root).items():
            memory = self._memory_by_id.get(memory_id)
            if memory is None:
                continue
            negative_count = sum(
                label_counts.get(label, 0)
                for label in NEGATIVE_MEMORY_FEEDBACK_LABELS
            )
            positive_count = sum(
                label_counts.get(label, 0)
                for label in POSITIVE_MEMORY_FEEDBACK_LABELS
            )
            if negative_count <= 3 or positive_count > 0:
                continue
            member = ConsolidationClusterMember(
                memory_id=memory.id,
                text=memory.text,
                created_at=memory.created_at,
                metadata=memory.metadata,
            )
            candidates.append(
                ConsolidationFeedbackCandidate(
                    memory_id=memory.id,
                    created_at=memory.created_at,
                    metadata=memory.metadata,
                    preview=member.preview(),
                    label_counts=dict(sorted(label_counts.items())),
                    negative_count=negative_count,
                    positive_count=positive_count,
                )
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate.negative_count,
                candidate.positive_count,
                candidate.created_at,
                candidate.memory_id,
            )
        )
        return candidates

    def _query_timestamps(self) -> list[datetime]:
        if not self._query_log_path.exists():
            return []
        timestamps: list[datetime] = []
        try:
            for raw in self._query_log_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                value = payload.get("ts")
                if not isinstance(value, str):
                    continue
                parsed = created_at_datetime(value)
                if parsed is not None:
                    timestamps.append(parsed)
        except OSError:
            return []
        timestamps.sort()
        return timestamps

    def _unretrieved_memory_candidates(self) -> list[ConsolidationUnretrievedCandidate]:
        query_timestamps = self._query_timestamps()
        if len(query_timestamps) < 1000:
            return []
        candidates: list[ConsolidationUnretrievedCandidate] = []
        for memory in self._memories:
            if memory.access_count != 0:
                continue
            created_at = created_at_datetime(memory.created_at)
            if created_at is None:
                continue
            queries_since_created = sum(
                1 for timestamp in query_timestamps if timestamp > created_at
            )
            if queries_since_created < 1000:
                continue
            member = ConsolidationClusterMember(
                memory_id=memory.id,
                text=memory.text,
                created_at=memory.created_at,
                metadata=memory.metadata,
            )
            candidates.append(
                ConsolidationUnretrievedCandidate(
                    memory_id=memory.id,
                    created_at=memory.created_at,
                    metadata=memory.metadata,
                    preview=member.preview(),
                    access_count=memory.access_count,
                    queries_since_created=queries_since_created,
                )
            )
        candidates.sort(
            key=lambda candidate: (
                -candidate.queries_since_created,
                candidate.created_at,
                candidate.memory_id,
            )
        )
        return candidates[:25]

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
                metadata_cleanup=self._metadata_variant_candidate_groups(),
                negative_feedback_memories=self._negative_feedback_candidates(),
                unretrieved_memories=self._unretrieved_memory_candidates(),
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
                        metadata=self._memory_by_id[member_id].metadata,
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
            metadata_cleanup=self._metadata_variant_candidate_groups(),
            negative_feedback_memories=self._negative_feedback_candidates(),
            unretrieved_memories=self._unretrieved_memory_candidates(),
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
        metadata_by_id = self.metadata_store.load_all()
        hydrated: list[MemoryRecord] = []
        for memory in self.store.list_memories():
            metadata = merge_metadata(metadata_by_id.get(memory.id), None)
            hydrated.append(
                MemoryRecord(
                    id=memory.id,
                    text=memory.text,
                    created_at=memory.created_at,
                    embedding=memory.embedding,
                    metadata=metadata,
                    importance=memory.importance,
                    access_count=memory.access_count,
                    last_accessed=memory.last_accessed,
                )
            )
        self._memories = hydrated
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
    with ProjectWriteLock(project.root):
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
        metadata_by_id: dict[str, MemoryMetadata] = {}
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
            metadata_store = MemoryMetadataStore(project.db_path.parent / METADATA_FILENAME, read_only=True)
            metadata_by_id = metadata_store.load_all()

        temp_db_path = project.db_path.with_name(
            f"{project.db_path.name}.reembed-{uuid.uuid4().hex}.tmp"
        )
        temp_store = GraphStore(temp_db_path, current_dimensions)
        try:
            for memory in memories:
                metadata = merge_metadata(metadata_by_id.get(memory.id), None)
                temp_store.add_memory(
                    MemoryRecord(
                        id=memory.id,
                        text=memory.text,
                        created_at=memory.created_at,
                        embedding=embed_document(
                            embedder,
                            compose_embedding_text(memory.text, metadata),
                        ),
                        metadata=metadata,
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
