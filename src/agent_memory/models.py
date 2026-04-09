from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class MemoryRecord:
    id: str
    text: str
    created_at: str
    embedding: list[float]
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SimilarityEdge:
    source_id: str
    target_id: str
    weight: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MemoryHit:
    memory_id: str
    text: str
    score: float
    created_at: str
    query_similarity: float = 0.0

    def preview(self, limit: int = 140) -> str:
        text = " ".join(self.text.split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1].rstrip()}…"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["preview"] = self.preview()
        return payload


@dataclass(slots=True)
class MemoryCluster:
    cluster_id: str
    score: float
    seed_ids: list[str]
    memory_ids: list[str]
    hits: list[MemoryHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "score": self.score,
            "seed_ids": self.seed_ids,
            "memory_ids": self.memory_ids,
            "hits": [hit.to_dict() for hit in self.hits],
        }


@dataclass(slots=True)
class SaveResult:
    memory_id: str
    created_at: str
    connected_neighbors: list[dict[str, float]]
    total_memories: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SaveManyResult:
    saved: list[SaveResult]
    total_memories: int

    def to_dict(self) -> dict[str, object]:
        return {
            "saved": [item.to_dict() for item in self.saved],
            "total_memories": self.total_memories,
        }


@dataclass(slots=True)
class ConsolidationClusterMember:
    memory_id: str
    text: str
    created_at: str

    def preview(self, limit: int = 140) -> str:
        text = " ".join(self.text.split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1].rstrip()}…"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["preview"] = self.preview()
        return payload


@dataclass(slots=True)
class ConsolidationClusterEdge:
    source_id: str
    target_id: str
    similarity: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ConsolidationCluster:
    cluster_id: str
    seed_memory_ids: list[str]
    member_ids: list[str]
    members: list[ConsolidationClusterMember]
    pair_edges: list[ConsolidationClusterEdge]
    average_similarity: float
    max_similarity: float

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "seed_memory_ids": self.seed_memory_ids,
            "member_ids": self.member_ids,
            "members": [member.to_dict() for member in self.members],
            "pair_edges": [edge.to_dict() for edge in self.pair_edges],
            "average_similarity": self.average_similarity,
            "max_similarity": self.max_similarity,
        }


@dataclass(slots=True)
class ConsolidationReport:
    threshold: float
    clusters: list[ConsolidationCluster]
    total_memories: int
    clustered_memory_count: int
    candidate_pair_count: int
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "total_memories": self.total_memories,
            "clustered_memory_count": self.clustered_memory_count,
            "candidate_pair_count": self.candidate_pair_count,
            "generated_at": self.generated_at,
        }


@dataclass(slots=True)
class MemoryStats:
    project_root: Path
    db_path: Path
    memory_count: int
    similarity_edge_count: int
    next_edge_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "db_path": str(self.db_path),
            "memory_count": self.memory_count,
            "similarity_edge_count": self.similarity_edge_count,
            "next_edge_count": self.next_edge_count,
        }
