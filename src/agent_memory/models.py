from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


CONSOLIDATION_SECTION_NAMES = (
    "clusters",
    "metadata_cleanup",
    "negative_feedback_memories",
    "unretrieved_memories",
)


def consolidation_instructions_dict() -> dict[str, object]:
    return {
        "calling_agent_task": (
            "Complete the consolidation pass, not just the command execution. "
            "Review the report, edit/delete/save selected memories when warranted, "
            "and mark consolidation complete only after that review is done."
        ),
        "prompt": (
            "Review this read-only Agent Memory cleanup worklist. Clean only the "
            "candidates that are redundant, noisy, badly tagged, or low utility; leave "
            "distinct useful memories unchanged."
        ),
        "output_handling": [
            (
                "The top-level `agent-memory consolidate --json` command writes the "
                "full compact worklist to the returned `report_path` and prints a "
                "short run summary."
            ),
            (
                "Read `report_path` before editing memories. If you only need one "
                "section, use `agent-memory consolidate --json --section <name>`."
            ),
            (
                "Do not load every memory body. Use `agent-memory consolidate --json "
                "--group <group_id>` for candidate detail, then `agent-memory show "
                "<memory_id> --json` only for memories you may edit."
            ),
        ],
        "section_actions": {
            "clusters": (
                "Review embedding-similar memories. Merge or delete only when members "
                "are redundant or noisy; keep memories that are independently useful."
            ),
            "metadata_cleanup": (
                "Normalize tag values only when the standalone tag values mean the "
                "same thing. This section does not imply memory-level duplication."
            ),
            "negative_feedback_memories": (
                "Rewrite or delete memories with more than three negative ratings and "
                "zero positive ratings. Editing a memory resets prior feedback counts."
            ),
            "unretrieved_memories": (
                "Review memories with access_count=0 after enough later recall queries. "
                "Improve text/tags or delete only if they are low utility."
            ),
        },
        "commands": {
            "section": "agent-memory consolidate --json --section <section>",
            "group": "agent-memory consolidate --json --group <group_id>",
            "show": "agent-memory show <memory_id> --json",
            "edit": "agent-memory edit <memory_id> --stdin",
            "delete": "agent-memory delete <memory_id> --yes",
            "complete": "agent-memory consolidation-complete --json",
        },
    }


@dataclass(slots=True)
class MemoryMetadata:
    title: str | None = None
    kind: str | None = None
    subsystem: str | None = None
    workstream: str | None = None
    environment: str | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.title,
                self.kind,
                self.subsystem,
                self.workstream,
                self.environment,
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def header_lines(self) -> list[str]:
        lines: list[str] = []
        if self.title:
            lines.append(f"Title: {self.title}")
        if self.kind:
            lines.append(f"Kind: {self.kind}")
        if self.subsystem:
            lines.append(f"Subsystem: {self.subsystem}")
        if self.workstream:
            lines.append(f"Workstream: {self.workstream}")
        if self.environment:
            lines.append(f"Environment: {self.environment}")
        return lines

    def compact_parts(self) -> list[str]:
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.kind:
            parts.append(f"kind: {self.kind}")
        if self.subsystem:
            parts.append(f"subsystem: {self.subsystem}")
        if self.workstream:
            parts.append(f"workstream: {self.workstream}")
        if self.environment:
            parts.append(f"env: {self.environment}")
        return parts


@dataclass(slots=True)
class MemoryRecord:
    id: str
    text: str
    created_at: str
    embedding: list[float]
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def display_text(self) -> str:
        lines = self.metadata.header_lines()
        if self.text:
            if lines:
                lines.append("")
            lines.append(self.text)
        return "\n".join(lines).strip()

    def prompt_text(self) -> str:
        parts = self.metadata.compact_parts()
        cleaned = " ".join(self.text.split())
        if cleaned:
            parts.append(cleaned)
        return " | ".join(part for part in parts if part)


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
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    query_similarity: float = 0.0

    def preview(self, limit: int = 140) -> str:
        text = self.prompt_text()
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1].rstrip()}…"

    def display_text(self) -> str:
        lines = self.metadata.header_lines()
        if self.text:
            if lines:
                lines.append("")
            lines.append(self.text)
        return "\n".join(lines).strip()

    def prompt_text(self) -> str:
        parts = self.metadata.compact_parts()
        cleaned = " ".join(self.text.split())
        if cleaned:
            parts.append(cleaned)
        return " | ".join(part for part in parts if part)

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
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)

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
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)

    def preview(self, limit: int = 140) -> str:
        parts = self.metadata.compact_parts()
        cleaned = " ".join(self.text.split())
        if cleaned:
            parts.append(cleaned)
        text = " | ".join(part for part in parts if part)
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1].rstrip()}…"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["preview"] = self.preview()
        return payload

    def to_compact_dict(self, *, include_preview: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "memory_id": self.memory_id,
            "created_at": self.created_at,
            "title": self.metadata.title,
            "kind": self.metadata.kind,
            "subsystem": self.metadata.subsystem,
            "workstream": self.metadata.workstream,
            "environment": self.metadata.environment,
        }
        if include_preview:
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
    reason: str = "embedding_similarity_cluster"
    recommended_action: str = "merge_or_delete_if_not_independently_useful"

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "seed_memory_ids": self.seed_memory_ids,
            "member_ids": self.member_ids,
            "members": [member.to_dict() for member in self.members],
            "pair_edges": [edge.to_dict() for edge in self.pair_edges],
            "average_similarity": self.average_similarity,
            "max_similarity": self.max_similarity,
        }

    def to_summary_dict(
        self,
        *,
        sample_member_limit: int = 5,
        include_preview: bool = False,
    ) -> dict[str, object]:
        return {
            "group_id": self.cluster_id,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "member_count": len(self.member_ids),
            "member_ids": self.member_ids,
            "members": [
                member.to_compact_dict(include_preview=include_preview)
                for member in self.members[:sample_member_limit]
            ],
            "average_similarity": self.average_similarity,
            "max_similarity": self.max_similarity,
            "pair_edge_count": len(self.pair_edges),
        }


@dataclass(slots=True)
class ConsolidationCandidateGroup:
    group_id: str
    reason: str
    member_ids: list[str]
    members: list[ConsolidationClusterMember]
    recommended_action: str
    signals: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "reason": self.reason,
            "member_ids": self.member_ids,
            "members": [member.to_dict() for member in self.members],
            "recommended_action": self.recommended_action,
            "signals": self.signals,
        }

    def to_summary_dict(
        self,
        *,
        member_id_limit: int = 0,
        sample_member_limit: int = 0,
        include_signal_memory_ids: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "group_id": self.group_id,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "member_count": len(self.member_ids),
            "signals": _compact_consolidation_signals(
                self.signals,
                include_memory_ids=include_signal_memory_ids,
            ),
        }
        if member_id_limit > 0:
            payload["member_ids_preview"] = self.member_ids[:member_id_limit]
            payload["omitted_member_count"] = max(0, len(self.member_ids) - member_id_limit)
        if sample_member_limit > 0:
            payload["sample_members"] = [
                {
                    "memory_id": member.memory_id,
                    "created_at": member.created_at,
                    "metadata": member.metadata.to_dict(),
                    "preview": member.preview(),
                }
                for member in self.members[:sample_member_limit]
            ]
        return payload

    def to_metadata_cleanup_dict(self) -> dict[str, object]:
        variants = self.signals.get("variants")
        if not isinstance(variants, list):
            variants = []
        values: list[dict[str, object]] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            value = variant.get("value")
            count = variant.get("count")
            if isinstance(value, str) and isinstance(count, int):
                values.append({"value": value, "count": count})
        canonical = None
        if values:
            canonical = sorted(values, key=lambda item: (-int(item["count"]), str(item["value"])))[0]["value"]
        return {
            "group_id": self.group_id,
            "field": self.signals.get("field"),
            "normalized_tag": self.signals.get("normalized_tag"),
            "recommended_action": "normalize_if_tags_mean_the_same_standalone",
            "suggested_canonical_value": canonical,
            "value_count": len(values),
            "affected_memory_count": len(self.member_ids),
            "values": values,
        }


@dataclass(slots=True)
class ConsolidationFeedbackCandidate:
    memory_id: str
    created_at: str
    metadata: MemoryMetadata
    preview: str
    label_counts: dict[str, int]
    negative_count: int
    positive_count: int

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "created_at": self.created_at,
            "title": self.metadata.title,
            "kind": self.metadata.kind,
            "subsystem": self.metadata.subsystem,
            "workstream": self.metadata.workstream,
            "environment": self.metadata.environment,
            "preview": self.preview,
            "label_counts": self.label_counts,
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
            "recommended_action": "rewrite_or_delete_then_feedback_resets_on_edit",
        }


@dataclass(slots=True)
class ConsolidationUnretrievedCandidate:
    memory_id: str
    created_at: str
    metadata: MemoryMetadata
    preview: str
    access_count: int
    queries_since_created: int

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "created_at": self.created_at,
            "title": self.metadata.title,
            "kind": self.metadata.kind,
            "subsystem": self.metadata.subsystem,
            "workstream": self.metadata.workstream,
            "environment": self.metadata.environment,
            "preview": self.preview,
            "access_count": self.access_count,
            "queries_since_created": self.queries_since_created,
            "recommended_action": "review_for_low_utility_or_bad_tags",
        }


@dataclass(slots=True)
class ConsolidationReport:
    threshold: float
    clusters: list[ConsolidationCluster]
    total_memories: int
    clustered_memory_count: int
    candidate_pair_count: int
    generated_at: str
    metadata_cleanup: list[ConsolidationCandidateGroup] = field(default_factory=list)
    negative_feedback_memories: list[ConsolidationFeedbackCandidate] = field(default_factory=list)
    unretrieved_memories: list[ConsolidationUnretrievedCandidate] = field(default_factory=list)

    def candidate_counts(self) -> dict[str, int]:
        return {
            "clusters": len(self.clusters),
            "metadata_cleanup": len(self.metadata_cleanup),
            "negative_feedback_memories": len(self.negative_feedback_memories),
            "unretrieved_memories": len(self.unretrieved_memories),
        }

    def to_dict(self) -> dict[str, object]:
        return self.to_summary_dict()

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "instructions": consolidation_instructions_dict(),
            "threshold": self.threshold,
            "total_memories": self.total_memories,
            "clustered_memory_count": self.clustered_memory_count,
            "candidate_pair_count": self.candidate_pair_count,
            "generated_at": self.generated_at,
            "candidate_counts": self.candidate_counts(),
            "unretrieved_policy": {
                "minimum_queries_since_created": 1000,
                "description": (
                    "Only memories with access_count=0 and at least this many later "
                    "recall queries are surfaced. The count intentionally includes "
                    "both direct recall calls and prompt-injection recall calls."
                ),
            },
            "clusters": [
                cluster.to_summary_dict()
                for cluster in self.clusters
            ],
            "metadata_cleanup": [
                group.to_metadata_cleanup_dict()
                for group in self.metadata_cleanup
            ],
            "negative_feedback_memories": [
                candidate.to_summary_dict()
                for candidate in self.negative_feedback_memories
            ],
            "unretrieved_memories": [
                candidate.to_summary_dict()
                for candidate in self.unretrieved_memories
            ],
        }

    def section_detail_dict(self, section_name: str) -> dict[str, object] | None:
        if section_name == "clusters":
            return {
                "instructions": consolidation_instructions_dict(),
                "section": section_name,
                "group_count": len(self.clusters),
                "groups": [
                    cluster.to_summary_dict()
                    for cluster in self.clusters
                ],
            }
        if section_name == "metadata_cleanup":
            return {
                "instructions": consolidation_instructions_dict(),
                "section": section_name,
                "group_count": len(self.metadata_cleanup),
                "groups": [
                    group.to_metadata_cleanup_dict()
                    for group in self.metadata_cleanup
                ],
            }
        if section_name == "negative_feedback_memories":
            return {
                "instructions": consolidation_instructions_dict(),
                "section": section_name,
                "memory_count": len(self.negative_feedback_memories),
                "memories": [
                    candidate.to_summary_dict()
                    for candidate in self.negative_feedback_memories
                ],
            }
        if section_name == "unretrieved_memories":
            return {
                "instructions": consolidation_instructions_dict(),
                "section": section_name,
                "memory_count": len(self.unretrieved_memories),
                "memories": [
                    candidate.to_summary_dict()
                    for candidate in self.unretrieved_memories
                ],
            }
        return None

    def group_detail_dict(self, group_id: str) -> dict[str, object] | None:
        for cluster in self.clusters:
            if cluster.cluster_id != group_id:
                continue
            payload = cluster.to_summary_dict(
                sample_member_limit=10,
                include_preview=True,
            )
            payload["instructions"] = consolidation_instructions_dict()
            payload["section"] = "clusters"
            payload["pair_edges"] = [edge.to_dict() for edge in cluster.pair_edges]
            return payload
        for group in self.metadata_cleanup:
            if group.group_id == group_id:
                payload = group.to_metadata_cleanup_dict()
                payload["instructions"] = consolidation_instructions_dict()
                payload["section"] = "metadata_cleanup"
                return payload
        return None


def _compact_consolidation_signals(
    value: object,
    *,
    include_memory_ids: bool = False,
) -> object:
    if isinstance(value, dict):
        compacted: dict[str, object] = {}
        for key, item in value.items():
            if key == "memory_ids" and isinstance(item, list):
                if include_memory_ids:
                    compacted["memory_ids"] = item
                else:
                    compacted["memory_id_count"] = len(item)
                continue
            compacted[key] = _compact_consolidation_signals(
                item,
                include_memory_ids=include_memory_ids,
            )
        return compacted
    if isinstance(value, list):
        return [
            _compact_consolidation_signals(
                item,
                include_memory_ids=include_memory_ids,
            )
            for item in value
        ]
    return value


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
