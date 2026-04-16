from __future__ import annotations

from pathlib import Path
import os

from mcp.server.fastmcp import FastMCP

from agent_memory.config import ConfigError, MemoryConfig, load_project
from agent_memory.engine import AgentMemory, open_memory_with_retry
from agent_memory.models import MemoryMetadata
from agent_memory.hooks.common import (
    consolidation_status,
    mark_consolidation_completed,
)
from agent_memory.retrieval_feedback import (
    parse_feedback_assignments,
    record_retrieval_feedback,
)


def _resolve_project_root(project_root: str | None) -> Path | None:
    value = project_root or os.environ.get("AGENT_MEMORY_PROJECT_ROOT")
    if not value:
        return None
    return Path(value).resolve()


def build_server(default_project_root: Path | None = None) -> FastMCP:
    project_hint = default_project_root or _resolve_project_root(None)
    project_scope = (
        f" This server is pinned to the exact project root {project_hint}."
        if project_hint is not None
        else ""
    )
    server = FastMCP(
        "Agent Memory",
        instructions=(
            "Project-scoped local associative memory graph for durable agent memory."
            f"{project_scope} "
            "Recall relevant memories before substantive work. After meaningful steps, persist the user turn, "
            "assistant turn, and 1-3 distilled durable memories with capture_turn. "
            "Recall ranks memories by the highest parent-similarity score from the query root through "
            "memory-memory similarity edges."
        ),
    )

    def _open(project_root: str | None = None, *, read_only: bool = False) -> AgentMemory:
        root = _resolve_project_root(project_root) or project_hint
        if root is None:
            return open_memory_with_retry(read_only=read_only)
        return open_memory_with_retry(root, exact=True, read_only=read_only)

    @server.tool()
    def save_memory(
        text: str,
        title: str,
        kind: str,
        subsystem: str,
        workstream: str,
        environment: str,
        project_root: str | None = None,
    ) -> dict[str, object]:
        """Persist one memory body plus explicit metadata fields."""
        memory = _open(project_root)
        try:
            metadata = MemoryMetadata(
                title=title,
                kind=kind,
                subsystem=subsystem,
                workstream=workstream,
                environment=environment,
            )
            return memory.save(text, metadata=metadata).to_dict()
        finally:
            memory.close()

    @server.tool()
    def edit_memory(
        memory_id: str,
        text: str | None = None,
        title: str | None = None,
        kind: str | None = None,
        subsystem: str | None = None,
        workstream: str | None = None,
        environment: str | None = None,
        project_root: str | None = None,
    ) -> dict[str, object]:
        """Update a memory body and/or metadata fields. Omitted fields retain their current values."""
        memory = _open(project_root)
        try:
            existing = memory.get(memory_id)
            if existing is None:
                raise KeyError(f"Memory {memory_id!r} does not exist.")
            metadata = MemoryMetadata(
                title=title,
                kind=kind,
                subsystem=subsystem,
                workstream=workstream,
                environment=environment,
            )
            resolved_text = text if text is not None else existing.text
            return memory.edit(memory_id, resolved_text, metadata=metadata).to_dict()
        finally:
            memory.close()

    @server.tool()
    def capture_turn(
        user_text: str | None = None,
        assistant_text: str | None = None,
        memories: list[str] | None = None,
        project_root: str | None = None,
    ) -> dict[str, object]:
        """Persist a turn transcript plus any distilled memories in one batch save."""
        memory = _open(project_root)
        try:
            return memory.capture_turn(
                user_text=user_text,
                assistant_text=assistant_text,
                memories=memories,
            ).to_dict()
        finally:
            memory.close()

    @server.tool()
    def recall_memories(
        query: str,
        limit: int = 15,
        project_root: str | None = None,
    ) -> dict[str, object]:
        """Recall the highest-scoring memories relevant to the query."""
        memory = _open(project_root, read_only=True)
        try:
            return memory.recall(query, limit=limit).to_dict()
        finally:
            memory.close()

    @server.tool()
    def memory_feedback(
        event_id: str,
        memory: list[str] | None = None,
        overall: str | None = None,
        why: str | None = None,
        better: str | None = None,
        missing: str | None = None,
        note: str | None = None,
        project_root: str | None = None,
    ) -> dict[str, object]:
        """Record structured feedback for one prompt-time memory injection event."""
        root = _resolve_project_root(project_root) or project_hint
        if root is None:
            raise ConfigError("A project root is required to record retrieval feedback.")
        return record_retrieval_feedback(
            root,
            event_id=event_id,
            overall=overall,
            memory_feedback=parse_feedback_assignments(memory or []),
            why=why,
            better=better,
            missing=missing,
            note=note,
        )

    @server.tool()
    def consolidate_memories(project_root: str | None = None) -> dict[str, object]:
        """Report read-only high-similarity memory clusters without mutating stored memories."""
        memory = _open(project_root, read_only=True)
        try:
            return memory.consolidate().to_dict()
        finally:
            memory.close()

    @server.tool()
    def consolidation_state(project_root: str | None = None) -> dict[str, object]:
        """Return the current daily memory consolidation status for the current project."""
        root = _resolve_project_root(project_root) or project_hint
        if root is None:
            raise ConfigError("A project root is required to inspect consolidation state.")
        return consolidation_status(root)

    @server.tool()
    def complete_memory_consolidation(project_root: str | None = None) -> dict[str, object]:
        """Record today's date as the last completed memory consolidation for the current project."""
        root = _resolve_project_root(project_root) or project_hint
        if root is None:
            raise ConfigError("A project root is required to complete memory consolidation.")
        return mark_consolidation_completed(root)

    @server.tool()
    def memory_stats(project_root: str | None = None) -> dict[str, object]:
        """Return basic graph statistics for the current project."""
        memory = _open(project_root, read_only=True)
        try:
            return memory.stats().to_dict()
        finally:
            memory.close()

    @server.tool()
    def rewire_memory_graph(project_root: str | None = None) -> dict[str, object]:
        """Rebuild local similarity and next edges from existing memory nodes and embeddings."""
        memory = _open(project_root)
        try:
            return memory.rewire().to_dict()
        finally:
            memory.close()

    return server


def serve(project_root: Path | None = None) -> None:
    resolved_root = None
    if project_root is not None:
        resolved_root = load_project(project_root).root
    build_server(resolved_root).run(transport="stdio")


def main() -> None:
    serve()
