from __future__ import annotations

from dataclasses import replace

from agent_memory.models import MemoryMetadata


def copy_metadata(metadata: MemoryMetadata | None) -> MemoryMetadata:
    if metadata is None:
        return MemoryMetadata()
    return replace(metadata)


def compose_embedding_text(body: str, metadata: MemoryMetadata) -> str:
    lines = metadata.header_lines()
    if body:
        if lines:
            lines.append("")
        lines.append(body)
    return "\n".join(lines).strip()


def merge_metadata(
    primary: MemoryMetadata | None,
    secondary: MemoryMetadata | None,
) -> MemoryMetadata:
    if primary is None and secondary is None:
        return MemoryMetadata()
    if primary is None:
        return copy_metadata(secondary)
    if secondary is None:
        return copy_metadata(primary)
    return MemoryMetadata(
        title=primary.title or secondary.title,
        kind=primary.kind or secondary.kind,
        subsystem=primary.subsystem or secondary.subsystem,
        workstream=primary.workstream or secondary.workstream,
        environment=primary.environment or secondary.environment,
    )
