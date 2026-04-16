from __future__ import annotations

import json
from pathlib import Path

from agent_memory.models import MemoryMetadata


METADATA_FILENAME = "memory-metadata.json"


class MemoryMetadataStore:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> dict[str, MemoryMetadata]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        metadata_by_id: dict[str, MemoryMetadata] = {}
        for memory_id, raw in payload.items():
            if not isinstance(memory_id, str) or not isinstance(raw, dict):
                continue
            metadata_by_id[memory_id] = MemoryMetadata(
                title=_clean_optional(raw.get("title")),
                kind=_clean_optional(raw.get("kind")),
                subsystem=_clean_optional(raw.get("subsystem")),
                workstream=_clean_optional(raw.get("workstream")),
                environment=_clean_optional(raw.get("environment")),
            )
        return metadata_by_id

    def upsert(self, memory_id: str, metadata: MemoryMetadata) -> None:
        if self.read_only:
            raise RuntimeError("Cannot write metadata in read-only mode.")
        payload = self._load_raw()
        if metadata.is_empty():
            payload.pop(memory_id, None)
        else:
            payload[memory_id] = metadata.to_dict()
        self._write_raw(payload)

    def delete(self, memory_id: str) -> None:
        if self.read_only:
            raise RuntimeError("Cannot delete metadata in read-only mode.")
        payload = self._load_raw()
        if memory_id in payload:
            payload.pop(memory_id, None)
            self._write_raw(payload)

    def _load_raw(self) -> dict[str, object]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return dict(payload)

    def _write_raw(self, payload: dict[str, object]) -> None:
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self.path.write_text(serialized, encoding="utf-8")


def _clean_optional(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None
