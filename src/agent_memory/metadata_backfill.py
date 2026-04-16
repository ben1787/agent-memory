from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent_memory.config import load_project
from agent_memory.engine import open_memory_with_retry, reembed_project
from agent_memory.memory_metadata import merge_metadata
from agent_memory.metadata_store import METADATA_FILENAME, MemoryMetadataStore
from agent_memory.models import MemoryMetadata, MemoryRecord


_LABEL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^user preference(?:\s*\([^)]*\))?:\s*", re.IGNORECASE), "preference"),
    (re.compile(r"^project decision(?:\s*\([^)]*\))?:\s*", re.IGNORECASE), "architecture"),
    (re.compile(r"^project invariant(?:\s*\([^)]*\))?:\s*", re.IGNORECASE), "architecture"),
    (re.compile(r"^project rule(?:\s*\([^)]*\))?:\s*", re.IGNORECASE), "workflow"),
]

_KIND_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("preference", ("user preference", "user wants", "preference", "explicit correction")),
    ("testing", ("pytest", "test ", "tests ", "parity rig", "replay", "fixture", "assert ")),
    ("incident", ("incident", "outage", "alarm", "failing", "error", "latency spike", "timeout")),
    ("architecture", ("source of truth", "project decision", "project invariant", "architecture", "authoritative")),
    ("workflow", ("never commit", "push directly", "release flow", "merge", "branch", "workflow")),
]

_SUBSYSTEM_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("AgentMemory", ("agent-memory", "agent memory")),
    ("Harbour", ("harbour",)),
    ("CodePipeline", ("codepipeline",)),
    ("CodeBuild", ("codebuild", "buildspec")),
    ("bi-python", ("bi-python", "bi python")),
    ("Nexus", ("nexus",)),
    ("FactorRepository", ("factor repository", "factor_repository")),
    ("Postgres", ("postgres", "psql", "jsonb", "dbo.", "rds")),
    ("DynamoDB", ("dynamodb",)),
    ("EC2", ("ec2",)),
    ("OpenAI", ("openai", "tools.", "tool schema")),
    ("Claude", ("claude code", "claude")),
    ("Codex", ("codex",)),
    ("Sessions", ("session_store", "_sessions", "session_id")),
    ("MCP", ("mcp",)),
    ("ParityRig", ("parity rig", "graphite_tests", "replay_captured")),
]

_WORKSTREAM_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("prompt_hooks", ("userpromptsubmit", "hooks must emit json", "additionalcontext", "hooks.json")),
    ("session_ids", ("session_store", "_sessions", "session_id", "strict-session")),
    ("tool_schema", ("tool schema", "tools.", "visible tool count", "json schema")),
    ("endpoint_validation", ("request validation", "/mcp/", "initialize", "hello tool", "endpoint")),
    ("prod_pipeline", ("codepipeline", "codebuild", "buildspec", "superseded", "sts token")),
    ("parity_testing", ("parity rig", "replay", "graphite_tests", "pytest")),
    ("database_schema", ("jsonb", "dbo.", "column is", "table", "payload_hash")),
    ("deploy_workflow", ("merge", "release flow", "push directly", "branch", "deploy")),
    ("runtime_behavior", ("runtime behavior", "localhost", "cold-path", "source of truth")),
    ("optimizer_restart", ("optimizer", "restart race")),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(slots=True)
class MetadataBackfillResult:
    total_memories: int
    candidate_memories: int
    updated_memories: int
    overwrite: bool
    reembedded: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "total_memories": self.total_memories,
            "candidate_memories": self.candidate_memories,
            "updated_memories": self.updated_memories,
            "overwrite": self.overwrite,
            "reembedded": self.reembedded,
        }


def review_memories_with_codex(
    records: list[MemoryRecord],
    *,
    model: str = "gpt-5.4-mini",
) -> list[tuple[str, MemoryMetadata]]:
    if not records:
        return []

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["memories"],
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "memory_id",
                        "title",
                        "kind",
                        "subsystem",
                        "workstream",
                        "environment",
                    ],
                    "properties": {
                        "memory_id": {"type": "string"},
                        "title": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "preference",
                                "workflow",
                                "operational",
                                "incident",
                                "architecture",
                                "testing",
                            ],
                        },
                        "subsystem": {"type": "string"},
                        "workstream": {"type": "string"},
                        "environment": {
                            "type": "string",
                            "enum": ["local", "dev", "qa", "prod", "unknown"],
                        },
                    },
                },
            },
        },
    }
    prompt = _build_codex_prompt(records)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as schema_file:
        json.dump(schema, schema_file)
        schema_path = Path(schema_file.name)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as output_file:
        output_path = Path(output_file.name)
    codex_home = Path(tempfile.mkdtemp())
    try:
        auth_source = Path.home() / ".codex" / "auth.json"
        if auth_source.exists():
            shutil.copy2(auth_source, codex_home / "auth.json")
        env = dict(os.environ)
        env["CODEX_HOME"] = str(codex_home)

        result = subprocess.run(
            [
                "codex",
                "exec",
                "-C",
                "/tmp",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "-m",
                model,
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
            ],
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if not output_path.exists():
            raise RuntimeError(
                "Codex metadata review did not produce an output file."
                + _format_subprocess_error(result)
            )
        try:
            reviewed = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Codex metadata review returned invalid JSON."
                + _format_subprocess_error(result)
            ) from exc
        if not isinstance(reviewed, dict):
            raise RuntimeError("Codex metadata review returned a non-object payload.")
        reviewed_items = reviewed.get("memories")
        if not isinstance(reviewed_items, list):
            raise RuntimeError("Codex metadata review returned no `memories` list.")
        reviewed_by_id: dict[str, MemoryMetadata] = {}
        for item in reviewed_items:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("memory_id")
            if not isinstance(memory_id, str) or not memory_id:
                continue
            reviewed_by_id[memory_id] = MemoryMetadata(
                title=_clean_scalar(item.get("title")) or "Untitled memory",
                kind=_clean_scalar(item.get("kind")) or "operational",
                subsystem=_clean_scalar(item.get("subsystem")) or "General",
                workstream=_clean_scalar(item.get("workstream")) or "general",
                environment=_clean_scalar(item.get("environment")) or "unknown",
            )
        missing = [record.id for record in records if record.id not in reviewed_by_id]
        if missing:
            raise RuntimeError(
                "Codex metadata review omitted memories: " + ", ".join(missing[:10])
            )
        return [(record.id, reviewed_by_id[record.id]) for record in records]
    finally:
        schema_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        shutil.rmtree(codex_home, ignore_errors=True)


def derive_metadata_from_text(
    text: str,
    *,
    fallback: MemoryMetadata | None = None,
) -> MemoryMetadata:
    cleaned = " ".join(text.split())
    label_kind, remainder = _strip_label(cleaned)
    title = _derive_title(remainder or cleaned)
    kind = label_kind or _match_rule(_KIND_RULES, cleaned) or "operational"
    subsystem = _match_rule(_SUBSYSTEM_RULES, cleaned) or "General"
    workstream = _match_rule(_WORKSTREAM_RULES, cleaned) or "general"
    environment = _derive_environment(cleaned)
    derived = MemoryMetadata(
        title=title,
        kind=kind,
        subsystem=subsystem,
        workstream=workstream,
        environment=environment,
    )
    return merge_metadata(derived, fallback)


def backfill_project_metadata(
    project_root: Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    reviewer: str = "codex",
    batch_size: int = 1,
    model: str = "gpt-5.4-mini",
) -> MetadataBackfillResult:
    project = load_project(project_root, exact=True)
    memory = open_memory_with_retry(project.root, exact=True, read_only=True)
    try:
        records = memory.list_all()
    finally:
        memory.close()

    selected = records if limit is None else records[:limit]
    review_records: list[MemoryRecord] = []
    heuristic_candidates: list[tuple[str, MemoryMetadata]] = []
    for record in selected:
        if not overwrite and not record.metadata.is_empty():
            continue
        if reviewer == "heuristic":
            derived = derive_metadata_from_text(
                record.text,
                fallback=None if overwrite else record.metadata,
            )
            if derived.to_dict() == record.metadata.to_dict():
                continue
            heuristic_candidates.append((record.id, derived))
        else:
            review_records.append(record)

    candidate_count = len(review_records) if reviewer == "codex" else len(heuristic_candidates)
    if candidate_count == 0:
        return MetadataBackfillResult(
            total_memories=len(records),
            candidate_memories=candidate_count,
            updated_memories=0,
            overwrite=overwrite,
            reembedded=False,
        )

    store = MemoryMetadataStore(project.db_path.parent / METADATA_FILENAME, read_only=False)
    updated_count = 0
    if reviewer == "heuristic":
        for memory_id, metadata in heuristic_candidates:
            store.upsert(memory_id, metadata)
            updated_count += 1
    else:
        for batch in _batched(review_records, max(1, batch_size)):
            reviewed = review_memories_with_codex(batch, model=model)
            for memory_id, metadata in reviewed:
                original = next(record for record in batch if record.id == memory_id)
                merged = merge_metadata(metadata, None if overwrite else original.metadata)
                if merged.to_dict() == original.metadata.to_dict():
                    continue
                store.upsert(memory_id, merged)
                updated_count += 1

    if updated_count == 0:
        return MetadataBackfillResult(
            total_memories=len(records),
            candidate_memories=candidate_count,
            updated_memories=0,
            overwrite=overwrite,
            reembedded=False,
        )

    reembed_project(project.root, exact=True, force=True)
    return MetadataBackfillResult(
        total_memories=len(records),
        candidate_memories=candidate_count,
        updated_memories=updated_count,
        overwrite=overwrite,
        reembedded=True,
    )


def _strip_label(text: str) -> tuple[str | None, str]:
    for pattern, kind in _LABEL_PATTERNS:
        match = pattern.match(text)
        if match:
            return kind, text[match.end() :].strip()
    return None, text


def _derive_title(text: str) -> str:
    for chunk in _SENTENCE_SPLIT.split(text):
        candidate = " ".join(chunk.split()).strip(" -:;,.")
        if candidate:
            return _trim_title(candidate)
    return "Untitled memory"


def _trim_title(text: str, *, max_words: int = 12, max_chars: int = 88) -> str:
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if len(text) <= max_chars:
        return text.rstrip(".")
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return (clipped or text[:max_chars]).rstrip(".")


def _match_rule(rules: list[tuple[str, tuple[str, ...]]], text: str) -> str | None:
    lowered = text.casefold()
    for value, needles in rules:
        if any(needle.casefold() in lowered for needle in needles):
            return value
    return None


def _derive_environment(text: str) -> str:
    lowered = text.casefold()
    matches: set[str] = set()
    if any(token in lowered for token in (" prod ", "production", "prod-", " prod.", "prod/")):
        matches.add("prod")
    if any(token in lowered for token in (" dev ", "development", "dev-", " dev.", "dev/")):
        matches.add("dev")
    if any(token in lowered for token in (" qa ", "staging", "stage-", "stage ", "qa-")):
        matches.add("qa")
    if any(token in lowered for token in (" localhost", " local ", "worktree", "on laptop")):
        matches.add("local")
    if len(matches) == 1:
        return next(iter(matches))
    return "unknown"


def _build_codex_prompt(records: list[MemoryRecord]) -> str:
    payload = [
        {
            "memory_id": record.id,
            "text": record.text,
        }
        for record in records
    ]
    return (
        "Return only JSON matching the schema.\n"
        "Return an object with a top-level `memories` array.\n"
        "Review each memory independently from its actual text.\n"
        "Do not use keyword lookup shortcuts or broad heuristics detached from the memory body.\n"
        "For each memory, produce:\n"
        "- title: short durable title\n"
        "- kind: one of preference, workflow, operational, incident, architecture, testing\n"
        "- subsystem: specific component/system/repo area\n"
        "- workstream: narrow topic within that subsystem\n"
        "- environment: local, dev, qa, prod, or unknown if mixed/unclear\n"
        "Prefer exact existing product/component names when obvious from the text.\n"
        "If a memory spans multiple environments or the environment is not explicit, use unknown.\n\n"
        "Memories:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _clean_scalar(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _format_subprocess_error(result: subprocess.CompletedProcess[str]) -> str:
    stderr = " ".join((result.stderr or "").split())
    stdout = " ".join((result.stdout or "").split())
    details = stderr or stdout
    if not details:
        return ""
    if len(details) > 280:
        details = details[:277] + "..."
    return f" codex exited {result.returncode}: {details}"


def _batched(records: list[MemoryRecord], size: int) -> list[list[MemoryRecord]]:
    return [records[index : index + size] for index in range(0, len(records), size)]
