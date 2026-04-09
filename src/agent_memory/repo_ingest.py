from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import os
from pathlib import Path
import re

from agent_memory.embeddings import embed_documents
from agent_memory.engine import open_memory_with_retry, truncate_to_words, word_count


PROSE_EXTENSIONS = {".md", ".txt", ".rst"}
CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".r", ".sh", ".sql"}
STRUCTURED_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".scss", ".css", ".html", ".ini", ".conf"}
INCLUDED_EXTENSIONS = PROSE_EXTENSIONS | CODE_EXTENSIONS | STRUCTURED_EXTENSIONS
EXCLUDED_DIRS = {
    ".agent-memory",
    ".agents",
    ".angular",
    ".claude",
    ".codex",
    ".codex-worktrees",
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "cdk.out",
    "coverage",
    "dist",
    "htmlcov_unit",
    "node_modules",
    "output",
    "test-results",
}
EXCLUDED_FILENAMES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "uv.lock",
    "yarn.lock",
}
PRIORITY_COMPONENTS = [
    "<root>",
    "bi-python",
    "eds-python",
    "eds-frontend",
    "R",
    "wp-frontend-api",
    "office-addin-api",
    "incident-analyzer",
    "bi-python-cdk-stacks",
    "eds-ai-hedge-fund",
    "codebase-graph-viewer",
]
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class ChunkRecord:
    path: Path
    relative_path: str
    component: str
    kind: str
    start_line: int
    end_line: int
    text: str

    def render(self, *, max_words: int) -> str:
        header = (
            f"Source: {self.relative_path}\n"
            f"Kind: {self.kind}\n"
            f"Lines: {self.start_line}-{self.end_line}\n"
        )
        available = max(10, max_words - word_count(header) - 1)
        body = truncate_to_words(self.text.strip(), available)
        return f"{header}\n{body}".strip()


@dataclass(slots=True)
class RepoImportResult:
    project_root: Path
    source_root: Path
    discovered_files: int
    candidate_files: int
    imported_files: int
    imported_memories: int
    skipped_existing_texts: int
    skipped_files: int
    limit_reached: bool
    total_memories: int
    component_counts: dict[str, int]
    skipped_by_reason: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "source_root": str(self.source_root),
            "discovered_files": self.discovered_files,
            "candidate_files": self.candidate_files,
            "imported_files": self.imported_files,
            "imported_memories": self.imported_memories,
            "skipped_existing_texts": self.skipped_existing_texts,
            "skipped_files": self.skipped_files,
            "limit_reached": self.limit_reached,
            "total_memories": self.total_memories,
            "component_counts": self.component_counts,
            "skipped_by_reason": self.skipped_by_reason,
        }


@dataclass(slots=True)
class _TextBlock:
    start_line: int
    end_line: int
    text: str


def _is_doc_like(path: Path) -> bool:
    lowered = path.name.lower()
    if path.suffix.lower() in PROSE_EXTENSIONS:
        return True
    if lowered in {"readme.md", "claude.md", "agents.md"}:
        return True
    if lowered.startswith("adr-"):
        return True
    return "docs" in {part.lower() for part in path.parts}


def _kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if _is_doc_like(path):
        return "doc"
    if suffix in CODE_EXTENSIONS:
        return "code"
    return "config"


def _component_for_relative_path(relative_path: Path) -> str:
    if len(relative_path.parts) <= 1:
        return "<root>"
    return relative_path.parts[0]


def _component_sort_key(name: str) -> tuple[int, str]:
    try:
        return (PRIORITY_COMPONENTS.index(name), name)
    except ValueError:
        return (len(PRIORITY_COMPONENTS), name)


def _file_sort_key(relative_path: Path) -> tuple[int, int, str]:
    name = relative_path.name.lower()
    suffix = relative_path.suffix.lower()
    if name in {"readme.md", "claude.md", "agents.md"} or name.startswith("adr-"):
        priority = 0
    elif _is_doc_like(relative_path):
        priority = 1
    elif suffix in CODE_EXTENSIONS:
        priority = 2
    else:
        priority = 3
    return (priority, len(relative_path.parts), relative_path.as_posix())


def _is_binary_sample(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    non_text = sum(byte < 9 or (13 < byte < 32) for byte in data)
    return (non_text / max(1, len(data))) > 0.2


def _should_include_file(path: Path, *, max_file_bytes: int, skipped_by_reason: dict[str, int]) -> bool:
    suffix = path.suffix.lower()
    if path.name in EXCLUDED_FILENAMES:
        skipped_by_reason["excluded_filename"] += 1
        return False
    if suffix not in INCLUDED_EXTENSIONS:
        skipped_by_reason["unsupported_extension"] += 1
        return False
    try:
        size = path.stat().st_size
    except OSError:
        skipped_by_reason["stat_error"] += 1
        return False
    if size <= 0:
        skipped_by_reason["empty_file"] += 1
        return False
    if size > max_file_bytes:
        skipped_by_reason["too_large"] += 1
        return False
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        skipped_by_reason["read_error"] += 1
        return False
    if _is_binary_sample(sample):
        skipped_by_reason["binary_like"] += 1
        return False
    return True


def _discover_candidate_files(source_root: Path, *, max_file_bytes: int) -> tuple[dict[str, list[Path]], int, dict[str, int]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    skipped_by_reason: dict[str, int] = defaultdict(int)
    discovered_files = 0
    for dirpath, dirnames, filenames in os.walk(source_root):
        dirnames[:] = [
            name for name in dirnames
            if name not in EXCLUDED_DIRS and not name.endswith(".egg-info")
        ]
        current_dir = Path(dirpath)
        for filename in filenames:
            discovered_files += 1
            path = current_dir / filename
            if not _should_include_file(path, max_file_bytes=max_file_bytes, skipped_by_reason=skipped_by_reason):
                continue
            relative_path = path.relative_to(source_root)
            component = _component_for_relative_path(relative_path)
            groups[component].append(path)
    for component, paths in groups.items():
        paths.sort(key=lambda candidate: _file_sort_key(candidate.relative_to(source_root)))
    return groups, discovered_files, dict(skipped_by_reason)


def _split_block_sentences(block: _TextBlock, *, max_words: int) -> list[_TextBlock]:
    sentences = [item.strip() for item in SENTENCE_SPLIT_RE.split(block.text) if item.strip()]
    if len(sentences) < 2:
        return [block]
    chunks: list[_TextBlock] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = "\n".join(current + [sentence]).strip()
        if current and word_count(candidate) > max_words:
            chunks.append(
                _TextBlock(
                    start_line=block.start_line,
                    end_line=block.end_line,
                    text="\n".join(current).strip(),
                )
            )
            current = [sentence]
            continue
        current.append(sentence)
    if current:
        chunks.append(
            _TextBlock(
                start_line=block.start_line,
                end_line=block.end_line,
                text="\n".join(current).strip(),
            )
        )
    return [chunk for chunk in chunks if chunk.text.strip()]


def _split_block_lines(block: _TextBlock, *, max_words: int) -> list[_TextBlock]:
    lines = block.text.splitlines()
    chunks: list[_TextBlock] = []
    current_lines: list[str] = []
    current_start = block.start_line
    current_end = block.start_line
    for offset, line in enumerate(lines):
        candidate_lines = current_lines + [line]
        candidate_text = "\n".join(candidate_lines).strip()
        if current_lines and word_count(candidate_text) > max_words:
            chunks.append(
                _TextBlock(
                    start_line=current_start,
                    end_line=current_end,
                    text="\n".join(current_lines).strip(),
                )
            )
            current_lines = [line]
            current_start = block.start_line + offset
            current_end = block.start_line + offset
            continue
        current_lines = candidate_lines
        current_end = block.start_line + offset
    if current_lines:
        chunks.append(
            _TextBlock(
                start_line=current_start,
                end_line=current_end,
                text="\n".join(current_lines).strip(),
            )
        )
    return [chunk for chunk in chunks if chunk.text.strip()]


def _iter_blocks(text: str) -> list[_TextBlock]:
    lines = text.splitlines()
    blocks: list[_TextBlock] = []
    start_line: int | None = None
    current: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if start_line is None:
                start_line = line_number
            current.append(line.rstrip())
            continue
        if start_line is None:
            continue
        blocks.append(
            _TextBlock(
                start_line=start_line,
                end_line=line_number - 1,
                text="\n".join(current).strip(),
            )
        )
        start_line = None
        current = []
    if start_line is not None and current:
        blocks.append(
            _TextBlock(
                start_line=start_line,
                end_line=len(lines),
                text="\n".join(current).strip(),
            )
        )
    return [block for block in blocks if block.text.strip()]


def _normalize_blocks(path: Path, text: str, *, max_words: int) -> list[_TextBlock]:
    kind = _kind_for_path(path)
    blocks: list[_TextBlock] = []
    for block in _iter_blocks(text):
        if word_count(block.text) <= max_words:
            blocks.append(block)
            continue
        if kind == "doc":
            split_blocks = _split_block_sentences(block, max_words=max_words)
            for item in split_blocks:
                if word_count(item.text) <= max_words:
                    blocks.append(item)
                else:
                    blocks.extend(_split_block_lines(item, max_words=max_words))
            continue
        blocks.extend(_split_block_lines(block, max_words=max_words))
    return [block for block in blocks if block.text.strip()]


def _chunk_file(
    path: Path,
    *,
    source_root: Path,
    max_words: int,
    max_chunks_per_file: int,
) -> list[ChunkRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    # Reserve room for metadata in the final memory payload.
    content_budget = max(20, max_words - 20)
    normalized_blocks = _normalize_blocks(path, text, max_words=content_budget)
    if not normalized_blocks:
        return []
    records: list[ChunkRecord] = []
    current: list[_TextBlock] = []
    for block in normalized_blocks:
        current_text = "\n\n".join(item.text for item in current)
        candidate_text = "\n\n".join([current_text, block.text]).strip() if current else block.text
        if current and word_count(candidate_text) > content_budget:
            records.append(
                ChunkRecord(
                    path=path,
                    relative_path=path.relative_to(source_root).as_posix(),
                    component=_component_for_relative_path(path.relative_to(source_root)),
                    kind=_kind_for_path(path),
                    start_line=current[0].start_line,
                    end_line=current[-1].end_line,
                    text="\n\n".join(item.text for item in current).strip(),
                )
            )
            current = [block]
        else:
            current.append(block)
        if len(records) >= max_chunks_per_file:
            return records[:max_chunks_per_file]
    if current and len(records) < max_chunks_per_file:
        records.append(
            ChunkRecord(
                path=path,
                relative_path=path.relative_to(source_root).as_posix(),
                component=_component_for_relative_path(path.relative_to(source_root)),
                kind=_kind_for_path(path),
                start_line=current[0].start_line,
                end_line=current[-1].end_line,
                text="\n\n".join(item.text for item in current).strip(),
            )
        )
    return records[:max_chunks_per_file]


def import_repo_corpus(
    cwd: Path,
    *,
    source_root: Path | None = None,
    max_memories: int = 3000,
    max_chunks_per_file: int = 6,
    max_file_bytes: int = 512 * 1024,
) -> RepoImportResult:
    memory = open_memory_with_retry(cwd, exact=False, read_only=False)
    try:
        project_root = memory.project.root
        resolved_source_root = (source_root or project_root).resolve()
        if not resolved_source_root.exists():
            raise FileNotFoundError(f"Source root does not exist: {resolved_source_root}")
        groups, discovered_files, skipped_by_reason = _discover_candidate_files(
            resolved_source_root,
            max_file_bytes=max_file_bytes,
        )
        candidate_files = sum(len(paths) for paths in groups.values())
        existing_texts = {item.text for item in memory.list_all()}
        planned_texts: set[str] = set()
        component_counts: dict[str, int] = defaultdict(int)
        imported_files = 0
        skipped_existing_texts = 0
        records: list[dict[str, object]] = []

        ordered_components = sorted(groups.keys(), key=_component_sort_key)
        queues = {component: deque(groups[component]) for component in ordered_components}

        while queues and len(records) < max_memories:
            progressed = False
            for component in list(ordered_components):
                queue = queues.get(component)
                if queue is None:
                    continue
                if not queue:
                    del queues[component]
                    continue
                path = queue.popleft()
                chunk_records = _chunk_file(
                    path,
                    source_root=resolved_source_root,
                    max_words=memory.config.max_memory_words,
                    max_chunks_per_file=max_chunks_per_file,
                )
                file_added = False
                for chunk in chunk_records:
                    rendered = chunk.render(max_words=memory.config.max_memory_words)
                    if rendered in existing_texts or rendered in planned_texts:
                        skipped_existing_texts += 1
                        continue
                    records.append({"text": rendered})
                    planned_texts.add(rendered)
                    component_counts[chunk.component] += 1
                    file_added = True
                    if len(records) >= max_memories:
                        break
                if file_added:
                    imported_files += 1
                progressed = True
                if len(records) >= max_memories:
                    break
            if not progressed:
                break

        if records:
            embeddings = embed_documents(memory.embedder, [str(record["text"]) for record in records])
            for record, embedding in zip(records, embeddings):
                record["embedding"] = embedding
            memory.import_memories(records)
        total_memories = memory.stats().memory_count
    finally:
        memory.close()

    skipped_files = candidate_files - imported_files
    return RepoImportResult(
        project_root=project_root,
        source_root=resolved_source_root,
        discovered_files=discovered_files,
        candidate_files=candidate_files,
        imported_files=imported_files,
        imported_memories=len(records),
        skipped_existing_texts=skipped_existing_texts,
        skipped_files=skipped_files,
        limit_reached=len(records) >= max_memories,
        total_memories=total_memories,
        component_counts=dict(sorted(component_counts.items())),
        skipped_by_reason=skipped_by_reason,
    )
