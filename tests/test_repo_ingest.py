from __future__ import annotations

from pathlib import Path

from agent_memory.config import MemoryConfig, init_project
from agent_memory.engine import open_memory_with_retry
from agent_memory.repo_ingest import import_repo_corpus


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_import_repo_corpus_imports_and_skips_duplicates(tmp_path: Path) -> None:
    init_project(
        tmp_path,
        config=MemoryConfig(
            embedding_backend="hash",
            embedding_dimensions=64,
            max_memory_words=80,
        ),
    )
    _write(
        tmp_path / "README.md",
        "# EDS\n\n"
        "This repo contains platform notes and operational guidance.\n\n"
        "The importer should keep file metadata so the source path is searchable.\n",
    )
    _write(
        tmp_path / "eds-python" / "service.py",
        "def compute_alpha():\n"
        "    value = 1\n"
        "    return value\n\n"
        "class ReportBuilder:\n"
        "    def build(self):\n"
        "        return {'status': 'ok'}\n",
    )
    _write(tmp_path / "node_modules" / "ignore.js", "console.log('skip me')\n")

    first = import_repo_corpus(tmp_path, max_memories=10, max_chunks_per_file=3, max_file_bytes=64 * 1024)
    second = import_repo_corpus(tmp_path, max_memories=10, max_chunks_per_file=3, max_file_bytes=64 * 1024)

    assert first.imported_memories >= 2
    assert first.total_memories == first.imported_memories
    assert second.imported_memories == 0
    assert second.skipped_existing_texts >= first.imported_memories

    memory = open_memory_with_retry(tmp_path, exact=True, read_only=True)
    try:
        texts = [item.text for item in memory.list_all()]
    finally:
        memory.close()

    assert any("Source: README.md" in text for text in texts)
    assert any("Source: eds-python/service.py" in text for text in texts)
    assert not any("ignore.js" in text for text in texts)


def test_import_repo_corpus_respects_max_chunks_per_file(tmp_path: Path) -> None:
    init_project(
        tmp_path,
        config=MemoryConfig(
            embedding_backend="hash",
            embedding_dimensions=64,
            max_memory_words=60,
        ),
    )
    _write(
        tmp_path / "docs" / "long.md",
        "\n\n".join(
            f"Paragraph {index}. " + "word " * 25
            for index in range(1, 8)
        )
        + "\n",
    )

    result = import_repo_corpus(tmp_path, max_memories=20, max_chunks_per_file=2, max_file_bytes=64 * 1024)

    assert result.imported_memories == 2
    assert result.imported_files == 1
