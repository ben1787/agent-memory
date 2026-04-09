from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time

import numpy as np
import tiktoken

from agent_memory.benchmark import BENCHMARK_CASES, build_benchmark_config
from agent_memory.end_to_end import flatten_paragraphs
from agent_memory.embeddings import build_embedder
from agent_memory.engine import AgentMemory, MemoryHit
from agent_memory.raw_corpus import load_raw_article_corpus


DEFAULT_RAW_ARTICLES_DIR = Path(__file__).resolve().parents[1] / "benchmark_corpus" / "raw_articles"
TOKEN_ENCODER = tiktoken.get_encoding("o200k_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark Agent Memory recall hot-path latency."
    )
    parser.add_argument(
        "--raw-articles-dir",
        type=Path,
        default=DEFAULT_RAW_ARTICLES_DIR,
        help="Canonical raw article folder.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["fastembed", "hash"],
        default="fastembed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of memories to materialize per recall.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per benchmark case.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="Benchmark only the in-memory read kernel, skipping DB-backed graph construction.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Existing Agent Memory project root to reuse instead of rebuilding a benchmark project.",
    )
    return parser.parse_args()


def avg(values: list[float]) -> float:
    return round(statistics.fmean(values), 3) if values else 0.0


def count_text_tokens(texts: list[str]) -> int:
    return sum(len(TOKEN_ENCODER.encode(text)) for text in texts)


def build_memory(
    raw_articles_dir: Path,
    embedding_backend: str,
) -> tuple[AgentMemory, Path, dict[str, int]]:
    corpus = load_raw_article_corpus(raw_articles_dir)
    if not corpus:
        raise SystemExit(f"No article data found in {raw_articles_dir}")

    config = build_benchmark_config(embedding_backend)
    embedder = build_embedder(config)
    paragraph_records = flatten_paragraphs(corpus, embedder)

    workspace = Path(tempfile.mkdtemp(prefix="agent-memory-hotpath-"))
    build_started = time.perf_counter()
    memory = AgentMemory.initialize(workspace, config=config, force=True, embedder=embedder)
    try:
        stats = memory.import_memories(paragraph_records).to_dict()
    except Exception:
        memory.close()
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    build_ms = round((time.perf_counter() - build_started) * 1000, 3)
    stats["build_ms"] = build_ms
    stats["article_count"] = len(corpus)
    stats["paragraph_count"] = len(paragraph_records)
    return memory, workspace, stats


def open_existing_memory(project_root: Path) -> tuple[AgentMemory, dict[str, int]]:
    memory = AgentMemory.open(project_root.resolve())
    stats = memory.stats().to_dict()
    stats["article_count"] = 0
    stats["paragraph_count"] = stats["memory_count"]
    return memory, stats


def build_kernel_dataset_from_memory(memory: AgentMemory) -> tuple[dict[str, object], dict[str, int]]:
    if memory._embedding_matrix is None:
        raise SystemExit("Existing project has no stored memory embeddings.")
    similarity_matrix = memory._embedding_matrix @ memory._embedding_matrix.T
    np.fill_diagonal(similarity_matrix, 0.0)
    neighbor_order = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    edge_count = int(similarity_matrix.size - len(memory._memory_ids_in_order))
    return (
        {
            "embedder": memory.embedder,
            "config": memory.config,
            "memory_ids": list(memory._memory_ids_in_order),
            "texts": [memory._memory_by_id[memory_id].text for memory_id in memory._memory_ids_in_order],
            "matrix": memory._embedding_matrix,
            "similarity_matrix": similarity_matrix,
            "neighbor_order": neighbor_order,
        },
        {
            "article_count": 0,
            "paragraph_count": len(memory._memory_ids_in_order),
            "similarity_edge_count": edge_count,
        },
    )


def build_kernel_dataset(
    raw_articles_dir: Path,
    embedding_backend: str,
) -> tuple[dict[str, object], dict[str, int]]:
    corpus = load_raw_article_corpus(raw_articles_dir)
    if not corpus:
        raise SystemExit(f"No article data found in {raw_articles_dir}")

    config = build_benchmark_config(embedding_backend)
    embedder = build_embedder(config)
    paragraph_records = flatten_paragraphs(corpus, embedder)
    memory_ids = [f"mem_{index}" for index in range(len(paragraph_records))]
    texts = [record["text"] for record in paragraph_records]
    matrix = np.asarray([record["embedding"] for record in paragraph_records], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized_matrix = matrix / norms
    similarity_matrix = normalized_matrix @ normalized_matrix.T
    np.fill_diagonal(similarity_matrix, 0.0)
    neighbor_order = np.argsort(similarity_matrix, axis=1)[:, ::-1]

    return (
        {
            "embedder": embedder,
            "config": config,
            "memory_ids": memory_ids,
            "texts": texts,
            "matrix": normalized_matrix,
            "similarity_matrix": similarity_matrix,
            "neighbor_order": neighbor_order,
        },
        {
            "article_count": len(corpus),
            "paragraph_count": len(paragraph_records),
            "similarity_edge_count": len(paragraph_records) * max(len(paragraph_records) - 1, 0),
        },
    )


def materialize_hits(
    memory: AgentMemory,
    ranked_scores: list[tuple[str, float, str | None, float]],
    query_scores: dict[str, float],
    limit: int,
) -> list[MemoryHit]:
    hits: list[MemoryHit] = []
    for memory_id, score, _, _ in ranked_scores[:limit]:
        record = memory._memory_by_id[memory_id]
        hits.append(
            MemoryHit(
                memory_id=memory_id,
                text=record.text,
                score=round(score, 4),
                query_similarity=round(query_scores.get(memory_id, 0.0), 4),
                created_at=record.created_at,
            )
        )
    return hits


def benchmark_case(memory: AgentMemory, query: str, limit: int) -> dict[str, float | int]:
    actual_started = time.perf_counter()
    result = memory.recall(query, limit=limit)
    actual_recall_ms = (time.perf_counter() - actual_started) * 1000

    embed_started = time.perf_counter()
    raw_query_embedding = memory.embedder.embed_text(query)
    embed_ms = (time.perf_counter() - embed_started) * 1000

    similarity_started = time.perf_counter()
    query_embedding = memory._normalize_embedding(raw_query_embedding)
    scores = memory._embedding_matrix @ query_embedding if memory._embedding_matrix is not None else []
    query_scores = {
        memory_id: float(score)
        for memory_id, score in zip(memory._memory_ids_in_order, scores, strict=False)
    }
    similarity_ms = (time.perf_counter() - similarity_started) * 1000

    propagation_started = time.perf_counter()
    ranked_scores = memory._rank_memories(query_scores=query_scores, limit=limit)
    propagation_ms = (time.perf_counter() - propagation_started) * 1000

    materialize_started = time.perf_counter()
    hits = materialize_hits(memory, ranked_scores, query_scores, limit)
    materialize_ms = (time.perf_counter() - materialize_started) * 1000

    return {
        "actual_recall_ms": round(actual_recall_ms, 3),
        "embed_ms": round(embed_ms, 3),
        "similarity_ms": round(similarity_ms, 3),
        "propagation_ms": round(propagation_ms, 3),
        "materialize_ms": round(materialize_ms, 3),
        "top_score": round(hits[0].score, 4) if hits else 0.0,
        "hit_count": len(result.hits),
    }


def rank_kernel_scores(
    query_scores: np.ndarray,
    similarity_matrix: np.ndarray,
    neighbor_order: np.ndarray,
    limit: int,
) -> list[tuple[int, float]]:
    initial_scores = {
        index: float(score) for index, score in enumerate(query_scores.tolist()) if score > 0.0
    }
    if not initial_scores:
        return sorted(
            [(index, float(score)) for index, score in enumerate(query_scores.tolist())],
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

    settled_scores: dict[int, float] = {}
    results: list[tuple[int, float]] = []
    heap: list[tuple[float, float, int, int, int, int]] = []
    serial = 0

    def push_candidate(score: float, node_index: int, parent_index: int, cursor: int) -> None:
        nonlocal serial
        if score <= 0.0:
            return
        heapq.heappush(
            heap,
            (
                -score,
                -float(query_scores[node_index]),
                serial,
                node_index,
                parent_index,
                cursor,
            ),
        )
        serial += 1

    def push_next_neighbor(parent_index: int, start_cursor: int) -> None:
        parent_score = settled_scores.get(parent_index, 0.0)
        row = similarity_matrix[parent_index]
        order = neighbor_order[parent_index]
        cursor = start_cursor
        while cursor < len(order):
            neighbor_index = int(order[cursor])
            if neighbor_index != parent_index and neighbor_index not in settled_scores:
                push_candidate(
                    parent_score * float(row[neighbor_index]),
                    neighbor_index,
                    parent_index,
                    cursor,
                )
                return
            cursor += 1

    for index, score in initial_scores.items():
        push_candidate(score, index, -1, -1)

    while heap and len(results) < limit:
        negative_score, _, _, index, parent_index, cursor = heapq.heappop(heap)
        score = -negative_score
        if parent_index >= 0:
            push_next_neighbor(parent_index, cursor + 1)
        if index in settled_scores:
            continue
        settled_scores[index] = score
        results.append((index, score))
        push_next_neighbor(index, 0)

    if len(results) < limit:
        fallback = sorted(
            [(index, float(score)) for index, score in enumerate(query_scores.tolist())],
            key=lambda item: item[1],
            reverse=True,
        )
        for index, score in fallback:
            if index in settled_scores:
                continue
            results.append((index, score))
            if len(results) >= limit:
                break

    return results


def benchmark_case_kernel(
    dataset: dict[str, object],
    query: str,
    limit: int,
) -> dict[str, float | int]:
    embedder = dataset["embedder"]
    matrix = dataset["matrix"]
    similarity_matrix = dataset["similarity_matrix"]
    neighbor_order = dataset["neighbor_order"]
    memory_ids = dataset["memory_ids"]
    texts = dataset["texts"]

    started = time.perf_counter()
    raw_query_embedding = embedder.embed_text(query)
    embed_ms = (time.perf_counter() - started) * 1000

    similarity_started = time.perf_counter()
    query_vector = np.asarray(raw_query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm != 0.0:
        query_vector = query_vector / query_norm
    query_scores = matrix @ query_vector
    similarity_ms = (time.perf_counter() - similarity_started) * 1000

    propagation_started = time.perf_counter()
    ranked_scores = rank_kernel_scores(query_scores, similarity_matrix, neighbor_order, limit)
    propagation_ms = (time.perf_counter() - propagation_started) * 1000

    materialize_started = time.perf_counter()
    hits = [
        (
            memory_ids[index],
            float(score),
            float(query_scores[index]),
            texts[index],
        )
        for index, score in ranked_scores[:limit]
    ]
    materialize_ms = (time.perf_counter() - materialize_started) * 1000

    return {
        "embed_ms": round(embed_ms, 3),
        "similarity_ms": round(similarity_ms, 3),
        "propagation_ms": round(propagation_ms, 3),
        "materialize_ms": round(materialize_ms, 3),
        "total_read_ms": round(embed_ms + similarity_ms + propagation_ms + materialize_ms, 3),
        "context_tokens": count_text_tokens([hit[3] for hit in hits]),
        "top_score": round(hits[0][1], 4) if hits else 0.0,
        "hit_count": len(hits),
    }


def benchmark_case_raw_kernel(
    dataset: dict[str, object],
    query: str,
    limit: int,
) -> dict[str, float | int]:
    embedder = dataset["embedder"]
    matrix = dataset["matrix"]
    memory_ids = dataset["memory_ids"]
    texts = dataset["texts"]

    started = time.perf_counter()
    raw_query_embedding = embedder.embed_text(query)
    embed_ms = (time.perf_counter() - started) * 1000

    similarity_started = time.perf_counter()
    query_vector = np.asarray(raw_query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm != 0.0:
        query_vector = query_vector / query_norm
    query_scores = matrix @ query_vector
    similarity_ms = (time.perf_counter() - similarity_started) * 1000

    selection_started = time.perf_counter()
    if limit >= len(query_scores):
        selected_indices = np.argsort(query_scores)[::-1]
    else:
        partition = np.argpartition(query_scores, -limit)[-limit:]
        selected_indices = partition[np.argsort(query_scores[partition])[::-1]]
    selection_ms = (time.perf_counter() - selection_started) * 1000

    materialize_started = time.perf_counter()
    hits = [
        (
            memory_ids[index],
            float(query_scores[index]),
            texts[index],
        )
        for index in selected_indices.tolist()
    ]
    materialize_ms = (time.perf_counter() - materialize_started) * 1000

    return {
        "embed_ms": round(embed_ms, 3),
        "similarity_ms": round(similarity_ms, 3),
        "selection_ms": round(selection_ms, 3),
        "materialize_ms": round(materialize_ms, 3),
        "total_read_ms": round(embed_ms + similarity_ms + selection_ms + materialize_ms, 3),
        "context_tokens": count_text_tokens([hit[2] for hit in hits]),
        "top_score": round(hits[0][1], 4) if hits else 0.0,
        "hit_count": len(hits),
    }


def summarize_kernel(
    dataset: dict[str, object],
    stats: dict[str, int],
    limit: int,
    repeats: int,
) -> dict[str, object]:
    case_reports: list[dict[str, object]] = []
    graph_totals: dict[str, list[float]] = {
        "embed_ms": [],
        "similarity_ms": [],
        "propagation_ms": [],
        "materialize_ms": [],
        "total_read_ms": [],
        "context_tokens": [],
    }
    raw_totals: dict[str, list[float]] = {
        "embed_ms": [],
        "similarity_ms": [],
        "selection_ms": [],
        "materialize_ms": [],
        "total_read_ms": [],
        "context_tokens": [],
    }

    for case in BENCHMARK_CASES:
        graph_samples = [benchmark_case_kernel(dataset, case.query, limit) for _ in range(repeats)]
        raw_samples = [benchmark_case_raw_kernel(dataset, case.query, limit) for _ in range(repeats)]
        report = {
            "case_id": case.case_id,
            "query": case.query,
            "graph_samples": graph_samples,
            "raw_samples": raw_samples,
            "graph_average": {
                metric: avg([float(sample[metric]) for sample in graph_samples])
                for metric in graph_totals
            },
            "raw_average": {
                metric: avg([float(sample[metric]) for sample in raw_samples])
                for metric in raw_totals
            },
        }
        case_reports.append(report)
        for metric in graph_totals:
            graph_totals[metric].extend(float(sample[metric]) for sample in graph_samples)
        for metric in raw_totals:
            raw_totals[metric].extend(float(sample[metric]) for sample in raw_samples)

    return {
        "workspace": None,
        "kernel_only": True,
        "stats": stats,
        "config": dataset["config"].to_dict(),
        "limit": limit,
        "repeats": repeats,
        "graph_summary": {metric: avg(values) for metric, values in graph_totals.items()},
        "raw_summary": {metric: avg(values) for metric, values in raw_totals.items()},
        "cases": case_reports,
    }


def summarize_db(
    memory: AgentMemory,
    workspace: Path | None,
    stats: dict[str, int],
    limit: int,
    repeats: int,
) -> dict[str, object]:
    memory.recall(BENCHMARK_CASES[0].query, limit=limit)

    case_reports: list[dict[str, object]] = []
    totals: dict[str, list[float]] = {
        "actual_recall_ms": [],
        "embed_ms": [],
        "similarity_ms": [],
        "propagation_ms": [],
        "materialize_ms": [],
    }

    for case in BENCHMARK_CASES:
        samples = [benchmark_case(memory, case.query, limit) for _ in range(repeats)]
        report = {
            "case_id": case.case_id,
            "query": case.query,
            "samples": samples,
            "average": {
                metric: avg([float(sample[metric]) for sample in samples])
                for metric in totals
            },
        }
        case_reports.append(report)
        for metric in totals:
            totals[metric].extend(float(sample[metric]) for sample in samples)

    return {
        "workspace": str(workspace) if workspace is not None else str(memory.project.root),
        "kernel_only": False,
        "stats": stats,
        "config": memory.config.to_dict(),
        "limit": limit,
        "repeats": repeats,
        "summary": {metric: avg(values) for metric, values in totals.items()},
        "cases": case_reports,
    }


def main() -> None:
    args = parse_args()
    if args.kernel_only:
        if args.project_root is not None:
            memory, _stats = open_existing_memory(args.project_root)
            try:
                dataset, stats = build_kernel_dataset_from_memory(memory)
                payload = summarize_kernel(dataset, stats, args.limit, args.repeats)
            finally:
                memory.close()
        else:
            dataset, stats = build_kernel_dataset(args.raw_articles_dir.resolve(), args.embedding_backend)
            payload = summarize_kernel(dataset, stats, args.limit, args.repeats)
    else:
        if args.project_root is not None:
            memory, stats = open_existing_memory(args.project_root)
            try:
                payload = summarize_db(memory, None, stats, args.limit, args.repeats)
            finally:
                memory.close()
        else:
            memory, workspace, stats = build_memory(args.raw_articles_dir.resolve(), args.embedding_backend)
            try:
                payload = summarize_db(memory, workspace, stats, args.limit, args.repeats)
            finally:
                memory.close()
                if not args.keep_workspace:
                    shutil.rmtree(workspace, ignore_errors=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("Recall hot-path microbenchmark")
    if payload["workspace"]:
        print(f"Workspace: {payload['workspace']}")
    print(
        f"Articles: {payload['stats']['article_count']}  Paragraph memories: {payload['stats']['paragraph_count']}  "
        f"Similarity edges: {payload['stats']['similarity_edge_count']}"
    )
    if not args.kernel_only:
        print(f"Graph build/import: {payload['stats']['build_ms']} ms")
    print(
        "Averages across "
        f"{len(BENCHMARK_CASES)} cases x {args.repeats} repeats:"
    )
    if args.kernel_only:
        print("  Graph path:")
        for metric, label in [
            ("embed_ms", "embed query"),
            ("similarity_ms", "query->all similarity"),
            ("propagation_ms", "path propagation"),
            ("materialize_ms", "hit materialization"),
            ("total_read_ms", "total read"),
            ("context_tokens", "context tokens"),
        ]:
            print(f"    {label}: {payload['graph_summary'][metric]}")
        print("  Raw path:")
        for metric, label in [
            ("embed_ms", "embed query"),
            ("similarity_ms", "query->all similarity"),
            ("selection_ms", "top-n selection"),
            ("materialize_ms", "hit materialization"),
            ("total_read_ms", "total read"),
            ("context_tokens", "context tokens"),
        ]:
            print(f"    {label}: {payload['raw_summary'][metric]}")
    else:
        for metric, label in [
            ("actual_recall_ms", "actual recall"),
            ("embed_ms", "embed query"),
            ("similarity_ms", "query->all similarity"),
            ("propagation_ms", "path propagation"),
            ("materialize_ms", "hit materialization"),
        ]:
            if metric in payload["summary"]:
                print(f"  {label}: {payload['summary'][metric]} ms")

    print("\nPer-case averages")
    for case in payload["cases"]:
        if args.kernel_only:
            graph_avg = case["graph_average"]
            raw_avg = case["raw_average"]
            print(
                f"- {case['case_id']}: "
                f"graph_total={graph_avg['total_read_ms']} ms graph_tokens={graph_avg['context_tokens']} "
                f"raw_total={raw_avg['total_read_ms']} ms raw_tokens={raw_avg['context_tokens']}"
            )
        else:
            average = case["average"]
            print(
                f"- {case['case_id']}: total={average['actual_recall_ms']} ms "
                f"embed={average['embed_ms']} similarity={average['similarity_ms']} "
                f"propagation={average['propagation_ms']} materialize={average['materialize_ms']}"
            )


if __name__ == "__main__":
    main()
