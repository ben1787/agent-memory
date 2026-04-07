from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import tempfile
import time

from agent_memory.benchmark import build_benchmark_config
from agent_memory.end_to_end import (
    DEFAULT_ANSWER_MODEL_ID,
    flatten_paragraphs,
    load_cached_corpus,
)
from agent_memory.embeddings import build_embedder
from agent_memory.engine import AgentMemory
from agent_memory.reporting import render_isolated_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark isolated raw and graph answering agents in separate subprocesses."
    )
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument("--paragraphs-per-article", type=int, default=0)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hash"], default="fastembed")
    parser.add_argument("--raw-top-k", type=int, default=12)
    parser.add_argument("--graph-max-clusters", type=int, default=2)
    parser.add_argument("--graph-hits-per-cluster", type=int, default=2)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--html-file", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def print_summary(payload: dict[str, object]) -> None:
    print(f"Model: {payload['model_id']}")
    print(
        f"Corpus: articles={payload['total_articles']} paragraphs/article={payload['paragraphs_per_article']}"
    )
    print(
        "Graph avg: "
        f"{payload['graph']['summary']['average_total_ms']} ms total "
        f"(retrieval={payload['graph']['summary']['average_retrieval_ms']} ms, "
        f"generation={payload['graph']['summary']['average_generation_ms']} ms), "
        f"context={payload['graph']['summary']['average_context_tokens']} tok, "
        f"context_score={payload['graph']['summary']['average_context_score']}"
    )
    print(
        "Raw avg:   "
        f"{payload['raw']['summary']['average_total_ms']} ms total "
        f"(retrieval={payload['raw']['summary']['average_retrieval_ms']} ms, "
        f"generation={payload['raw']['summary']['average_generation_ms']} ms), "
        f"context={payload['raw']['summary']['average_context_tokens']} tok, "
        f"context_score={payload['raw']['summary']['average_context_score']}"
    )


def main() -> None:
    args = parse_args()
    corpus = load_cached_corpus(args.cache_file)
    if args.paragraphs_per_article > 0:
        limited_corpus = {
            title: paragraphs[:args.paragraphs_per_article]
            for title, paragraphs in list(corpus.items())[:args.total_articles]
        }
    else:
        limited_corpus = {
            title: paragraphs
            for title, paragraphs in list(corpus.items())[:args.total_articles]
        }

    temp_root = Path(tempfile.mkdtemp(prefix="agent-memory-isolated-"))
    raw_cache = temp_root / "raw_corpus.json"
    raw_cache.write_text(json.dumps(limited_corpus, indent=2) + "\n")

    config = build_benchmark_config(args.embedding_backend)
    embedder = build_embedder(config)
    paragraph_records = flatten_paragraphs(limited_corpus, embedder)

    graph_root = temp_root / "graph_agent"
    memory = AgentMemory.initialize(graph_root, config=config, force=True, embedder=embedder)
    try:
        for paragraph in paragraph_records:
            memory.save(paragraph["text"], embedding=paragraph["embedding"])
    finally:
        memory.close()
        del memory
        gc.collect()
        time.sleep(0.5)

    raw_output = temp_root / "raw_results.json"
    graph_output = temp_root / "graph_results.json"

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_raw_agent.py",
            "--cache-file",
            str(raw_cache),
            "--total-articles",
            str(args.total_articles),
            "--paragraphs-per-article",
            str(args.paragraphs_per_article),
            "--embedding-backend",
            args.embedding_backend,
            "--top-k",
            str(args.raw_top_k),
            "--output-file",
            str(raw_output),
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_graph_agent.py",
            "--project-root",
            str(graph_root),
            "--max-clusters",
            str(args.graph_max_clusters),
            "--max-hits-per-cluster",
            str(args.graph_hits_per_cluster),
            "--output-file",
            str(graph_output),
        ],
        check=True,
    )

    payload = {
        "model_id": DEFAULT_ANSWER_MODEL_ID,
        "total_articles": args.total_articles,
        "paragraphs_per_article": args.paragraphs_per_article,
        "raw": json.loads(raw_output.read_text()),
        "graph": json.loads(graph_output.read_text()),
        "workspace": str(temp_root),
    }

    if args.output_file:
        args.output_file.write_text(json.dumps(payload, indent=2) + "\n")
    if args.html_file:
        args.html_file.write_text(render_isolated_benchmark_report(payload))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_summary(payload)

    if not args.keep_workspace:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
