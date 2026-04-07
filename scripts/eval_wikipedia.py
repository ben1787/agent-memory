from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_memory.benchmark import (
    ARTICLE_GROUPS,
    build_benchmark_config,
    print_benchmark_report,
    run_wikipedia_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Agent Memory on Wikipedia articles.")
    parser.add_argument(
        "--paragraphs-per-article",
        type=int,
        default=4,
        help="Maximum number of paragraphs to ingest from each article.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["fastembed", "hash"],
        default="fastembed",
        help="Embedding backend to use during evaluation.",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep the temporary evaluation project on disk.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full evaluation report as JSON.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Optional JSON corpus cache to reuse instead of fetching Wikipedia live.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_benchmark_config(args.embedding_backend)

    corpus = None
    if args.cache_file is not None:
        payload = json.loads(args.cache_file.read_text())
        benchmark_titles = {
            title for titles in ARTICLE_GROUPS.values() for title in titles
        }
        corpus = {
            title: paragraphs[: args.paragraphs_per_article]
            for title, paragraphs in payload.items()
            if title in benchmark_titles
        }

    payload = run_wikipedia_benchmark(
        article_limit_per_title=args.paragraphs_per_article,
        embedding_backend=args.embedding_backend,
        keep_workspace=args.keep_workspace,
        config=config,
        corpus=corpus,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print_benchmark_report(payload)


if __name__ == "__main__":
    main()
