from __future__ import annotations

import argparse
import json

from agent_memory.benchmark import (
    build_benchmark_config,
    load_wikipedia_corpus,
    run_benchmark_on_corpus,
)
from agent_memory.embeddings import build_embedder


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the default Agent Memory benchmark config.")
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
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs to execute with the default config.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print all sweep results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = build_benchmark_config(args.embedding_backend)
    corpus = load_wikipedia_corpus(args.paragraphs_per_article)
    embedder = build_embedder(base_config)

    results = []
    for run_index in range(args.runs):
        payload = run_benchmark_on_corpus(
            corpus,
            base_config,
            keep_workspace=False,
            embedder=embedder,
        )
        results.append(
            {
                "run": run_index + 1,
                "config": base_config.to_dict(),
                "summary": payload["summary"],
            }
        )

    results.sort(key=lambda item: item["summary"]["benchmark_score"], reverse=True)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("Benchmark runs")
    for result in results:
        summary = result["summary"]
        config = result["config"]
        print("\n---")
        print(
            f"run={result['run']} score={summary['benchmark_score']} "
            f"strict_pass={summary['strict_pass_count']} "
            f"overall_recall={summary['average_overall_recall']}"
        )
        print(f"   config={config}")
        print(f"   case_scores={summary['case_scores']}")


if __name__ == "__main__":
    main()
