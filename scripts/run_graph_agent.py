from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_memory.benchmark import BENCHMARK_CASES
from agent_memory.end_to_end import (
    LocalSeq2SeqAnswerer,
    build_graph_context,
    evaluate_path,
    summarize_results,
)
from agent_memory.engine import AgentMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the graph-only answering agent.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-clusters", type=int, default=2)
    parser.add_argument("--max-hits-per-cluster", type=int, default=2)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory = AgentMemory.open(args.project_root)
    try:
        answerer = LocalSeq2SeqAnswerer(model_id=args.model_id)
        answerer.warmup()
        results = []
        for case in BENCHMARK_CASES:
            contexts, meta = build_graph_context(
                memory,
                case.query,
                max_clusters=args.max_clusters,
                max_hits_per_cluster=args.max_hits_per_cluster,
            )
            results.append(
                evaluate_path(
                    label="graph",
                    case=case,
                    contexts=contexts,
                    retrieval_meta=meta,
                    answerer=answerer,
                )
            )
        payload = {
            "system": "graph",
            "model_id": args.model_id,
            "summary": summarize_results(results),
            "results": results,
        }
        args.output_file.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
