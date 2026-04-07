from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_memory.benchmark import BENCHMARK_CASES, build_benchmark_config
from agent_memory.embeddings import build_embedder
from agent_memory.end_to_end import (
    DEFAULT_ANSWER_MODEL_ID,
    LocalSeq2SeqAnswerer,
    RawCorpusRetriever,
    evaluate_path,
    flatten_paragraphs,
    build_raw_context,
    load_cached_corpus,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the raw-documents-only answering agent.")
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument("--paragraphs-per-article", type=int, default=0)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hash"], default="fastembed")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


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
    config = build_benchmark_config(args.embedding_backend)
    embedder = build_embedder(config)
    paragraphs = flatten_paragraphs(limited_corpus, embedder)
    retriever = RawCorpusRetriever(paragraphs, embedder)
    answerer = LocalSeq2SeqAnswerer()
    answerer.warmup()

    results = []
    for case in BENCHMARK_CASES:
        contexts, meta = build_raw_context(retriever, case.query, top_k=args.top_k)
        results.append(
            evaluate_path(
                label="raw",
                case=case,
                contexts=contexts,
                retrieval_meta=meta,
                answerer=answerer,
            )
        )
    payload = {
        "system": "raw",
        "model_id": DEFAULT_ANSWER_MODEL_ID,
        "summary": summarize_results(results),
        "results": results,
    }
    args.output_file.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
