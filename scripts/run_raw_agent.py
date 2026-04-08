from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agents import Agent, function_tool

from agent_memory.benchmark import BENCHMARK_CASES, build_benchmark_config
from agent_memory.embeddings import build_embedder
from agent_memory.end_to_end import (
    ANSWER_INSTRUCTIONS_SUFFIX,
    DEFAULT_ANSWER_MODEL_ID,
    AnswerFormat,
    HitRecord,
    OpenAIAgentAnswerer,
    RawCorpusRetriever,
    evaluate_path,
    flatten_paragraphs,
    hits_to_context_items,
    load_cached_corpus,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the raw-corpus OpenAI agent.")
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument("--paragraphs-per-article", type=int, default=0)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hash"], default="fastembed")
    parser.add_argument("--model-id", default=DEFAULT_ANSWER_MODEL_ID)
    # Reason: default raised to 20 to match compare_isolated_agents. Kept
    # in sync across all three agents so the comparison stays apples-to-apples.
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def build_raw_agent(
    *,
    retriever: RawCorpusRetriever,
    model: str,
    hit_sink: list[HitRecord],
    retrieval_time_ms: list[float],
    call_log: list[dict[str, Any]],
    recall_limit: int,
) -> Agent[Any]:
    # Reason: the raw-corpus agent retrieves directly over the in-process
    # paragraph list (not through AgentMemory). Same embedder so the
    # cosine similarity is computed the same way, but the documents are
    # the source paragraphs rather than graph-stored memory entries.

    @function_tool
    def recall_raw_paragraphs(query: str, limit: int = recall_limit) -> str:
        """Retrieve paragraphs from the raw source corpus by cosine similarity.

        Returns a JSON list of paragraphs, each with reference_id (e.g. "Graph theory ¶3"),
        title, score, and full text. Call multiple times with refined queries if the
        first pass is insufficient.
        """
        bounded_limit = max(1, min(int(limit), 50))
        started = time.perf_counter()
        items = retriever.retrieve(query, top_k=bounded_limit)
        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieval_time_ms.append(elapsed_ms)
        payload = []
        per_call_hits: list[dict[str, Any]] = []
        for item in items:
            hit_sink.append(
                HitRecord(
                    reference_id=item.reference_id,
                    title=item.title,
                    text=item.text,
                    score=float(item.score),
                    locator=item.locator,
                )
            )
            payload.append(
                {
                    "reference_id": item.reference_id,
                    "title": item.title,
                    "score": item.score,
                    "text": item.text,
                }
            )
            per_call_hits.append(
                {
                    "memory_id": item.reference_id,
                    "title": item.title,
                    "score": float(item.score),
                    "query_similarity": float(item.score),
                }
            )
        call_log.append(
            {
                "query": query,
                "limit": bounded_limit,
                "retrieval_ms": round(elapsed_ms, 3),
                "hit_count": len(per_call_hits),
                "hits": per_call_hits,
            }
        )
        return json.dumps(payload, indent=2)

    return Agent(
        name="Raw Corpus Agent",
        instructions=(
            "Answer the question using only the raw corpus recall tool. "
            "Call `recall_raw_paragraphs` with the question (or a refined sub-query) "
            "to retrieve relevant paragraphs directly from the source articles. "
            "You may call the tool more than once with different phrasings if the "
            "first pass is insufficient. Cite only the reference IDs you actually used. "
            + ANSWER_INSTRUCTIONS_SUFFIX
        ),
        model=model,
        tools=[recall_raw_paragraphs],
        output_type=AnswerFormat,
    )


def main() -> None:
    args = parse_args()
    corpus = load_cached_corpus(args.cache_file)
    if args.paragraphs_per_article > 0:
        limited_corpus = {
            title: paragraphs[: args.paragraphs_per_article]
            for title, paragraphs in list(corpus.items())[: args.total_articles]
        }
    else:
        limited_corpus = {
            title: paragraphs
            for title, paragraphs in list(corpus.items())[: args.total_articles]
        }
    config = build_benchmark_config(args.embedding_backend)
    embedder = build_embedder(config)
    paragraphs = flatten_paragraphs(limited_corpus, embedder)
    retriever = RawCorpusRetriever(paragraphs, embedder)
    answerer = OpenAIAgentAnswerer(model=args.model_id, max_turns=args.max_turns)

    results: list[dict[str, object]] = []
    for case in BENCHMARK_CASES:
        case_hits: list[HitRecord] = []
        case_retrieval_ms: list[float] = []
        case_call_log: list[dict[str, Any]] = []
        agent = build_raw_agent(
            retriever=retriever,
            model=args.model_id,
            hit_sink=case_hits,
            retrieval_time_ms=case_retrieval_ms,
            call_log=case_call_log,
            recall_limit=args.recall_limit,
        )
        agent_result = answerer.run(agent, case.query, case_hits)
        contexts = hits_to_context_items(case_hits)
        meta = {
            "retrieval_ms": round(sum(case_retrieval_ms), 3),
            "tool_calls": len(case_retrieval_ms),
            "tool_call_log": case_call_log,
        }
        results.append(
            evaluate_path(
                label="raw",
                case=case,
                contexts=contexts,
                retrieval_meta=meta,
                agent_result=agent_result,
            )
        )
    payload = {
        "system": "raw",
        "model_id": args.model_id,
        "summary": summarize_results(results),
        "results": results,
    }
    args.output_file.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
