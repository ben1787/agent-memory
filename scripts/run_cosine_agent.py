from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agents import Agent, function_tool

from agent_memory.benchmark import BENCHMARK_CASES, parse_title
from agent_memory.end_to_end import (
    ANSWER_INSTRUCTIONS_SUFFIX,
    DEFAULT_ANSWER_MODEL_ID,
    AnswerFormat,
    HitRecord,
    OpenAIAgentAnswerer,
    evaluate_path,
    hits_to_context_items,
    summarize_results,
)
from agent_memory.engine import AgentMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the flat-cosine OpenAI agent.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_ANSWER_MODEL_ID)
    # Reason: default raised to 20 to match compare_isolated_agents. See
    # comment there for rationale (closes iteration-gap losses).
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def build_cosine_agent(
    *,
    memory: AgentMemory,
    model: str,
    hit_sink: list[HitRecord],
    retrieval_time_ms: list[float],
    call_log: list[dict[str, Any]],
    recall_limit: int,
) -> Agent[Any]:
    # Reason: this is the graph agent's twin — same store, same embedder,
    # but the tool calls `recall_cosine` (flat top-N nearest neighbors)
    # instead of `recall` (PPV graph spreading). Any quality difference
    # between this and the graph agent is attributable purely to the
    # graph-expansion step.

    @function_tool
    def recall_cosine_memories(query: str, limit: int = recall_limit) -> str:
        """Retrieve memories by flat cosine similarity (no graph walk).

        Returns a JSON list of hits ranked purely by embedding similarity to the query,
        with memory_id, title, score, and full text. Call multiple times with refined
        queries if the first pass is insufficient.
        """
        bounded_limit = max(1, min(int(limit), 50))
        started = time.perf_counter()
        result = memory.recall_cosine(query, limit=bounded_limit)
        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieval_time_ms.append(elapsed_ms)
        payload = []
        per_call_hits: list[dict[str, Any]] = []
        for hit in result.hits:
            title = parse_title(hit.text)
            hit_sink.append(
                HitRecord(
                    reference_id=hit.memory_id,
                    title=title,
                    text=hit.text,
                    score=float(hit.query_similarity or 0.0),
                    locator=None,
                )
            )
            payload.append(
                {
                    "memory_id": hit.memory_id,
                    "title": title,
                    "score": hit.score,
                    "query_similarity": hit.query_similarity,
                    "text": hit.text,
                }
            )
            per_call_hits.append(
                {
                    "memory_id": hit.memory_id,
                    "title": title,
                    "score": float(hit.score),
                    "query_similarity": float(hit.query_similarity or 0.0),
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
        name="Cosine Recall Agent",
        instructions=(
            "Answer the question using only the cosine recall tool. "
            "Call `recall_cosine_memories` with the question (or a refined sub-query) "
            "to retrieve relevant memories from the local Agent Memory store. "
            "You may call the tool more than once with different phrasings if the "
            "first pass is insufficient. Cite only the memory IDs you actually used. "
            + ANSWER_INSTRUCTIONS_SUFFIX
        ),
        model=model,
        tools=[recall_cosine_memories],
        output_type=AnswerFormat,
    )


def main() -> None:
    args = parse_args()
    memory = AgentMemory.open(args.project_root)
    try:
        answerer = OpenAIAgentAnswerer(model=args.model_id, max_turns=args.max_turns)
        results: list[dict[str, object]] = []
        for case in BENCHMARK_CASES:
            case_hits: list[HitRecord] = []
            case_retrieval_ms: list[float] = []
            case_call_log: list[dict[str, Any]] = []
            agent = build_cosine_agent(
                memory=memory,
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
                    label="cosine",
                    case=case,
                    contexts=contexts,
                    retrieval_meta=meta,
                    agent_result=agent_result,
                )
            )
        payload = {
            "system": "cosine",
            "model_id": args.model_id,
            "summary": summarize_results(results),
            "results": results,
        }
        args.output_file.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
