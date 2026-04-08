"""Run a single benchmark case across raw/graph/cosine agents and dump
the per-tool-call log so we can inspect exactly what each agent asked,
what came back, and how it iterated.

Reason: the full benchmark only tells us aggregate scores; to diagnose
why graph stops calling its tool sooner than raw we need the per-call
queries side by side. This script reuses the same builders the full
benchmark uses so we get identical behavior, then prints a compact
trace per agent.

Usage:
  python scripts/probe_single_case.py --case-id technical-level-9
  python scripts/probe_single_case.py --case-id science-level-6 --recall-limit 12
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agent_memory.benchmark import BENCHMARK_CASES, build_benchmark_config
from agent_memory.embeddings import build_embedder
from agent_memory.end_to_end import (
    DEFAULT_ANSWER_MODEL_ID,
    HitRecord,
    OpenAIAgentAnswerer,
    RawCorpusRetriever,
    flatten_paragraphs,
    hits_to_context_items,
    load_cached_corpus,
)
from agent_memory.engine import AgentMemory

# Import the builders directly from the agent scripts so we never drift
# from production benchmark behavior. The script files live in scripts/
# so we add it to sys.path the cheap way: import via importlib.
import importlib.util

SCRIPTS = Path(__file__).parent


def _load_builder(filename: str, attr: str):
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), SCRIPTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


build_graph_agent = _load_builder("run_graph_agent.py", "build_graph_agent")
build_cosine_agent = _load_builder("run_cosine_agent.py", "build_cosine_agent")
build_raw_agent = _load_builder("run_raw_agent.py", "build_raw_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True, help="e.g. technical-level-9")
    parser.add_argument("--cache-file", type=Path, default=Path("reports/wikipedia_cache.json"))
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--model-id", default=DEFAULT_ANSWER_MODEL_ID)
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args()


def find_case(case_id: str):
    for case in BENCHMARK_CASES:
        if case.case_id == case_id:
            return case
    raise SystemExit(f"unknown case_id={case_id}; expected one of: {[c.case_id for c in BENCHMARK_CASES][:5]}...")


def run_one(label: str, build_fn, query: str, *, model_id: str, max_turns: int, **builder_kwargs) -> dict[str, Any]:
    case_hits: list[HitRecord] = []
    case_retrieval_ms: list[float] = []
    case_call_log: list[dict[str, Any]] = []
    agent = build_fn(
        model=model_id,
        hit_sink=case_hits,
        retrieval_time_ms=case_retrieval_ms,
        call_log=case_call_log,
        **builder_kwargs,
    )
    answerer = OpenAIAgentAnswerer(model=model_id, max_turns=max_turns)
    result = answerer.run(agent, query, case_hits)
    contexts = hits_to_context_items(case_hits)
    return {
        "label": label,
        "answer": result.answer,
        "tool_call_count": result.tool_call_count,
        "model_turns": result.model_turns,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "generation_ms": round(result.generation_ms, 1),
        "total_retrieval_ms": round(sum(case_retrieval_ms), 1),
        "tool_call_log": case_call_log,
        "deduped_context_count": len(contexts),
        "deduped_context_ids": [c.reference_id for c in contexts],
    }


def print_trace(label: str, trace: dict[str, Any]) -> None:
    print(f"\n{'='*90}\n{label.upper()}")
    print(f"  tool_calls={trace['tool_call_count']}  turns={trace['model_turns']}  "
          f"prompt_tokens={trace['input_tokens']}  gen_ms={trace['generation_ms']}  "
          f"retrieval_ms={trace['total_retrieval_ms']}")
    print(f"  deduped union: {trace['deduped_context_count']} unique items")
    for i, call in enumerate(trace["tool_call_log"], start=1):
        print(f"  call {i}: limit={call['limit']} ms={call['retrieval_ms']} hits={call['hit_count']}")
        print(f"    query: {call['query']}")
        for hit in call["hits"][:5]:
            print(f"      {hit['memory_id'][:40]:40} score={hit['score']:.4f} | {hit.get('title','?')}")
        if len(call["hits"]) > 5:
            print(f"      ... +{len(call['hits']) - 5} more")
    print(f"\n  ANSWER ({len(trace['answer'])} chars):")
    print("  " + trace["answer"][:1200].replace("\n", "\n  "))


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    case = find_case(args.case_id)

    corpus = load_cached_corpus(args.cache_file)
    limited = {title: paragraphs for title, paragraphs in list(corpus.items())[: args.total_articles]}

    config = build_benchmark_config("fastembed")
    embedder = build_embedder(config)
    paragraphs = flatten_paragraphs(limited, embedder)

    # Build the graph store on disk so the graph + cosine agents can use it
    import tempfile
    workspace = Path(tempfile.mkdtemp(prefix="probe-single-case-"))
    memory = AgentMemory.initialize(workspace, config=config, force=True, embedder=embedder)
    try:
        for p in paragraphs:
            memory.save(p["text"], embedding=p["embedding"])
    finally:
        memory.close()

    memory = AgentMemory.open(workspace)
    retriever = RawCorpusRetriever(paragraphs, embedder)
    common = dict(model_id=args.model_id, max_turns=args.max_turns)

    print(f"\nCASE: {case.case_id}")
    print(f"Q: {case.query}")
    print(f"Expected titles: {case.expected_titles}")
    print(f"Required terms: {case.required_terms}")

    try:
        raw_trace = run_one("raw", build_raw_agent, case.query,
                            retriever=retriever, recall_limit=args.recall_limit, **common)
        graph_trace = run_one("graph", build_graph_agent, case.query,
                              memory=memory, recall_limit=args.recall_limit, **common)
        cosine_trace = run_one("cosine", build_cosine_agent, case.query,
                               memory=memory, recall_limit=args.recall_limit, **common)
    finally:
        memory.close()
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)

    for name, trace in (("raw", raw_trace), ("graph", graph_trace), ("cosine", cosine_trace)):
        print_trace(name, trace)

    if args.output_file:
        args.output_file.write_text(json.dumps(
            {"case_id": case.case_id, "query": case.query,
             "raw": raw_trace, "graph": graph_trace, "cosine": cosine_trace},
            indent=2,
        ) + "\n")


if __name__ == "__main__":
    main()
