"""Benchmark raw / graph / cosine answering agents in one process.

Reason: previous version shelled out to three subprocesses (each cold-loading
fastembed, with the raw subprocess re-embedding the entire corpus from scratch
and a tempdir graph store that got deleted at the end of every run) and ran
its 60 OpenAI calls sequentially via Runner.run_sync. End-to-end ~8 minutes.

This version:
- Persists the embedded graph store + raw paragraph embeddings on disk under
  reports/.benchmark_store/{corpus_hash}/, so the embedding step happens once
  per unique corpus and is reused on every subsequent run.
- Runs everything in one process — no subprocess fan-out, no JSON ferrying,
  one fastembed instance shared by all paths.
- Opens the AgentMemory store read_only so concurrent recall calls cannot
  race on touch_memories / last_recalled writes.
- Schedules all (case, agent) pairs through asyncio.gather using the new
  OpenAIAgentAnswerer.run_async, so 60 OpenAI requests run concurrently
  instead of sequentially. Wall time drops from ~8 min to ~15 s on warm cache.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

from agent_memory.benchmark import BENCHMARK_CASES, build_benchmark_config
from agent_memory.embeddings import build_embedder
from agent_memory.end_to_end import (
    DEFAULT_ANSWER_MODEL_ID,
    HitRecord,
    JudgeVerdict,
    OpenAIAgentAnswerer,
    RawCorpusRetriever,
    build_judge_agent,
    evaluate_path,
    flatten_paragraphs,
    hits_to_context_items,
    load_cached_corpus,
    summarize_results,
)
from agents import Runner
import random
from agent_memory.engine import AgentMemory
from agent_memory.reporting import render_isolated_benchmark_report


SCRIPT_DIR = Path(__file__).parent


def _load_factory(filename: str, attr: str):
    # Reason: the per-agent build_*_agent factories live in sibling scripts
    # that were originally CLI entry points. Importing them via importlib
    # keeps a single source of truth for tool definitions without forcing
    # those scripts onto sys.path or restructuring them as a package.
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr)


build_raw_agent = _load_factory("run_raw_agent.py", "build_raw_agent")
build_graph_agent = _load_factory("run_graph_agent.py", "build_graph_agent")
build_cosine_agent = _load_factory("run_cosine_agent.py", "build_cosine_agent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw, graph, and cosine answering agents in one process."
    )
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--total-articles", type=int, default=100)
    parser.add_argument("--paragraphs-per-article", type=int, default=0)
    parser.add_argument("--embedding-backend", choices=["fastembed", "hash"], default="fastembed")
    # Reason: all three retrieval tools expose a `limit` parameter the agent
    # can override; this sets the default the agent sees when it just calls
    # the tool with no explicit limit. Same default across all three so the
    # comparison stays apples-to-apples.
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--model-id", default=None,
                        help="Override the default OpenAI model (defaults to DEFAULT_ANSWER_MODEL_ID).")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--html-file", type=Path)
    # Reason: cache MUST live outside the agent-memory repo because the repo
    # itself owns a parent .agent-memory/ store and the engine refuses nested
    # stores along an ancestor chain (config.py:242). ~/.cache/ is the
    # standard XDG-style location and survives across runs.
    parser.add_argument("--cache-root", type=Path,
                        default=Path.home() / ".cache" / "agent-memory-benchmark",
                        help="Where to keep persisted graph store + paragraph embeddings between runs.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuilding the persistent embedded store even if it already exists.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-as-judge subjective scoring pass (saves ~20 OpenAI calls).")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _corpus_cache_key(limited_corpus: dict[str, list[str]], embedding_backend: str) -> str:
    # Reason: hash the canonicalized corpus + embedding backend so any change
    # to either invalidates the cached store automatically. Truncated to 16
    # chars because that's already 64 bits of collision resistance and keeps
    # the directory name readable.
    canonical = json.dumps(
        {"corpus": limited_corpus, "backend": embedding_backend},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _ensure_persisted_cache(
    cache_dir: Path,
    limited_corpus: dict[str, list[str]],
    config: Any,
    embedder: Any,
    rebuild: bool,
) -> tuple[Path, list[dict[str, object]]]:
    # Reason: build the embedded graph store + raw-paragraph embeddings exactly
    # once per (corpus, backend) and reuse them. Returns the AgentMemory project
    # root and the in-memory paragraph list (with embeddings) for the raw retriever.
    graph_root = cache_dir / "graph"
    raw_pkl = cache_dir / "raw_paragraphs.pkl"

    if not rebuild and graph_root.exists() and raw_pkl.exists():
        with raw_pkl.open("rb") as fh:
            paragraphs = pickle.load(fh)
        return graph_root, paragraphs

    # Cold build path. Embed once, populate the store, write the sidecar.
    if cache_dir.exists() and rebuild:
        import shutil
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    paragraphs = flatten_paragraphs(limited_corpus, embedder)
    memory = AgentMemory.initialize(graph_root, config=config, force=True, embedder=embedder)
    try:
        for paragraph in paragraphs:
            memory.save(paragraph["text"], embedding=paragraph["embedding"])
    finally:
        memory.close()

    with raw_pkl.open("wb") as fh:
        pickle.dump(paragraphs, fh)
    return graph_root, paragraphs


async def _run_single_case(
    *,
    label: str,
    build_fn: Any,
    case: Any,
    answerer: OpenAIAgentAnswerer,
    model_id: str,
    recall_limit: int,
    builder_kwargs: dict[str, Any],
) -> dict[str, object]:
    # Reason: each (label, case) gets its own fresh closure state. The
    # build_*_agent factories accept hit_sink/retrieval_time_ms/call_log as
    # parameters, so the only thing we need is to allocate them per task and
    # never share. That keeps async fan-out completely isolated.
    case_hits: list[HitRecord] = []
    case_retrieval_ms: list[float] = []
    case_call_log: list[dict[str, Any]] = []
    agent = build_fn(
        model=model_id,
        hit_sink=case_hits,
        retrieval_time_ms=case_retrieval_ms,
        call_log=case_call_log,
        recall_limit=recall_limit,
        **builder_kwargs,
    )
    agent_result = await answerer.run_async(agent, case.query, case_hits)
    contexts = hits_to_context_items(case_hits)
    meta = {
        "retrieval_ms": round(sum(case_retrieval_ms), 3),
        "tool_calls": len(case_retrieval_ms),
        "tool_call_log": case_call_log,
    }
    return evaluate_path(
        label=label,
        case=case,
        contexts=contexts,
        retrieval_meta=meta,
        agent_result=agent_result,
    )


async def _run_all_cases(
    *,
    memory: AgentMemory,
    retriever: RawCorpusRetriever,
    answerer: OpenAIAgentAnswerer,
    model_id: str,
    recall_limit: int,
) -> dict[str, list[dict[str, object]]]:
    tasks: list[tuple[str, Any]] = []
    for case in BENCHMARK_CASES:
        tasks.append((
            "raw",
            _run_single_case(
                label="raw",
                build_fn=build_raw_agent,
                case=case,
                answerer=answerer,
                model_id=model_id,
                recall_limit=recall_limit,
                builder_kwargs={"retriever": retriever},
            ),
        ))
        tasks.append((
            "graph",
            _run_single_case(
                label="graph",
                build_fn=build_graph_agent,
                case=case,
                answerer=answerer,
                model_id=model_id,
                recall_limit=recall_limit,
                builder_kwargs={"memory": memory},
            ),
        ))
        tasks.append((
            "cosine",
            _run_single_case(
                label="cosine",
                build_fn=build_cosine_agent,
                case=case,
                answerer=answerer,
                model_id=model_id,
                recall_limit=recall_limit,
                builder_kwargs={"memory": memory},
            ),
        ))

    coros = [coro for _, coro in tasks]
    completed = await asyncio.gather(*coros)
    by_label: dict[str, list[dict[str, object]]] = {"raw": [], "graph": [], "cosine": []}
    for (label, _), result in zip(tasks, completed):
        by_label[label].append(result)
    return by_label


async def _judge_one_case(
    *,
    case_id: str,
    query: str,
    answers_by_system: dict[str, str],
    answerer: OpenAIAgentAnswerer,
    model_id: str,
) -> dict[str, Any]:
    # Reason: anonymize the systems by random A/B/C permutation per case so
    # the judge cannot pattern-match on style or position. We track the
    # mapping so we can re-attribute scores back to systems after the call.
    systems = list(answers_by_system.keys())
    random.shuffle(systems)
    letter_to_system = {"A": systems[0], "B": systems[1], "C": systems[2]}

    prompt = (
        f"QUESTION:\n{query}\n\n"
        f"ANSWER A:\n{answers_by_system[letter_to_system['A']]}\n\n"
        f"ANSWER B:\n{answers_by_system[letter_to_system['B']]}\n\n"
        f"ANSWER C:\n{answers_by_system[letter_to_system['C']]}"
    )
    judge = build_judge_agent(model=model_id)
    result = await Runner.run(judge, prompt, run_config=answerer.run_config, max_turns=2)
    verdict: JudgeVerdict = result.final_output  # type: ignore[assignment]

    scores_by_system = {
        letter_to_system["A"]: verdict.score_a.model_dump(),
        letter_to_system["B"]: verdict.score_b.model_dump(),
        letter_to_system["C"]: verdict.score_c.model_dump(),
    }
    winner_letter = verdict.overall_winner.strip().upper()
    if winner_letter in letter_to_system:
        winner_system = letter_to_system[winner_letter]
    else:
        winner_system = "tie"
    return {
        "case_id": case_id,
        "letter_to_system": letter_to_system,
        "scores": scores_by_system,
        "winner_letter": winner_letter,
        "winner_system": winner_system,
        "reasoning": verdict.reasoning,
    }


async def _run_llm_judge(
    *,
    by_label: dict[str, list[dict[str, object]]],
    answerer: OpenAIAgentAnswerer,
    model_id: str,
) -> dict[str, Any]:
    # Reason: build per-case answer triples keyed by case_id so we can pair
    # them up correctly even though gather() preserves submission order. We
    # then run all 20 judge calls in parallel and aggregate.
    raw_by_case = {r["case_id"]: r for r in by_label["raw"]}
    graph_by_case = {r["case_id"]: r for r in by_label["graph"]}
    cosine_by_case = {r["case_id"]: r for r in by_label["cosine"]}

    coros = []
    case_ids: list[str] = []
    for case_id, raw_row in raw_by_case.items():
        if case_id not in graph_by_case or case_id not in cosine_by_case:
            continue
        case_ids.append(case_id)
        coros.append(_judge_one_case(
            case_id=case_id,
            query=str(raw_row["query"]),
            answers_by_system={
                "raw": str(raw_row["answer"]),
                "graph": str(graph_by_case[case_id]["answer"]),
                "cosine": str(cosine_by_case[case_id]["answer"]),
            },
            answerer=answerer,
            model_id=model_id,
        ))

    per_case = await asyncio.gather(*coros)

    # Aggregate: per-system mean scores per criterion, win counts.
    criteria = ("completeness", "accuracy", "depth", "clarity")
    sums: dict[str, dict[str, float]] = {s: {c: 0.0 for c in criteria} for s in ("raw", "graph", "cosine")}
    counts: dict[str, int] = {s: 0 for s in ("raw", "graph", "cosine")}
    wins: dict[str, int] = {"raw": 0, "graph": 0, "cosine": 0, "tie": 0}
    for verdict in per_case:
        for system, score in verdict["scores"].items():
            counts[system] += 1
            for c in criteria:
                sums[system][c] += float(score[c])
        wins[verdict["winner_system"]] = wins.get(verdict["winner_system"], 0) + 1

    averages: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    for system in ("raw", "graph", "cosine"):
        n = max(counts[system], 1)
        avg = {c: round(sums[system][c] / n, 3) for c in criteria}
        averages[system] = avg
        overall[system] = round(sum(avg.values()) / len(criteria), 3)

    return {
        "per_case": per_case,
        "averages": averages,
        "overall": overall,
        "wins": wins,
        "criteria": list(criteria),
    }


def _compute_graph_cosine_overlap(
    graph_results: list[dict[str, object]],
    cosine_results: list[dict[str, object]],
) -> dict[str, object]:
    cosine_by_case = {r["case_id"]: r for r in cosine_results}
    per_case: list[dict[str, object]] = []
    jaccards: list[float] = []
    for graph_row in graph_results:
        case_id = graph_row["case_id"]
        cosine_row = cosine_by_case.get(case_id)
        if cosine_row is None:
            continue
        graph_ids = {ref["reference_id"] for ref in graph_row.get("context_references", [])}
        cosine_ids = {ref["reference_id"] for ref in cosine_row.get("context_references", [])}
        intersection = graph_ids & cosine_ids
        union = graph_ids | cosine_ids
        jaccard = len(intersection) / len(union) if union else 0.0
        jaccards.append(jaccard)
        per_case.append(
            {
                "case_id": case_id,
                "graph_count": len(graph_ids),
                "cosine_count": len(cosine_ids),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": round(jaccard, 4),
                "shared_ids": sorted(intersection),
                "graph_only_ids": sorted(graph_ids - cosine_ids),
                "cosine_only_ids": sorted(cosine_ids - graph_ids),
            }
        )
    mean_jaccard = round(sum(jaccards) / len(jaccards), 4) if jaccards else 0.0
    return {
        "mean_jaccard": mean_jaccard,
        "cases_compared": len(per_case),
        "per_case": per_case,
    }


def print_summary(payload: dict[str, object]) -> None:
    print(f"Model: {payload['model_id']}")
    print(
        f"Corpus: articles={payload['total_articles']} paragraphs/article={payload['paragraphs_per_article']}"
    )
    judge = payload.get("llm_judge") or {}
    judge_overall = judge.get("overall", {}) if judge else {}
    judge_wins = judge.get("wins", {}) if judge else {}
    # Reason: print both scores side by side. context_score is the
    # deterministic keyword-coverage rubric (good as a sanity check on
    # retrieval); judge_score is the subjective LLM-as-judge mean over 4
    # criteria (the score we actually trust for answer quality).
    header = f"{'system':8} {'total_ms':>10} {'ctx_tok':>9} {'prompt_tok':>11} {'tool_calls':>11} {'context':>9} {'judge':>7} {'wins':>5}"
    print(header)
    print("-" * len(header))
    for label in ("graph", "raw", "cosine"):
        s = payload[label]["summary"]
        judge_score = judge_overall.get(label, "—")
        wins = judge_wins.get(label, "—")
        print(
            f"{label:8} "
            f"{s['average_total_ms']:>10.0f} "
            f"{s['average_context_tokens']:>9.0f} "
            f"{s['average_prompt_tokens']:>11.0f} "
            f"{s.get('average_tool_calls','—'):>11} "
            f"{s['average_context_score']:>9.4f} "
            f"{judge_score if judge_score == '—' else f'{judge_score:.3f}':>7} "
            f"{wins:>5}"
        )
    if judge:
        ties = judge_wins.get("tie", 0)
        criteria = judge.get("criteria", [])
        averages = judge.get("averages", {})
        print()
        print(f"LLM judge wins: graph={judge_wins.get('graph',0)} raw={judge_wins.get('raw',0)} cosine={judge_wins.get('cosine',0)} tie={ties}")
        if criteria and averages:
            print(f"Per-criterion averages ({', '.join(criteria)}):")
            for label in ("graph", "raw", "cosine"):
                vals = averages.get(label, {})
                print(f"  {label:7} " + "  ".join(f"{c}={vals.get(c,0):.2f}" for c in criteria))
    overlap = payload.get("graph_vs_cosine_overlap")
    if overlap:
        print()
        print(
            f"Graph vs Cosine retrieval overlap: mean Jaccard={overlap['mean_jaccard']} "
            f"across {overlap['cases_compared']} cases"
        )


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set before running this benchmark.")

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

    cache_key = _corpus_cache_key(limited_corpus, args.embedding_backend)
    cache_dir = args.cache_root / cache_key
    cache_status = "hit" if (cache_dir / "graph").exists() and (cache_dir / "raw_paragraphs.pkl").exists() else "miss"
    print(f"Cache: {cache_dir} ({cache_status})")

    setup_started = time.perf_counter()
    graph_root, paragraphs = _ensure_persisted_cache(
        cache_dir, limited_corpus, config, embedder, rebuild=args.rebuild_cache
    )
    setup_ms = (time.perf_counter() - setup_started) * 1000
    print(f"Setup: {setup_ms:.0f} ms ({len(paragraphs)} paragraphs)")

    # Reason: read_only=True keeps recall() from writing touch_memories /
    # last_recalled, which would otherwise race when multiple asyncio tasks
    # call recall on the same store concurrently.
    memory = AgentMemory.open(graph_root, read_only=True)
    retriever = RawCorpusRetriever(paragraphs, embedder)
    model_id = args.model_id or DEFAULT_ANSWER_MODEL_ID
    answerer = OpenAIAgentAnswerer(model=model_id, max_turns=args.max_turns)

    bench_started = time.perf_counter()
    try:
        by_label = asyncio.run(_run_all_cases(
            memory=memory,
            retriever=retriever,
            answerer=answerer,
            model_id=model_id,
            recall_limit=args.recall_limit,
        ))
    finally:
        memory.close()
    bench_ms = (time.perf_counter() - bench_started) * 1000
    print(f"Benchmark: {bench_ms:.0f} ms ({len(BENCHMARK_CASES)} cases x 3 agents = {len(BENCHMARK_CASES)*3} runs)")

    # Reason: preserve the case ordering from BENCHMARK_CASES even though
    # asyncio.gather returns in submission order — _run_all_cases already
    # interleaves submissions case-by-case in groups of three, so each
    # by_label list is in case order.
    raw_payload = {
        "system": "raw",
        "model_id": model_id,
        "summary": summarize_results(by_label["raw"]),
        "results": by_label["raw"],
    }
    graph_payload = {
        "system": "graph",
        "model_id": model_id,
        "summary": summarize_results(by_label["graph"]),
        "results": by_label["graph"],
    }
    cosine_payload = {
        "system": "cosine",
        "model_id": model_id,
        "summary": summarize_results(by_label["cosine"]),
        "results": by_label["cosine"],
    }

    payload: dict[str, object] = {
        "model_id": model_id,
        "total_articles": args.total_articles,
        "paragraphs_per_article": args.paragraphs_per_article,
        "raw": raw_payload,
        "graph": graph_payload,
        "cosine": cosine_payload,
        "cache_dir": str(cache_dir),
        "setup_ms": round(setup_ms, 1),
        "benchmark_ms": round(bench_ms, 1),
    }
    payload["graph_vs_cosine_overlap"] = _compute_graph_cosine_overlap(
        graph_payload["results"], cosine_payload["results"]
    )

    if not args.no_judge:
        # Reason: 20 extra OpenAI calls (one per case, each scoring all 3
        # answers anonymized as A/B/C). Runs in parallel via asyncio.gather
        # so wall-time impact is bounded by the slowest single judge call.
        judge_started = time.perf_counter()
        judge_summary = asyncio.run(_run_llm_judge(
            by_label=by_label,
            answerer=answerer,
            model_id=model_id,
        ))
        judge_ms = (time.perf_counter() - judge_started) * 1000
        judge_summary["judge_ms"] = round(judge_ms, 1)
        payload["llm_judge"] = judge_summary
        print(f"Judge: {judge_ms:.0f} ms ({len(BENCHMARK_CASES)} cases)")

    if args.output_file:
        args.output_file.write_text(json.dumps(payload, indent=2) + "\n")
    if args.html_file:
        args.html_file.write_text(render_isolated_benchmark_report(payload))

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_summary(payload)


if __name__ == "__main__":
    main()
