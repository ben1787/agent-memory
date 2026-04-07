from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from agents import Agent, ModelSettings, RunConfig, Runner, ShellCallOutcome, ShellCommandOutput, ShellResult, ShellTool, function_tool
from pydantic import BaseModel, Field

from agent_memory.engine import AgentMemory


SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def safe_slug(title: str) -> str:
    slug = SAFE_FILENAME_PATTERN.sub("-", title.strip()).strip("-._")
    return slug or "article"


class RawAnswer(BaseModel):
    answer: str
    references: list[str] = Field(default_factory=list)
    inspected_files: list[str] = Field(default_factory=list)


class GraphAnswer(BaseModel):
    answer: str
    references: list[str] = Field(default_factory=list)
    checked_memory_ids: list[str] = Field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw-file and graph retrieval using the OpenAI Agents SDK."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/Users/benjaminlieblich/repos/agent-memory/reports/codex-benchmark-workspace"),
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Restrict the run to one or more benchmark case IDs.",
    )
    parser.add_argument(
        "--capture-debug",
        action="store_true",
        help="Persist tool-call and token-usage traces for each agent run.",
    )
    return parser.parse_args()


def load_cases(
    cases_file: Path,
    limit: int | None,
    case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    cases = json.loads(cases_file.read_text())
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case["case_id"] in wanted]
    if limit is not None:
        return cases[:limit]
    return cases


def load_existing_results(output_json: Path) -> list[dict[str, Any]]:
    if not output_json.exists():
        return []
    try:
        payload = json.loads(output_json.read_text())
    except json.JSONDecodeError:
        return []
    cases = payload.get("cases")
    if isinstance(cases, list):
        return cases
    return []


SAFE_SHELL_PREFIXES = {
    "pwd",
    "ls",
    "find",
    "rg",
    "cat",
    "sed",
    "head",
    "tail",
    "wc",
    "grep",
}
DISALLOWED_SHELL_TOKENS = {
    "..",
    "/",
    "~",
    ".agent-memory",
    "graph_project",
    "python",
    "python3",
    "node",
    "curl",
    "wget",
    "git",
    "env",
}


def _validate_shell_command(command: str) -> str | None:
    stripped = command.strip()
    if not stripped:
        return "Empty command is not allowed."
    first = stripped.split()[0]
    if first not in SAFE_SHELL_PREFIXES:
        return (
            "Only read-only inspection commands are allowed: "
            + ", ".join(sorted(SAFE_SHELL_PREFIXES))
        )
    for token in DISALLOWED_SHELL_TOKENS:
        if token in stripped:
            return f"Command contains a disallowed token: {token}"
    return None


def build_raw_tools(raw_articles_dir: Path) -> list[Any]:
    def run_shell(request: Any) -> ShellResult:
        outputs: list[ShellCommandOutput] = []
        timeout_ms = request.data.action.timeout_ms or 20_000
        for command in request.data.action.commands:
            validation_error = _validate_shell_command(command)
            if validation_error is not None:
                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stderr=validation_error,
                        outcome=ShellCallOutcome(type="exit", exit_code=2),
                    )
                )
                continue

            try:
                completed = subprocess.run(
                    ["/bin/zsh", "-lc", command],
                    cwd=raw_articles_dir,
                    capture_output=True,
                    text=True,
                    timeout=max(timeout_ms / 1000, 1),
                )
                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        outcome=ShellCallOutcome(
                            type="exit",
                            exit_code=completed.returncode,
                        ),
                    )
                )
            except subprocess.TimeoutExpired as exc:
                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stdout=exc.stdout or "",
                        stderr=(exc.stderr or "") + "\nCommand timed out.",
                        outcome=ShellCallOutcome(type="timeout"),
                    )
                )

        return ShellResult(output=outputs)

    return [
        ShellTool(
            executor=run_shell,
            environment={"type": "local"},
        )
    ]


def compact_recall_payload(result: Any) -> dict[str, Any]:
    return {
        "root": "query",
        "top_query_similarity": round(result.seed_score, 4),
        "hit_count": len(result.hits),
        "hits": [
            {
                "memory_id": hit.memory_id,
                "score": hit.score,
                "query_similarity": hit.query_similarity,
                "text": hit.text,
            }
            for hit in result.hits
        ],
    }


def build_graph_tools(graph_project_root: Path) -> list[Any]:
    @function_tool
    def recall_memories(
        query: str,
        limit: int = 15,
    ) -> str:
        """Recall the highest-scoring memories from the local Agent Memory graph."""
        memory = AgentMemory.open(graph_project_root)
        try:
            result = memory.recall(query, limit=max(1, min(limit, 50)))
        finally:
            memory.close()
        return json.dumps(compact_recall_payload(result), indent=2)

    return [recall_memories]


def build_agents(model: str, workspace_root: Path) -> tuple[Agent[Any], Agent[Any]]:
    raw_articles_dir = workspace_root / "raw_articles"
    graph_project_root = workspace_root / "graph_project"

    raw_agent = Agent(
        name="Raw Benchmark Agent",
        instructions=(
            "Answer questions using only the raw article files available in the current shell working directory. "
            "You may inspect files with shell commands such as `ls`, `rg --files`, `rg -n`, `sed -n`, and `cat`. "
            "There is an `INDEX.md` file in the directory that you may open if useful. "
            "Do not assume article contents without opening the files yourself. "
            "Cite only paragraph labels you actually used, and list only the files you actually opened. "
            "The `answer` field must be a short plain-English string, not an object, list, or markdown block."
        ),
        model=model,
        tools=build_raw_tools(raw_articles_dir),
        output_type=RawAnswer,
    )

    graph_agent = Agent(
        name="Graph Benchmark Agent",
        instructions=(
            "Answer questions using only the local Agent Memory graph recall tool. "
            "Use recalled memories as your evidence, cite only the memory IDs you actually used, "
            "and list the memory IDs you inspected while forming the answer. "
            "You may call the tool multiple times: start with a broad query, read the returned memories, "
            "then refine the query if needed. The tool returns memories ordered from highest to lowest "
            "path-product score from the query root. You may set `limit`, but keep it reasonable; "
            "the tool caps it at 50. "
            "The `answer` field must be a short plain-English string, not an object, list, or markdown block."
        ),
        model=model,
        tools=build_graph_tools(graph_project_root),
        output_type=GraphAnswer,
    )
    return raw_agent, graph_agent


def summarize_usage(response: Any, index: int) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"index": index}
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    output = getattr(response, "output", None) or []
    return {
        "index": index,
        "requests": getattr(usage, "requests", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_tokens": getattr(input_details, "cached_tokens", None),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
        "output_item_types": [getattr(item, "type", type(item).__name__) for item in output],
    }


def summarize_trace(result: Any) -> dict[str, Any]:
    tool_calls: list[dict[str, Any]] = []
    call_index: dict[str, dict[str, Any]] = {}
    pending_calls: list[dict[str, Any]] = []

    for item in result.new_items:
        item_type = type(item).__name__
        if item_type == "ToolCallItem":
            raw_item = getattr(item, "raw_item", None)
            call = {
                "name": getattr(raw_item, "name", None),
                "arguments": getattr(raw_item, "arguments", None),
                "call_id": getattr(raw_item, "call_id", None),
                "status": getattr(raw_item, "status", None),
            }
            tool_calls.append(call)
            call_id = call.get("call_id")
            if call_id:
                call_index[call_id] = call
            else:
                pending_calls.append(call)
            continue

        if item_type != "ToolCallOutputItem":
            continue

        raw_item = getattr(item, "raw_item", None)
        call_id = raw_item.get("call_id") if isinstance(raw_item, dict) else None
        output = getattr(item, "output", "")
        output_text = output if isinstance(output, str) else json.dumps(output, default=str)
        entry = call_index.get(call_id)
        if entry is None and pending_calls:
            entry = pending_calls.pop(0)
        if entry is None:
            entry = {"name": None, "arguments": None, "call_id": call_id}
            tool_calls.append(entry)
        entry["output_chars"] = len(output_text)
        entry["output_preview"] = output_text[:2000]
        entry["output_error"] = "An error occurred while running the tool." in output_text

    responses = [summarize_usage(response, index) for index, response in enumerate(result.raw_responses)]
    total_input_tokens = sum(item.get("input_tokens") or 0 for item in responses)
    total_output_tokens = sum(item.get("output_tokens") or 0 for item in responses)
    total_cached_tokens = sum(item.get("cached_tokens") or 0 for item in responses)
    return {
        "model_turns": len(result.raw_responses),
        "tool_call_count": len(tool_calls),
        "tool_calls": tool_calls,
        "responses": responses,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cached_tokens": total_cached_tokens,
    }


def run_agent(
    agent: Agent[Any],
    prompt: str,
    run_config: RunConfig,
) -> tuple[BaseModel, float, dict[str, Any]]:
    started = time.perf_counter()
    result = Runner.run_sync(agent, prompt, run_config=run_config, max_turns=20)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    final = result.final_output
    if not isinstance(final, BaseModel):
        raise TypeError(f"Unexpected final output type: {type(final)!r}")
    return final, elapsed_ms, summarize_trace(result)


def normalize_raw(result: RawAnswer, elapsed_ms: float, trace: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = result.model_dump()
    payload["elapsed_ms"] = elapsed_ms
    if trace is not None:
        payload["trace"] = trace
    return payload


def normalize_graph(
    result: GraphAnswer,
    elapsed_ms: float,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = result.model_dump()
    payload["elapsed_ms"] = elapsed_ms
    if trace is not None:
        payload["trace"] = trace
    return payload


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set before running this benchmark.")

    workspace_root = args.workspace_root.resolve()
    cases = load_cases(workspace_root / "benchmark_cases.json", args.limit, args.case_id)
    raw_agent, graph_agent = build_agents(args.model, workspace_root)
    existing_results = load_existing_results(args.output_json)
    completed_case_ids = {item["case_id"] for item in existing_results}
    run_config = RunConfig(
        model=args.model,
        tracing_disabled=True,
        model_settings=ModelSettings(
            max_tokens=800,
            verbosity="low",
            store=False,
        ),
    )

    results: list[dict[str, Any]] = list(existing_results)
    remaining_cases = [case for case in cases if case["case_id"] not in completed_case_ids]
    total_cases = len(cases)
    if existing_results:
        print(
            f"Resuming from {len(existing_results)} completed case(s); {len(remaining_cases)} remaining.",
            flush=True,
        )
    for index, case in enumerate(remaining_cases, start=len(existing_results) + 1):
        query = case["query"]
        print(f"[{index}/{total_cases}] {case['case_id']} raw", flush=True)
        raw_prompt = (
            f"Case {case['case_id']}.\n"
            f"Question: {query}\n\n"
            "Return structured output only. "
            "Keep `answer` concise: 2-4 plain sentences max. "
            "Do not return nested objects."
        )
        raw_result, raw_elapsed_ms, raw_trace = run_agent(raw_agent, raw_prompt, run_config)
        print(
            f"[{index}/{total_cases}] {case['case_id']} raw done in {raw_elapsed_ms:.1f} ms",
            flush=True,
        )

        print(f"[{index}/{total_cases}] {case['case_id']} graph", flush=True)
        graph_prompt = (
            f"Case {case['case_id']}.\n"
            f"Question: {query}\n\n"
            "Return structured output only. "
            "Keep `answer` concise: 2-4 plain sentences max. "
            "Do not return nested objects."
        )
        graph_result, graph_elapsed_ms, graph_trace = run_agent(graph_agent, graph_prompt, run_config)
        print(
            f"[{index}/{total_cases}] {case['case_id']} graph done in {graph_elapsed_ms:.1f} ms",
            flush=True,
        )

        case_payload = {
            "case_id": case["case_id"],
            "query": query,
            "raw": normalize_raw(
                raw_result,
                raw_elapsed_ms,
                trace=raw_trace if args.capture_debug else None,
            ),
            "graph": normalize_graph(
                graph_result,
                graph_elapsed_ms,
                trace=graph_trace if args.capture_debug else None,
            ),
        }
        results.append(case_payload)

        payload = {
            "runner": "openai-agents-sdk",
            "model": args.model,
            "workspace_root": str(workspace_root),
            "cases": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        if args.output_html:
            from subprocess import run

            run(
                [
                    sys.executable,
                    "scripts/render_codex_skill_benchmark.py",
                    "--input-file",
                    str(args.output_json),
                    "--output-file",
                    str(args.output_html),
                ],
                check=True,
            )

    payload = {
        "runner": "openai-agents-sdk",
        "model": args.model,
        "workspace_root": str(workspace_root),
        "cases": results,
    }

    print(f"Wrote benchmark JSON to {args.output_json}")
    if args.output_html:
        print(f"Wrote benchmark HTML to {args.output_html}")


if __name__ == "__main__":
    main()
