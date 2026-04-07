---
name: codex-skill-benchmark-orchestrator
description: Use when the main agent must benchmark two fresh Codex skills against the same question set, collect structured JSON outputs from each subagent, measure elapsed time per side, and render a side-by-side HTML report.
---

# Codex Skill Benchmark Orchestrator

Use this skill for the top-level benchmark run.

## Inputs

Expect:
- a benchmark workspace with:
  - `raw_articles/`
  - `graph_project/`
  - `benchmark_cases.json`
- two subagent skills:
  - `graph-benchmark-answer`
  - `raw-file-benchmark-answer`

## Benchmark workflow

For each benchmark question:

1. Spawn one fresh graph subagent with the graph skill.
2. Spawn one fresh raw subagent with the raw-file skill.
3. Give both agents the same question.
4. Require JSON-only output from each agent.
5. Capture elapsed time for each side from spawn to completion.
6. Store the raw completion text as well as parsed JSON.

## Expected subagent output

Graph skill:

```json
{
  "answer": "...",
  "references": ["mem_xxx"],
  "checked_memory_ids": ["mem_xxx", "mem_yyy"]
}
```

Raw skill:

```json
{
  "answer": "...",
  "references": ["Graph theory ¶1"],
  "inspected_files": ["Graph-theory.md"]
}
```

## Result record shape

Store one record per case with:
- `case_id`
- `query`
- `graph.elapsed_ms`
- `graph.answer`
- `graph.references`
- `graph.checked_memory_ids`
- `graph.raw_completion`
- `raw.elapsed_ms`
- `raw.answer`
- `raw.references`
- `raw.inspected_files`
- `raw.raw_completion`

## Reporting

- Write the collected results to JSON.
- Render a side-by-side HTML report from that JSON.
- The HTML should show:
  - question
  - graph answer and refs
  - raw answer and refs
  - timings
  - raw inspected files
  - graph checked memory IDs

## Notes

- Keep each subagent fresh. Do not reuse question-answering agents across cases.
- If a subagent returns invalid JSON, keep the raw completion text and mark the parse failure in the result record.
- Bounded parallelism is fine, but per-question graph/raw pairs are easier to time and compare.
