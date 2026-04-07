---
name: graph-benchmark-answer
description: Use when a benchmark subagent must answer a question using only the local Agent Memory graph for retrieval. Run fresh on a question, use `agent-memory recall --json` against the provided graph project, and cite only the memory IDs actually used.
---

# Graph Benchmark Answer

Use this skill only for the graph side of the benchmark.

## Inputs

Expect:
- a benchmark question
- a graph project root that contains `.agent-memory/memory.kuzu`

## Retrieval workflow

1. Run:

```bash
uv run agent-memory recall "<QUESTION>" --cwd "<GRAPH_PROJECT_ROOT>" --json
```

2. Inspect the returned clusters and hits.
3. Decide which recalled memories are actually relevant.
4. Answer only from those recalled memories.

## Output rules

- Do not inspect raw article files.
- Do not use outside knowledge unless the retrieved memories are clearly insufficient.
- Return JSON only, with this shape:

```json
{
  "answer": "short answer text",
  "references": ["mem_xxx", "mem_yyy"],
  "checked_memory_ids": ["mem_xxx", "mem_yyy", "mem_zzz"]
}
```

- `references` must contain only the memory IDs you actually used in the answer.
- `checked_memory_ids` can include extra recalled memories you inspected but did not cite.
- If the graph recall is insufficient, say so briefly in `answer` and still include the memory IDs you checked.

## Notes

- The query is only the seed for retrieval. Your evidence comes from the recalled memory hits.
- Prefer a small set of precise memory IDs over citing every retrieved hit.
