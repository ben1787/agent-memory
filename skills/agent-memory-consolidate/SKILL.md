---
name: agent-memory-consolidate
description: Review Agent Memory consolidation candidates and decide whether to keep them or rewrite them into fewer, more orthogonal memories.
---

# Agent Memory Consolidate

Use this skill for the daily Agent Memory consolidation pass.

## Workflow

1. Check the current consolidation mode:

```bash
agent-memory consolidation-status --json
```

2. Fetch the current cleanup worklist:

```bash
agent-memory consolidate --json
```

3. Review every relevant section of the report and decide what to do.

Rules:
- The consolidation report is read-only.
- The default JSON is a compact worklist. It intentionally does not dump full memory bodies.
- The JSON includes an `instructions` block with section actions, drilldown commands, and truncation handling. Follow it when the report is handed to you without this skill text.
- The report has only similarity clusters, standalone metadata tag cleanup, repeatedly negative-rated memories, and sufficiently tried-but-unretrieved memories.
- To inspect one candidate, run `agent-memory consolidate --json --group <group_id>`.
- To inspect a specific memory body, run `agent-memory show <memory_id> --json`.
- Clusters are built from cosine similarity `>= 0.85`.
- Clusters may overlap.
- `metadata_cleanup` surfaces similar standalone tag values only; do not assume memory-level redundancy from tag cleanup.
- `negative_feedback_memories` surfaces memories with more than three negative per-memory ratings and zero positive ratings. Editing such a memory resets its prior feedback.
- `unretrieved_memories` surfaces memories with zero accesses only after enough later recall queries exist to make non-retrieval meaningful. Count both direct recall calls and prompt-injection recall calls.
- Do not do contradiction resolution or timestamp-based truth arbitration in this pass.
- Leave candidates alone if they already look clean and distinct.
- If a candidate group is redundant or messy, replace it with fewer, more orthogonal memories.
- In `dry_run`, do not mutate the memory store.

4. Decide whether to keep each candidate group or replace it with fewer, more orthogonal memories.

Do not load every memory body in the project. Use the compact worklist to choose a small number of clusters or memory IDs, then drill into only those bodies.

5. Apply the edits immediately using the existing memory-editing commands:

```bash
agent-memory show <memory_id> --json
agent-memory edit <memory_id> "<new text>"
agent-memory delete <memory_id> --yes
agent-memory save "<new memory>"
```

6. Mark the run complete:

```bash
agent-memory consolidation-complete --json
```
