---
name: consolidate
description: Review today's Agent Memory consolidation candidates and decide whether to leave them alone or replace them with fewer, more orthogonal memories.
---

# Consolidate Agent Memory

Use this skill for the daily Agent Memory consolidation pass.

## Workflow

1. Inspect the current status:

```bash
agent-memory consolidation-status --json
```

2. Fetch the current cleanup worklist:

```bash
agent-memory consolidate --json
```

The top-level command writes the full compact worklist to `.agent-memory/consolidation-report.json` and prints a short JSON run summary with `report_path`. Read that file before editing memories. The stdout summary is intentionally small so terminal truncation does not hide the next step.

3. Review every relevant section of the returned report.

Rules:
- The report is read-only.
- The default JSON stdout is a compact run summary. The full compact worklist is written to `report_path`; it intentionally does not dump full memory bodies.
- The JSON includes an `instructions` block with section actions, drilldown commands, and truncation handling. Follow it when the report is handed to you without this skill text.
- The report has only similarity clusters, standalone metadata tag cleanup, repeatedly negative-rated memories, and sufficiently tried-but-unretrieved memories.
- To inspect one candidate, run `agent-memory consolidate --json --group <group_id>`.
- To inspect a specific memory body, run `agent-memory show <memory_id> --json`.
- `clusters` groups memories whose cosine similarity is at least `0.85`.
- Clusters may overlap. Do not assume they are a partition of the memory store.
- `metadata_cleanup` surfaces similar standalone tag values only; do not assume memory-level redundancy from tag cleanup.
- `negative_feedback_memories` surfaces memories with more than three negative per-memory ratings and zero positive ratings. Editing such a memory resets its prior feedback.
- `unretrieved_memories` surfaces memories with zero accesses only after enough later recall queries exist to make non-retrieval meaningful. Count both direct recall calls and prompt-injection recall calls.
- Contradiction resolution and timestamp-based truth arbitration are out of scope for this pass.
- If a candidate group already looks appropriately distinct, leave it alone.
- If a candidate group is redundant or noisy, replace it with fewer, more orthogonal memories.

4. Decide whether to keep each candidate group or replace it with fewer, more orthogonal memories.

Do not load every memory body in the project. Use the compact worklist to choose a small number of clusters or memory IDs, then drill into only those bodies.

5. Apply the changes immediately with the existing CLI tools:

```bash
agent-memory show <memory_id> --json
agent-memory edit <memory_id> "<new text>"
agent-memory delete <memory_id> --yes
agent-memory save "<new memory>"
```

6. Prefer preserving useful facts over aggressive deletion.

7. When you are done applying the plan, mark the run complete:

```bash
agent-memory consolidation-complete --json
```

## Output

- Summarize which clusters you reviewed.
- Summarize which non-cluster cleanup sections you reviewed when relevant.
- Say which memories you left unchanged.
- List the affected memory IDs and the new memories you created.
