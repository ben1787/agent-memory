---
name: consolidate
description: Review today's high-similarity Agent Memory clusters and decide whether to leave them alone or replace them with fewer, more orthogonal memories.
---

# Consolidate Agent Memory

Use this skill for the daily Agent Memory consolidation pass.

## Workflow

1. Inspect the current status:

```bash
agent-memory consolidation-status --json
```

2. Fetch the current cluster report:

```bash
agent-memory consolidate --json
```

3. Review the returned clusters.

Rules:
- The report is read-only. It groups memories whose cosine similarity is at least `0.92`.
- Clusters may overlap. Do not assume they are a partition of the memory store.
- Contradiction resolution and timestamp-based truth arbitration are out of scope for this pass.
- If a cluster already looks appropriately distinct, leave it alone.
- If a cluster is redundant or noisy, replace it with fewer, more orthogonal memories.

4. Decide whether to keep each cluster or replace it with fewer, more orthogonal memories.

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
- Say which memories you left unchanged.
- List the affected memory IDs and the new memories you created.
