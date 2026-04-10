---
name: agent-memory-consolidate
description: Review Agent Memory similarity clusters and decide whether to keep them or rewrite them into fewer, more orthogonal memories.
---

# Agent Memory Consolidate

Use this skill for the daily Agent Memory consolidation pass.

## Workflow

1. Check the current consolidation mode:

```bash
agent-memory consolidation-status --json
```

2. Fetch the current cluster report:

```bash
agent-memory consolidate --json
```

3. Review the clusters and decide what to do.

Rules:
- The cluster report is read-only.
- Clusters are built from cosine similarity `>= 0.92`.
- Clusters may overlap.
- Do not do contradiction resolution or timestamp-based truth arbitration in this pass.
- Leave clusters alone if they already look clean and distinct.
- If a cluster is redundant or messy, replace it with fewer, more orthogonal memories.
- In `dry_run`, do not mutate the memory store.

4. Decide whether to keep each cluster or replace it with fewer, more orthogonal memories.

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
