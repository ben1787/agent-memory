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

2. If the mode is `dry_run`, start the review:

```bash
agent-memory consolidation-start --json
```

3. Fetch the current cluster report:

```bash
agent-memory consolidate --json
```

4. Review the clusters and decide what to do.

Rules:
- The cluster report is read-only.
- Clusters are built from cosine similarity `>= 0.92`.
- Clusters may overlap.
- Do not do contradiction resolution or timestamp-based truth arbitration in this pass.
- Leave clusters alone if they already look clean and distinct.
- If a cluster is redundant or messy, replace it with fewer, more orthogonal memories.
- In `dry_run`, do not mutate the memory store.

5. In `dry_run`, produce a proposed action plan only.

That proposal should include:
- clusters to keep unchanged
- memory IDs to delete
- memory IDs to edit
- new memories to save
- expected final result after application

Wait for explicit approval before applying anything.

6. After approval, switch to apply mode:

```bash
agent-memory consolidation-approve --json
```

7. Use the existing memory-editing commands as needed:

```bash
agent-memory show <memory_id> --json
agent-memory edit <memory_id> "<new text>"
agent-memory delete <memory_id> --yes
agent-memory save "<new memory>"
```

8. Mark the run complete:

```bash
agent-memory consolidation-complete --json
```
