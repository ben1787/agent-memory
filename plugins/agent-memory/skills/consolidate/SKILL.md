---
name: consolidate
description: Review today's high-similarity Agent Memory clusters and decide whether to leave them alone or replace them with fewer, more orthogonal memories.
---

# Consolidate Agent Memory

Use this skill for the daily Agent Memory consolidation pass.

## Workflow

1. Inspect the current mode:

```bash
agent-memory consolidation-status --json
```

2. If the mode is still `dry_run`, start the review:

```bash
agent-memory consolidation-start --json
```

3. Fetch the current cluster report:

```bash
agent-memory consolidate --json
```

4. Review the returned clusters.

Rules:
- The report is read-only. It groups memories whose cosine similarity is at least `0.92`.
- Clusters may overlap. Do not assume they are a partition of the memory store.
- Contradiction resolution and timestamp-based truth arbitration are out of scope for this pass.
- If a cluster already looks appropriately distinct, leave it alone.
- If a cluster is redundant or noisy, replace it with fewer, more orthogonal memories.
- During `dry_run`, do not mutate the memory store.

5. In `dry_run`, produce a proposal instead of running changes.

Your proposal should say:
- which clusters you would leave alone
- which memory IDs you would delete
- which memory IDs you would edit
- which new memories you would save
- what you expect the resulting memory set to look like

Wait for explicit approval before applying any edits.

6. After approval, flip the run into apply mode:

```bash
agent-memory consolidation-approve --json
```

7. Then carry out the approved changes with the existing CLI tools:

```bash
agent-memory show <memory_id> --json
agent-memory edit <memory_id> "<new text>"
agent-memory delete <memory_id> --yes
agent-memory save "<new memory>"
```

8. Prefer preserving useful facts over aggressive deletion.

9. When you are done applying the approved plan, mark the run complete:

```bash
agent-memory consolidation-complete --json
```

## Output

- Summarize which clusters you reviewed.
- Say which memories you left unchanged.
- In `dry_run`, list the proposed changes only and say approval is required.
- In apply mode, list the affected memory IDs and the new memories you created.
