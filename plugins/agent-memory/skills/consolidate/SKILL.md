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

3. Review every relevant section of the returned report.

Rules:
- The report is read-only.
- `clusters` groups memories whose cosine similarity is at least `0.92`.
- Clusters may overlap. Do not assume they are a partition of the memory store.
- `duplicate_groups` surfaces exact text duplicates, same titles, and very similar titles.
- `metadata_variant_groups` surfaces metadata spelling variants such as hyphen/underscore/case/plural drift.
- `metadata_cohorts` surfaces larger same-metadata groups worth scanning together.
- `recent_bursts` surfaces same-day topic bursts that often contain episode notes or redundant saves.
- `quality_flag_groups` surfaces deterministic risk flags such as very short memories, raw transcript markers, PR URLs, commit-like tokens, branch names, one-off process directives, and dated status notes.
- Contradiction resolution and timestamp-based truth arbitration are out of scope for this pass.
- If a candidate group already looks appropriately distinct, leave it alone.
- If a candidate group is redundant or noisy, replace it with fewer, more orthogonal memories.

4. Decide whether to keep each candidate group or replace it with fewer, more orthogonal memories.

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
