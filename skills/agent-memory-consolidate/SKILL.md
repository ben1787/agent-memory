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
- Clusters are built from cosine similarity `>= 0.92`.
- Clusters may overlap.
- `duplicate_groups` surfaces exact text duplicates, same titles, and very similar titles.
- `metadata_variant_groups` surfaces metadata spelling variants such as hyphen/underscore/case/plural drift.
- `metadata_cohorts` surfaces larger same-metadata groups worth scanning together.
- `recent_bursts` surfaces same-day topic bursts that often contain episode notes or redundant saves.
- `quality_flag_groups` surfaces deterministic risk flags such as very short memories, raw transcript markers, PR URLs, commit-like tokens, branch names, one-off process directives, and dated status notes.
- Do not do contradiction resolution or timestamp-based truth arbitration in this pass.
- Leave candidates alone if they already look clean and distinct.
- If a candidate group is redundant or messy, replace it with fewer, more orthogonal memories.
- In `dry_run`, do not mutate the memory store.

4. Decide whether to keep each candidate group or replace it with fewer, more orthogonal memories.

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
