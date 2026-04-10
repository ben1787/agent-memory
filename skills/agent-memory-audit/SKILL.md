---
name: agent-memory-audit
description: Audit an existing Agent Memory store against the current memory-worthiness policy, identify memories that should be deleted or rewritten, and apply cleanup store-by-store.
---

# Agent Memory Audit

Use this skill when an existing memory store has drifted away from the current Agent Memory save policy and needs cleanup.

## What should not be in memory

Delete memories that are any of the following:

- Raw transcript fragments, including `User message:` and `Assistant reply:`.
- Standalone greetings like `Hi` or `Hello`.
- Session handoff blobs like "This session is being continued from a previous conversation...".
- One-off process directives such as "use this worktree", "push this branch", "commit and push", or lane ownership instructions.
- Branch / commit / PR / capture-id / payload-hash snapshots that only mattered for a single debugging or release moment.
- Dated run snapshots such as build timings, machine-health observations, live worker counts, or one day's parity status.
- Slack drafting and chatty status summaries.
- Grep-easy facts that can be rediscovered instantly from the repo.
- Stale release/version snapshots that have been superseded.
- Exact duplicates when one canonical copy is enough.

Keep or rewrite memories that are durable, repo-specific, and likely to save future time:

- Workflow rules.
- Architecture maps.
- Search shortcuts.
- Environment quirks.
- Validation or release constraints.
- Stable bug patterns and root causes.
- Recurring product or customer facts.

## Store access

To audit a specific folder's memories, either change into the folder or pass `--cwd`:

```bash
cd /absolute/project/path
agent-memory list --all --json
```

or

```bash
agent-memory list --all --json --cwd /absolute/project/path
```

## Batch review

Do not use regex filters, deterministic scoring, or any other string-matching classifier.
Review the memories directly in batches and decide case-by-case with subjective judgment.

```bash
agent-memory list --all --json --cwd /absolute/project/path
```

If the store is large, review it in slices. Pull up to about 50 memories at once, then assess them in smaller mental groups if needed:

```bash
agent-memory list --all --json --cwd /absolute/project/path | jq '.memories[0:50]'
agent-memory list --all --json --cwd /absolute/project/path | jq '.memories[50:100]'
```

Use `agent-memory show <memory_id> --cwd /absolute/project/path --json` only when you need to inspect one memory more closely.

## Assessment standard

For each memory, ask:

- Would this still help a future agent working in this repo?
- Is it durable, repo-specific, and hard enough to rediscover that it belongs in memory?
- Is it a real workflow rule, architecture fact, environment quirk, validation constraint, root cause, or recurring project fact?
- Or is it just a transcript fragment, one-off branch episode, stale status note, chatty narration, or easy-to-grep fact?

Then choose one of three actions:

- `keep` if the memory is already good.
- `edit` if the fact is still useful but the wording is stale, noisy, or too tied to a specific moment.
- `delete` if the memory should not exist at all.

## Cleanup workflow

1. Load a batch of memories from the target store.
2. Read them directly and judge them one by one with the policy above.
3. Rewrite durable memories when the fact is still good but the wording is stale, noisy, or too tied to a specific moment.
4. Delete memories that are transcript junk, stale episode notes, branch choreography, or otherwise not worth keeping.
5. Keep moving through the store until every batch has been reviewed.

Use the standard CLI for mutations:

```bash
agent-memory show <memory_id> --cwd /absolute/project/path --json
agent-memory edit <memory_id> --cwd /absolute/project/path "<new text>"
agent-memory delete <memory_id> --cwd /absolute/project/path --yes
agent-memory save --cwd /absolute/project/path "<new memory>"
```

## Rewrite standard

When rewriting a memory, aim for this shape:

- `Scope + Category`
- one sentence for the durable fact
- one sentence for why it matters
- optional exception
- confidence high/med/low

Do not preserve chatty narration, commit choreography, or raw conversation phrasing.
