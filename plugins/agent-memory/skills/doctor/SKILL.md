---
name: doctor
description: Inspect Agent Memory initialization and local integration state in the current repository. Use when debugging plugin setup or project memory wiring.
disable-model-invocation: true
---

Inspect Agent Memory in the current repository.

Rules:
- Run `agent-memory doctor --json` with the Bash tool in the current working directory.
- Summarize whether Agent Memory is initialized, whether the CLI is available, which local integration files exist, and any warnings.
- If the project is not initialized, say that plainly and suggest `/agent-memory:init`.
