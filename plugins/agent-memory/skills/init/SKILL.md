---
name: init
description: Initialize Agent Memory in the current repository using the plugin-managed CLI. Use when the user wants persistent project memory set up for Claude Code.
disable-model-invocation: true
---

Initialize Agent Memory for the current repository.

Rules:
- Run the setup with the Bash tool, not by describing the command.
- Default to `agent-memory init --no-install-claude-hooks` because this plugin already provides the Claude prompt hook.
- If `$ARGUMENTS` mentions `mcp`, include `--with-mcp`.
- If `$ARGUMENTS` mentions `force`, include `--force`.
- Keep the default Codex integration unless the user explicitly asked to skip it.
- Run the command from the current working directory.
- Summarize the created files and any warnings after the command finishes.
- If the repo is already initialized, report that directly and do not retry with `--force` unless the user explicitly asked for it.
