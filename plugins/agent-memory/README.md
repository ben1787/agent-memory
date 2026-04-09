# Agent Memory Claude Plugin

This plugin makes `agent-memory` available inside Claude Code without a separate manual install.

What it does:

- Adds an `agent-memory` executable to Claude's Bash tool PATH while the plugin is enabled
- Bootstraps the real `agent-memory` binary into persistent plugin data on first use
- Adds `/agent-memory:init`, `/agent-memory:doctor`, and `/agent-memory:consolidate` skills
- Adds Claude `UserPromptSubmit` and `Stop` hooks so initialized repos get Agent Memory guidance, turn capture, and daily consolidation scheduling

Install from this repository's marketplace:

```text
/plugin marketplace add ben1787/agent-memory
/plugin install agent-memory@agent-memory-plugins
/reload-plugins
```

Initialize the current repo:

```text
/agent-memory:init
```

Or initialize from Claude's Bash tool directly:

```bash
agent-memory init --no-install-claude-hooks
```

Notes:

- The plugin bootstraps the release pinned in `release-version.txt`, so the plugin bundle and installed binary stay in sync.
- Updating the plugin is what moves users to a newer `agent-memory` release, and the next plugin-backed `agent-memory` launch refreshes the managed binary automatically.
- Set `AGENT_MEMORY_VERSION` before launching Claude Code only for development or testing overrides.
- Set `AGENT_MEMORY_LOCAL_TARBALL` to a built release archive only when you need to test an unreleased pinned version before publishing it.
- If you want repo-local Claude hooks for teammates who are not using the plugin, run `agent-memory init` manually and omit `--no-install-claude-hooks`.
