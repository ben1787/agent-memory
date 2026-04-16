# Agent Memory

`agent-memory` is a local-first associative memory store for AI coding agents.

It treats every saved memory as a node, builds similarity edges automatically from embeddings, and retrieves information as natural clusters from a graph traversal seeded by the query embedding.

It is designed to give agents a **persistent project memory** that survives across sessions, conversations, and CLIs — without any cloud service, API key, or background daemon.

## Why this shape

- **Local-first**: data lives in `.agent-memory/memory.kuzu` inside your project
- **Project-scoped**: each initialized folder gets its own memory store, with at most one store along any ancestor chain
- **Graph-native**: memories are nodes; `SIMILAR` and `NEXT` edges are persisted in a [Kuzu](https://kuzudb.com/) database
- **CLI-first by default**: agents access memory through `agent-memory recall` / `agent-memory save` via their shell tool, with prompt hooks able to inject strong prompt-matched recall automatically at a parent-score floor of `0.7`, and no MCP wiring required
- **Recoverable**: every save / edit / delete is logged so `agent-memory undo` can walk back mistakes
- **Single binary**: ships as a self-contained executable, no Python on the user's machine required

## Install

### Claude Code plugin marketplace

If you use Claude Code, the easiest install path is the plugin marketplace.
This plugin is not in Claude Code's default marketplace. Add this repository as a custom marketplace first:

```text
/plugin marketplace add ben1787/agent-memory
/plugin install agent-memory@agent-memory-plugins
/reload-plugins
```

That gives Claude Code a plugin-managed `agent-memory` executable plus `/agent-memory:init` and `/agent-memory:doctor` skills. To initialize the current repo after installing the plugin, run:

```text
/agent-memory:init
```

The marketplace plugin pins a specific released `agent-memory` version and installs that exact release on first use. Updating the plugin moves users to a newer `agent-memory` release, and the plugin refreshes its managed binary to that pinned version on the next use so the bundle and installed binary stay in sync.

### Standalone CLI install

The fastest standalone path:

```bash
curl -LsSf https://raw.githubusercontent.com/ben1787/agent-memory/main/install.sh | sh
```

That detects your platform, downloads the right prebuilt binary from the latest GitHub release, verifies its sha256, drops it at `~/.local/bin/agent-memory`, and ensures `~/.local/bin` is on your `PATH`. After it finishes, `agent-memory` is on PATH from any directory.

### From source (developers)

If you're hacking on agent-memory itself, install from a local checkout with `uv`:

```bash
git clone https://github.com/ben1787/agent-memory
cd agent-memory
uv tool install --reinstall .
```

### Self-update

```bash
agent-memory upgrade
```

Hits the GitHub releases API, downloads the right binary for your platform, verifies the checksum, and replaces the running binary in place. agent-memory also runs a non-blocking 24h staleness check from any command and prints a one-line "new version available" notice when a newer release is published.

## Quickstart

Initialize a project:

```bash
cd /path/to/your/repo
agent-memory init
```

That creates `.agent-memory/` for the store, installs `UserPromptSubmit` hooks for both Claude Code and Codex (if you use those agents) so they can inject strong prompt-matched memory before the model call at the current `0.7` parent-score floor plus periodic memory guidance, and injects an `Agent Memory` instructions block into `CLAUDE.md` / `AGENTS.md` if those files exist.

Save a memory:

```bash
cat <<'EOF' | agent-memory save
The billing webhook handler lives in services/billing/webhooks.py.
EOF
```

For agents and anything with quotes, backticks, dollar signs, or newlines, prefer piped stdin so the shell cannot rewrite the text before `agent-memory` sees it. For a very short shell-safe one-liner, quoted positional args are still fine:

```bash
agent-memory save "The billing webhook handler lives in services/billing/webhooks.py."
```

Recall memories:

```bash
agent-memory recall "billing webhook handler"
```

Phrase recall queries the way the answer would phrase itself, not the way a question would. ✅ `"billing webhook handler"` ❌ `"how do I find the webhook code"`.

## Recovery commands

Mistakes happen. The recovery flow:

```bash
agent-memory list --recent 10              # most recent memories with their ids
agent-memory show <memory_id>              # full text + metadata of one memory
agent-memory edit <memory_id> "<new text>" # one-shot replacement (re-embeds)
cat <<'EOF' | agent-memory edit <memory_id> # multi-line / shell-hostile content
<replacement text>
EOF
agent-memory edit <memory_id>              # opens $EDITOR with current text
agent-memory delete <memory_id> --yes      # remove a memory entirely
agent-memory undo                          # reverse the most recent save / edit / delete
```

Every destructive operation is appended to `.agent-memory/operations.log`, so `undo` can walk back chains of mistakes — and so can a future you reading the audit trail.

## Project nesting

Agent Memory enforces **at most one store along any ancestor chain**. If you try to `init` inside a directory that already has a store in some ancestor, or above a directory that already has a store, the install refuses with a clear error and a remediation suggestion. This prevents the lookup ambiguity that would otherwise come from one store silently shadowing another.

## CLI surface

| Command | What it does |
|---|---|
| `agent-memory init [path]` | Set up a project store + install agent hooks + inject instructions block |
| `agent-memory uninstall [path]` | Reverse `init` (use `--remove-store` to also delete `.agent-memory/`) |
| `agent-memory uninstall-all [path]` | Clean-room uninstall for the current project plus machine-level binaries, bundles, Claude plugin data/cache, and installer PATH hooks |
| `agent-memory save "<text>"` | Save one or more memories (positional or `--stdin`) |
| `agent-memory recall <query>` | Retrieve clustered memories |
| `agent-memory list [--recent N \| --all]` | List memories newest-first |
| `agent-memory show <id>` | Full text + metadata of one memory |
| `agent-memory edit <id> [<new text>]` | Edit a memory in place (one-shot, `--stdin`, or `$EDITOR`) |
| `agent-memory delete <id> --yes` | Delete a memory |
| `agent-memory undo` | Reverse the most recent save / edit / delete |
| `agent-memory upgrade` | Self-update to the latest GitHub release |
| `agent-memory stats` | Memory + relationship counts |
| `agent-memory hook-log` | Recent UserPromptSubmit hook activity (debug aid) |
| `agent-memory --version` | Print version |

Run `agent-memory --help` for the full list.

## Full uninstall

For a project-only uninstall:

```bash
agent-memory uninstall --remove-store
```

For a clean-room uninstall that also removes machine-level binaries, extracted bundles, Claude plugin cache/data, Claude marketplace registry entries, and installer-managed PATH/session hooks:

```bash
agent-memory uninstall-all
```

Run `uninstall-all` from inside an initialized repo, or pass a repo path explicitly, if you also want that repo's `.agent-memory/` store removed as part of the same command.

## How agents discover it

`agent-memory init` installs a `UserPromptSubmit` hook into both:

- `.codex/hooks.json` (Codex repo-local hook config)
- `.claude/settings.local.json` (Claude Code local settings)

The hook command is portable across machines:

```text
AGENT_MEMORY_PROJECT_ROOT=/abs/path/to/repo agent-memory _hook claude-user-prompt-submit
```

It dispatches to the binary on `PATH`, so the same hook config works on any machine with `agent-memory` installed (regardless of install method). On every prompt, the hook can recall against the raw user prompt and inject only strong matches at the current `0.7` parent-score floor; on the configured `1 + X` cadence, it also injects the broader memory/save guidance.

For redundancy, `init` also injects an `Agent Memory` section into `CLAUDE.md` and `AGENTS.md` if those files already exist. The block tells the agent how to use the CLI even if the prompt-submit hook ever breaks or gets disabled.

## Optional: MCP server

`agent-memory` ships with an MCP server (`agent-memory serve-mcp`) that exposes `save_memory`, `recall_memories`, `consolidate_memories`, etc. as MCP tools. The MCP path is **not installed by default** — it adds three points of wiring (`.mcp.json`, `.codex/config.toml`, `.claude/settings.local.json` `enabledMcpjsonServers`) and most users get just as much utility from the CLI + the prompt-submit hook injection.

To opt in:

```bash
agent-memory init --with-mcp
```

## Storage layout

```text
.agent-memory/
  config.json          # store config
  instructions.md      # default instructions block (legacy)
  memory.kuzu          # the kuzu database
  operations.log       # append-only audit trail (powers `undo`)
  hook-events.jsonl    # hook firing log (powers `hook-log`)
```

## Retrieval algorithm

`agent-memory recall` runs a lazy max-neighbor graph traversal:

1. Embed the query.
2. Treat the query as a temporary root node with direct similarity to every saved memory.
3. Put those direct query scores into a max-heap.
4. Repeatedly pop the best current parent-similarity value, settle that memory, and enqueue only the next best unused neighbor from that settled memory.
5. Stop once the top `N` memories are settled.
6. Return memories ordered by descending parent-similarity score.

This keeps retrieval parameter-light and avoids walking the whole graph when only the top hits matter.

## Embeddings

Two backends:

- `fastembed` (default): semantic local embeddings via `snowflake/snowflake-arctic-embed-m`
- `hash`: deterministic fallback for tests and fully offline bootstrap

When a project's configured embedding backend/model/dimensions change, Agent Memory records the store's previous embedding signature and automatically re-embeds the project store on the next open. Successful re-embeds also prune stale fastembed model caches, keeping only the active configured model. You can trigger the rebuild explicitly with:

```bash
agent-memory reembed
```

To prune stale fastembed model caches without forcing a re-embed:

```bash
agent-memory prune-model-cache
```

## Development

```bash
git clone https://github.com/ben1787/agent-memory
cd agent-memory
uv sync
uv run pytest tests/
```

To build the standalone binary locally:

```bash
uv pip install pyinstaller
pyinstaller pyinstaller/agent-memory.spec --clean --noconfirm
./dist/agent-memory --version
```

The `.github/workflows/release.yml` matrix runs the same `pyinstaller` invocation on macOS arm64/x86_64, Linux x86_64/arm64, and Windows x86_64 runners on every `v*` tag, then uploads the artifacts to a GitHub Release.

## Versioning

Tagged releases follow semver. Cut a release with:

```bash
git tag vX.Y.Z
git push --tags
```

The Actions release workflow handles everything else.

## License

MIT.
