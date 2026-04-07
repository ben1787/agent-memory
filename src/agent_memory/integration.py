from __future__ import annotations

import json
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


MCP_FILENAME = ".mcp.json"
GITIGNORE_FILENAME = ".gitignore"
CLAUDE_SETTINGS_LOCAL_FILENAME = "settings.local.json"
CODEX_CONFIG_FILENAME = "config.toml"
CODEX_HOOKS_FILENAME = "hooks.json"
DEFAULT_SERVER_NAME = "agent-memory"
DEFAULT_CODEX_TRUST_LEVEL = "trusted"
GITIGNORE_ENTRY = ".agent-memory/"
LOCAL_EXCLUDE_ENTRIES = [
    ".agent-memory/",
    ".claude/settings.local.json",
    ".codex/config.toml",
    ".codex/hooks.json",
    ".mcp.json",
]

INSTRUCTIONS_FILENAMES = ("CLAUDE.md", "AGENTS.md")
INSTRUCTIONS_BEGIN_MARKER = "<!-- agent-memory:begin -->"
INSTRUCTIONS_END_MARKER = "<!-- agent-memory:end -->"
INSTRUCTIONS_BLOCK = """<!-- agent-memory:begin -->
## Agent Memory — persistent project knowledge

This project has **Agent Memory** installed at `.agent-memory/` — a project-scoped vector store of durable project knowledge that survives across sessions, agents, and CLI invocations. It is the long-term memory you do not get from this file alone. A `UserPromptSubmit` hook also injects a short reminder of this on every turn; if that reminder is missing from your context, the hook is broken — surface that and stop.

You access it through the `agent-memory` CLI, invoked via your shell tool (Bash or equivalent). There is no MCP server to wire up — just the CLI.

### Recall before non-trivial work

Before starting any non-trivial task — debugging, implementing, refactoring, exploring an unfamiliar area — ask yourself: *"Has prior work in this repo already discovered something that would change how I approach this?"* If even a little, recall first:

```
agent-memory recall <task-shaped query>
```

Phrase the query the way the answer would phrase itself, not the way a question would. ✅ `"factor model nexus repository pattern"`  ❌ `"how do I add a factor model"`. Recall is cheap; do it.

### Save when something durable surfaced

Before sending your final answer, ask: *"Did I just learn 0–3 durable things future-me would want to know without re-reading this whole conversation?"* If yes, save them — short, complete sentences, one fact per memory:

```
agent-memory save "<memory 1>" "<memory 2>"
```

For memories with newlines, quotes, or other shell-hostile characters, pipe instead:

```
echo "$MEMORY_TEXT" | agent-memory save --stdin
```

**Save these:**
- Architectural decisions and the *why* behind them
- File / module locations that were hard to find
- Gotchas, footguns, bugs that took real time to diagnose
- User preferences, especially explicit corrections
- Non-obvious cross-component relationships
- Pointers to external systems (dashboards, ticket projects, monitors)

**Do NOT save:**
- Anything already written in this CLAUDE.md / AGENTS.md (it's loaded every turn already)
- Generic programming knowledge or framework docs
- Verbatim chat or transcript dumps
- Ephemeral debugging state or in-progress task notes
- Long prose — prefer one fact per memory, ≤30 words

### Recover when you got it wrong

You will save things you regret. That is expected — there is a recovery flow.

```
agent-memory list --recent 10               # most recent memories with their ids
agent-memory show <memory_id>               # full text + metadata of one memory
agent-memory edit <memory_id> "<new text>"  # replace text in place (re-embeds + rebuilds similarity edges)
agent-memory edit <memory_id> --stdin       # multi-line / shell-hostile content
agent-memory delete <memory_id> --yes       # remove a memory entirely
agent-memory undo                           # reverse the most recent save / edit / delete
```

`undo` works across all destructive operations and is backed by `.agent-memory/operations.log`. Chain undos to walk back multiple operations. Saved the wrong thing? `undo` it. Edit toward something worse? `undo` it. Deleted something you shouldn't have? `undo` brings it back with the original id.

### Where it lives

- Store: `.agent-memory/memory.kuzu` (durable, gitignored)
- Operations log: `.agent-memory/operations.log` (append-only audit trail; powers `undo`)
- Hook log: `.agent-memory/hook-events.jsonl` (use this to verify the prompt-submit hook is firing)
- Re-run install: `agent-memory init` from the project root
<!-- agent-memory:end -->
"""


def _toml_literal(value: str) -> str:
    """Render a string as a TOML literal string (single-quoted, no escapes).

    TOML basic strings (double-quoted) interpret backslash escapes, so a Windows
    path like ``C:\\Users\\...`` would be parsed as ``\\U`` (a unicode escape) and
    fail to load. Literal strings have no escapes at all, which is exactly what
    we want for filesystem paths.

    Reason: Windows paths contain backslashes which break TOML basic strings.
    """
    if "'" not in value:
        return f"'{value}'"
    # Defensive fallback: a single quote inside a filesystem path is rare on
    # every platform. Multi-line literal strings ('''...''') cannot contain
    # three consecutive single quotes; only truly pathological paths reach here.
    return f"'''{value}'''"


@dataclass(slots=True)
class IntegrationResult:
    path: Path
    status: str
    details: str


def suggest_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return current


def _server_payload(project_root: Path) -> dict[str, object]:
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [
            "-m",
            "agent_memory.cli",
            "serve-mcp",
            "--cwd",
            str(project_root),
        ],
        "cwd": str(project_root),
        "env": {
            "AGENT_MEMORY_PROJECT_ROOT": str(project_root),
        },
    }


def _codex_server_payload(project_root: Path) -> dict[str, object]:
    return {
        "command": sys.executable,
        "args": [
            "-m",
            "agent_memory.cli",
            "serve-mcp",
            "--cwd",
            str(project_root),
        ],
        "cwd": str(project_root),
        "env": {
            "AGENT_MEMORY_PROJECT_ROOT": str(project_root),
        },
    }


def _render_codex_mcp_server_block(project_root: Path, server_name: str = DEFAULT_SERVER_NAME) -> str:
    payload = _codex_server_payload(project_root)
    # Reason: paths (command/args/cwd/env values) must render as TOML literal
    # strings on Windows; basic strings would interpret backslashes as escapes.
    lines = [
        f'[mcp_servers."{server_name}"]',
        f"command = {_toml_literal(str(payload['command']))}",
        "args = [",
    ]
    for arg in payload["args"]:
        lines.append(f"  {_toml_literal(str(arg))},")
    lines.extend(
        [
            "]",
            f"cwd = {_toml_literal(str(payload['cwd']))}",
            "",
            f'[mcp_servers."{server_name}".env]',
        ]
    )
    for key, value in payload["env"].items():
        lines.append(f"{key} = {_toml_literal(str(value))}")
    return "\n".join(lines) + "\n"


def _codex_project_trust_header(project_root: Path) -> str:
    """Render the [projects.<path>] header using a TOML literal-string key.

    Reason: same rationale as _render_codex_mcp_server_block — TOML basic
    strings break on Windows backslashes.
    """
    return f"[projects.{_toml_literal(str(project_root))}]"


def _render_codex_project_trust_block(
    project_root: Path,
    *,
    trust_level: str = DEFAULT_CODEX_TRUST_LEVEL,
) -> str:
    return (
        f"{_codex_project_trust_header(project_root)}\n"
        f'trust_level = {json.dumps(trust_level)}\n'
    )


def _render_toml(lines: list[str]) -> str:
    rendered = "\n".join(lines).rstrip()
    return rendered + "\n" if rendered else ""


def _set_codex_mcp_server_cannot_fail(
    existing_text: str,
    project_root: Path,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
) -> str:
    server_header = f'[mcp_servers."{server_name}"]'
    env_header = f'[mcp_servers."{server_name}".env]'
    target_headers = {server_header, env_header}
    lines = existing_text.splitlines()
    output: list[str] = []
    skipping = False

    for line in lines:
        stripped = line.strip()
        is_table = stripped.startswith("[") and stripped.endswith("]")
        if is_table:
            if stripped in target_headers:
                skipping = True
                continue
            if skipping:
                skipping = False
        if not skipping:
            output.append(line)

    rendered = "\n".join(output).rstrip()
    block = _render_codex_mcp_server_block(project_root, server_name=server_name).rstrip()
    if rendered:
        return rendered + "\n\n" + block + "\n"
    return block + "\n"


def _set_codex_project_trust_cannot_fail(
    existing_text: str,
    project_root: Path,
    *,
    trust_level: str = DEFAULT_CODEX_TRUST_LEVEL,
) -> str:
    # Reason: the target header must match the TOML-literal-string format we
    # render with, so existing-block detection works on Windows paths.
    target_header = _codex_project_trust_header(project_root)
    rendered_line = f'trust_level = {json.dumps(trust_level)}'
    lines = existing_text.splitlines()
    output: list[str] = []
    found_target = False
    in_target = False
    trust_written = False

    for line in lines:
        stripped = line.strip()
        is_table = stripped.startswith("[") and stripped.endswith("]")
        if is_table:
            if in_target and not trust_written:
                output.append(rendered_line)
                trust_written = True
            in_target = stripped == target_header
            if in_target:
                found_target = True
            output.append(line)
            continue

        if in_target and stripped.startswith("trust_level"):
            output.append(rendered_line)
            trust_written = True
            continue

        output.append(line)

    if in_target and not trust_written:
        output.append(rendered_line)
        trust_written = True

    rendered = "\n".join(output).rstrip()
    if found_target:
        return rendered + "\n"

    block = _render_codex_project_trust_block(
        project_root,
        trust_level=trust_level,
    ).rstrip()
    if rendered:
        return rendered + "\n\n" + block + "\n"
    return block + "\n"


def _drop_toml_tables_cannot_fail(existing_text: str, target_headers: set[str]) -> str:
    lines = existing_text.splitlines()
    output: list[str] = []
    skipping = False

    for line in lines:
        stripped = line.strip()
        is_table = stripped.startswith("[") and stripped.endswith("]")
        if is_table:
            if stripped in target_headers:
                skipping = True
                continue
            if skipping:
                skipping = False
        if not skipping:
            output.append(line)

    return _render_toml(output)


def _remove_toml_key_in_section_cannot_fail(existing_text: str, target_header: str, key_name: str) -> str:
    lines = existing_text.splitlines()
    output: list[str] = []
    in_target = False
    section_lines: list[str] = []

    def flush_section() -> None:
        nonlocal section_lines
        if not section_lines:
            return
        if in_target:
            payload_lines = [
                line
                for line in section_lines[1:]
                if line.strip() and not line.strip().startswith("#")
            ]
            if payload_lines:
                output.extend(section_lines)
        else:
            output.extend(section_lines)
        section_lines = []

    for line in lines:
        stripped = line.strip()
        is_table = stripped.startswith("[") and stripped.endswith("]")
        if is_table:
            flush_section()
            in_target = stripped == target_header
            section_lines = [line]
            continue

        if in_target and stripped.startswith(f"{key_name} "):
            continue

        section_lines.append(line)

    flush_section()
    return _render_toml(output)


def _resolve_git_dir(project_root: Path) -> Path | None:
    git_path = project_root / ".git"
    if git_path.is_dir():
        return git_path
    if git_path.is_file():
        content = git_path.read_text(encoding='utf-8').strip()
        prefix = "gitdir: "
        if content.startswith(prefix):
            resolved = (project_root / content[len(prefix) :]).resolve()
            return resolved
    return None


def install_mcp_server(project_root: Path, server_name: str = DEFAULT_SERVER_NAME) -> IntegrationResult:
    mcp_path = project_root / MCP_FILENAME
    if mcp_path.exists():
        payload = json.loads(mcp_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected {mcp_path} to contain a JSON object")
    else:
        payload = {}

    servers = payload.get("mcpServers")
    if servers is None:
        servers = {}
        payload["mcpServers"] = servers
    if not isinstance(servers, dict):
        raise ValueError(f"Expected {mcp_path} mcpServers to be a JSON object")

    desired = _server_payload(project_root)
    existing = servers.get(server_name)
    if existing == desired:
        return IntegrationResult(
            path=mcp_path,
            status="unchanged",
            details=f"{MCP_FILENAME} already contains the {server_name} server entry.",
        )

    status = "created" if server_name not in servers else "updated"
    servers[server_name] = desired
    mcp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    return IntegrationResult(
        path=mcp_path,
        status=status,
        details=f"{status.title()} {server_name} MCP entry in {MCP_FILENAME}.",
    )


def uninstall_mcp_server(project_root: Path, server_name: str = DEFAULT_SERVER_NAME) -> IntegrationResult:
    mcp_path = project_root / MCP_FILENAME
    if not mcp_path.exists():
        return IntegrationResult(
            path=mcp_path,
            status="unchanged",
            details=f"{MCP_FILENAME} does not exist.",
        )

    payload = json.loads(mcp_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {mcp_path} to contain a JSON object")

    servers = payload.get("mcpServers")
    if not isinstance(servers, dict) or server_name not in servers:
        return IntegrationResult(
            path=mcp_path,
            status="unchanged",
            details=f"{MCP_FILENAME} does not contain the {server_name} server entry.",
        )

    del servers[server_name]
    if not servers:
        payload.pop("mcpServers", None)

    if payload:
        mcp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
        return IntegrationResult(
            path=mcp_path,
            status="updated",
            details=f"Removed {server_name} MCP entry from {MCP_FILENAME}.",
        )

    mcp_path.unlink()
    return IntegrationResult(
        path=mcp_path,
        status="removed",
        details=f"Removed {server_name} MCP entry and deleted empty {MCP_FILENAME}.",
    )


def install_codex_mcp_server(project_root: Path, server_name: str = DEFAULT_SERVER_NAME) -> IntegrationResult:
    codex_dir = project_root / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    config_path = codex_dir / CODEX_CONFIG_FILENAME
    desired = _codex_server_payload(project_root)
    existed = config_path.exists()

    if existed:
        existing_text = config_path.read_text(encoding='utf-8')
        try:
            payload = tomllib.loads(existing_text)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Expected {config_path} to contain valid TOML") from exc
        mcp_servers = payload.get("mcp_servers")
        if mcp_servers is not None and not isinstance(mcp_servers, dict):
            raise ValueError(f"Expected {config_path} mcp_servers to be a TOML table")
        existing = (mcp_servers or {}).get(server_name)
        if existing == desired:
            return IntegrationResult(
                path=config_path,
                status="unchanged",
                details="Codex repo-local config already contains the agent-memory MCP entry.",
            )
        rendered = _set_codex_mcp_server_cannot_fail(existing_text, project_root, server_name=server_name)
        status = "updated"
    else:
        rendered = _render_codex_mcp_server_block(project_root, server_name=server_name)
        status = "created"

    config_path.write_text(rendered, encoding='utf-8')
    return IntegrationResult(
        path=config_path,
        status=status,
        details="Installed Codex repo-local MCP entry in .codex/config.toml.",
    )


def uninstall_codex_mcp_server(project_root: Path, server_name: str = DEFAULT_SERVER_NAME) -> IntegrationResult:
    config_path = project_root / ".codex" / CODEX_CONFIG_FILENAME
    if not config_path.exists():
        return IntegrationResult(
            path=config_path,
            status="unchanged",
            details=".codex/config.toml does not exist.",
        )

    existing_text = config_path.read_text(encoding='utf-8')
    payload = tomllib.loads(existing_text)
    mcp_servers = payload.get("mcp_servers")
    if not isinstance(mcp_servers, dict) or server_name not in mcp_servers:
        return IntegrationResult(
            path=config_path,
            status="unchanged",
            details="Codex repo-local config does not contain the agent-memory MCP entry.",
        )

    rendered = _drop_toml_tables_cannot_fail(
        existing_text,
        {
            f'[mcp_servers."{server_name}"]',
            f'[mcp_servers."{server_name}".env]',
        },
    )
    if rendered:
        config_path.write_text(rendered, encoding='utf-8')
        return IntegrationResult(
            path=config_path,
            status="updated",
            details="Removed Codex repo-local MCP entry from .codex/config.toml.",
        )

    config_path.unlink()
    return IntegrationResult(
        path=config_path,
        status="removed",
        details="Removed Codex repo-local MCP entry and deleted empty .codex/config.toml.",
    )


def codex_project_trust_state(
    project_root: Path,
    *,
    codex_home: Path | None = None,
) -> tuple[bool | None, str | None]:
    resolved_home = (codex_home or (Path.home() / ".codex")).resolve()
    config_path = resolved_home / CODEX_CONFIG_FILENAME
    if not config_path.exists():
        return False, None

    try:
        payload = tomllib.loads(config_path.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as exc:
        return None, f"Failed to parse {config_path}: {exc}"

    projects = payload.get("projects")
    if projects is None:
        return False, None
    if not isinstance(projects, dict):
        return None, f"Expected {config_path} projects to be a TOML table."

    project_entry = projects.get(str(project_root.resolve()))
    if project_entry is None:
        return False, None
    if not isinstance(project_entry, dict):
        return None, f"Expected {config_path} projects.{project_root.resolve()} to be a TOML table."

    return project_entry.get("trust_level") == DEFAULT_CODEX_TRUST_LEVEL, None


def install_codex_project_trust(
    project_root: Path,
    *,
    codex_home: Path | None = None,
    trust_level: str = DEFAULT_CODEX_TRUST_LEVEL,
) -> IntegrationResult:
    resolved_root = project_root.resolve()
    resolved_home = (codex_home or (Path.home() / ".codex")).resolve()
    resolved_home.mkdir(parents=True, exist_ok=True)
    config_path = resolved_home / CODEX_CONFIG_FILENAME

    if config_path.exists():
        existing_text = config_path.read_text(encoding='utf-8')
        trusted, error = codex_project_trust_state(resolved_root, codex_home=resolved_home)
        if error:
            raise ValueError(error)
        if trusted:
            return IntegrationResult(
                path=config_path,
                status="unchanged",
                details=f"Codex global config already trusts {resolved_root}.",
            )
        rendered = _set_codex_project_trust_cannot_fail(
            existing_text,
            resolved_root,
            trust_level=trust_level,
        )
        status = "updated"
    else:
        rendered = _render_codex_project_trust_block(
            resolved_root,
            trust_level=trust_level,
        )
        status = "created"

    config_path.write_text(rendered, encoding='utf-8')
    return IntegrationResult(
        path=config_path,
        status=status,
        details=f"Added Codex trusted-project entry for {resolved_root} in {config_path}.",
    )


def uninstall_codex_project_trust(
    project_root: Path,
    *,
    codex_home: Path | None = None,
) -> IntegrationResult:
    resolved_root = project_root.resolve()
    resolved_home = (codex_home or (Path.home() / ".codex")).resolve()
    config_path = resolved_home / CODEX_CONFIG_FILENAME
    if not config_path.exists():
        return IntegrationResult(
            path=config_path,
            status="unchanged",
            details=f"{config_path} does not exist.",
        )

    trusted, error = codex_project_trust_state(resolved_root, codex_home=resolved_home)
    if error:
        raise ValueError(error)
    if not trusted:
        return IntegrationResult(
            path=config_path,
            status="unchanged",
            details=f"Codex global config does not trust {resolved_root}.",
        )

    rendered = _remove_toml_key_in_section_cannot_fail(
        config_path.read_text(encoding='utf-8'),
        _codex_project_trust_header(resolved_root),
        "trust_level",
    )
    if rendered:
        config_path.write_text(rendered, encoding='utf-8')
        return IntegrationResult(
            path=config_path,
            status="updated",
            details=f"Removed Codex trusted-project entry for {resolved_root} from {config_path}.",
        )

    config_path.unlink()
    return IntegrationResult(
        path=config_path,
        status="removed",
        details=f"Removed Codex trusted-project entry and deleted empty {config_path}.",
    )


def ensure_gitignore_entry(project_root: Path, entry: str = GITIGNORE_ENTRY) -> IntegrationResult:
    gitignore_path = project_root / GITIGNORE_FILENAME
    normalized_entry = entry.strip()
    if not gitignore_path.exists():
        gitignore_path.write_text(normalized_entry + "\n", encoding='utf-8')
        return IntegrationResult(
            path=gitignore_path,
            status="created",
            details=f"Created {GITIGNORE_FILENAME} with {normalized_entry}.",
        )

    existing_lines = gitignore_path.read_text(encoding='utf-8').splitlines()
    if normalized_entry in existing_lines:
        return IntegrationResult(
            path=gitignore_path,
            status="unchanged",
            details=f"{GITIGNORE_FILENAME} already ignores {normalized_entry}.",
        )

    content = gitignore_path.read_text(encoding='utf-8')
    suffix = "" if content.endswith("\n") or not content else "\n"
    gitignore_path.write_text(content + suffix + normalized_entry + "\n", encoding='utf-8')
    return IntegrationResult(
        path=gitignore_path,
        status="updated",
        details=f"Added {normalized_entry} to {GITIGNORE_FILENAME}.",
    )


def ensure_gitignore_entries(
    project_root: Path,
    entries: list[str],
) -> IntegrationResult:
    gitignore_path = project_root / GITIGNORE_FILENAME
    normalized_entries = [entry.strip() for entry in entries if entry.strip()]
    if not gitignore_path.exists():
        gitignore_path.write_text("\n".join(normalized_entries) + "\n", encoding='utf-8')
        return IntegrationResult(
            path=gitignore_path,
            status="created",
            details=f"Created {GITIGNORE_FILENAME} with {', '.join(normalized_entries)}.",
        )

    existing_lines = gitignore_path.read_text(encoding='utf-8').splitlines()
    missing = [entry for entry in normalized_entries if entry not in existing_lines]
    if not missing:
        return IntegrationResult(
            path=gitignore_path,
            status="unchanged",
            details=f"{GITIGNORE_FILENAME} already ignores {', '.join(normalized_entries)}.",
        )

    content = gitignore_path.read_text(encoding='utf-8')
    suffix = "" if content.endswith("\n") or not content else "\n"
    gitignore_path.write_text(content + suffix + "\n".join(missing) + "\n", encoding='utf-8')
    return IntegrationResult(
        path=gitignore_path,
        status="updated",
        details=f"Added {', '.join(missing)} to {GITIGNORE_FILENAME}.",
    )


def ensure_local_git_excludes(
    project_root: Path,
    entries: list[str] | None = None,
) -> IntegrationResult:
    git_dir = _resolve_git_dir(project_root)
    if git_dir is None:
        return ensure_gitignore_entries(project_root, entries or LOCAL_EXCLUDE_ENTRIES)

    exclude_path = git_dir / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_entries = [entry.strip() for entry in (entries or LOCAL_EXCLUDE_ENTRIES) if entry.strip()]

    if not exclude_path.exists():
        exclude_path.write_text("", encoding='utf-8')

    existing_lines = exclude_path.read_text(encoding='utf-8').splitlines()
    missing = [entry for entry in normalized_entries if entry not in existing_lines]
    if not missing:
        return IntegrationResult(
            path=exclude_path,
            status="unchanged",
            details=f"Local git exclude already contains {', '.join(normalized_entries)}.",
        )

    content = exclude_path.read_text(encoding='utf-8')
    suffix = "" if content.endswith("\n") or not content else "\n"
    exclude_path.write_text(content + suffix + "\n".join(missing) + "\n", encoding='utf-8')
    status = "created" if not content else "updated"
    return IntegrationResult(
        path=exclude_path,
        status=status,
        details=f"Added {', '.join(missing)} to .git/info/exclude.",
    )


def remove_local_git_excludes(
    project_root: Path,
    entries: list[str] | None = None,
) -> IntegrationResult:
    normalized_entries = [entry.strip() for entry in (entries or LOCAL_EXCLUDE_ENTRIES) if entry.strip()]
    git_dir = _resolve_git_dir(project_root)
    if git_dir is None:
        target_path = project_root / GITIGNORE_FILENAME
    else:
        target_path = git_dir / "info" / "exclude"

    if not target_path.exists():
        return IntegrationResult(
            path=target_path,
            status="unchanged",
            details=f"{target_path} does not exist.",
        )

    existing_lines = target_path.read_text(encoding='utf-8').splitlines()
    kept_lines = [line for line in existing_lines if line.strip() not in normalized_entries]
    removed = [line for line in existing_lines if line.strip() in normalized_entries]
    if not removed:
        return IntegrationResult(
            path=target_path,
            status="unchanged",
            details=f"{target_path} does not contain {', '.join(normalized_entries)}.",
        )

    target_path.write_text(_render_toml(kept_lines), encoding='utf-8')
    return IntegrationResult(
        path=target_path,
        status="updated",
        details=f"Removed {', '.join(sorted(set(entry.strip() for entry in removed)))} from {target_path.name}.",
    )


def _shell_command_for_hook(project_root: Path, hook_name: str) -> str:
    """Build a portable hook command line.

    Emits:
        AGENT_MEMORY_PROJECT_ROOT=<root> PATH=$HOME/.local/bin:$PATH agent-memory _hook <hook-name>

    The PATH prefix is critical: Claude Code (and Codex) launch hook commands
    via `/bin/sh -c "..."`, which uses a stripped-down PATH of
    `/usr/bin:/bin:/usr/sbin:/sbin` and does NOT source the user's shell rc.
    Without the prefix, `agent-memory` resolves correctly when you test the
    command from your interactive shell but fails silently when the real hook
    fires, because `~/.local/bin` is not on the subshell PATH.

    `$HOME` IS reliably set in the hook subprocess environment, so prepending
    `$HOME/.local/bin` keeps the command machine-portable (no user-specific
    absolute path baked in) while ensuring the binary actually resolves on
    every machine where install.sh dropped it in the default location.

    If a user installed agent-memory somewhere other than `$HOME/.local/bin`
    (e.g. `/opt/local/bin` via Homebrew, or a custom INSTALL_DIR), they can
    re-init with `agent-memory init` and the binary will be on PATH for the
    init process; for the hook command itself, system locations like /usr/local/bin
    are already on the /bin/sh PATH or covered by Homebrew's path injection.

    The previous form embedded an absolute Python interpreter path, which was
    machine-specific. The form before that was just `agent-memory _hook ...`
    with no PATH prefix, which silently failed in real hook subprocesses.
    """
    exports = f"AGENT_MEMORY_PROJECT_ROOT={shlex.quote(str(project_root))}"
    path_prefix = "PATH=$HOME/.local/bin:$PATH"
    return f"{exports} {path_prefix} agent-memory _hook {hook_name}"


# Mapping from the new portable hook-name slug to the underlying Python module.
# Used by the `_hook` CLI subcommand group to dispatch to the right module.
HOOK_NAME_TO_MODULE: dict[str, str] = {
    "claude-user-prompt-submit": "agent_memory.hooks.claude_user_prompt_submit",
    "codex-user-prompt-submit": "agent_memory.hooks.codex_user_prompt_submit",
}


def _merge_hook(settings: dict[str, object], event_name: str, hook_payload: dict[str, object]) -> bool:
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Expected Claude settings hooks to be a JSON object")

    groups = hooks.setdefault(event_name, [])
    if not isinstance(groups, list):
        raise ValueError(f"Expected Claude settings hooks.{event_name} to be a JSON array")

    desired_command = hook_payload.get("command")
    for group in groups:
        if not isinstance(group, dict):
            continue
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            continue
        for handler in handlers:
            if isinstance(handler, dict) and handler.get("command") == desired_command:
                changed = False
                for key, value in hook_payload.items():
                    if handler.get(key) != value:
                        handler[key] = value
                        changed = True
                return changed

    groups.append({"hooks": [hook_payload]})
    return True


def install_claude_hooks(
    project_root: Path,
    *,
    register_mcp_server: bool = False,
) -> IntegrationResult:
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / CLAUDE_SETTINGS_LOCAL_FILENAME
    existed = settings_path.exists()

    if existed:
        payload = json.loads(settings_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected {settings_path} to contain a JSON object")
    else:
        payload = {}

    changed = False
    if register_mcp_server:
        enabled_servers = payload.get("enabledMcpjsonServers")
        if enabled_servers is None:
            payload["enabledMcpjsonServers"] = [DEFAULT_SERVER_NAME]
            changed = True
        elif isinstance(enabled_servers, list):
            if DEFAULT_SERVER_NAME not in enabled_servers:
                enabled_servers.append(DEFAULT_SERVER_NAME)
                changed = True
        else:
            raise ValueError(f"Expected {settings_path} enabledMcpjsonServers to be a JSON array")

    prompt_hook = {
        "type": "command",
        "command": _shell_command_for_hook(project_root, "claude-user-prompt-submit"),
        "timeout": 10,
    }
    # Drop ALL prior agent-memory hook command shapes before re-merging the
    # current desired form. The substring `_hook claude-user-prompt-submit`
    # matches both broken intermediate forms (no-PATH-prefix) and the current
    # PATH-prefix form, so the strip+merge pair is idempotent: re-running init
    # always converges on the same canonical command, no matter what was there
    # before.
    changed = _remove_hook_commands(
        payload,
        "UserPromptSubmit",
        [
            "agent_memory.hooks.claude_user_prompt_submit",  # legacy: python -m form
            "_hook claude-user-prompt-submit",  # all portable forms (broken + current)
        ],
    ) or changed
    changed = _merge_hook(payload, "UserPromptSubmit", prompt_hook) or changed
    changed = _remove_hook_commands(
        payload,
        "Stop",
        [
            "agent_memory.hooks.claude_stop_capture",
            "_hook claude-stop-capture",
        ],
    ) or changed

    if not changed and settings_path.exists():
        return IntegrationResult(
            path=settings_path,
            status="unchanged",
            details="Claude local hooks already configured for agent-memory.",
        )

    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    return IntegrationResult(
        path=settings_path,
        status="created" if not existed else "updated",
        details="Installed Claude local prompt hook for memory reminders and recall guidance.",
    )


def _prune_hook_payload(payload: dict[str, object]) -> dict[str, object]:
    hooks = payload.get("hooks")
    if isinstance(hooks, dict):
        empty_events = [event for event, groups in hooks.items() if not groups]
        for event in empty_events:
            hooks.pop(event, None)
        if not hooks:
            payload.pop("hooks", None)
    return payload


def uninstall_claude_hooks(project_root: Path) -> IntegrationResult:
    settings_path = project_root / ".claude" / CLAUDE_SETTINGS_LOCAL_FILENAME
    if not settings_path.exists():
        return IntegrationResult(
            path=settings_path,
            status="unchanged",
            details=".claude/settings.local.json does not exist.",
        )

    payload = json.loads(settings_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {settings_path} to contain a JSON object")

    changed = False
    enabled_servers = payload.get("enabledMcpjsonServers")
    if isinstance(enabled_servers, list) and DEFAULT_SERVER_NAME in enabled_servers:
        payload["enabledMcpjsonServers"] = [item for item in enabled_servers if item != DEFAULT_SERVER_NAME]
        if not payload["enabledMcpjsonServers"]:
            payload.pop("enabledMcpjsonServers", None)
        changed = True

    # Match BOTH the legacy `python -m agent_memory.hooks.*` form and the
    # current portable `agent-memory _hook *` form so uninstall is robust
    # across upgrades.
    changed = _remove_hook_commands(
        payload,
        "UserPromptSubmit",
        [
            "agent_memory.hooks.claude_user_prompt_submit",
            "_hook claude-user-prompt-submit",
        ],
    ) or changed
    changed = _remove_hook_commands(
        payload,
        "Stop",
        [
            "agent_memory.hooks.claude_stop_capture",
            "_hook claude-stop-capture",
        ],
    ) or changed
    payload = _prune_hook_payload(payload)

    if not changed:
        return IntegrationResult(
            path=settings_path,
            status="unchanged",
            details="Claude local settings do not contain agent-memory hook configuration.",
        )

    if payload:
        settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
        return IntegrationResult(
            path=settings_path,
            status="updated",
            details="Removed agent-memory Claude local hook configuration.",
        )

    settings_path.unlink()
    return IntegrationResult(
        path=settings_path,
        status="removed",
        details="Removed agent-memory Claude local hook configuration and deleted empty settings.local.json.",
    )


def _merge_event_hook(payload: dict[str, object], event_name: str, hook_payload: dict[str, object]) -> bool:
    hooks = payload.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Expected Codex hooks root to contain a JSON object under hooks")

    groups = hooks.setdefault(event_name, [])
    if not isinstance(groups, list):
        raise ValueError(f"Expected Codex hooks {event_name} to be a JSON array")

    desired_command = hook_payload.get("command")
    for group in groups:
        if not isinstance(group, dict):
            continue
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            continue
        for handler in handlers:
            if isinstance(handler, dict) and handler.get("command") == desired_command:
                changed = False
                for key, value in hook_payload.items():
                    if handler.get(key) != value:
                        handler[key] = value
                        changed = True
                return changed

    groups.append({"hooks": [hook_payload]})
    return True


def _remove_hook_commands(payload: dict[str, object], event_name: str, command_substrings: list[str]) -> bool:
    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return False
    groups = hooks.get(event_name)
    if not isinstance(groups, list):
        return False

    changed = False
    kept_groups: list[dict[str, object]] = []
    for group in groups:
        if not isinstance(group, dict):
            kept_groups.append(group)
            continue
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            kept_groups.append(group)
            continue
        kept_handlers = []
        for handler in handlers:
            if not isinstance(handler, dict):
                kept_handlers.append(handler)
                continue
            command = str(handler.get("command") or "")
            if any(part in command for part in command_substrings):
                changed = True
                continue
            kept_handlers.append(handler)
        if kept_handlers:
            kept_group = dict(group)
            kept_group["hooks"] = kept_handlers
            kept_groups.append(kept_group)
        else:
            changed = True
    hooks[event_name] = kept_groups
    return changed


def _set_features_cannot_fail(existing_text: str, *, codex_hooks: bool) -> str:
    lines = existing_text.splitlines()
    output: list[str] = []
    in_features = False
    features_seen = False
    codex_hooks_written = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        is_table = stripped.startswith("[") and stripped.endswith("]")
        if is_table:
            if in_features and not codex_hooks_written:
                output.append(f"codex_hooks = {'true' if codex_hooks else 'false'}")
                codex_hooks_written = True
            in_features = stripped == "[features]"
            features_seen = features_seen or in_features
            output.append(line)
            continue

        if in_features and stripped.startswith("codex_hooks"):
            output.append(f"codex_hooks = {'true' if codex_hooks else 'false'}")
            codex_hooks_written = True
            continue

        output.append(line)

        if index == len(lines) - 1 and in_features and not codex_hooks_written:
            output.append(f"codex_hooks = {'true' if codex_hooks else 'false'}")
            codex_hooks_written = True

    if not features_seen:
        if output and output[-1].strip():
            output.append("")
        output.append("[features]")
        output.append(f"codex_hooks = {'true' if codex_hooks else 'false'}")

    return "\n".join(output).rstrip() + "\n"


def install_codex_feature_flag(project_root: Path) -> IntegrationResult:
    codex_dir = project_root / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    config_path = codex_dir / CODEX_CONFIG_FILENAME
    existed = config_path.exists()

    if existed:
        existing_text = config_path.read_text(encoding='utf-8')
        payload = tomllib.loads(existing_text)
        features = payload.get("features")
        if isinstance(features, dict) and features.get("codex_hooks") is True:
            return IntegrationResult(
                path=config_path,
                status="unchanged",
                details="Codex repo-local config already enables codex_hooks.",
            )
        rendered = _set_features_cannot_fail(existing_text, codex_hooks=True)
        status = "updated"
    else:
        rendered = "[features]\ncodex_hooks = true\n"
        status = "created"

    config_path.write_text(rendered, encoding='utf-8')
    return IntegrationResult(
        path=config_path,
        status=status,
        details="Enabled Codex repo-local hooks via .codex/config.toml.",
    )


def uninstall_codex_feature_flag(project_root: Path) -> IntegrationResult:
    config_path = project_root / ".codex" / CODEX_CONFIG_FILENAME
    if not config_path.exists():
        return IntegrationResult(
            path=config_path,
            status="unchanged",
            details=".codex/config.toml does not exist.",
        )

    existing_text = config_path.read_text(encoding='utf-8')
    payload = tomllib.loads(existing_text)
    features = payload.get("features")
    if not isinstance(features, dict) or "codex_hooks" not in features:
        return IntegrationResult(
            path=config_path,
            status="unchanged",
            details="Codex repo-local config does not contain codex_hooks.",
        )

    hooks_path = project_root / ".codex" / CODEX_HOOKS_FILENAME
    if hooks_path.exists():
        hooks_payload = json.loads(hooks_path.read_text(encoding='utf-8'))
        hooks = hooks_payload.get("hooks")
        if isinstance(hooks, dict) and any(groups for groups in hooks.values()):
            return IntegrationResult(
                path=config_path,
                status="unchanged",
                details="Other Codex repo-local hooks still exist; leaving codex_hooks enabled.",
            )

    rendered = _remove_toml_key_in_section_cannot_fail(existing_text, "[features]", "codex_hooks")
    if rendered:
        config_path.write_text(rendered, encoding='utf-8')
        return IntegrationResult(
            path=config_path,
            status="updated",
            details="Removed codex_hooks from .codex/config.toml.",
        )

    config_path.unlink()
    return IntegrationResult(
        path=config_path,
        status="removed",
        details="Removed codex_hooks and deleted empty .codex/config.toml.",
    )


def install_codex_hooks(project_root: Path) -> IntegrationResult:
    codex_dir = project_root / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    hooks_path = codex_dir / CODEX_HOOKS_FILENAME
    existed = hooks_path.exists()

    if existed:
        payload = json.loads(hooks_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected {hooks_path} to contain a JSON object")
    else:
        payload = {}

    prompt_hook = {
        "type": "command",
        "command": _shell_command_for_hook(project_root, "codex-user-prompt-submit"),
        "timeout": 10,
        "statusMessage": "Recalling project memory",
    }
    # Drop ALL prior agent-memory hook command shapes before re-merging the
    # canonical form. See install_claude_hooks for the rationale — same logic.
    changed = _remove_hook_commands(
        payload,
        "UserPromptSubmit",
        [
            "agent_memory.hooks.codex_user_prompt_submit",  # legacy: python -m form
            "_hook codex-user-prompt-submit",  # all portable forms (broken + current)
        ],
    )
    changed = _merge_event_hook(payload, "UserPromptSubmit", prompt_hook) or changed
    changed = _remove_hook_commands(
        payload,
        "Stop",
        [
            "agent_memory.hooks.codex_stop_capture",
            "_hook codex-stop-capture",
        ],
    ) or changed
    payload = _prune_hook_payload(payload)

    if not changed and existed:
        return IntegrationResult(
            path=hooks_path,
            status="unchanged",
            details="Codex repo-local hooks already configured for agent-memory.",
        )

    hooks_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    return IntegrationResult(
        path=hooks_path,
        status="created" if not existed else "updated",
        details="Installed Codex repo-local prompt hook for memory reminders and recall guidance.",
    )


def uninstall_codex_hooks(project_root: Path) -> IntegrationResult:
    hooks_path = project_root / ".codex" / CODEX_HOOKS_FILENAME
    if not hooks_path.exists():
        return IntegrationResult(
            path=hooks_path,
            status="unchanged",
            details=".codex/hooks.json does not exist.",
        )

    payload = json.loads(hooks_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {hooks_path} to contain a JSON object")

    # Match BOTH the legacy `python -m agent_memory.hooks.*` form and the
    # current portable `agent-memory _hook *` form.
    changed = _remove_hook_commands(
        payload,
        "UserPromptSubmit",
        [
            "agent_memory.hooks.codex_user_prompt_submit",
            "_hook codex-user-prompt-submit",
        ],
    )
    changed = _remove_hook_commands(
        payload,
        "Stop",
        [
            "agent_memory.hooks.codex_stop_capture",
            "_hook codex-stop-capture",
        ],
    ) or changed
    payload = _prune_hook_payload(payload)

    if not changed:
        return IntegrationResult(
            path=hooks_path,
            status="unchanged",
            details="Codex repo-local hooks do not contain agent-memory hook configuration.",
        )

    if payload:
        hooks_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
        return IntegrationResult(
            path=hooks_path,
            status="updated",
            details="Removed agent-memory Codex repo-local hook configuration.",
        )

    hooks_path.unlink()
    return IntegrationResult(
        path=hooks_path,
        status="removed",
        details="Removed agent-memory Codex repo-local hook configuration and deleted empty hooks.json.",
    )


def _inject_instructions_block(existing_text: str) -> tuple[str, bool]:
    """Insert or replace the agent-memory block. Returns (new_text, changed)."""
    begin = INSTRUCTIONS_BEGIN_MARKER
    end = INSTRUCTIONS_END_MARKER
    block = INSTRUCTIONS_BLOCK.rstrip("\n")

    begin_idx = existing_text.find(begin)
    end_idx = existing_text.find(end)
    if begin_idx != -1 and end_idx != -1 and end_idx > begin_idx:
        prefix = existing_text[:begin_idx]
        suffix = existing_text[end_idx + len(end):]
        new_text = prefix + block + suffix
        return (new_text, new_text != existing_text)

    # No marker — inject after the first H1 line if present, otherwise at top.
    lines = existing_text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            insert_at = i + 1
            break

    prefix = "".join(lines[:insert_at]).rstrip("\n")
    suffix = "".join(lines[insert_at:]).lstrip("\n")
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
        parts.append("\n\n")
    parts.append(block)
    if suffix:
        parts.append("\n\n")
        parts.append(suffix)
    injected = "".join(parts)
    if not injected.endswith("\n"):
        injected += "\n"
    return (injected, True)


def _strip_instructions_block(existing_text: str) -> tuple[str, bool]:
    begin = INSTRUCTIONS_BEGIN_MARKER
    end = INSTRUCTIONS_END_MARKER
    begin_idx = existing_text.find(begin)
    end_idx = existing_text.find(end)
    if begin_idx == -1 or end_idx == -1 or end_idx < begin_idx:
        return (existing_text, False)
    after_end = end_idx + len(end)
    # Eat one trailing newline so we don't leave a stranded blank line.
    if after_end < len(existing_text) and existing_text[after_end] == "\n":
        after_end += 1
    new_text = existing_text[:begin_idx].rstrip(" \t") + existing_text[after_end:]
    return (new_text, True)


def install_memory_instructions(project_root: Path) -> list[IntegrationResult]:
    """Inject the Agent Memory instructions block into CLAUDE.md and AGENTS.md.

    Only updates files that already exist — never creates them. Idempotent via
    HTML-comment markers; subsequent runs replace the marker block in place.
    """
    results: list[IntegrationResult] = []
    for filename in INSTRUCTIONS_FILENAMES:
        path = project_root / filename
        if not path.exists():
            results.append(
                IntegrationResult(
                    path=path,
                    status="skipped",
                    details=f"{filename} does not exist; not creating it.",
                )
            )
            continue
        existing = path.read_text(encoding='utf-8')
        new_text, changed = _inject_instructions_block(existing)
        if not changed:
            results.append(
                IntegrationResult(
                    path=path,
                    status="unchanged",
                    details=f"Agent Memory instructions already current in {filename}.",
                )
            )
            continue
        path.write_text(new_text, encoding='utf-8')
        had_marker_before = INSTRUCTIONS_BEGIN_MARKER in existing
        results.append(
            IntegrationResult(
                path=path,
                status="updated" if had_marker_before else "created",
                details=f"{'Refreshed' if had_marker_before else 'Inserted'} Agent Memory instructions block in {filename}.",
            )
        )
    return results


def uninstall_memory_instructions(project_root: Path) -> list[IntegrationResult]:
    results: list[IntegrationResult] = []
    for filename in INSTRUCTIONS_FILENAMES:
        path = project_root / filename
        if not path.exists():
            results.append(
                IntegrationResult(
                    path=path,
                    status="unchanged",
                    details=f"{filename} does not exist.",
                )
            )
            continue
        existing = path.read_text(encoding='utf-8')
        new_text, changed = _strip_instructions_block(existing)
        if not changed:
            results.append(
                IntegrationResult(
                    path=path,
                    status="unchanged",
                    details=f"No Agent Memory instructions block found in {filename}.",
                )
            )
            continue
        path.write_text(new_text, encoding='utf-8')
        results.append(
            IntegrationResult(
                path=path,
                status="updated",
                details=f"Removed Agent Memory instructions block from {filename}.",
            )
        )
    return results
