from __future__ import annotations

import json
from collections import deque
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import typer

# Windows defaults the console to cp1252, which crashes on emoji or any non-Latin-1
# character that may appear in memory text or integration details. Force UTF-8 on
# both stdout and stderr so the CLI can safely echo arbitrary user content.
# Reason: prevents UnicodeEncodeError on Windows when the CLI prints unicode.
if sys.platform.startswith("win"):
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

from agent_memory.config import ConfigError, MemoryConfig, init_project, is_project_root, load_project
from agent_memory.embeddings import prune_fastembed_model_cache
from agent_memory.engine import AgentMemory, open_memory_with_retry, reembed_project
from agent_memory.integration import (
    codex_project_trust_state,
    ensure_local_git_excludes,
    install_claude_hooks,
    install_codex_feature_flag,
    install_codex_hooks,
    install_codex_mcp_server,
    install_codex_project_trust,
    install_mcp_server,
    install_memory_instructions,
    remove_local_git_excludes,
    suggest_project_root,
    uninstall_claude_hooks,
    uninstall_codex_feature_flag,
    uninstall_codex_hooks,
    uninstall_codex_mcp_server,
    uninstall_codex_project_trust,
    uninstall_mcp_server,
    uninstall_memory_instructions,
)
from agent_memory.mcp_server import serve
from agent_memory.repo_ingest import import_repo_corpus
# smoke_test is POSIX-only (uses pty/fcntl/termios). Import lazily from inside
# the command so `agent-memory --help` and every other command still work on
# Windows, where the module cannot be imported at all.
from agent_memory.store import GraphStore
from agent_memory.hooks.common import (
    approve_consolidation_apply,
    consolidation_status,
    hook_log_entries,
    mark_consolidation_completed,
    mark_consolidation_started,
)


from agent_memory import __display_version__, __version__


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"agent-memory {__display_version__}")
        raise typer.Exit()


app = typer.Typer(
    no_args_is_help=True,
    help=(
        "Project-scoped local memory for agents. "
        "Each initialized folder gets its own .agent-memory store. "
        "Use `agent-memory init` inside a repo to create the store, install "
        "Codex/Claude local prompt hooks plus Codex project trust for automatic memory instructions, then `save`, `recall`, `list`, `edit`, `delete`, `undo`."
    ),
)


@app.callback()
def _main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print agent-memory version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Top-level callback so `agent-memory --version` works without a subcommand.
    Also where we plug in the non-blocking 24h staleness check."""
    # Lazy import to keep cold-start fast for hot-path commands.
    try:
        from agent_memory.upgrade import check_for_upgrade_in_background

        notice = check_for_upgrade_in_background()
        if notice:
            # Print to stderr so it never pollutes JSON output.
            typer.echo(notice, err=True)
    except Exception:
        # Never fail a real command because the staleness check threw.
        pass


# Internal hook dispatch group: used by .codex/hooks.json and .claude/settings.local.json
# entries to invoke hook handlers via the on-PATH `agent-memory` binary instead of
# embedding an absolute python interpreter path. Hidden from --help so users don't
# discover it accidentally — these are not user-facing commands.
hook_app = typer.Typer(no_args_is_help=True, hidden=True)
app.add_typer(
    hook_app,
    name="_hook",
    hidden=True,
    help="Internal: hook handlers invoked by Codex/Claude local hooks.",
)


@hook_app.command(
    name="claude-user-prompt-submit",
    help="Internal hook handler for Claude Code's UserPromptSubmit event.",
    hidden=True,
)
def _hook_claude_user_prompt_submit() -> None:
    from agent_memory.hooks.claude_user_prompt_submit import main as _main

    _main()


@hook_app.command(
    name="codex-user-prompt-submit",
    help="Internal hook handler for Codex's UserPromptSubmit event.",
    hidden=True,
)
def _hook_codex_user_prompt_submit() -> None:
    from agent_memory.hooks.codex_user_prompt_submit import main as _main

    _main()


@hook_app.command(
    name="claude-stop-capture",
    help="Internal hook handler for Claude Code's Stop event.",
    hidden=True,
)
def _hook_claude_stop_capture() -> None:
    from agent_memory.hooks.claude_stop_capture import main as _main

    _main()


@hook_app.command(
    name="codex-stop-capture",
    help="Internal hook handler for Codex's Stop event.",
    hidden=True,
)
def _hook_codex_stop_capture() -> None:
    from agent_memory.hooks.codex_stop_capture import main as _main

    _main()


def _emit(payload: dict[str, object], as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def _open_memory(cwd: Path, *, read_only: bool = False) -> AgentMemory:
    try:
        return open_memory_with_retry(
            cwd,
            exact=is_project_root(cwd),
            read_only=read_only,
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_project(cwd: Path):
    memory = _open_memory(cwd, read_only=True)
    project = memory.project
    memory.close()
    return project


def _resolve_init_path(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    suggested = suggest_project_root(Path.cwd())
    if sys.stdin.isatty():
        response = typer.prompt(
            "Project root for Agent Memory",
            default=str(suggested),
        )
        return Path(response).expanduser().resolve()
    return suggested


def _resolve_project_path(path: Path | None) -> Path:
    candidate = (path or Path.cwd()).resolve()
    if is_project_root(candidate):
        return candidate
    try:
        return load_project(candidate).root
    except ConfigError:
        return suggest_project_root(candidate)


def _default_smoke_reinstall_from(cwd: Path | None = None) -> Path | None:
    candidate = (cwd or Path.cwd()).resolve()
    pyproject_path = candidate / "pyproject.toml"
    cli_path = candidate / "src" / "agent_memory" / "cli.py"
    if not pyproject_path.exists() or not cli_path.exists():
        return None
    try:
        payload = tomllib.loads(pyproject_path.read_text(encoding='utf-8'))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = payload.get("project")
    if not isinstance(project, dict):
        return None
    if project.get("name") != "agent-memory":
        return None
    return candidate


def _result_payload(result) -> dict[str, object]:
    return {
        "path": str(result.path),
        "status": result.status,
        "details": result.details,
    }


def _remove_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        return


def _short_error_text(text: str, *, limit: int = 160) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _codex_config_has_server(project_root: Path, server_name: str = "agent-memory") -> tuple[bool | None, str | None]:
    config_path = project_root / ".codex" / "config.toml"
    if not config_path.exists():
        return False, None
    try:
        payload = tomllib.loads(config_path.read_text(encoding='utf-8'))
    except tomllib.TOMLDecodeError as exc:
        return None, f"Failed to parse {config_path}: {exc}"
    mcp_servers = payload.get("mcp_servers")
    if mcp_servers is None:
        return False, None
    if not isinstance(mcp_servers, dict):
        return None, f"Expected {config_path} mcp_servers to be a TOML table."
    return server_name in mcp_servers, None


def _codex_feature_state(project_root: Path, feature_name: str = "codex_hooks") -> tuple[bool | None, str | None]:
    codex_path = shutil.which("codex")
    if not codex_path:
        return None, "Codex CLI not found on PATH."

    try:
        result = subprocess.run(
            [codex_path, "-C", str(project_root), "features", "list"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return None, f"Failed to run Codex CLI: {exc}"

    if result.returncode != 0:
        details = _short_error_text(result.stderr or result.stdout or "unknown Codex CLI failure")
        return None, f"`codex features list` failed: {details}"

    pattern = re.compile(rf"^{re.escape(feature_name)}\s+.+?\s+(true|false)\s*$", re.MULTILINE)
    match = pattern.search(result.stdout)
    if not match:
        return None, f"Could not parse `{feature_name}` from `codex features list`."
    return match.group(1) == "true", None


def _doctor_payload(cwd: Path) -> dict[str, object]:
    project_root = suggest_project_root(cwd.resolve())
    initialized = (project_root / ".agent-memory" / "config.json").exists()
    files = {
        "initialized": initialized,
        "instructions": (project_root / ".agent-memory" / "instructions.md").exists(),
        "mcp_json": (project_root / ".mcp.json").exists(),
        "codex_config": (project_root / ".codex" / "config.toml").exists(),
        "codex_hooks": (project_root / ".codex" / "hooks.json").exists(),
        "claude_settings": (project_root / ".claude" / "settings.local.json").exists(),
    }

    codex_path = shutil.which("codex")
    agent_memory_path = shutil.which("agent-memory")
    codex_hooks_effective = None
    codex_feature_error = None
    codex_mcp_server = None
    codex_mcp_error = None
    codex_project_trusted = None
    codex_trust_error = None
    if files["codex_config"]:
        codex_project_trusted, codex_trust_error = codex_project_trust_state(project_root)
    if codex_path:
        codex_hooks_effective, codex_feature_error = _codex_feature_state(project_root)
        codex_mcp_server, codex_mcp_error = _codex_config_has_server(project_root)

    warnings: list[str] = []
    if not initialized:
        warnings.append("Agent Memory is not initialized in this project root yet.")
    if codex_path is None and files["codex_config"]:
        warnings.append("Codex CLI not found on PATH; cannot verify `codex_hooks` for this project.")
    if codex_path and files["codex_hooks"]:
        warnings.append("Use a fresh interactive Codex session rooted in this repo to validate hook injection.")
        warnings.append("`codex exec` may ignore repo-local hooks and project MCP wiring.")
    if files["codex_config"] and codex_project_trusted is False:
        warnings.append("Codex ignores repo-local `.codex/config.toml` until this project is trusted in `~/.codex/config.toml`.")
    if files["mcp_json"] and codex_mcp_server is False:
        warnings.append("Current Codex builds load repo-local MCP servers from `.codex/config.toml`; `.mcp.json` alone may not be enough.")
    if codex_hooks_effective is False:
        warnings.append("`codex features list` reports `codex_hooks` as disabled in the current Codex CLI environment.")
    if codex_feature_error:
        warnings.append(codex_feature_error)
    if codex_mcp_error:
        warnings.append(codex_mcp_error)
    if codex_trust_error:
        warnings.append(codex_trust_error)

    return {
        "project_root": str(project_root),
        "paths": {
            "agent_memory": agent_memory_path,
            "codex": codex_path,
        },
        "files": files,
        "codex_hooks_effective": codex_hooks_effective,
        "codex_mcp_server": codex_mcp_server,
        "codex_project_trusted": codex_project_trusted,
        "warnings": warnings,
    }


def _run_init(
    path: Path | None,
    embedding_backend: str,
    force: bool,
    install_mcp: bool,
    install_local_excludes: bool,
    install_codex: bool,
    install_codex_trust: bool,
    install_claude: bool,
) -> None:
    resolved_path = _resolve_init_path(path)
    config = MemoryConfig(embedding_backend=embedding_backend)
    try:
        project = init_project(resolved_path, config=config, force=force)
        store = GraphStore(project.db_path, config.embedding_dimensions)
        store.close()
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Initialized Agent Memory at {project.root}")
    typer.echo(f"Config: {project.config_path}")
    typer.echo(f"DB: {project.db_path}")
    typer.echo(f"Instructions: {project.instructions_path}")

    if install_local_excludes:
        exclude_result = ensure_local_git_excludes(project.root)
        typer.echo(f"Local ignore: {exclude_result.details}")

    if install_mcp:
        try:
            mcp_result = install_mcp_server(project.root)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"MCP: {mcp_result.details}")
        typer.echo("Project note: `.mcp.json` helps clients that honor repo-local MCP JSON directly.")

    if install_codex:
        try:
            codex_config_result = install_codex_feature_flag(project.root)
            codex_hooks_result = install_codex_hooks(project.root)
            codex_mcp_result = install_codex_mcp_server(project.root) if install_mcp else None
            codex_trust_result = (
                install_codex_project_trust(project.root) if install_codex_trust else None
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"Codex config: {codex_config_result.details}")
        typer.echo(f"Codex hooks: {codex_hooks_result.details}")
        if codex_mcp_result is not None:
            typer.echo(f"Codex MCP: {codex_mcp_result.details}")
        if codex_trust_result is not None:
            typer.echo(f"Codex trust: {codex_trust_result.details}")
            typer.echo("Codex note: current Codex builds load repo-local `.codex/config.toml` only for trusted projects.")
        typer.echo("Codex note: start a fresh interactive Codex session rooted in this repo to load repo-local hooks.")

    if install_claude:
        try:
            claude_result = install_claude_hooks(project.root, register_mcp_server=install_mcp)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"Claude hooks: {claude_result.details}")
        typer.echo("Claude Code will now inject memory instructions before prompts in this repo.")

    instructions_results = install_memory_instructions(project.root)
    for result in instructions_results:
        typer.echo(f"Instructions ({result.path.name}): {result.details}")

    typer.echo(
        "Access pattern: agents call `agent-memory recall <query>` and "
        "`agent-memory save \"<memory>\"` via their shell tool. The CLI is the canonical interface; "
        "MCP is opt-in via --with-mcp."
    )


def _run_uninstall(
    path: Path | None,
    *,
    remove_store: bool,
    remove_codex_trust: bool,
    as_json: bool,
) -> None:
    project_root = _resolve_project_path(path)
    local_exclude_entries = [entry for entry in [
        ".claude/settings.local.json",
        ".codex/config.toml",
        ".codex/hooks.json",
        ".mcp.json",
    ]]
    if remove_store:
        local_exclude_entries.insert(0, ".agent-memory/")

    results: list[tuple[str, object]] = [
        ("mcp", uninstall_mcp_server(project_root)),
        ("codex_hooks", uninstall_codex_hooks(project_root)),
        ("codex_feature_flag", uninstall_codex_feature_flag(project_root)),
        ("codex_mcp", uninstall_codex_mcp_server(project_root)),
        ("claude_hooks", uninstall_claude_hooks(project_root)),
        ("local_excludes", remove_local_git_excludes(project_root, entries=local_exclude_entries)),
    ]
    for instructions_result in uninstall_memory_instructions(project_root):
        results.append((f"instructions:{instructions_result.path.name}", instructions_result))

    if remove_codex_trust:
        results.append(("codex_trust", uninstall_codex_project_trust(project_root)))

    # Clean up now-empty .codex/ and .claude/ directories so a clean uninstall
    # leaves no trace. Only rmdir if the directory is genuinely empty — never
    # touch a directory the user has populated with their own files.
    for sub in (".codex", ".claude"):
        dir_path = project_root / sub
        try:
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
        except OSError:
            # Permissions / race / non-empty: leave it alone, the file removals
            # have already done the user-visible work.
            pass

    store_path = project_root / ".agent-memory"
    if remove_store:
        if store_path.exists():
            shutil.rmtree(store_path)
            store_result = {
                "path": str(store_path),
                "status": "removed",
                "details": "Deleted .agent-memory/ and all stored project memory data.",
            }
        else:
            store_result = {
                "path": str(store_path),
                "status": "unchanged",
                "details": ".agent-memory/ does not exist.",
            }
    else:
        store_result = {
            "path": str(store_path),
            "status": "kept",
            "details": "Left .agent-memory/ in place. Pass --remove-store for a full clean-room uninstall.",
        }

    _remove_if_empty(project_root / ".codex")
    _remove_if_empty(project_root / ".claude")

    payload = {
        "project_root": str(project_root),
        "remove_store": remove_store,
        "remove_codex_trust": remove_codex_trust,
        "results": {name: _result_payload(result) for name, result in results},
        "store": store_result,
    }

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Project root: {project_root}")
    for name, result in results:
        typer.echo(f"{name}: {result.details}")
    typer.echo(f"store: {store_result['details']}")


@app.command(
    help=(
        "Initialize a project-local memory store and wire repo-local integration in one step. "
        "By default this creates .agent-memory/, hides local integration files from git, installs Codex and Claude "
        "UserPromptSubmit hooks for automatic memory instructions, adds this repo to Codex trusted projects, and "
        "injects an Agent Memory section into CLAUDE.md/AGENTS.md if those files exist. The default install is "
        "CLI-only — agents access memory through `agent-memory recall` / `agent-memory save` via their shell tool. "
        "Pass --with-mcp to additionally wire repo-local MCP server entries (.mcp.json, .codex/config.toml, "
        ".claude/settings.local.json enabledMcpjsonServers) for clients that prefer native MCP tool calls."
    )
)
def init(
    path: Path | None = typer.Argument(
        None,
        help="Project directory that should hold the local memory store.",
        resolve_path=True,
    ),
    embedding_backend: str = typer.Option(
        "fastembed",
        "--embedding-backend",
        help="Embedding backend to use. `fastembed` is the default, `hash` is deterministic and test-friendly.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite the existing config if the project is already initialized.",
    ),
    install_mcp: bool = typer.Option(
        False,
        "--with-mcp/--without-mcp",
        help="Opt-in: also wire repo-local MCP server entries (.mcp.json, Codex MCP block, Claude enabledMcpjsonServers). Default is CLI-only.",
    ),
    install_local_excludes: bool = typer.Option(
        True,
        "--install-local-excludes/--no-install-local-excludes",
        help="Hide local agent-memory integration files via .git/info/exclude when possible.",
    ),
    install_codex: bool = typer.Option(
        True,
        "--install-codex-hooks/--no-install-codex-hooks",
        help="Install Codex repo-local prompt hooks and enable the codex_hooks feature flag for automatic memory instructions before prompts.",
    ),
    install_codex_trust: bool = typer.Option(
        True,
        "--install-codex-trust/--no-install-codex-trust",
        help="Add this repo to Codex trusted projects in ~/.codex/config.toml so repo-local Codex config is loaded.",
    ),
    install_claude: bool = typer.Option(
        True,
        "--install-claude-hooks/--no-install-claude-hooks",
        help="Install Claude local prompt hooks for automatic memory instructions before prompts.",
    ),
) -> None:
    _run_init(
        path=path,
        embedding_backend=embedding_backend,
        force=force,
        install_mcp=install_mcp,
        install_local_excludes=install_local_excludes,
        install_codex=install_codex,
        install_codex_trust=install_codex_trust,
        install_claude=install_claude,
    )


@app.command(
    help=(
        "Interactive one-shot setup flow. Alias for `init` with the same defaults: hooks + instruction "
        "injection, CLI-only access. Pass --with-mcp to additionally wire repo-local MCP server entries."
    )
)
def setup(
    path: Path | None = typer.Argument(
        None,
        help="Project directory that should hold the local memory store.",
        resolve_path=True,
    ),
    embedding_backend: str = typer.Option(
        "fastembed",
        "--embedding-backend",
        help="Embedding backend to use. `fastembed` is the default, `hash` is deterministic and test-friendly.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite the existing config if the project is already initialized.",
    ),
    install_mcp: bool = typer.Option(
        False,
        "--with-mcp/--without-mcp",
        help="Opt-in: also wire repo-local MCP server entries. Default is CLI-only.",
    ),
) -> None:
    _run_init(
        path=path,
        embedding_backend=embedding_backend,
        force=force,
        install_mcp=install_mcp,
        install_local_excludes=True,
        install_codex=True,
        install_codex_trust=True,
        install_claude=True,
    )


@app.command(
    help=(
        "Remove Agent Memory integration from a project. By default this removes repo-local MCP/hooks/config "
        "and the project's Codex trust entry, but keeps `.agent-memory/` so saved memories are not destroyed "
        "unless you pass `--remove-store`."
    )
)
def uninstall(
    path: Path | None = typer.Argument(
        None,
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    remove_store: bool = typer.Option(
        False,
        "--remove-store/--keep-store",
        help="Delete `.agent-memory/` for a full clean-room uninstall.",
    ),
    remove_codex_trust: bool = typer.Option(
        True,
        "--remove-codex-trust/--keep-codex-trust",
        help="Remove this repo's trusted-project entry from ~/.codex/config.toml.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    _run_uninstall(
        path=path,
        remove_store=remove_store,
        remove_codex_trust=remove_codex_trust,
        as_json=as_json,
    )


@app.command(
    help=(
        "Save one or more memories into the current project store. "
        "Pass multiple quoted arguments to batch-save in one command, "
        "or pipe a single memory body via --stdin to avoid shell-escaping multi-line text."
    )
)
def save(
    texts: list[str] = typer.Argument(
        None,
        help="One or more memory texts to persist. Omit when using --stdin.",
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    from_stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read a single memory body from stdin instead of taking it as an argument. Useful for memories with quotes, newlines, or other shell-hostile characters.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    if from_stdin:
        if texts:
            raise typer.BadParameter("--stdin cannot be combined with positional memory arguments.")
        body = sys.stdin.read().strip()
        if not body:
            raise typer.BadParameter("--stdin received empty input; nothing to save.")
        texts = [body]
    elif not texts:
        raise typer.BadParameter("Provide at least one memory argument, or use --stdin.")

    memory = _open_memory(cwd)
    try:
        result = memory.save(texts)
    finally:
        memory.close()
    payload = result.to_dict()
    if as_json:
        _emit(payload, True)
        return
    saved = payload["saved"]
    if len(saved) == 1:
        item = saved[0]
        typer.echo(f"Saved {item['memory_id']}")
        typer.echo(f"Created at: {item['created_at']}")
        typer.echo(f"Connected neighbors: {len(item['connected_neighbors'])}")
        for neighbor in item["connected_neighbors"]:
            typer.echo(
                f"  {neighbor['memory_id']}  similarity={neighbor['similarity']}"
            )
    else:
        typer.echo(f"Saved {len(saved)} memories")
        for item in saved:
            typer.echo(f"  {item['memory_id']}  created_at={item['created_at']}")
    typer.echo(f"Total memories: {payload['total_memories']}")


def _format_memory_record(record, *, show_full_text: bool = True) -> dict:
    payload = {
        "memory_id": record.id,
        "created_at": record.created_at,
        "text": record.text if show_full_text else (record.text[:140] + "…" if len(record.text) > 140 else record.text),
        "access_count": record.access_count,
        "last_accessed": record.last_accessed,
    }
    return payload


@app.command(
    name="list",
    help=(
        "List memories in the project store. Defaults to the 10 most-recently-created. "
        "Use --all to dump every memory."
    ),
)
def list_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    recent: int = typer.Option(
        10,
        "--recent",
        "-n",
        help="Number of most-recent memories to show. Ignored if --all is set.",
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        help="List every memory in the store, not just the most recent.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory = _open_memory(cwd, read_only=True)
    try:
        records = memory.list_all() if show_all else memory.list_recent(recent)
        # list_all is created_at ASC; for newest-first display reverse it.
        if show_all:
            records = list(reversed(records))
        total = memory.stats().memory_count
    finally:
        memory.close()

    if as_json:
        _emit(
            {
                "total_memories": total,
                "shown": len(records),
                "memories": [_format_memory_record(r) for r in records],
            },
            True,
        )
        return

    if not records:
        typer.echo("(no memories yet — try `agent-memory save \"...\"`)")
        return
    typer.echo(f"Showing {len(records)} of {total} memories (newest first):")
    for record in records:
        typer.echo("")
        typer.echo(f"  {record.id}  ({record.created_at})")
        typer.echo(f"    {record.text}")


@app.command(
    name="show",
    help="Show the full text and metadata of a single memory by id.",
)
def show_command(
    memory_id: str = typer.Argument(..., help="Memory id, e.g. mem_abc123def456."),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory = _open_memory(cwd, read_only=True)
    try:
        record = memory.get(memory_id)
    finally:
        memory.close()
    if record is None:
        typer.echo(f"No memory with id {memory_id!r}.", err=True)
        raise typer.Exit(code=1)
    if as_json:
        _emit(_format_memory_record(record), True)
        return
    typer.echo(f"id: {record.id}")
    typer.echo(f"created_at: {record.created_at}")
    typer.echo(f"access_count: {record.access_count}")
    if record.last_accessed:
        typer.echo(f"last_accessed: {record.last_accessed}")
    typer.echo("")
    typer.echo(record.text)


@app.command(
    name="edit",
    help=(
        "Edit a memory's text in place and re-embed it. "
        "Three input modes: pass new text as a positional argument (one-shot), "
        "use --stdin for multi-line / shell-hostile content, or omit text entirely "
        "to open $EDITOR with the current memory body prefilled."
    ),
)
def edit_command(
    memory_id: str = typer.Argument(..., help="Memory id, e.g. mem_abc123def456."),
    new_text: str | None = typer.Argument(
        None,
        help="New text for the memory. Omit to use --stdin or $EDITOR.",
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    from_stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read the new memory body from stdin instead of taking it as an argument.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    if from_stdin and new_text is not None:
        raise typer.BadParameter("--stdin cannot be combined with a positional new-text argument.")

    if from_stdin:
        body = sys.stdin.read().strip()
        if not body:
            raise typer.BadParameter("--stdin received empty input; nothing to write.")
        resolved_text: str | None = body
    else:
        resolved_text = new_text

    memory = _open_memory(cwd)
    try:
        if resolved_text is None:
            # $EDITOR mode: prefill a temp file with the current text, let the user edit, read it back.
            existing = memory.get(memory_id)
            if existing is None:
                typer.echo(f"No memory with id {memory_id!r}.", err=True)
                raise typer.Exit(code=1)
            edited = typer.edit(existing.text, extension=".md")
            if edited is None:
                typer.echo("Edit aborted (no editor or no save). Memory unchanged.")
                raise typer.Exit(code=1)
            resolved_text = edited.strip()
            if resolved_text == existing.text.strip():
                typer.echo("No changes detected. Memory unchanged.")
                raise typer.Exit(code=0)
            if not resolved_text:
                typer.echo("Editor produced empty content. Memory unchanged.", err=True)
                raise typer.Exit(code=1)

        try:
            updated = memory.edit(memory_id, resolved_text)
        except KeyError:
            typer.echo(f"No memory with id {memory_id!r}.", err=True)
            raise typer.Exit(code=1)
    finally:
        memory.close()

    if as_json:
        _emit(_format_memory_record(updated), True)
        return
    typer.echo(f"Updated {updated.id}")
    typer.echo(updated.text)


@app.command(
    name="delete",
    help="Delete a single memory by id. Pass --yes to skip the confirmation prompt.",
)
def delete_command(
    memory_id: str = typer.Argument(..., help="Memory id, e.g. mem_abc123def456."),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt. Required for non-interactive use.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory = _open_memory(cwd)
    try:
        existing = memory.get(memory_id)
        if existing is None:
            typer.echo(f"No memory with id {memory_id!r}.", err=True)
            raise typer.Exit(code=1)

        if not yes:
            preview = existing.text if len(existing.text) <= 200 else existing.text[:200] + "…"
            typer.echo(f"About to delete {existing.id}:")
            typer.echo(f"  {preview}")
            confirm = typer.confirm("Proceed?", default=False)
            if not confirm:
                typer.echo("Aborted.")
                raise typer.Exit(code=1)

        deleted = memory.delete(memory_id)
        total = memory.stats().memory_count
    finally:
        memory.close()

    if as_json:
        _emit(
            {
                "deleted": _format_memory_record(deleted),
                "total_memories": total,
                "undo_hint": "Run `agent-memory undo` to restore.",
            },
            True,
        )
        return
    typer.echo(f"Deleted {deleted.id}. Total memories: {total}.")
    typer.echo("Run `agent-memory undo` to restore it.")


@app.command(
    name="undo",
    help=(
        "Reverse the most recent destructive memory operation (save, edit, or delete) "
        "in this project. Each undo appends a record to the operations log so a future "
        "undo can find the next-most-recent operation."
    ),
)
def undo_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory = _open_memory(cwd)
    try:
        result = memory.undo()
        total = memory.stats().memory_count
    finally:
        memory.close()
    payload = {**result, "total_memories": total}
    if as_json:
        _emit(payload, True)
        return
    if not result.get("reverted"):
        typer.echo(result.get("reason") or "Nothing to undo.")
        raise typer.Exit(code=1)
    typer.echo(f"Reverted {result['reverted']} (seq {result['seq']}).")
    typer.echo(result["details"])
    typer.echo(f"Total memories: {total}.")


@app.command(
    help=(
        "Capture one full interaction turn in the background: the user message, the assistant reply, "
        "and any extra distilled memories. This is the recommended hook for automatic background writes."
    )
)
def capture_turn(
    user: str | None = typer.Option(None, "--user", help="The user message from the turn."),
    assistant: str | None = typer.Option(None, "--assistant", help="The assistant reply from the turn."),
    memory: list[str] = typer.Option(
        [],
        "--memory",
        help="Additional distilled memories to save alongside the raw turn.",
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory_store = _open_memory(cwd)
    try:
        result = memory_store.capture_turn(
            user_text=user,
            assistant_text=assistant,
            memories=memory,
        )
    finally:
        memory_store.close()
    payload = result.to_dict()
    if as_json:
        _emit(payload, True)
        return
    typer.echo(f"Captured {len(payload['saved'])} memories from turn")
    for item in payload["saved"]:
        typer.echo(f"  {item['memory_id']}  created_at={item['created_at']}")
    typer.echo(f"Total memories: {payload['total_memories']}")


@app.command(
    help=(
        "Recall the highest-scoring memories for a query. "
        "Results are ordered by descending parent-similarity score from the query root."
    )
)
def recall(
    query_parts: list[str] = typer.Argument(
        ...,
        help="Query text used to seed recall. Multiple words may be passed without shell quoting.",
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    limit: int = typer.Option(15, "--limit", min=1),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        raise typer.BadParameter("Query text cannot be empty.")
    memory = _open_memory(cwd, read_only=True)
    try:
        result = memory.recall(query, limit=limit)
    finally:
        memory.close()
    payload = result.to_dict()
    if as_json:
        _emit(payload, True)
        return
    typer.echo(f'Recall for: "{payload["query"]}"')
    for index, node in enumerate(payload["nodes"]):
        if index:
            typer.echo("")
        typer.echo(
            f'{node["alias"]} <- {node["source"]}  similarity={node["source_similarity"]}'
        )
        typer.echo(f'  {node["created_at"]}  {node["memory_id"]}')
        typer.echo(f'  {node["text"]}')


@app.command(
    help=(
        "Report overlapping high-similarity memory clusters without mutating stored memories."
    )
)
def consolidate(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory = _open_memory(cwd, read_only=True)
    try:
        report = memory.consolidate()
    finally:
        memory.close()
    payload = report.to_dict()
    if as_json:
        _emit(payload, True)
        return
    typer.echo(
        f"Clusters: {len(payload['clusters'])}  threshold={payload['threshold']}  "
        f"clustered_memories={payload['clustered_memory_count']}/{payload['total_memories']}"
    )
    for cluster in payload["clusters"]:
        typer.echo(
            f"  {cluster['cluster_id']}  size={len(cluster['member_ids'])}  "
            f"avg={cluster['average_similarity']}  max={cluster['max_similarity']}"
        )
        typer.echo(f"    members: {' ~ '.join(cluster['member_ids'])}")


@app.command(
    "consolidation-status",
    help="Show the current daily memory consolidation scheduler state for this project."
)
def consolidation_status_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    payload = consolidation_status(load_project(cwd, exact=is_project_root(cwd)).root)
    _emit(payload, as_json)


@app.command(
    "consolidation-start",
    help="Mark today's memory consolidation workflow as started for this project."
)
def consolidation_start(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    payload = mark_consolidation_started(load_project(cwd, exact=is_project_root(cwd)).root)
    _emit(payload, as_json)


@app.command(
    "consolidation-complete",
    help="Mark today's memory consolidation workflow as completed for this project."
)
def consolidation_complete(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    payload = mark_consolidation_completed(load_project(cwd, exact=is_project_root(cwd)).root)
    _emit(payload, as_json)


@app.command(
    "consolidation-approve",
    help="Approve today's dry-run memory consolidation plan so the host LLM can apply the changes."
)
def consolidation_approve(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    payload = approve_consolidation_apply(load_project(cwd, exact=is_project_root(cwd)).root)
    _emit(payload, as_json)


@app.command(
    help=(
        "Rebuild persisted SIMILAR and NEXT edges from stored nodes and embeddings. "
        "This is mainly a maintenance/debug command; reads use the stored nodes and embeddings."
    )
)
def rewire(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory = _open_memory(cwd)
    try:
        payload = memory.rewire().to_dict()
    finally:
        memory.close()
    _emit(payload, as_json)


@app.command(
    help="Show basic counts for memories and persisted relationships in the current project store."
)
def stats(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    memory = _open_memory(cwd, read_only=True)
    try:
        payload = memory.stats().to_dict()
    finally:
        memory.close()
    _emit(payload, as_json)


@app.command(
    name="import-repo",
    help=(
        "Scan a source tree, chunk supported text/code files into memory-sized records, "
        "and bulk import them into the current project store. Intended for bootstrapping "
        "large repos such as EDS without hand-saving thousands of memories."
    ),
)
def import_repo_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the initialized project store.",
        resolve_path=True,
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Source tree to scan. Defaults to the current project root.",
        resolve_path=True,
    ),
    max_memories: int = typer.Option(
        3000,
        "--max-memories",
        min=1,
        help="Stop after importing roughly this many new memories.",
    ),
    max_chunks_per_file: int = typer.Option(
        6,
        "--max-chunks-per-file",
        min=1,
        help="Prevent a few large files from dominating the import.",
    ),
    max_file_kb: int = typer.Option(
        512,
        "--max-file-kb",
        min=1,
        help="Skip files larger than this many KB.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    try:
        result = import_repo_corpus(
            cwd,
            source_root=root,
            max_memories=max_memories,
            max_chunks_per_file=max_chunks_per_file,
            max_file_bytes=max_file_kb * 1024,
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = result.to_dict()
    if as_json:
        _emit(payload, True)
        return

    typer.echo(f"Project root: {payload['project_root']}")
    typer.echo(f"Source root: {payload['source_root']}")
    typer.echo(
        f"Imported {payload['imported_memories']} memories from {payload['imported_files']} files "
        f"(candidate_files={payload['candidate_files']}, discovered_files={payload['discovered_files']})."
    )
    typer.echo(f"Total memories: {payload['total_memories']}")
    if payload["skipped_existing_texts"]:
        typer.echo(f"Skipped existing identical texts: {payload['skipped_existing_texts']}")
    component_counts = payload["component_counts"]
    assert isinstance(component_counts, dict)
    if component_counts:
        typer.echo("Imported by component:")
        for component, count in component_counts.items():
            typer.echo(f"  {component}: {count}")


@app.command(
    name="reembed",
    help=(
        "Re-embed every memory in the current project store using the configured local embedding model. "
        "This rewrites the database when the store's recorded embedding backend/model/dimensions differ "
        "from the current config, or when --force is passed."
    ),
)
def reembed_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild the store even if its recorded embedding settings already match the config.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    try:
        result = reembed_project(
            cwd,
            exact=is_project_root(cwd),
            force=force,
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = result.to_dict()
    if as_json:
        _emit(payload, True)
        return

    if result.reembedded:
        typer.echo(f"Re-embedded {result.memory_count} memories.")
    else:
        typer.echo("Store embeddings already match the current config.")
    typer.echo(
        "store: "
        f"{result.previous_store_backend}/{result.previous_store_model}/{result.previous_store_dimensions}"
    )
    typer.echo(
        "config: "
        f"{result.current_store_backend}/{result.current_store_model}/{result.current_store_dimensions}"
    )
    if result.cache_prune is not None and result.cache_prune.pruned:
        typer.echo(
            f"Pruned {len(result.cache_prune.pruned)} cached model(s), "
            f"freed {_format_bytes(result.cache_prune.freed_bytes)}."
        )


@app.command(
    name="prune-model-cache",
    help=(
        "Delete stale fastembed model caches and keep only the embedding model configured for the current project. "
        "Useful after a model switch or re-embed so old ONNX weights don't linger on disk."
    ),
)
def prune_model_cache_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    try:
        project = load_project(cwd, exact=is_project_root(cwd))
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if project.config.embedding_backend.lower() != "fastembed":
        payload = {
            "cache_dir": None,
            "kept_models": [],
            "pruned": [],
            "freed_bytes": 0,
            "details": f"No fastembed model cache to prune for backend `{project.config.embedding_backend}`.",
        }
        _emit(payload, as_json)
        return

    result = prune_fastembed_model_cache([project.config.embedding_model])
    payload = result.to_dict()
    if as_json:
        _emit(payload, True)
        return

    typer.echo(f"Kept model cache: {project.config.embedding_model}")
    typer.echo(f"Cache dir: {result.cache_dir}")
    if not result.pruned:
        typer.echo("No stale model caches found.")
        return
    typer.echo(
        f"Pruned {len(result.pruned)} cached model(s), freed {_format_bytes(result.freed_bytes)}."
    )
    for entry in result.pruned:
        typer.echo(f"  {entry.model_name or entry.hf_source or entry.path.name}")


@app.command(
    "serve-mcp",
    help=(
        "Run the local MCP server for one exact project root. "
        "Use --cwd with the initialized repo root to avoid ambiguous project selection."
    ),
)
def serve_mcp(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    )
) -> None:
    try:
        serve(cwd.resolve())
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command(
    help=(
        "Print the project-local agent instructions block from .agent-memory/instructions.md. "
        "Use this in wrappers or system prompts instead of mutating a repo-wide AGENTS.md."
    )
)
def instructions(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
) -> None:
    project = _load_project(cwd)
    typer.echo(project.instructions_path.read_text(encoding='utf-8'))


@app.command(
    help=(
        "Inspect the local Agent Memory integration files and Codex/Claude wiring without assuming the "
        "host will actually load them. This is useful when debugging fresh-session setup versus "
        "`codex exec` behavior."
    )
)
def doctor(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    payload = _doctor_payload(cwd)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Project root: {payload['project_root']}")
    paths = payload["paths"]
    assert isinstance(paths, dict)
    typer.echo(f"agent-memory CLI: {paths.get('agent_memory') or 'not found'}")
    typer.echo(f"Codex CLI: {paths.get('codex') or 'not found'}")

    files = payload["files"]
    assert isinstance(files, dict)
    typer.echo(f"Initialized: {files.get('initialized')}")
    typer.echo(f"Instructions file: {files.get('instructions')}")
    typer.echo(f".mcp.json: {files.get('mcp_json')}")
    typer.echo(f".codex/config.toml: {files.get('codex_config')}")
    typer.echo(f".codex/hooks.json: {files.get('codex_hooks')}")
    typer.echo(f".claude/settings.local.json: {files.get('claude_settings')}")
    typer.echo(f"Codex codex_hooks effective: {payload['codex_hooks_effective']}")
    typer.echo(f"Codex MCP server configured: {payload['codex_mcp_server']}")
    typer.echo(f"Codex project trusted: {payload['codex_project_trusted']}")

    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    if warnings:
        typer.echo("Warnings:")
        for warning in warnings:
            typer.echo(f"- {warning}")


@app.command(
    "smoke-test",
    help=(
        "Run a disposable end-to-end Codex smoke test that verifies uninstall/reinstall, "
        "live UserPromptSubmit hook injection, memory writes on the first prompt, and memory "
        "reads on the second prompt."
    ),
)
def smoke_test(
    project: Path | None = typer.Option(
        None,
        "--project",
        help="Existing project root to test instead of a disposable temp repo. Requires --destructive.",
        resolve_path=True,
    ),
    destructive: bool = typer.Option(
        False,
        "--destructive",
        help="Allow smoke-test to uninstall and reinstall an explicit --project in place.",
    ),
    reinstall_from: Path | None = typer.Option(
        None,
        "--reinstall-from",
        help=(
            "Reinstall the current checkout into `uv tool` before the smoke test runs. "
            "Defaults to the current directory when run from the agent-memory source checkout."
        ),
        resolve_path=True,
    ),
    keep_repo: bool = typer.Option(
        False,
        "--keep-repo",
        help="Keep the disposable temp repo on disk after the smoke test finishes.",
    ),
    timeout_seconds: int = typer.Option(
        120,
        "--timeout-seconds",
        min=30,
        help="Timeout for each live Codex smoke-test phase.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    try:
        # Lazy import: smoke_test uses pty/fcntl/termios which don't exist on
        # Windows. Importing at module scope would break `agent-memory` on
        # Windows entirely.
        from agent_memory.smoke_test import SmokeTestError, run_codex_smoke_test
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            f"smoke-test is POSIX-only and cannot run on this platform: {exc}"
        ) from exc
    resolved_reinstall_from = reinstall_from or _default_smoke_reinstall_from()
    try:
        result = run_codex_smoke_test(
            project_root=project,
            destructive=destructive,
            reinstall_from=resolved_reinstall_from,
            keep_repo=keep_repo,
            timeout_seconds=timeout_seconds,
        )
    except SmokeTestError as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = result.to_dict()
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Repo root: {payload['repo_root']}")
    typer.echo(f"First session file: {payload['first_session_file']}")
    typer.echo(f"Second session file: {payload['second_session_file']}")
    typer.echo(f"Used temp repo: {payload['used_temp_repo']}")
    typer.echo(f"Uninstall verified: {payload['uninstall_verified']}")
    typer.echo(f"Baseline memory count: {payload['baseline_memory_count']}")
    typer.echo(f"Post-save memory count: {payload['post_save_memory_count']}")
    typer.echo(f"Hook events: {payload['hook_event_count']}")
    typer.echo(f"First pre-submit verified: {payload['first_pre_submit_verified']}")
    typer.echo(f"Second pre-submit verified: {payload['second_pre_submit_verified']}")
    typer.echo(f"Save path verified: {payload['save_path_verified']}")
    typer.echo(f"Read path verified: {payload['read_path_verified']}")
    typer.echo(f"Recall top hit: {payload['recall_top_hit']}")
    typer.echo(f"First final answer: {payload['first_final_answer']}")
    typer.echo(f"Second final answer: {payload['second_final_answer']}")
    typer.echo(f"Recall top hit: {payload['recall_top_hit']}")
    typer.echo(f"First final answer: {payload['first_final_answer']}")
    typer.echo(f"Second final answer: {payload['second_final_answer']}")


@app.command(
    "hook-log",
    help=(
        "Show recent local hook events for this project so you can verify whether UserPromptSubmit/Stop are firing "
        "and what they injected or queued."
    ),
)
def hook_log(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    tail: int = typer.Option(25, "--tail", min=1, help="How many recent hook log entries to show."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    project = load_project(cwd, exact=is_project_root(cwd))
    entries = hook_log_entries(project.root)
    selected = entries[-tail:]
    if as_json:
        typer.echo(json.dumps(selected, indent=2))
        return
    if not selected:
        typer.echo("No hook log entries yet.")
        return
    for entry in selected:
        payload = entry.get("payload") or {}
        typer.echo(f"{entry.get('timestamp')}  {entry.get('hook')}  {entry.get('action')}")
        if payload:
            typer.echo(f"  {json.dumps(payload, ensure_ascii=True)}")


@app.command(
    name="upgrade",
    help=(
        "Upgrade agent-memory to the latest GitHub release. Hits the releases API, "
        "downloads the right binary for your platform, verifies the sha256 checksum, "
        "and replaces the running binary in place. No-op if you are already on the latest version."
    ),
)
def upgrade_command(
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    from agent_memory.upgrade import perform_upgrade

    result = perform_upgrade()
    if as_json:
        _emit(result, True)
        return
    status = result.get("status")
    details = result.get("details", "")
    if status == "upgraded":
        typer.echo(f"OK: {details}")
        typer.echo(f"  binary: {result.get('binary_path')}")
        return
    if status == "up-to-date":
        typer.echo(details)
        return
    typer.echo(details, err=True)
    raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
