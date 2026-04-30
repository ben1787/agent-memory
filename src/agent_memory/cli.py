from __future__ import annotations

import json
from collections import deque
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

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

from agent_memory.config import (
    ConfigError,
    MemoryConfig,
    init_project,
    is_project_root,
    load_linked_project_roots,
    load_project,
    write_linked_project_roots,
)
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
from agent_memory.legacy_memory import entry_to_metadata, parse_legacy_memory_markdown
from agent_memory.metadata_backfill import backfill_project_metadata, review_memories_with_codex
from agent_memory.models import CONSOLIDATION_SECTION_NAMES, MemoryMetadata
from agent_memory.mcp_server import serve
from agent_memory.project_registry import (
    list_registered_project_roots,
    register_project_root,
    registry_path,
    unregister_project_root,
)
from agent_memory.retrieval_feedback import (
    OVERALL_FEEDBACK_LABELS,
    MEMORY_FEEDBACK_LABELS,
    parse_feedback_assignments,
    record_retrieval_feedback,
)
from agent_memory.repo_ingest import import_repo_corpus
# smoke_test is POSIX-only (uses pty/fcntl/termios). Import lazily from inside
# the command so `agent-memory --help` and every other command still work on
# Windows, where the module cannot be imported at all.
from agent_memory.store import GraphStore
from agent_memory.hooks.common import (
    consolidation_status,
    hook_log_entries,
    mark_consolidation_completed,
)
from agent_memory.write_lock import ProjectWriteLock


from agent_memory import __display_version__, __version__


CLAUDE_PLUGIN_MARKETPLACE = "agent-memory-plugins"
CLAUDE_PLUGIN_NAME = "agent-memory"
CLAUDE_PLUGIN_ENTRY = f"{CLAUDE_PLUGIN_NAME}@{CLAUDE_PLUGIN_MARKETPLACE}"
CLAUDE_PLUGIN_DATA_DIRNAME = f"{CLAUDE_PLUGIN_NAME}-{CLAUDE_PLUGIN_MARKETPLACE}"
CLAUDE_INSTALLER_PATH_HOOK_MARKER = "AGENT_MEMORY_INSTALLER_PATH_HOOK_v1"
INSTALLER_RC_COMMENT = "# added by agent-memory installer"


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
        "Codex/Claude local prompt hooks plus Codex project trust for automatic prompt-time recall injection and memory instructions, then `save`, `recall`, `list`, `edit`, `delete`, `undo`."
    ),
)


def _refresh_project_integration_if_needed() -> None:
    try:
        project = load_project(Path.cwd())
    except ConfigError:
        return
    try:
        from agent_memory.integration import refresh_project_integration

        register_project_root(project.root)
        refresh_project_integration(project, current_version=__display_version__)
    except Exception:
        # Never fail a real command because integration refresh failed.
        pass


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
    auto_upgrade_enabled = os.environ.get("AGENT_MEMORY_DISABLE_AUTO_UPGRADE") != "1"
    if len(sys.argv) > 1 and sys.argv[1] == "_hook":
        auto_upgrade_enabled = False

    project = None
    if auto_upgrade_enabled:
        try:
            project = load_project(Path.cwd())
            if not project.config.auto_upgrade:
                auto_upgrade_enabled = False
        except ConfigError:
            project = None

    # Lazy import to keep cold-start fast for hot-path commands.
    try:
        from agent_memory.upgrade import check_for_upgrade_in_background, perform_upgrade

        notice = check_for_upgrade_in_background()
        if notice and auto_upgrade_enabled:
            result = perform_upgrade()
            if result.get("status") == "upgraded":
                binary_path = result.get("binary_path")
                if isinstance(binary_path, str) and binary_path:
                    os.environ["AGENT_MEMORY_DISABLE_AUTO_UPGRADE"] = "1"
                    os.execv(binary_path, [binary_path, *sys.argv[1:]])
            elif notice:
                # Print to stderr so it never pollutes JSON output.
                typer.echo(notice, err=True)
        elif notice:
            # Print to stderr so it never pollutes JSON output.
            typer.echo(notice, err=True)
    except Exception:
        # Never fail a real command because the staleness check threw.
        pass
    _refresh_project_integration_if_needed()


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


def _read_nonempty_stdin(*, empty_message: str) -> str:
    body = sys.stdin.read().strip()
    if not body:
        raise typer.BadParameter(empty_message)
    return body


def _clean_metadata_option(value: str | None, *, current: str | None = None) -> str | None:
    if value is None:
        return current
    cleaned = " ".join(value.split())
    return cleaned or None


def _build_memory_metadata(
    *,
    title: str | None,
    kind: str | None,
    subsystem: str | None,
    workstream: str | None,
    environment: str | None,
    fallback: MemoryMetadata | None = None,
    require_complete: bool,
) -> MemoryMetadata:
    metadata = MemoryMetadata(
        title=_clean_metadata_option(title, current=fallback.title if fallback else None),
        kind=_clean_metadata_option(kind, current=fallback.kind if fallback else None),
        subsystem=_clean_metadata_option(subsystem, current=fallback.subsystem if fallback else None),
        workstream=_clean_metadata_option(workstream, current=fallback.workstream if fallback else None),
        environment=_clean_metadata_option(
            environment,
            current=fallback.environment if fallback else None,
        ),
    )
    if not require_complete:
        return metadata
    missing = [
        flag
        for flag, value in (
            ("--title", metadata.title),
            ("--kind", metadata.kind),
            ("--subsystem", metadata.subsystem),
            ("--workstream", metadata.workstream),
            ("--environment", metadata.environment),
        )
        if not value
    ]
    if missing:
        raise typer.BadParameter(
            "save requires explicit metadata: " + ", ".join(missing)
        )
    return metadata


def _metadata_flags_provided(
    *,
    title: str | None,
    kind: str | None,
    subsystem: str | None,
    workstream: str | None,
    environment: str | None,
) -> bool:
    return any(
        value is not None
        for value in (title, kind, subsystem, workstream, environment)
    )


def _coerce_optional_feedback_text(
    payload: dict[str, Any],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise typer.BadParameter(
            f"Feedback stdin payload field `{key}` must be a string when provided."
        )
    cleaned = value.strip()
    return cleaned or None


def _parse_feedback_stdin_memory(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, list):
        raise typer.BadParameter(
            "Feedback stdin payload field `memory` must be a string, a list of strings, "
            "or a list of {ref|alias|memory_id, label} objects."
        )

    parsed: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                parsed.append(cleaned)
            continue
        if not isinstance(item, dict):
            raise typer.BadParameter(
                "Feedback stdin payload field `memory` must contain only strings or objects."
            )
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise typer.BadParameter(
                "Feedback stdin memory objects must include a non-empty string `label`."
            )
        ref = item.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            for alt in ("alias", "memory_id"):
                candidate = item.get(alt)
                if isinstance(candidate, str) and candidate.strip():
                    ref = candidate
                    break
        if not isinstance(ref, str) or not ref.strip():
            raise typer.BadParameter(
                "Feedback stdin memory objects must include one of `ref`, `alias`, or `memory_id`."
            )
        parsed.append(f"{ref.strip()}={label.strip()}")
    return parsed


def _parse_feedback_stdin_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            "Feedback stdin must be a JSON object with keys like "
            "`overall`, `why`, `better`, `missing`, `note`, and `memory`."
        ) from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter(
            "Feedback stdin must be a JSON object, not a list or plain string."
        )

    allowed = {"overall", "why", "better", "missing", "note", "memory", "memory_feedback"}
    unknown = sorted(key for key in payload if key not in allowed)
    if unknown:
        raise typer.BadParameter(
            "Unknown feedback stdin payload field(s): "
            + ", ".join(unknown)
            + ". Allowed fields: overall, why, better, missing, note, memory."
        )

    if "memory" in payload and "memory_feedback" in payload:
        raise typer.BadParameter(
            "Feedback stdin payload cannot include both `memory` and `memory_feedback`."
        )

    memory_items = _parse_feedback_stdin_memory(
        payload.get("memory", payload.get("memory_feedback"))
    )
    return {
        "memory": memory_items,
        "overall": _coerce_optional_feedback_text(payload, "overall"),
        "why": _coerce_optional_feedback_text(payload, "why"),
        "better": _coerce_optional_feedback_text(payload, "better"),
        "missing": _coerce_optional_feedback_text(payload, "missing"),
        "note": _coerce_optional_feedback_text(payload, "note"),
    }


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


def _discover_existing_project_root(path: Path | None) -> Path | None:
    candidate = (path or Path.cwd()).resolve()
    if is_project_root(candidate):
        return candidate
    try:
        return load_project(candidate).root
    except ConfigError:
        return None


def _persist_linked_roots(project, linked_roots: list[Path]) -> None:
    normalized = [str(root.resolve()) for root in linked_roots]
    if load_linked_project_roots(project.root) == normalized:
        return
    write_linked_project_roots(project.root, normalized)


def _linked_roots(project) -> list[Path]:
    roots: list[Path] = []
    seen = {project.root.resolve()}
    for raw_root in load_linked_project_roots(project.root):
        if not raw_root:
            continue
        try:
            resolved = Path(raw_root).expanduser().resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _validate_link_root(project_root: Path, link_root: Path) -> Path:
    resolved_project_root = project_root.resolve()
    resolved_link_root = link_root.resolve()
    if resolved_link_root == resolved_project_root:
        raise typer.BadParameter("Cannot link the project root to itself.", param_hint="--link-root")
    if not resolved_link_root.exists() or not resolved_link_root.is_dir():
        raise typer.BadParameter(
            f"Linked repo path does not exist or is not a directory: {resolved_link_root}",
            param_hint="--link-root",
        )
    if not resolved_link_root.is_relative_to(resolved_project_root):
        raise typer.BadParameter(
            "Linked repos must live under the shared project root so CLI lookups stay unambiguous.",
            param_hint="--link-root",
        )
    if is_project_root(resolved_link_root):
        raise typer.BadParameter(
            f"{resolved_link_root} already has its own .agent-memory store. Uninstall that store before linking it to {resolved_project_root}.",
            param_hint="--link-root",
        )
    return resolved_link_root


def _install_linked_root(
    project,
    linked_root: Path,
    *,
    install_mcp: bool,
    install_local_excludes: bool,
    install_codex: bool,
    install_codex_trust: bool,
    install_claude: bool,
) -> dict[str, object]:
    results: list[dict[str, object]] = []

    if install_local_excludes:
        results.append(_result_payload(ensure_local_git_excludes(linked_root)))
    if install_mcp:
        results.append(
            _result_payload(
                install_mcp_server(
                    linked_root,
                    memory_project_root=project.root,
                )
            )
        )
    if install_codex:
        results.append(_result_payload(install_codex_feature_flag(linked_root)))
        results.append(
            _result_payload(
                install_codex_hooks(
                    linked_root,
                    memory_project_root=project.root,
                )
            )
        )
        if install_mcp:
            results.append(
                _result_payload(
                    install_codex_mcp_server(
                        linked_root,
                        memory_project_root=project.root,
                    )
                )
            )
        if install_codex_trust:
            results.append(_result_payload(install_codex_project_trust(linked_root)))
    if install_claude:
        results.append(
            _result_payload(
                install_claude_hooks(
                    linked_root,
                    register_mcp_server=install_mcp,
                    memory_project_root=project.root,
                )
            )
        )

    for instructions_result in install_memory_instructions(
        linked_root,
        memory_project_root=project.root,
    ):
        results.append(_result_payload(instructions_result))

    return {
        "linked_root": str(linked_root),
        "results": results,
    }


def _link_project_roots(
    project,
    link_roots: list[Path],
    *,
    install_mcp: bool,
    install_local_excludes: bool,
    install_codex: bool,
    install_codex_trust: bool,
    install_claude: bool,
) -> list[dict[str, object]]:
    existing = _linked_roots(project)
    linked_results: list[dict[str, object]] = []
    for link_root in link_roots:
        resolved_link_root = _validate_link_root(project.root, link_root)
        if resolved_link_root not in existing:
            existing.append(resolved_link_root)
        linked_results.append(
            _install_linked_root(
                project,
                resolved_link_root,
                install_mcp=install_mcp,
                install_local_excludes=install_local_excludes,
                install_codex=install_codex,
                install_codex_trust=install_codex_trust,
                install_claude=install_claude,
            )
        )
    _persist_linked_roots(project, existing)
    return linked_results


def _refresh_integrations_payload(
    *,
    cwd: Path,
    all_known: bool,
) -> dict[str, object]:
    roots = list_registered_project_roots() if all_known else []
    try:
        current_project = load_project(cwd)
    except ConfigError:
        current_project = None
    if current_project is not None and current_project.root not in roots:
        roots.append(current_project.root)
    ordered_roots = sorted({root.resolve() for root in roots}, key=lambda root: str(root))

    payloads: list[dict[str, object]] = []
    missing_roots: list[str] = []
    for root in ordered_roots:
        if not root.exists():
            missing_roots.append(str(root))
            unregister_project_root(root)
            continue
        try:
            project = load_project(root, exact=True)
        except ConfigError:
            missing_roots.append(str(root))
            unregister_project_root(root)
            continue
        from agent_memory.integration import refresh_project_integration

        register_project_root(project.root)
        payloads.append(
            refresh_project_integration(
                project,
                current_version=__display_version__,
                force=True,
            )
        )
    return {
        "registry_path": str(registry_path()),
        "refreshed_projects": payloads,
        "missing_roots": missing_roots,
    }


def _print_refresh_integrations_payload(payload: dict[str, object]) -> None:
    typer.echo(f"Registry: {payload['registry_path']}")
    refreshed = payload["refreshed_projects"]
    assert isinstance(refreshed, list)
    if not refreshed:
        typer.echo("No registered Agent Memory projects were refreshed.")
    for project_payload in refreshed:
        if not isinstance(project_payload, dict):
            continue
        typer.echo(f"Project root: {project_payload.get('project_root')}")
        roots = project_payload.get("refreshed_roots")
        if isinstance(roots, list) and roots:
            typer.echo("  Refreshed roots:")
            for root in roots:
                typer.echo(f"  - {root}")
        skipped = project_payload.get("skipped_missing_roots")
        if isinstance(skipped, list) and skipped:
            typer.echo("  Missing linked roots:")
            for root in skipped:
                typer.echo(f"  - {root}")

    missing_roots = payload.get("missing_roots")
    if isinstance(missing_roots, list) and missing_roots:
        typer.echo("Removed missing registered roots:")
        for root in missing_roots:
            typer.echo(f"- {root}")


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


def _path_payload(path: Path, status: str, details: str) -> dict[str, object]:
    return {
        "path": str(path),
        "status": status,
        "details": details,
    }


def _remove_path(path: Path, *, kind: str) -> dict[str, object]:
    if not path.exists() and not path.is_symlink():
        return _path_payload(path, "unchanged", f"{kind} does not exist.")
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            return _path_payload(path, "removed", f"Deleted {kind}.")
        path.unlink()
        return _path_payload(path, "removed", f"Deleted {kind}.")
    except OSError as exc:
        return _path_payload(path, "error", f"Failed to delete {kind}: {exc}")


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    stop_at = stop_at.resolve()
    while True:
        try:
            current = current.resolve()
        except OSError:
            break
        if current == stop_at or not str(current).startswith(str(stop_at)):
            break
        try:
            current.rmdir()
        except OSError:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent


def _default_installer_path_line() -> str:
    return f'export PATH="{Path.home() / ".local/bin"}:$PATH"'


def _cleanup_shell_rc_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return _path_payload(path, "unchanged", "Shell rc file does not exist.")
    try:
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    except OSError as exc:
        return _path_payload(path, "error", f"Failed to read shell rc file: {exc}")

    path_line = _default_installer_path_line()
    updated: list[str] = []
    removed = 0
    index = 0
    while index < len(lines):
        stripped = lines[index].rstrip("\r\n")
        next_stripped = lines[index + 1].rstrip("\r\n") if index + 1 < len(lines) else None
        if stripped == INSTALLER_RC_COMMENT and next_stripped == path_line:
            removed += 2
            index += 2
            continue
        if stripped == path_line:
            removed += 1
            index += 1
            continue
        updated.append(lines[index])
        index += 1

    if removed == 0:
        return _path_payload(path, "unchanged", "No agent-memory PATH lines found.")

    try:
        path.write_text("".join(updated), encoding='utf-8')
    except OSError as exc:
        return _path_payload(path, "error", f"Failed to rewrite shell rc file: {exc}")
    return _path_payload(path, "updated", f"Removed {removed} installer-managed PATH line(s).")


def _cleanup_claude_settings_path_hook(settings_path: Path) -> dict[str, object]:
    if not settings_path.exists():
        return _path_payload(settings_path, "unchanged", "Claude settings file does not exist.")
    try:
        payload = json.loads(settings_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return _path_payload(settings_path, "error", f"Failed to parse Claude settings: {exc}")

    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return _path_payload(settings_path, "unchanged", "No Claude hooks block found.")
    session_start = hooks.get("SessionStart")
    if not isinstance(session_start, list):
        return _path_payload(settings_path, "unchanged", "No SessionStart hooks found.")

    changed = False
    cleaned_groups: list[object] = []
    removed_entries = 0
    for group in session_start:
        if not isinstance(group, dict):
            cleaned_groups.append(group)
            continue
        entries = group.get("hooks")
        if not isinstance(entries, list):
            cleaned_groups.append(group)
            continue
        kept_entries: list[object] = []
        for entry in entries:
            if isinstance(entry, dict) and CLAUDE_INSTALLER_PATH_HOOK_MARKER in str(entry.get("command", "")):
                removed_entries += 1
                changed = True
                continue
            kept_entries.append(entry)
        if kept_entries:
            next_group = dict(group)
            next_group["hooks"] = kept_entries
            cleaned_groups.append(next_group)
        else:
            changed = True

    if not changed:
        return _path_payload(settings_path, "unchanged", "No agent-memory Claude SessionStart hook found.")

    if cleaned_groups:
        hooks["SessionStart"] = cleaned_groups
    else:
        hooks.pop("SessionStart", None)
    if not hooks:
        payload.pop("hooks", None)

    try:
        settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    except OSError as exc:
        return _path_payload(settings_path, "error", f"Failed to rewrite Claude settings: {exc}")
    return _path_payload(settings_path, "updated", f"Removed {removed_entries} agent-memory SessionStart hook(s).")


def _cleanup_json_mapping_entry(path: Path, key: str, *, nested_field: str | None = None) -> dict[str, object]:
    if not path.exists():
        return _path_payload(path, "unchanged", "JSON file does not exist.")
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return _path_payload(path, "error", f"Failed to parse JSON file: {exc}")

    target: Any
    if nested_field is None:
        target = payload
    else:
        target = payload.get(nested_field)
        if not isinstance(target, dict):
            return _path_payload(path, "unchanged", f"No `{nested_field}` map found.")

    if not isinstance(target, dict) or key not in target:
        return _path_payload(path, "unchanged", f"No `{key}` entry found.")

    target.pop(key, None)
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding='utf-8')
    except OSError as exc:
        return _path_payload(path, "error", f"Failed to rewrite JSON file: {exc}")
    return _path_payload(path, "updated", f"Removed `{key}` entry.")


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
    link_roots: list[Path],
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
        with ProjectWriteLock(project.root):
            store = GraphStore(project.db_path, config.embedding_dimensions)
            store.close()
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc

    register_project_root(project.root)

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

    if link_roots:
        linked_results = _link_project_roots(
            project,
            link_roots,
            install_mcp=install_mcp,
            install_local_excludes=install_local_excludes,
            install_codex=install_codex,
            install_codex_trust=install_codex_trust,
            install_claude=install_claude,
        )
        for linked_payload in linked_results:
            typer.echo(f"Linked repo: {linked_payload['linked_root']}")
            results = linked_payload["results"]
            assert isinstance(results, list)
            for result in results:
                if isinstance(result, dict):
                    typer.echo(f"  {result.get('details')}")

    typer.echo(
        "Access pattern: agents call `agent-memory recall <query>` and "
        "`agent-memory save \"<memory>\"` via their shell tool. The CLI is the canonical interface; "
        "MCP is opt-in via --with-mcp."
    )


def _project_uninstall_payload(
    project_root: Path,
    *,
    remove_store: bool,
    remove_codex_trust: bool,
) -> dict[str, object]:
    project = None
    if is_project_root(project_root):
        try:
            project = load_project(project_root, exact=True)
        except ConfigError:
            project = None
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

    linked_root_payloads: list[dict[str, object]] = []
    if remove_store and project is not None:
        for linked_root in _linked_roots(project):
            linked_results: list[tuple[str, object]] = [
                ("mcp", uninstall_mcp_server(linked_root)),
                ("codex_hooks", uninstall_codex_hooks(linked_root)),
                ("codex_feature_flag", uninstall_codex_feature_flag(linked_root)),
                ("codex_mcp", uninstall_codex_mcp_server(linked_root)),
                ("claude_hooks", uninstall_claude_hooks(linked_root)),
                (
                    "local_excludes",
                    remove_local_git_excludes(
                        linked_root,
                        entries=[
                            ".claude/settings.local.json",
                            ".codex/config.toml",
                            ".codex/hooks.json",
                            ".mcp.json",
                        ],
                    ),
                ),
            ]
            for instructions_result in uninstall_memory_instructions(linked_root):
                linked_results.append((f"instructions:{instructions_result.path.name}", instructions_result))
            if remove_codex_trust:
                linked_results.append(("codex_trust", uninstall_codex_project_trust(linked_root)))
            _remove_if_empty(linked_root / ".codex")
            _remove_if_empty(linked_root / ".claude")
            linked_root_payloads.append(
                {
                    "project_root": str(linked_root),
                    "results": {name: _result_payload(result) for name, result in linked_results},
                }
            )

    payload = {
        "project_root": str(project_root),
        "remove_store": remove_store,
        "remove_codex_trust": remove_codex_trust,
        "results": {name: _result_payload(result) for name, result in results},
        "store": store_result,
        "linked_roots": linked_root_payloads,
    }

    return payload


def _run_uninstall(
    path: Path | None,
    *,
    remove_store: bool,
    remove_codex_trust: bool,
    as_json: bool,
) -> None:
    project_root = _resolve_project_path(path)
    payload = _project_uninstall_payload(
        project_root,
        remove_store=remove_store,
        remove_codex_trust=remove_codex_trust,
    )
    if remove_store:
        unregister_project_root(project_root)

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Project root: {payload['project_root']}")
    for name, result in payload["results"].items():
        if isinstance(result, dict):
            typer.echo(f"{name}: {result.get('details')}")
    store_result = payload["store"]
    if isinstance(store_result, dict):
        typer.echo(f"store: {store_result.get('details')}")
    linked_payloads = payload.get("linked_roots")
    if isinstance(linked_payloads, list):
        for linked_payload in linked_payloads:
            if not isinstance(linked_payload, dict):
                continue
            typer.echo(f"linked_root: {linked_payload.get('project_root')}")
            results = linked_payload.get("results")
            if isinstance(results, dict):
                for name, result in results.items():
                    if isinstance(result, dict):
                        typer.echo(f"  {name}: {result.get('details')}")


def _system_uninstall_payload() -> dict[str, object]:
    from agent_memory.upgrade import _cache_dir, _resolve_running_binary_path

    home = Path.home()
    candidate_binaries: list[Path] = []
    executable_names = {"agent-memory", "agent-memory.exe"}

    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name in executable_names and argv0.is_absolute() and (argv0.exists() or argv0.is_symlink()):
        candidate_binaries.append(argv0)
    elif sys.argv and sys.argv[0]:
        raw = Path(sys.argv[0])
        if raw.name in executable_names and (raw.exists() or raw.is_symlink()):
            candidate_binaries.append(raw.resolve())

    on_path = shutil.which("agent-memory")
    if on_path:
        on_path_candidate = Path(on_path)
        if on_path_candidate.name in executable_names:
            candidate_binaries.append(on_path_candidate)

    for default_binary in (
        home / ".local" / "bin" / "agent-memory",
        Path("/usr/local/bin/agent-memory"),
        Path("/opt/homebrew/bin/agent-memory"),
    ):
        candidate_binaries.append(default_binary)

    seen_binary_paths: set[Path] = set()
    binary_results: list[dict[str, object]] = []
    for candidate in candidate_binaries:
        try:
            key = candidate.resolve(strict=False)
        except OSError:
            key = candidate
        if key in seen_binary_paths:
            continue
        seen_binary_paths.add(key)
        if candidate.exists() or candidate.is_symlink():
            result = _remove_path(candidate, kind="agent-memory executable")
            binary_results.append(result)
            if result.get("status") == "removed":
                _prune_empty_parents(candidate.parent, stop_at=home)

    running_binary = _resolve_running_binary_path()
    default_libexec_root = home / ".local" / "share" / "agent-memory"
    libexec_candidates: list[Path] = [default_libexec_root]
    if running_binary is not None:
        plugin_data_root = home / ".claude" / "plugins" / "data" / CLAUDE_PLUGIN_DATA_DIRNAME
        if plugin_data_root not in running_binary.parents and len(running_binary.parents) >= 2:
            libexec_candidates.append(running_binary.parent.parent)

    libexec_results: list[dict[str, object]] = []
    seen_libexec: set[Path] = set()
    for candidate in libexec_candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen_libexec:
            continue
        seen_libexec.add(normalized)
        result = _remove_path(candidate, kind="agent-memory libexec bundle")
        libexec_results.append(result)
        if result.get("status") == "removed":
            _prune_empty_parents(candidate.parent, stop_at=home)

    shell_rc_results = [
        _cleanup_shell_rc_file(path)
        for path in (
            Path(os.environ.get("ZDOTDIR") or str(home)) / ".zshrc",
            Path(os.environ.get("ZDOTDIR") or str(home)) / ".zshenv",
            home / ".bashrc",
            home / ".bash_profile",
            home / ".profile",
        )
    ]

    claude_settings_path = home / ".claude" / "settings.json"
    claude_registry_root = home / ".claude" / "plugins"
    plugin_cache_root = claude_registry_root / "cache" / CLAUDE_PLUGIN_MARKETPLACE
    plugin_data_root = claude_registry_root / "data" / CLAUDE_PLUGIN_DATA_DIRNAME
    marketplace_clone_root = claude_registry_root / "marketplaces" / CLAUDE_PLUGIN_MARKETPLACE

    claude_registry_results = {
        "settings_path_hook": _cleanup_claude_settings_path_hook(claude_settings_path),
        "settings_enabled_plugins": _cleanup_json_mapping_entry(
            claude_settings_path,
            CLAUDE_PLUGIN_ENTRY,
            nested_field="enabledPlugins",
        ),
        "settings_extra_marketplaces": _cleanup_json_mapping_entry(
            claude_settings_path,
            CLAUDE_PLUGIN_MARKETPLACE,
            nested_field="extraKnownMarketplaces",
        ),
        "known_marketplaces": _cleanup_json_mapping_entry(
            claude_registry_root / "known_marketplaces.json",
            CLAUDE_PLUGIN_MARKETPLACE,
        ),
        "installed_plugins": _cleanup_json_mapping_entry(
            claude_registry_root / "installed_plugins.json",
            CLAUDE_PLUGIN_ENTRY,
            nested_field="plugins",
        ),
        "plugin_cache": _remove_path(plugin_cache_root, kind="Claude plugin cache"),
        "plugin_data": _remove_path(plugin_data_root, kind="Claude plugin data"),
        "marketplace_clone": _remove_path(marketplace_clone_root, kind="Claude marketplace clone"),
    }

    for result in claude_registry_results.values():
        if isinstance(result, dict) and result.get("status") == "removed":
            result_path = Path(str(result["path"]))
            _prune_empty_parents(result_path.parent, stop_at=home)

    update_cache_result = _remove_path(_cache_dir(), kind="agent-memory update cache")
    if update_cache_result.get("status") == "removed":
        _prune_empty_parents(Path(str(update_cache_result["path"])).parent, stop_at=home)

    return {
        "executables": binary_results,
        "libexec_bundles": libexec_results,
        "shell_rc_files": shell_rc_results,
        "claude": claude_registry_results,
        "update_cache": update_cache_result,
    }


@app.command(
    help=(
        "Initialize a project-local memory store and wire repo-local integration in one step. "
        "By default this creates .agent-memory/, hides local integration files from git, installs Codex and Claude "
        "UserPromptSubmit hooks for automatic prompt-time recall injection and memory instructions, adds this repo to Codex trusted projects, and "
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
    link_roots: list[Path] = typer.Option(
        [],
        "--link-root",
        help="Descendant repo root to wire to this shared store as a linked integration root. Repeat for multiple child repos.",
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
        help="Install Codex repo-local prompt hooks and enable the codex_hooks feature flag for automatic prompt-time recall injection and memory instructions.",
    ),
    install_codex_trust: bool = typer.Option(
        True,
        "--install-codex-trust/--no-install-codex-trust",
        help="Add this repo to Codex trusted projects in ~/.codex/config.toml so repo-local Codex config is loaded.",
    ),
    install_claude: bool = typer.Option(
        True,
        "--install-claude-hooks/--no-install-claude-hooks",
        help="Install Claude local prompt hooks for automatic prompt-time recall injection and memory instructions.",
    ),
) -> None:
    _run_init(
        path=path,
        link_roots=link_roots,
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
    link_roots: list[Path] = typer.Option(
        [],
        "--link-root",
        help="Descendant repo root to wire to this shared store as a linked integration root. Repeat for multiple child repos.",
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
        link_roots=link_roots,
        embedding_backend=embedding_backend,
        force=force,
        install_mcp=install_mcp,
        install_local_excludes=True,
        install_codex=True,
        install_codex_trust=True,
        install_claude=True,
    )


@app.command(
    name="link-root",
    help=(
        "Wire one or more descendant repos to the current project's shared Agent Memory store. "
        "This installs repo-local hooks/instructions in the child repos but points them at the parent store."
    ),
)
def link_root_command(
    link_roots: list[Path] = typer.Argument(
        ...,
        help="Descendant repo root(s) to wire to the current shared store.",
        resolve_path=True,
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project root (or any path inside it) that owns the shared Agent Memory store.",
        resolve_path=True,
    ),
    install_mcp: bool = typer.Option(
        False,
        "--with-mcp/--without-mcp",
        help="Also install repo-local MCP server entries in linked repos.",
    ),
    install_local_excludes: bool = typer.Option(
        True,
        "--install-local-excludes/--no-install-local-excludes",
        help="Hide local linked-repo integration files via .git/info/exclude when possible.",
    ),
    install_codex: bool = typer.Option(
        True,
        "--install-codex-hooks/--no-install-codex-hooks",
        help="Install Codex repo-local prompt hooks in linked repos.",
    ),
    install_codex_trust: bool = typer.Option(
        True,
        "--install-codex-trust/--no-install-codex-trust",
        help="Trust linked repos in Codex so their repo-local config loads.",
    ),
    install_claude: bool = typer.Option(
        True,
        "--install-claude-hooks/--no-install-claude-hooks",
        help="Install Claude local prompt hooks in linked repos.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    try:
        project = load_project(cwd, exact=is_project_root(cwd))
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    register_project_root(project.root)
    linked_results = _link_project_roots(
        project,
        link_roots,
        install_mcp=install_mcp,
        install_local_excludes=install_local_excludes,
        install_codex=install_codex,
        install_codex_trust=install_codex_trust,
        install_claude=install_claude,
    )
    payload = {
        "project_root": str(project.root),
        "linked_roots": linked_results,
        "registered_linked_roots": [str(root) for root in _linked_roots(project)],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"Project root: {project.root}")
    for linked_payload in linked_results:
        typer.echo(f"Linked repo: {linked_payload['linked_root']}")
        results = linked_payload["results"]
        assert isinstance(results, list)
        for result in results:
            if isinstance(result, dict):
                typer.echo(f"  {result.get('details')}")


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
    name="uninstall-all",
    help=(
        "Clean-room uninstall. If run inside an initialized project (or with an explicit path), "
        "this removes that project's Agent Memory integration and `.agent-memory/` store, then "
        "also removes machine-level agent-memory artifacts: standalone binaries, extracted bundles, "
        "installer PATH/session hooks, Claude plugin cache/data, Claude plugin registry entries, "
        "and the update-check cache."
    ),
)
def uninstall_all(
    path: Path | None = typer.Argument(
        None,
        help="Optional project directory or any path inside an initialized project.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    project_root = _discover_existing_project_root(path)
    project_payload: dict[str, object] | None = None
    if project_root is not None:
        project_payload = _project_uninstall_payload(
            project_root,
            remove_store=True,
            remove_codex_trust=True,
        )
        unregister_project_root(project_root)

    system_payload = _system_uninstall_payload()
    payload = {
        "project": project_payload,
        "system": system_payload,
        "notes": [
            "If Claude Code is currently running, reload or restart it after uninstall so it drops any in-memory plugin state.",
        ],
    }

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    if project_payload is not None:
        typer.echo(f"Project root: {project_payload['project_root']}")
        project_results = project_payload.get("results")
        if isinstance(project_results, dict):
            for name, result in project_results.items():
                if isinstance(result, dict):
                    typer.echo(f"{name}: {result.get('details')}")
        store_result = project_payload.get("store")
        if isinstance(store_result, dict):
            typer.echo(f"store: {store_result.get('details')}")
    else:
        typer.echo("Project root: none (no initialized Agent Memory project found from the provided path).")

    typer.echo("System cleanup:")
    for result in system_payload["executables"]:
        if isinstance(result, dict):
            typer.echo(f"  executable {result.get('path')}: {result.get('details')}")
    for result in system_payload["libexec_bundles"]:
        if isinstance(result, dict):
            typer.echo(f"  libexec {result.get('path')}: {result.get('details')}")
    for result in system_payload["shell_rc_files"]:
        if isinstance(result, dict) and result.get("status") != "unchanged":
            typer.echo(f"  shell rc {result.get('path')}: {result.get('details')}")
    claude_payload = system_payload["claude"]
    if isinstance(claude_payload, dict):
        for name, result in claude_payload.items():
            if isinstance(result, dict) and result.get("status") != "unchanged":
                typer.echo(f"  claude {name}: {result.get('details')}")
    cache_result = system_payload["update_cache"]
    if isinstance(cache_result, dict):
        typer.echo(f"  update cache: {cache_result.get('details')}")
    typer.echo(payload["notes"][0])


@app.command(
    help=(
        "Save one memory into the current project store with explicit metadata fields. "
        "Pass the body as a positional argument for short shell-safe text, or prefer "
        "piping the body on stdin for agents and shell-hostile content. If stdin is "
        "already piped and no positional text is given, save reads from stdin automatically."
    )
)
def save(
    text: str | None = typer.Argument(
        None,
        help="Memory body to persist. Omit when using --stdin or --batch.",
    ),
    title: str | None = typer.Option(None, "--title", help="Short durable title for the memory."),
    kind: str | None = typer.Option(None, "--kind", help="Memory kind, e.g. operational or preference."),
    subsystem: str | None = typer.Option(None, "--subsystem", help="Primary subsystem this memory belongs to."),
    workstream: str | None = typer.Option(None, "--workstream", help="Narrow workstream within the subsystem."),
    environment: str | None = typer.Option(None, "--environment", help="Environment scope, e.g. local, dev, qa, prod."),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    from_stdin: bool = typer.Option(
        False,
        "--stdin",
        help=(
            "Explicitly read a single memory body from stdin instead of taking it as an "
            "argument. Optional when stdin is already piped and no positional text is given."
        ),
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        help=(
            "Read a JSON list of memory objects from stdin and save them in one engine session. "
            "Each entry: {title, kind, subsystem, workstream, environment, text}. Always emits JSON."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    if batch:
        if text is not None or from_stdin or any(
            value is not None for value in (title, kind, subsystem, workstream, environment)
        ):
            raise typer.BadParameter(
                "--batch reads its inputs from stdin; do not pass text, --stdin, or metadata flags."
            )
        _run_save_batch(cwd=cwd)
        return

    missing = [
        flag
        for flag, value in (
            ("--title", title),
            ("--kind", kind),
            ("--subsystem", subsystem),
            ("--workstream", workstream),
            ("--environment", environment),
        )
        if value is None
    ]
    if missing:
        raise typer.BadParameter("save requires explicit metadata: " + ", ".join(missing))
    if from_stdin and text is not None:
        raise typer.BadParameter("--stdin cannot be combined with a positional memory body.")

    if from_stdin:
        resolved_text = _read_nonempty_stdin(
            empty_message="--stdin received empty input; nothing to save."
        )
    elif text is None:
        if sys.stdin.isatty():
            raise typer.BadParameter("Provide memory text, or pipe a body on stdin.")
        resolved_text = _read_nonempty_stdin(
            empty_message="stdin was piped but empty; provide memory text or pipe a body to save."
        )
    else:
        resolved_text = text

    metadata = _build_memory_metadata(
        title=title,
        kind=kind,
        subsystem=subsystem,
        workstream=workstream,
        environment=environment,
        require_complete=True,
    )

    memory = _open_memory(cwd)
    try:
        result = memory.save(resolved_text, metadata=metadata)
    finally:
        memory.close()
    payload = result.to_dict()
    if as_json:
        _emit(payload, True)
        return
    item = payload["saved"][0]
    typer.echo(f"Saved {item['memory_id']}")
    typer.echo(f"Created at: {item['created_at']}")
    item_metadata = item.get("metadata")
    if isinstance(item_metadata, dict):
        if item_metadata.get("title"):
            typer.echo(f"Title: {item_metadata['title']}")
        for key, label in (
            ("kind", "Kind"),
            ("subsystem", "Subsystem"),
            ("workstream", "Workstream"),
            ("environment", "Environment"),
        ):
            value = item_metadata.get(key)
            if isinstance(value, str) and value:
                typer.echo(f"{label}: {value}")
    typer.echo(f"Connected neighbors: {len(item['connected_neighbors'])}")
    for neighbor in item["connected_neighbors"]:
        typer.echo(
            f"  {neighbor['memory_id']}  similarity={neighbor['similarity']}"
        )
    typer.echo(f"Total memories: {payload['total_memories']}")


def _read_batch_json_list(label: str) -> list:
    """Read stdin and parse a JSON array. Used by --batch on save/edit/recall."""
    raw = _read_nonempty_stdin(
        empty_message=f"--batch expects a JSON array on stdin; got empty input ({label})."
    )
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--batch ({label}) received invalid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise typer.BadParameter(f"--batch ({label}) must be a JSON array.")
    return items


def _coerce_optional_str(entry: dict, field: str, *, label: str, index: int) -> str | None:
    value = entry.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise typer.BadParameter(
            f"--batch ({label}) entry #{index} `{field}` must be a string when provided."
        )
    return value


def _partial_metadata_from_entry(entry: dict, *, label: str, index: int) -> MemoryMetadata | None:
    fields = {
        field: _coerce_optional_str(entry, field, label=label, index=index)
        for field in ("title", "kind", "subsystem", "workstream", "environment")
    }
    if not any(value is not None for value in fields.values()):
        return None
    return MemoryMetadata(
        title=_clean_metadata_option(fields["title"]),
        kind=_clean_metadata_option(fields["kind"]),
        subsystem=_clean_metadata_option(fields["subsystem"]),
        workstream=_clean_metadata_option(fields["workstream"]),
        environment=_clean_metadata_option(fields["environment"]),
    )


def _run_edit_batch(*, cwd: Path) -> None:
    """Apply a JSON list of edit objects from stdin in one engine session."""
    items = _read_batch_json_list("edit")
    engine_items: list[dict] = []
    for index, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise typer.BadParameter(f"--batch (edit) entry #{index} is not a JSON object.")
        memory_id = entry.get("id")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise typer.BadParameter(f"--batch (edit) entry #{index} is missing a string `id`.")
        text_value = _coerce_optional_str(entry, "text", label="edit", index=index)
        partial_metadata = _partial_metadata_from_entry(entry, label="edit", index=index)
        engine_items.append({"id": memory_id, "text": text_value, "metadata": partial_metadata})

    memory = _open_memory(cwd)
    try:
        outcomes = memory.edit_many(engine_items)
    finally:
        memory.close()

    rows: list[dict[str, object]] = []
    any_failed = False
    for outcome in outcomes:
        row: dict[str, object] = {"memory_id": outcome.memory_id, "status": outcome.status}
        if outcome.error:
            row["error"] = outcome.error
            any_failed = True
        if outcome.record is not None:
            row["record"] = _format_memory_record(outcome.record)
        rows.append(row)

    typer.echo(json.dumps(rows, indent=2))
    if any_failed:
        raise typer.Exit(code=1)


def _run_save_batch(*, cwd: Path) -> None:
    """Save a JSON list of memory objects from stdin in one engine session."""
    items = _read_batch_json_list("save")
    engine_items: list[dict] = []
    for index, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise typer.BadParameter(f"--batch (save) entry #{index} is not a JSON object.")
        text_value = entry.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise typer.BadParameter(
                f"--batch (save) entry #{index} is missing a non-empty string `text`."
            )
        metadata = _build_memory_metadata(
            title=_coerce_optional_str(entry, "title", label="save", index=index),
            kind=_coerce_optional_str(entry, "kind", label="save", index=index),
            subsystem=_coerce_optional_str(entry, "subsystem", label="save", index=index),
            workstream=_coerce_optional_str(entry, "workstream", label="save", index=index),
            environment=_coerce_optional_str(entry, "environment", label="save", index=index),
            require_complete=True,
        )
        engine_items.append({"text": text_value, "metadata": metadata})

    memory = _open_memory(cwd)
    try:
        result = memory.save_many(engine_items)
    finally:
        memory.close()
    typer.echo(json.dumps(result.to_dict()["saved"], indent=2))


def _run_recall_batch(*, cwd: Path, limit: int) -> None:
    """Recall for each query in a JSON list on stdin; emit a list of trees."""
    items = _read_batch_json_list("recall")
    queries: list[str] = []
    for index, entry in enumerate(items):
        if isinstance(entry, str):
            text = entry.strip()
        elif isinstance(entry, dict):
            raw = entry.get("query")
            if not isinstance(raw, str):
                raise typer.BadParameter(
                    f"--batch (recall) entry #{index} object must include a string `query`."
                )
            text = raw.strip()
        else:
            raise typer.BadParameter(
                f"--batch (recall) entry #{index} must be a string or {{query}} object."
            )
        if not text:
            raise typer.BadParameter(f"--batch (recall) entry #{index} query is empty.")
        queries.append(text)

    memory = _open_memory(cwd, read_only=True)
    try:
        results = memory.recall_many(queries, limit=limit)
    finally:
        memory.close()
    typer.echo(json.dumps([result.to_dict() for result in results], indent=2))


def _format_memory_record(record, *, show_full_text: bool = True) -> dict:
    display_text = record.display_text()
    payload = {
        "memory_id": record.id,
        "created_at": record.created_at,
        "text": record.text if show_full_text else (record.text[:140] + "…" if len(record.text) > 140 else record.text),
        "display_text": display_text if show_full_text else (display_text[:140] + "…" if len(display_text) > 140 else display_text),
        "metadata": record.metadata.to_dict(),
        "title": record.metadata.title,
        "kind": record.metadata.kind,
        "subsystem": record.metadata.subsystem,
        "workstream": record.metadata.workstream,
        "environment": record.metadata.environment,
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
        for line in record.display_text().splitlines():
            typer.echo(f"    {line}")


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
    typer.echo(record.display_text())


@app.command(
    name="edit",
    help=(
        "Edit a memory's body text and/or metadata in place and re-embed it. "
        "Pass new text as a positional argument for a one-shot edit, pipe or use "
        "--stdin for multi-line content, or omit text entirely to update metadata "
        "only or open $EDITOR with the current body prefilled. Use --batch to read "
        "a JSON list of edit objects from stdin and apply them in one engine session."
    ),
)
def edit_command(
    memory_id: str | None = typer.Argument(
        None,
        help="Memory id, e.g. mem_abc123def456. Omit when using --from.",
    ),
    new_text: str | None = typer.Argument(
        None,
        help="New text for the memory. Omit to use --stdin or $EDITOR.",
    ),
    title: str | None = typer.Option(None, "--title", help="Updated title. Omit to keep the current title."),
    kind: str | None = typer.Option(None, "--kind", help="Updated kind. Omit to keep the current kind."),
    subsystem: str | None = typer.Option(
        None,
        "--subsystem",
        help="Updated subsystem. Omit to keep the current subsystem.",
    ),
    workstream: str | None = typer.Option(
        None,
        "--workstream",
        help="Updated workstream. Omit to keep the current workstream.",
    ),
    environment: str | None = typer.Option(
        None,
        "--environment",
        help="Updated environment. Omit to keep the current environment.",
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
        help=(
            "Explicitly read the new memory body from stdin instead of taking it as an "
            "argument. Optional when stdin is already piped and no positional text is given."
        ),
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        help=(
            "Read a JSON list of edit objects from stdin and apply them in one engine "
            "session. Each entry: {id, title?, kind?, subsystem?, workstream?, "
            "environment?, text?}. Omitted fields keep current values. Always emits JSON."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    if batch:
        if memory_id is not None or new_text is not None or from_stdin:
            raise typer.BadParameter(
                "--batch reads its inputs from stdin; do not pass a memory id, text, or --stdin."
            )
        _run_edit_batch(cwd=cwd)
        return

    if memory_id is None:
        raise typer.BadParameter("memory_id is required unless --batch is set.")

    if from_stdin and new_text is not None:
        raise typer.BadParameter("--stdin cannot be combined with a positional new-text argument.")

    metadata_requested = _metadata_flags_provided(
        title=title,
        kind=kind,
        subsystem=subsystem,
        workstream=workstream,
        environment=environment,
    )

    if from_stdin:
        resolved_text: str | None = _read_nonempty_stdin(
            empty_message="--stdin received empty input; nothing to write."
        )
    elif new_text is None and not sys.stdin.isatty() and not metadata_requested:
        resolved_text = _read_nonempty_stdin(
            empty_message="stdin was piped but empty; provide new text or pipe a body to write."
        )
    else:
        resolved_text = new_text

    memory = _open_memory(cwd)
    try:
        existing = memory.get(memory_id)
        if existing is None:
            typer.echo(f"No memory with id {memory_id!r}.", err=True)
            raise typer.Exit(code=1)

        metadata = _build_memory_metadata(
            title=title,
            kind=kind,
            subsystem=subsystem,
            workstream=workstream,
            environment=environment,
            fallback=existing.metadata,
            require_complete=False,
        )

        if resolved_text is None:
            if metadata_requested:
                resolved_text = existing.text
            else:
                edited = typer.edit(existing.text, extension=".md")
                if edited is None:
                    typer.echo("Edit aborted (no editor or no save). Memory unchanged.")
                    raise typer.Exit(code=1)
                resolved_text = edited.strip()
                if not resolved_text:
                    typer.echo("Editor produced empty content. Memory unchanged.", err=True)
                    raise typer.Exit(code=1)

        if (
            resolved_text.strip() == existing.text.strip()
            and metadata.to_dict() == existing.metadata.to_dict()
        ):
            typer.echo("No changes detected. Memory unchanged.")
            raise typer.Exit(code=0)

        updated = memory.edit(memory_id, resolved_text, metadata=metadata)
    finally:
        memory.close()

    if as_json:
        _emit(_format_memory_record(updated), True)
        return
    typer.echo(f"Updated {updated.id}")
    typer.echo(updated.display_text())


@app.command(
    name="backfill-metadata",
    help=(
        "Generate explicit metadata for existing memories in a project store, write the "
        "metadata sidecar, and re-embed the project so metadata participates in recall."
    ),
)
def backfill_metadata_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Regenerate metadata even for memories that already have metadata.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Only backfill the first N memories. Useful for sampling or staged migrations.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview the migration counts and sample reviewed metadata without writing anything.",
    ),
    reviewer: str = typer.Option(
        "codex",
        "--reviewer",
        help="Metadata reviewer to use. Only `codex` is supported for this command.",
    ),
    model: str = typer.Option(
        "gpt-5.4-mini",
        "--model",
        help="Codex model to use when reviewer=codex.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        min=1,
        help="How many memories to send per review request. Default 1 reviews each memory independently.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    if reviewer != "codex":
        raise typer.BadParameter("Only `codex` is supported for this command.", param_hint="--reviewer")

    project = _load_project(cwd)
    memory = _open_memory(project.root, read_only=True)
    try:
        records = memory.list_all()
    finally:
        memory.close()

    selected = records if limit is None else records[:limit]
    candidate_records = [
        record
        for record in selected
        if overwrite or record.metadata.is_empty()
    ]
    candidates: list[dict[str, object]] = []
    sample_size = min(10, len(candidate_records))
    if sample_size:
        if reviewer == "codex":
            for record in candidate_records[:sample_size]:
                reviewed = review_memories_with_codex([record], model=model)
                _, metadata = reviewed[0]
                candidates.append(
                    {
                        "memory_id": record.id,
                        "metadata": metadata.to_dict(),
                        "preview": record.text[:140] + ("…" if len(record.text) > 140 else ""),
                    }
                )

    if dry_run:
        payload = {
            "project_root": str(project.root),
            "total_memories": len(records),
            "candidate_memories": len(candidate_records),
            "updated_memories": len(candidate_records),
            "overwrite": overwrite,
            "reviewer": reviewer,
            "model": model if reviewer == "codex" else None,
            "batch_size": batch_size,
            "sample": candidates[:10],
        }
        _emit(payload, as_json)
        return

    result = backfill_project_metadata(
        project.root,
        overwrite=overwrite,
        limit=limit,
        reviewer=reviewer,
        batch_size=batch_size,
        model=model,
    )
    payload = {
        "project_root": str(project.root),
        **result.to_dict(),
    }
    _emit(payload, as_json)


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
        "Record structured feedback about a prompt-time memory injection event. "
        "Use the event id and aliases shown in the injected Agent Memory context."
    )
)
def feedback(
    event_id: str = typer.Argument(
        ...,
        help="Retrieval event id from the injected prompt context, e.g. evt_ab12cd34ef.",
    ),
    from_stdin: bool = typer.Option(
        False,
        "--stdin",
        help=(
            "Read a JSON payload for overall/why/better/missing/note/memory from stdin. "
            "Optional when stdin is already piped and no inline feedback flags are given."
        ),
    ),
    memory: list[str] = typer.Option(
        [],
        "--memory",
        help=(
            "Per-memory feedback as alias-or-memory-id=label, e.g. A=helpful or mem_xxx=stale. "
            f"Labels: {', '.join(sorted(MEMORY_FEEDBACK_LABELS))}."
        ),
    ),
    overall: str | None = typer.Option(
        None,
        "--overall",
        help=f"Overall label for the whole injection event: {', '.join(sorted(OVERALL_FEEDBACK_LABELS))}.",
    ),
    why: str | None = typer.Option(
        None,
        "--why",
        help="Short event-level explanation of why the recalled set was or was not useful.",
    ),
    better: str | None = typer.Option(
        None,
        "--better",
        help="Short event-level note on what would have made the recalled set better.",
    ),
    missing: str | None = typer.Option(
        None,
        "--missing",
        help="Optional short note about what should have surfaced but did not.",
    ),
    note: str | None = typer.Option(
        None,
        "--note",
        help="Optional short freeform note about the retrieval quality.",
    ),
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
    inline_feedback_present = bool(memory or overall or why or better or missing or note)
    if from_stdin:
        if inline_feedback_present:
            raise typer.BadParameter(
                "--stdin cannot be combined with --memory, --overall, --why, --better, --missing, or --note."
            )
        stdin_payload = _parse_feedback_stdin_payload(
            _read_nonempty_stdin(
                empty_message="--stdin received empty input; provide a JSON feedback payload."
            )
        )
        memory = list(stdin_payload["memory"])
        overall = stdin_payload["overall"]
        why = stdin_payload["why"]
        better = stdin_payload["better"]
        missing = stdin_payload["missing"]
        note = stdin_payload["note"]
    elif not inline_feedback_present and not sys.stdin.isatty():
        stdin_payload = _parse_feedback_stdin_payload(
            _read_nonempty_stdin(
                empty_message="stdin was piped but empty; provide a JSON feedback payload or inline flags."
            )
        )
        memory = list(stdin_payload["memory"])
        overall = stdin_payload["overall"]
        why = stdin_payload["why"]
        better = stdin_payload["better"]
        missing = stdin_payload["missing"]
        note = stdin_payload["note"]
    try:
        assignments = parse_feedback_assignments(memory)
        payload = record_retrieval_feedback(
            project.root,
            event_id=event_id,
            overall=overall,
            memory_feedback=assignments,
            why=why,
            better=better,
            missing=missing,
            note=note,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if as_json:
        _emit(payload, True)
        return

    typer.echo(f"Recorded retrieval feedback for {payload['event_id']}.")
    if payload.get("overall"):
        typer.echo(f"Overall: {payload['overall']}")
    if payload.get("why"):
        typer.echo(f"Why: {payload['why']}")
    if payload.get("better"):
        typer.echo(f"Better: {payload['better']}")
    items = payload.get("memory_feedback")
    if isinstance(items, list) and items:
        typer.echo("Per-memory feedback:")
        for item in items:
            if not isinstance(item, dict):
                continue
            typer.echo(
                f"  {item.get('alias')} ({item.get('memory_id')}): {item.get('label')}"
            )
    if payload.get("missing"):
        typer.echo(f"Missing: {payload['missing']}")
    if payload.get("note"):
        typer.echo(f"Note: {payload['note']}")


@app.command(
    help=(
        "Recall the highest-scoring memories for a query. "
        "Results are ordered by descending parent-similarity score from the query root."
    )
)
def recall(
    query_parts: list[str] = typer.Argument(
        None,
        help="Query text used to seed recall. Multiple words may be passed without shell quoting.",
    ),
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    limit: int = typer.Option(15, "--limit", min=1),
    batch: bool = typer.Option(
        False,
        "--batch",
        help=(
            "Read a JSON list of query strings (or {query} objects) from stdin and emit a "
            "list of recall trees, one per query, in input order. Always emits JSON."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    if batch:
        if query_parts:
            raise typer.BadParameter("--batch reads queries from stdin; do not pass positional query text.")
        _run_recall_batch(cwd=cwd, limit=limit)
        return

    query = " ".join(part for part in (query_parts or []) if part).strip()
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
    as_json: bool = typer.Option(False, "--json", help="Print compact JSON output."),
    group_id: str | None = typer.Option(
        None,
        "--group",
        help="Return one candidate group by id with all member ids and a few previews.",
    ),
    section: str | None = typer.Option(
        None,
        "--section",
        help="Return all compact groups for one consolidation section.",
    ),
) -> None:
    if group_id is not None and section is not None:
        raise typer.BadParameter("Use either --group or --section, not both.")
    memory = _open_memory(cwd, read_only=True)
    try:
        report = memory.consolidate()
    finally:
        memory.close()
    summary_payload = report.to_summary_dict()
    if group_id is not None:
        payload = report.group_detail_dict(group_id)
        if payload is None:
            raise typer.BadParameter(f"No consolidation group found with id {group_id!r}.")
        if as_json:
            _emit(payload, True)
            return
        typer.echo(f"{payload['group_id']}  action={payload['recommended_action']}")
        member_ids = payload.get("member_ids")
        if isinstance(member_ids, list):
            typer.echo(f"members: {' '.join(str(member_id) for member_id in member_ids)}")
        for member in payload.get("members", []):
            if isinstance(member, dict):
                typer.echo(
                    f"  {member.get('memory_id')}: "
                    f"{member.get('preview') or member.get('title') or ''}"
                )
        return
    if section is not None:
        payload = report.section_detail_dict(section)
        if payload is None:
            expected = ", ".join(CONSOLIDATION_SECTION_NAMES)
            raise typer.BadParameter(
                f"No consolidation section found with name {section!r}. "
                f"Expected one of: {expected}."
            )
        if as_json:
            _emit(payload, True)
            return
        if "groups" in payload:
            typer.echo(f"{payload['section']}  groups={payload['group_count']}")
            for group in payload["groups"]:
                if isinstance(group, dict):
                    typer.echo(
                        f"  {group['group_id']}  "
                        f"action={group['recommended_action']}"
                    )
        else:
            typer.echo(f"{payload['section']}  memories={payload['memory_count']}")
            for memory in payload["memories"]:
                if isinstance(memory, dict):
                    typer.echo(
                        f"  {memory['memory_id']}  {memory.get('preview', '')}"
                    )
        return
    if as_json:
        _emit(summary_payload, True)
        return
    payload = summary_payload
    typer.echo(
        f"Clusters: {len(payload['clusters'])}  threshold={payload['threshold']}  "
        f"clustered_memories={payload['clustered_memory_count']}/{payload['total_memories']}"
    )
    counts = payload.get("cleanup_candidate_counts")
    if not isinstance(counts, dict):
        counts = payload.get("candidate_counts")
    if isinstance(counts, dict):
        typer.echo(
            "Cleanup candidates: "
            f"clusters={counts.get('clusters', 0)}  "
            f"metadata_cleanup={counts.get('metadata_cleanup', 0)}  "
            f"negative_feedback={counts.get('negative_feedback_memories', 0)}  "
            f"unretrieved={counts.get('unretrieved_memories', 0)}"
        )
    for cluster in payload["clusters"]:
        typer.echo(
            f"  {cluster['group_id']}  size={cluster['member_count']}  "
            f"avg={cluster['average_similarity']}  max={cluster['max_similarity']}"
        )
        typer.echo(f"    members: {' '.join(cluster['member_ids'])}")


@app.command(
    "consolidation-status",
    help="Show the current daily memory consolidation status for this project."
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
    "consolidation-complete",
    help="Record today's date as the last completed memory consolidation for this project."
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
    name="migrate-memory-md",
    help=(
        "Import curated bullet notes from a legacy `MEMORY.md` file into the current project store. "
        "This skips rules/preamble sections and focuses on durable notes plus point-in-time notes."
    ),
)
def migrate_memory_md_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the initialized project store.",
        resolve_path=True,
    ),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Legacy MEMORY.md file to import. Defaults to <project-root>/MEMORY.md.",
        resolve_path=True,
    ),
    subsystem: str | None = typer.Option(
        None,
        "--subsystem",
        help="Metadata subsystem for imported memories. Defaults to the project root folder name.",
    ),
    workstream: str | None = typer.Option(
        None,
        "--workstream",
        help="Override the workstream metadata for every imported memory.",
    ),
    environment: str | None = typer.Option(
        None,
        "--environment",
        help="Override the environment metadata for every imported memory.",
    ),
    kind: str | None = typer.Option(
        None,
        "--kind",
        help="Override the kind metadata for every imported memory.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview what would be imported without writing any memories.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    try:
        project = load_project(cwd, exact=is_project_root(cwd))
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    register_project_root(project.root)
    source_path = (path or (project.root / "MEMORY.md")).resolve()
    if not source_path.exists():
        raise typer.BadParameter(f"Legacy memory file does not exist: {source_path}")

    entries = parse_legacy_memory_markdown(source_path)
    resolved_subsystem = subsystem or project.root.name
    preview = [
        {
            "line_number": entry.line_number,
            "section_path": list(entry.section_path),
            "text": entry.text,
            "metadata": entry_to_metadata(
                entry,
                default_subsystem=resolved_subsystem,
                workstream_override=workstream,
                environment_override=environment,
                kind_override=kind,
            ).to_dict(),
        }
        for entry in entries[:10]
    ]

    if dry_run:
        payload = {
            "project_root": str(project.root),
            "source_path": str(source_path),
            "discovered_entries": len(entries),
            "preview": preview,
        }
        _emit(payload, as_json)
        return

    memory = _open_memory(project.root)
    saved: list[dict[str, object]] = []
    skipped_duplicates = 0
    skipped_errors: list[dict[str, object]] = []
    try:
        existing_texts = {
            " ".join(record.text.split()).casefold()
            for record in memory.list_all()
        }
        for entry in entries:
            normalized_text = " ".join(entry.text.split()).casefold()
            if normalized_text in existing_texts:
                skipped_duplicates += 1
                continue
            metadata = entry_to_metadata(
                entry,
                default_subsystem=resolved_subsystem,
                workstream_override=workstream,
                environment_override=environment,
                kind_override=kind,
            )
            try:
                result = memory.save(entry.text, metadata=metadata)
            except ValueError as exc:
                skipped_errors.append(
                    {
                        "line_number": entry.line_number,
                        "text": entry.text,
                        "error": str(exc),
                    }
                )
                continue
            saved_item = result.saved[0].to_dict()
            saved.append(saved_item)
            existing_texts.add(normalized_text)
    finally:
        memory.close()

    payload = {
        "project_root": str(project.root),
        "source_path": str(source_path),
        "discovered_entries": len(entries),
        "saved_count": len(saved),
        "skipped_duplicates": skipped_duplicates,
        "skipped_errors": skipped_errors,
        "saved": saved,
    }
    _emit(payload, as_json)


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
    name="refresh-integrations",
    help=(
        "Rewrite Agent Memory hooks/instructions for the current project family. "
        "Use --all-known to refresh every registered project family on this machine."
    ),
)
def refresh_integrations_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    all_known: bool = typer.Option(
        False,
        "--all-known",
        help="Refresh every registered Agent Memory project family, not just the current cwd project.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    payload = _refresh_integrations_payload(cwd=cwd, all_known=all_known)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    _print_refresh_integrations_payload(payload)


@app.command(
    name="update",
    help=(
        "Refresh Agent Memory hooks/instructions everywhere this machine already uses the current binary. "
        "By default this updates every registered project family, such as shared stores and their linked repos."
    ),
)
def update_command(
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Project directory or any path inside the project.",
        resolve_path=True,
    ),
    current_project_only: bool = typer.Option(
        False,
        "--current-project-only",
        help="Refresh only the cwd project family instead of every registered Agent Memory project family.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    payload = _refresh_integrations_payload(cwd=cwd, all_known=not current_project_only)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    _print_refresh_integrations_payload(payload)


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
    cwd: Path = typer.Option(
        Path("."),
        "--cwd",
        help="Optional project directory. If it resolves to an Agent Memory project, upgrade also refreshes that project family.",
        resolve_path=True,
    ),
    as_json: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    from agent_memory.upgrade import perform_upgrade

    result = perform_upgrade()
    refresh_payload: dict[str, object] | None = None
    if result.get("status") == "upgraded":
        binary_path = result.get("binary_path")
        if isinstance(binary_path, str) and binary_path:
            try:
                refresh_run = subprocess.run(
                    [
                        binary_path,
                        "refresh-integrations",
                        "--all-known",
                        "--cwd",
                        str(cwd),
                        "--json",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if refresh_run.returncode == 0 and refresh_run.stdout.strip():
                    refresh_payload = json.loads(refresh_run.stdout)
                else:
                    refresh_payload = {
                        "error": refresh_run.stderr.strip() or refresh_run.stdout.strip() or "refresh-integrations failed after upgrade",
                    }
            except (OSError, json.JSONDecodeError) as exc:
                refresh_payload = {"error": str(exc)}
        else:
            refresh_payload = _refresh_integrations_payload(cwd=cwd, all_known=True)
    elif result.get("status") == "up-to-date":
        refresh_payload = _refresh_integrations_payload(cwd=cwd, all_known=True)

    if refresh_payload is not None:
        result["refresh"] = refresh_payload

    if as_json:
        _emit(result, True)
        return
    status = result.get("status")
    details = result.get("details", "")
    if status == "upgraded":
        typer.echo(f"OK: {details}")
        typer.echo(f"  binary: {result.get('binary_path')}")
        if refresh_payload is not None:
            typer.echo("  integrations: refreshed registered project families")
        return
    if status == "up-to-date":
        typer.echo(details)
        if refresh_payload is not None:
            typer.echo("Refreshed registered project families.")
        return
    typer.echo(details, err=True)
    raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
