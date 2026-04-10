from __future__ import annotations

import json
import os
import sys
import traceback

import re

from agent_memory.hooks.common import (
    load_memory_config,
    log_hook_event,
    pending_consolidation_instruction,
    project_root_from_env_or_cwd,
    read_hook_input,
    sync_prompt_artifacts,
    summarize_hook_payload,
    truncate_words,
)


INSTRUCTION_FULL = """Agent Memory

Storing memories:
  Save only durable, repo-specific facts that will materially speed up future work or prevent likely mistakes.
  Save only if at least 2:
    - Likely to matter again
    - Hard to rediscover quickly
    - Changes tools/files/search path
    - Missing it wastes time or causes bad assumptions
    - Stable beyond this session
  Prefer: workflow rules, architecture map, search shortcuts, environment quirks, external system behavior, validation/release constraints, recurring customer/project facts.
  Do not save: temp branch/PR/test state, logs/transcripts, generic advice, grep-easy facts, speculation, soon-changing details.
  Format: Scope + Category; 1-sentence fact; 1-sentence why; optional exception; confidence high/med/low.
  Worthiness test: if it won’t save time, prevent a likely error, or narrow the search path, don’t save.
  After work, save memories if the criteria are met with `agent-memory save "<memory>" "<memory>"`. Use `--stdin` for quotes/newlines.
  If you saved something wrong: use `list --recent 5`, then `edit`/`delete`, or `undo`.

Reading memories:
  - If the answer isn’t clear from context or code, consider `agent-memory recall <task-shaped query>` before manual searching through files or the internet."""

DEFAULT_CONTEXT_INTERVAL = 10


def _parse_turn_id(raw: object) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        match = re.search(r"\d+", raw)
        if match:
            return int(match.group())
    return None


def _should_inject_context(turn_id: int | None, interval: int) -> bool:
    if interval <= 1:
        return True
    if turn_id is None:
        return True
    return (turn_id - 1) % interval == 0


def _emit_noop() -> None:
    sys.stdout.write("{}")
    sys.stdout.flush()


def main() -> None:
    try:
        payload = read_hook_input()
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            _emit_noop()
            return

        project_root = project_root_from_env_or_cwd(payload.get("cwd"))
        sync_prompt_artifacts(project_root)
        hook_summary = summarize_hook_payload(payload)
        log_hook_event(
            project_root,
            hook_name="codex_user_prompt_submit",
            action="start",
            payload=hook_summary,
        )
        config = load_memory_config(project_root)
        interval = DEFAULT_CONTEXT_INTERVAL
        if config is not None:
            interval = max(1, int(getattr(config, "prompt_context_turn_interval", DEFAULT_CONTEXT_INTERVAL)))

        turn_id = _parse_turn_id(payload.get("turn_id"))
        if not _should_inject_context(turn_id, interval):
            log_hook_event(
                project_root,
                hook_name="codex_user_prompt_submit",
                action="skip_context",
                payload={
                    **hook_summary,
                    "interval": interval,
                    "reason": "non_interval_turn",
                },
            )
            _emit_noop()
            return

        additional_context = INSTRUCTION_FULL
        consolidation_instruction = pending_consolidation_instruction(project_root)
        if consolidation_instruction:
            additional_context = f"{additional_context}\n\n{consolidation_instruction}"
        log_hook_event(
            project_root,
            hook_name="codex_user_prompt_submit",
            action="inject_context",
            payload={
                **hook_summary,
                "context_preview": truncate_words(additional_context, 80),
            },
        )
        sys.stdout.write(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": additional_context,
                    }
                }
            )
        )
        sys.stdout.flush()
    except Exception:
        try:
            project_root = project_root_from_env_or_cwd(payload.get("cwd") if "payload" in locals() else None)
            error_payload = summarize_hook_payload(payload) if "payload" in locals() else {}
            log_hook_event(
                project_root,
                hook_name="codex_user_prompt_submit",
                action="error",
                payload={
                    **error_payload,
                    "error": traceback.format_exc(limit=3),
                },
            )
        except Exception:
            pass
        if os.environ.get("AGENT_MEMORY_DEBUG_HOOKS") == "1":
            traceback.print_exc(file=sys.stderr)
        _emit_noop()


if __name__ == "__main__":
    main()
