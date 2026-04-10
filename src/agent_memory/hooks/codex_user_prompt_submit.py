from __future__ import annotations

import json
import os
import sys
import traceback

import re

from agent_memory.hooks.common import (
    auto_recall_matches,
    load_memory_config,
    log_hook_event,
    pending_consolidation_instruction,
    project_root_from_env_or_cwd,
    read_hook_input,
    render_auto_recall_block,
    render_guidance_context,
    sync_prompt_artifacts,
    summarize_hook_payload,
    truncate_words,
)

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
        inject_guidance = _should_inject_context(turn_id, interval)
        if not inject_guidance:
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
        sections: list[str] = []
        recalled_memories, recall_metadata = auto_recall_matches(project_root, prompt)
        if inject_guidance:
            sections.append(
                render_guidance_context(
                    recalled_memories,
                    consolidation_instruction=pending_consolidation_instruction(project_root),
                )
            )
        elif recalled_memories:
            sections.append(render_auto_recall_block(recalled_memories))

        if not sections:
            _emit_noop()
            return

        additional_context = "\n\n".join(sections)
        log_hook_event(
            project_root,
            hook_name="codex_user_prompt_submit",
            action="inject_context",
            payload={
                **hook_summary,
                "inject_guidance": inject_guidance,
                "inject_auto_recall": recalled_memories is not None,
                "auto_recall": recall_metadata,
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
