from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback

from agent_memory.hooks.common import (
    latest_transcript_turn,
    log_hook_event,
    open_memory_with_retry,
    project_root_from_env_or_cwd,
    read_hook_input,
    summarize_hook_payload,
    truncate_words,
)


def main() -> None:
    try:
        payload = read_hook_input()
        project_root = project_root_from_env_or_cwd(payload.get("cwd"))
        hook_summary = summarize_hook_payload(payload)
        log_hook_event(
            project_root,
            hook_name="codex_stop_capture",
            action="start",
            payload=hook_summary,
        )

        user_text = None
        assistant_text = None

        transcript_path = payload.get("transcript_path")
        if isinstance(transcript_path, str) and transcript_path.strip():
            resolved = Path(transcript_path).expanduser()
            if resolved.exists():
                user_text, assistant_text, _ = latest_transcript_turn(resolved)

        if assistant_text is None:
            fallback_assistant = payload.get("last_assistant_message")
            if isinstance(fallback_assistant, str) and fallback_assistant.strip():
                assistant_text = fallback_assistant.strip()

        if user_text is None:
            fallback_prompt = payload.get("prompt")
            if isinstance(fallback_prompt, str) and fallback_prompt.strip():
                user_text = fallback_prompt.strip()

        if not user_text and not assistant_text:
            log_hook_event(
                project_root,
                hook_name="codex_stop_capture",
                action="skip",
                payload={
                    **hook_summary,
                    "reason": "No turn text available to capture.",
                },
            )
            return

        memory = open_memory_with_retry(project_root)
        try:
            result = memory.capture_turn(
                user_text=user_text,
                assistant_text=assistant_text,
            )
        finally:
            memory.close()

        log_hook_event(
            project_root,
            hook_name="codex_stop_capture",
            action="captured",
            payload={
                **hook_summary,
                "saved_count": len(result.saved),
                "user_preview": truncate_words(user_text or "", 20),
                "assistant_preview": truncate_words(assistant_text or "", 20),
            },
        )
    except Exception:
        try:
            project_root = project_root_from_env_or_cwd(payload.get("cwd") if "payload" in locals() else None)
            error_payload = summarize_hook_payload(payload) if "payload" in locals() else {}
            log_hook_event(
                project_root,
                hook_name="codex_stop_capture",
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


if __name__ == "__main__":
    main()
