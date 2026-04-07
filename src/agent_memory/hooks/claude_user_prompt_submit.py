from __future__ import annotations

import json
import os
import sys
import traceback

from agent_memory.hooks.common import (
    log_hook_event,
    project_root_from_env_or_cwd,
    read_hook_input,
    summarize_hook_payload,
    truncate_words,
)


INSTRUCTION = """Agent Memory:
- Before answering anything non-trivial, run `agent-memory recall <task-shaped query>` via your shell tool to check for prior project knowledge. Phrase the query like the answer would, not like a question.
- After the work, decide whether you learned 0-3 durable things future-you would want without re-reading this conversation. If yes, save them with `agent-memory save "<memory>" "<memory>"`. For memories with quotes/newlines, pipe via `agent-memory save --stdin`.
- Save only stable, useful project knowledge: decisions and the *why*, file locations, gotchas, user preferences, cross-component relationships, external system pointers.
- Do not save noise, generic programming knowledge, transcript dumps, or anything already in CLAUDE.md/AGENTS.md.
- If you saved something wrong: `agent-memory list --recent 5` to find the id, then `agent-memory edit <id> "<corrected>"` to fix or `agent-memory delete <id> --yes` to remove. `agent-memory undo` reverts the most recent save/edit/delete."""


def main() -> None:
    try:
        payload = read_hook_input()
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            sys.stdout.write("{}")
            sys.stdout.flush()
            return

        project_root = project_root_from_env_or_cwd(payload.get("cwd"))
        log_hook_event(
            project_root,
            hook_name="claude_user_prompt_submit",
            action="start",
            payload=summarize_hook_payload(payload),
        )
        additional_context = INSTRUCTION
        log_hook_event(
            project_root,
            hook_name="claude_user_prompt_submit",
            action="inject_context",
            payload={
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
            log_hook_event(
                project_root,
                hook_name="claude_user_prompt_submit",
                action="error",
                payload={"error": traceback.format_exc(limit=3)},
            )
        except Exception:
            pass
        if os.environ.get("AGENT_MEMORY_DEBUG_HOOKS") == "1":
            traceback.print_exc(file=sys.stderr)
        sys.stdout.write("{}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
