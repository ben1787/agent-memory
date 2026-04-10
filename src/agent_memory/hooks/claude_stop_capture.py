from __future__ import annotations

import os
import sys
import traceback

from agent_memory.hooks.common import read_hook_input


def main() -> None:
    try:
        read_hook_input()
    except Exception:
        if os.environ.get("AGENT_MEMORY_DEBUG_HOOKS") == "1":
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
