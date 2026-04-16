from __future__ import annotations

import threading
import time
from pathlib import Path

from agent_memory.write_lock import ProjectWriteLock


def test_project_write_lock_serializes_concurrent_writers(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    (project_root / ".agent-memory").mkdir(parents=True)

    first = ProjectWriteLock(project_root, poll_interval_s=0.01)
    second = ProjectWriteLock(project_root, poll_interval_s=0.01)
    acquired: list[str] = []

    def worker() -> None:
        with second:
            acquired.append("second")

    with first:
        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.05)
        assert acquired == []
    thread.join(timeout=1.0)

    assert acquired == ["second"]
