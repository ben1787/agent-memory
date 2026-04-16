from __future__ import annotations

import json
import os
from pathlib import Path
import random
import socket
import time


LOCK_FILENAME = "write.lock"
DEFAULT_STALE_AFTER_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 0.1


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class ProjectWriteLock:
    def __init__(
        self,
        project_root: Path,
        *,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_SECONDS,
        stale_after_s: float = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self.project_root = project_root.resolve()
        self.path = self.project_root / ".agent-memory" / LOCK_FILENAME
        self.poll_interval_s = poll_interval_s
        self.stale_after_s = stale_after_s
        self._held = False

    def acquire(self, *, timeout_s: float | None = None) -> "ProjectWriteLock":
        started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                fd = os.open(
                    str(self.path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                if self._break_stale_lock():
                    continue
                if timeout_s is not None and (time.monotonic() - started) >= timeout_s:
                    raise TimeoutError(f"Timed out waiting for write lock at {self.path}.")
                time.sleep(self.poll_interval_s + random.uniform(0.0, self.poll_interval_s))
                continue

            payload = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at": time.time(),
            }
            try:
                os.write(fd, json.dumps(payload).encode("utf-8"))
            finally:
                os.close(fd)
            self._held = True
            return self

    def release(self) -> None:
        if not self._held:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._held = False

    def _break_stale_lock(self) -> bool:
        if not self.path.exists():
            return False
        if not self._lock_is_stale():
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    def _lock_is_stale(self) -> bool:
        try:
            stat = self.path.stat()
        except OSError:
            return False

        age_s = max(0.0, time.time() - stat.st_mtime)
        if age_s >= self.stale_after_s:
            return True

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return age_s >= 5.0

        hostname = payload.get("hostname")
        pid = payload.get("pid")
        if hostname == socket.gethostname() and isinstance(pid, int):
            return not _pid_is_running(pid)
        return False

    def __enter__(self) -> "ProjectWriteLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
