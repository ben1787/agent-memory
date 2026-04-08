"""Append-only log of recall queries.

Each line in `.agent-memory/queries.jsonl` is a JSON object capturing a single
recall query — just the question text, the method that handled it, and a
timestamp. No result IDs, no scores, no top-k. The point is to capture the
*production query distribution* so that future algorithm changes (PPV graph
spreading vs flat cosine vs whatever comes next) can be replayed against the
real questions agents have asked, instead of synthetic benchmarks.

Failure mode: logging is best-effort. A logging failure must never break a
recall — if the file is read-only, the disk is full, or the directory has
been deleted, we swallow the error and let recall return normally. The whole
point is to be invisible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


QUERY_LOG_FILENAME = "queries.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_query(log_path: Path, query: str, method: str) -> None:
    """Append a single query line to the log.

    Best-effort: any IOError is swallowed so recall is never broken by a
    logging failure.
    """
    entry = {
        "ts": _utc_now_iso(),
        "method": method,
        "query": query,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        # Reason: query logging must never break a recall. If the disk is
        # full, the directory is gone, or the file is read-only, we accept
        # the data loss and let the caller proceed.
        pass
