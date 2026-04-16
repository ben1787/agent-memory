from __future__ import annotations

import json
import os
from pathlib import Path


REGISTRY_FILENAME = "known-projects.json"


def _data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "agent-memory"


def registry_path() -> Path:
    return _data_dir() / REGISTRY_FILENAME


def _normalize_roots(roots: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            continue
        unique[str(resolved)] = resolved
    return [unique[key] for key in sorted(unique)]


def list_registered_project_roots() -> list[Path]:
    path = registry_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("project_roots") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    roots = [Path(str(item)) for item in items if isinstance(item, str) and item.strip()]
    return _normalize_roots(roots)


def _write_roots(roots: list[Path]) -> list[Path]:
    normalized = _normalize_roots(roots)
    path = registry_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"project_roots": [str(root) for root in normalized]}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return normalized


def register_project_root(root: Path) -> list[Path]:
    current = list_registered_project_roots()
    current.append(root)
    return _write_roots(current)


def unregister_project_root(root: Path) -> list[Path]:
    try:
        resolved = root.expanduser().resolve()
    except OSError:
        resolved = root.expanduser()
    current = [candidate for candidate in list_registered_project_roots() if candidate != resolved]
    path = registry_path()
    if not current:
        try:
            path.unlink()
        except OSError:
            pass
        return []
    return _write_roots(current)
