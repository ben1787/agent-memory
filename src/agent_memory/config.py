from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


APP_DIR_NAME = ".agent-memory"
CONFIG_FILENAME = "config.json"
DB_FILENAME = "memory.kuzu"
INSTRUCTIONS_FILENAME = "instructions.md"


class ConfigError(RuntimeError):
    """Raised when project configuration cannot be found or parsed."""


def _config_path(root: Path) -> Path:
    return root / APP_DIR_NAME / CONFIG_FILENAME


@dataclass(slots=True)
class MemoryConfig:
    version: int = 4
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    max_memory_words: int = 1000
    duplicate_threshold: float = 0.97
    overlap_threshold: float = 0.90
    lexical_duplicate_threshold: float = 0.95

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MemoryConfig":
        data = dict(payload)

        data.pop("similarity_threshold", None)
        data.pop("edge_similarity_threshold", None)
        data.pop("query_similarity_threshold", None)
        data.pop("write_similarity_threshold", None)
        data.pop("read_similarity_threshold", None)
        data.pop("max_seed_nodes", None)
        data.pop("max_neighbors", None)
        data.pop("max_frontier_nodes_per_hop", None)
        data.pop("max_hops", None)

        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProjectContext:
    root: Path
    app_dir: Path
    db_path: Path
    config_path: Path
    instructions_path: Path
    config: MemoryConfig


def default_instructions() -> str:
    return """# Agent Memory Instructions

This project uses Agent Memory for project-scoped long-term memory.

Rules:
- Treat this memory store as specific to the current project root.
- If there is any ambiguity about which project root is active, ask the user instead of guessing.
- Before substantive work, decide whether to call `recall_memories` with a concrete query tied to the current task.
- During work, decide whether there are 0-3 durable memories worth saving with `save_memory`.
- If the MCP tools are unavailable in the current client, fall back to `agent-memory recall` and `agent-memory save` in the current project root.
- Prefer stable facts, decisions, file locations, constraints, preferences, and discovered relationships.
- Avoid saving noise, repeated wording, or giant raw dumps when a shorter memory will do.
- Save memories into this project store only; never write into some parent or sibling project by accident.

Suggested workflow:
1. Consider whether recalling stored memories would help with the current task.
2. If yes, call `agent-memory recall "<task-specific query>"`.
3. Do the work.
4. If the work produced durable project knowledge, save 0-3 concise memories with `agent-memory save`.

For MCP clients, use the project root explicitly so reads and writes stay scoped correctly.
"""


def is_project_root(path: Path | None) -> bool:
    if path is None:
        return False
    return _config_path(path.resolve()).exists()


def find_project_roots(start: Path | None = None) -> list[Path]:
    current = (start or Path.cwd()).resolve()
    return [candidate for candidate in [current, *current.parents] if _config_path(candidate).exists()]


def find_project_root(start: Path | None = None, exact: bool = False) -> Path:
    current = (start or Path.cwd()).resolve()
    if exact:
        if _config_path(current).exists():
            return current
        raise ConfigError(
            "Expected an agent-memory project exactly at "
            f"{current}. Pass an initialized project root, not a child path."
        )

    candidates = find_project_roots(current)
    if not candidates:
        raise ConfigError(
            "No agent-memory project found. Run `agent-memory init` in your project first."
        )
    if len(candidates) > 1:
        raise ConfigError(
            "Multiple agent-memory projects are visible from "
            f"{current}: {', '.join(str(candidate) for candidate in candidates)}. "
            "Specify the exact project root so reads and writes cannot go to the wrong store."
        )
    return candidates[0]


def load_project(start: Path | None = None, exact: bool = False) -> ProjectContext:
    root = find_project_root(start, exact=exact)
    app_dir = root / APP_DIR_NAME
    config_path = app_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise ConfigError(f"Missing config file at {config_path}")
    config = MemoryConfig.from_dict(json.loads(config_path.read_text(encoding='utf-8')))
    return ProjectContext(
        root=root,
        app_dir=app_dir,
        db_path=app_dir / DB_FILENAME,
        config_path=config_path,
        instructions_path=app_dir / INSTRUCTIONS_FILENAME,
        config=config,
    )


# Common cruft directory names that should never be walked when checking for
# nested agent-memory stores. Skipping these keeps the descendant scan O(small)
# even on huge monorepos.
_DESCENDANT_SCAN_SKIP = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".jj",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "target",
        "build",
        "dist",
        ".next",
        ".nuxt",
        ".gradle",
        ".idea",
        ".vscode",
        ".cache",
        APP_DIR_NAME,  # never recurse into a found store
    }
)


def find_ancestor_store(start: Path) -> Path | None:
    """Walk upward from `start` (exclusive) and return the nearest ancestor
    that already contains an `.agent-memory/` store. Returns None if there
    is no ancestor store. Used by init_project to enforce the
    "at most one store along any ancestor chain" invariant.
    """
    current = start.resolve()
    for ancestor in current.parents:
        if (ancestor / APP_DIR_NAME / CONFIG_FILENAME).exists():
            return ancestor
    return None


def find_descendant_stores(root: Path, max_depth: int = 6) -> list[Path]:
    """Walk downward from `root` looking for nested `.agent-memory/` stores.

    Bounded to `max_depth` levels of subdirectories and skips well-known
    cruft directories (.git, node_modules, build, etc.) so the scan stays
    fast even on large monorepos.
    """
    results: list[Path] = []
    root_resolved = root.resolve()

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(directory.iterdir())
        except (PermissionError, OSError):
            return
        for entry in entries:
            try:
                if not entry.is_dir() or entry.is_symlink():
                    continue
            except OSError:
                continue
            if entry.name in _DESCENDANT_SCAN_SKIP:
                continue
            if entry.name == APP_DIR_NAME:
                continue
            # Hit: a descendant has its own store.
            if (entry / APP_DIR_NAME / CONFIG_FILENAME).exists():
                results.append(entry.resolve())
                # Do not recurse into a directory that already has a store —
                # any further nesting underneath it is its own problem.
                continue
            walk(entry, depth + 1)

    walk(root_resolved, 0)
    return results


def init_project(
    root: Path,
    config: MemoryConfig | None = None,
    force: bool = False,
) -> ProjectContext:
    resolved_root = root.resolve()
    app_dir = resolved_root / APP_DIR_NAME
    config_path = app_dir / CONFIG_FILENAME
    db_path = app_dir / DB_FILENAME
    instructions_path = app_dir / INSTRUCTIONS_FILENAME

    # --- Nesting check (always enforced, even with --force) -------------------
    # Agent Memory's lookup walks upward to find the nearest ancestor store, so
    # nesting one project root inside another silently shadows the outer one
    # from inside the inner subtree. Refuse to create such configurations at
    # all so the user is forced to make a deliberate choice.
    ancestor = find_ancestor_store(resolved_root)
    if ancestor is not None and ancestor != resolved_root:
        raise ConfigError(
            f"Cannot initialize Agent Memory at {resolved_root}: an ancestor at "
            f"{ancestor} already has its own .agent-memory/ store. "
            "Agent Memory does not allow nested stores along an ancestor chain — "
            "the outer store would silently shadow the inner one in some lookup paths.\n\n"
            "Suggestions:\n"
            f"  - Use the existing store at {ancestor} (cd there and re-run your command), or\n"
            f"  - Uninstall the outer store first: `agent-memory uninstall {ancestor}`, then re-run init here, or\n"
            f"  - Pick a different project location that does not live under {ancestor}."
        )

    descendants = find_descendant_stores(resolved_root)
    if descendants:
        descendant_list = "\n".join(f"  - {d}" for d in descendants)
        first = descendants[0]
        raise ConfigError(
            f"Cannot initialize Agent Memory at {resolved_root}: one or more descendant "
            f"directories already have their own .agent-memory/ store:\n"
            f"{descendant_list}\n\n"
            "Agent Memory does not allow nested stores — only one store is permitted "
            "anywhere along an ancestor chain.\n\n"
            "Suggestions:\n"
            f"  - Uninstall the descendant store(s) first: `agent-memory uninstall {first}`, then re-run init here, or\n"
            "  - Initialize Agent Memory at the descendant location instead of the parent."
        )

    app_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and not force:
        raise ConfigError(
            f"Agent Memory is already initialized at {resolved_root}. "
            "Use `--force` to overwrite the config."
        )
    resolved_config = config or MemoryConfig()
    config_path.write_text(json.dumps(resolved_config.to_dict(), indent=2) + "\n", encoding='utf-8')
    if force or not instructions_path.exists():
        instructions_path.write_text(default_instructions(), encoding='utf-8')
    return ProjectContext(
        root=resolved_root,
        app_dir=app_dir,
        db_path=db_path,
        config_path=config_path,
        instructions_path=instructions_path,
        config=resolved_config,
    )
