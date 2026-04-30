from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


APP_DIR_NAME = ".agent-memory"
CONFIG_FILENAME = "config.json"
DB_FILENAME = "memory.kuzu"
INSTRUCTIONS_FILENAME = "instructions.md"
LINKED_ROOTS_FILENAME = "linked-roots.json"


class ConfigError(RuntimeError):
    """Raised when project configuration cannot be found or parsed."""


def _config_path(root: Path) -> Path:
    return root / APP_DIR_NAME / CONFIG_FILENAME


def _linked_roots_path(root: Path) -> Path:
    return root / APP_DIR_NAME / LINKED_ROOTS_FILENAME


CURRENT_CONFIG_VERSION = 8
LEGACY_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
LEGACY_DEFAULT_DIMENSIONS = 384
DEFAULT_MODEL = "snowflake/snowflake-arctic-embed-m"
DEFAULT_DIMENSIONS = 768
DEFAULT_MAX_MEMORY_WORDS = 250
DEFAULT_PROMPT_CONTEXT_TURN_INTERVAL = 10
DEFAULT_AUTO_UPGRADE = True


@dataclass(slots=True)
class MemoryConfig:
    version: int = CURRENT_CONFIG_VERSION
    embedding_backend: str = "fastembed"
    embedding_model: str = DEFAULT_MODEL
    embedding_dimensions: int = DEFAULT_DIMENSIONS
    stored_embedding_backend: str | None = None
    stored_embedding_model: str | None = None
    stored_embedding_dimensions: int | None = None
    integration_version: str | None = None
    max_memory_words: int = DEFAULT_MAX_MEMORY_WORDS
    prompt_context_turn_interval: int = DEFAULT_PROMPT_CONTEXT_TURN_INTERVAL
    auto_upgrade: bool = DEFAULT_AUTO_UPGRADE
    duplicate_threshold: float = 0.97
    overlap_threshold: float = 0.90
    lexical_duplicate_threshold: float = 0.95
    consolidation_similarity_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.stored_embedding_backend is None:
            self.stored_embedding_backend = self.embedding_backend
        if self.stored_embedding_model is None:
            self.stored_embedding_model = self.embedding_model
        if self.stored_embedding_dimensions is None:
            self.stored_embedding_dimensions = self.embedding_dimensions

    def desired_embedding_signature(self) -> tuple[str, str, int]:
        return (
            self.embedding_backend,
            self.embedding_model,
            self.embedding_dimensions,
        )

    def stored_embedding_signature(self) -> tuple[str, str, int]:
        return (
            self.stored_embedding_backend or self.embedding_backend,
            self.stored_embedding_model or self.embedding_model,
            self.stored_embedding_dimensions or self.embedding_dimensions,
        )

    def needs_reembed(self) -> bool:
        return self.stored_embedding_signature() != self.desired_embedding_signature()

    def with_store_current(self) -> "MemoryConfig":
        payload = self.to_dict()
        payload["stored_embedding_backend"] = self.embedding_backend
        payload["stored_embedding_model"] = self.embedding_model
        payload["stored_embedding_dimensions"] = self.embedding_dimensions
        return MemoryConfig.from_dict(payload)

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

        version = int(data.get("version", 4))
        original_backend = str(data.get("embedding_backend", "fastembed"))
        original_model = str(data.get("embedding_model", LEGACY_DEFAULT_MODEL))
        original_dimensions = int(data.get("embedding_dimensions", LEGACY_DEFAULT_DIMENSIONS))

        data.setdefault("stored_embedding_backend", original_backend)
        data.setdefault("stored_embedding_model", original_model)
        data.setdefault("stored_embedding_dimensions", original_dimensions)
        data.setdefault("integration_version", None)
        data.setdefault("prompt_context_turn_interval", DEFAULT_PROMPT_CONTEXT_TURN_INTERVAL)
        data.setdefault("auto_upgrade", DEFAULT_AUTO_UPGRADE)

        if version < CURRENT_CONFIG_VERSION:
            if (
                version < 8
                and float(data.get("consolidation_similarity_threshold", 0.92)) == 0.92
            ):
                data["consolidation_similarity_threshold"] = 0.8
            if (
                original_backend == "fastembed"
                and original_model == LEGACY_DEFAULT_MODEL
                and original_dimensions == LEGACY_DEFAULT_DIMENSIONS
            ):
                data["embedding_model"] = DEFAULT_MODEL
                data["embedding_dimensions"] = DEFAULT_DIMENSIONS
            max_words = int(data.get("max_memory_words", 1000))
            if max_words in (1000, 200):
                data["max_memory_words"] = DEFAULT_MAX_MEMORY_WORDS
            if "integration_version" not in data:
                data["integration_version"] = None
            data["version"] = CURRENT_CONFIG_VERSION

        allowed_fields = {field.name for field in fields(cls)}
        filtered = {
            key: value
            for key, value in data.items()
            if key in allowed_fields
        }
        return cls(**filtered)

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
- Strong matches from the current user prompt may be recalled automatically into context before the model call. The current auto-recall floor is a parent score of 0.7.
- If prior project knowledge might help and the automatic recall is missing or incomplete, consider calling `recall_memories` with a concrete query tied to the current task.
- After finishing the work for the turn, decide whether there are 0-3 durable memories worth saving with `save_memory`.
- One strong operational fact is enough: a file/module location, hook or threshold rule, install/update gotcha, runtime quirk, or explicit user correction.
- When saving, pass explicit metadata arguments: `title`, `kind`, `subsystem`, `workstream`, and `environment`. Keep the memory body text plain, self-contained, and written for a reader with no conversational context.
- If the MCP tools are unavailable in the current client, fall back to `agent-memory recall` and `agent-memory save` in the current project root.
- Prefer stable facts, decisions, file locations, constraints, preferences, and discovered relationships.
- Do not save terse fragments or giant raw dumps. Target 50-250 words with concrete detail about what changed, where it lives, why it matters, and how to use the fact correctly.
- Save memories into this project store only; never write into some parent or sibling project by accident.

Suggested workflow:
1. Check any automatically injected Agent Memory context first.
2. If prior project knowledge might help and the injected context is missing or incomplete, call `agent-memory recall "<task-specific query>"`.
3. Do the work.
4. Before the final answer, if the work produced durable project knowledge, save 0-3 self-contained memories with explicit metadata fields and a 50-250 word body.
5. Save it if it would save future code inspection, prevent a likely wrong assumption, or narrow the search path.
6. Reuse existing metadata spellings when they fit, and only invent a new value when nothing existing matches.

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
    raw_config = json.loads(config_path.read_text(encoding='utf-8'))
    legacy_linked_roots = raw_config.pop("linked_project_roots", None)
    config = MemoryConfig.from_dict(raw_config)
    normalized = config.to_dict()
    if normalized != raw_config:
        config_path.write_text(json.dumps(normalized, indent=2) + "\n", encoding='utf-8')
    if legacy_linked_roots is not None:
        write_linked_project_roots(root, legacy_linked_roots)
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


def _normalize_linked_root_strings(root: Path, roots: list[object]) -> list[str]:
    normalized: dict[str, str] = {}
    for item in roots:
        if not isinstance(item, str) or not item.strip():
            continue
        try:
            resolved = str(Path(item).expanduser().resolve())
        except OSError:
            resolved = str((root / item).expanduser())
        normalized[resolved] = resolved
    return [normalized[key] for key in sorted(normalized)]


def load_linked_project_roots(root: Path) -> list[str]:
    path = _linked_roots_path(root.resolve())
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("linked_project_roots") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return _normalize_linked_root_strings(root.resolve(), items)


def write_linked_project_roots(root: Path, roots: list[object]) -> list[str]:
    resolved_root = root.resolve()
    normalized = _normalize_linked_root_strings(resolved_root, roots)
    path = _linked_roots_path(resolved_root)
    if not normalized:
        try:
            path.unlink()
        except OSError:
            pass
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"linked_project_roots": normalized}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return normalized
