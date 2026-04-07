from __future__ import annotations

from pathlib import Path

import pytest

from agent_memory.config import ConfigError, find_project_root, init_project, load_project


def test_load_project_finds_unique_ancestor(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    nested = project_root / "src" / "feature"
    nested.mkdir(parents=True)
    init_project(project_root)

    project = load_project(nested)

    assert project.root == project_root.resolve()
    assert project.instructions_path.exists()
    assert "Agent Memory Instructions" in project.instructions_path.read_text()


def test_init_refuses_when_ancestor_already_has_store(tmp_path: Path) -> None:
    """Nesting a new store inside an existing one is forbidden — the outer
    store would silently shadow the inner one in some lookup paths."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    init_project(outer)

    with pytest.raises(ConfigError) as exc:
        init_project(inner)

    message = str(exc.value)
    assert "Cannot initialize" in message
    assert "ancestor" in message
    assert str(outer.resolve()) in message
    # The error should suggest a remediation.
    assert "uninstall" in message.lower()


def test_init_refuses_when_descendant_already_has_store(tmp_path: Path) -> None:
    """Initializing in a parent of an existing store is also forbidden —
    the new parent would silently shadow the existing inner store."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    init_project(inner)

    with pytest.raises(ConfigError) as exc:
        init_project(outer)

    message = str(exc.value)
    assert "Cannot initialize" in message
    assert "descendant" in message
    assert str(inner.resolve()) in message


def test_init_allows_siblings(tmp_path: Path) -> None:
    """Sibling directories are not ancestor-or-descendant of each other,
    so they can each have their own store without violating the invariant."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    init_project(a)
    init_project(b)  # Must not raise.

    assert (a / ".agent-memory" / "config.json").exists()
    assert (b / ".agent-memory" / "config.json").exists()


def test_init_skips_descendant_scan_into_node_modules(tmp_path: Path) -> None:
    """The descendant scan must skip well-known cruft directories so it stays
    fast on real-world repos. Specifically: if `node_modules/.agent-memory/`
    exists for some weird reason, init at the project root must NOT find it."""
    project = tmp_path / "project"
    weird_nested = project / "node_modules" / "some-pkg"
    weird_nested.mkdir(parents=True)
    init_project(weird_nested)
    # Now try to init at the project root. The scan must skip node_modules
    # and let the install proceed. (This is technically violating the invariant
    # in a hidden corner of the tree, but it's a vanishingly rare scenario and
    # the alternative — scanning every node_modules dir on every init — is much
    # worse for the common case.)
    init_project(project)
    assert (project / ".agent-memory" / "config.json").exists()


def test_exact_project_root_must_match_initialized_root(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    child = project_root / "pkg"
    child.mkdir(parents=True)
    init_project(project_root)

    with pytest.raises(ConfigError) as exc:
        load_project(child, exact=True)

    assert "Expected an agent-memory project exactly at" in str(exc.value)
