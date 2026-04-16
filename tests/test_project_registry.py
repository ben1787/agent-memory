from __future__ import annotations

from pathlib import Path

from agent_memory.project_registry import (
    list_registered_project_roots,
    register_project_root,
    registry_path,
    unregister_project_root,
)


def test_project_registry_registers_and_unregisters_unique_roots(tmp_path: Path, monkeypatch) -> None:
    data_home = tmp_path / "data-home"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    first = tmp_path / "repo-a"
    second = tmp_path / "repo-b"
    first.mkdir()
    second.mkdir()

    register_project_root(first)
    register_project_root(second)
    register_project_root(first)

    assert list_registered_project_roots() == [first.resolve(), second.resolve()]
    assert registry_path() == data_home / "agent-memory" / "known-projects.json"

    remaining = unregister_project_root(first)
    assert remaining == [second.resolve()]
    assert list_registered_project_roots() == [second.resolve()]

    unregister_project_root(second)
    assert list_registered_project_roots() == []
    assert not registry_path().exists()
