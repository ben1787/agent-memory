from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_registry_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-home"))
