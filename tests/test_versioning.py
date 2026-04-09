from __future__ import annotations

import json
import tomllib
from pathlib import Path

from agent_memory import __display_version__, __release_tag__, __version__
from agent_memory.versioning import version_key


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_version_converts_to_release_display_and_tag() -> None:
    assert __version__ == "0.2.3"
    assert __display_version__ == "0.2.3"
    assert __release_tag__ == "v0.2.3"


def test_version_key_handles_package_and_release_forms() -> None:
    assert version_key(__version__) == version_key(__release_tag__)
    assert version_key("0.2.4") > version_key(__version__)


def test_pyproject_version_matches_package_version() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["version"] == __version__


def test_plugin_release_pin_matches_package_release_tag() -> None:
    pinned = (REPO_ROOT / "plugins" / "agent-memory" / "release-version.txt").read_text(encoding="utf-8").strip()
    assert pinned == __release_tag__


def test_plugin_and_marketplace_versions_match_display_version() -> None:
    plugin_payload = json.loads(
        (REPO_ROOT / "plugins" / "agent-memory" / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    marketplace_payload = json.loads(
        (REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )

    assert plugin_payload["version"] == __display_version__
    assert marketplace_payload["metadata"]["version"] == __display_version__
    assert marketplace_payload["plugins"][0]["version"] == __display_version__
