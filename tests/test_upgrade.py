from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from agent_memory import upgrade


def test_parse_version_handles_v_prefix_and_prerelease() -> None:
    assert upgrade._parse_version("v1.2.3") == (1, 2, 3)
    assert upgrade._parse_version("1.2.3") == (1, 2, 3)
    # Prerelease suffix is stripped for comparison purposes.
    assert upgrade._parse_version("v1.2.3-rc.1") == (1, 2, 3)
    # Non-numeric chunks become 0 rather than raising.
    assert upgrade._parse_version("v1.2.foo") == (1, 2, 0)


def test_detect_asset_name_returns_none_on_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(upgrade.platform, "system", lambda: "Plan9")
    assert upgrade._detect_asset_name() is None


def test_detect_asset_name_macos_arm64(monkeypatch) -> None:
    monkeypatch.setattr(upgrade.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(upgrade.platform, "machine", lambda: "arm64")
    assert upgrade._detect_asset_name() == "agent-memory-macos-arm64"


def test_detect_asset_name_linux_x86(monkeypatch) -> None:
    monkeypatch.setattr(upgrade.platform, "system", lambda: "Linux")
    monkeypatch.setattr(upgrade.platform, "machine", lambda: "x86_64")
    assert upgrade._detect_asset_name() == "agent-memory-linux-x86_64"


def test_check_for_upgrade_uses_cache_when_fresh(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(upgrade, "_cache_dir", lambda: cache_dir)
    cache_file = cache_dir / "update-check.json"
    # Pretend we checked 1 hour ago and the latest tag was way newer.
    cache_file.write_text(
        json.dumps({"checked_at": time.time() - 3600, "latest_tag": "v999.0.0"})
    )

    # _resolve_latest_release should NOT be called because the cache is fresh.
    with patch.object(upgrade, "_resolve_latest_release") as resolve_mock:
        notice = upgrade.check_for_upgrade_in_background()
        resolve_mock.assert_not_called()

    assert notice is not None
    assert "v999.0.0" in notice
    assert "agent-memory upgrade" in notice


def test_check_for_upgrade_returns_none_when_already_latest(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(upgrade, "_cache_dir", lambda: cache_dir)
    # Stale cache → fresh fetch.
    cache_file = cache_dir / "update-check.json"
    cache_file.write_text(json.dumps({"checked_at": 0, "latest_tag": "v0.0.1"}))

    fake_latest = upgrade.LatestRelease(
        tag=f"v{upgrade.__version__}",
        version_tuple=upgrade._parse_version(upgrade.__version__),
        asset_url="",
        asset_sha_url="",
    )
    with patch.object(upgrade, "_resolve_latest_release", return_value=fake_latest):
        notice = upgrade.check_for_upgrade_in_background()
    assert notice is None


def test_check_for_upgrade_handles_network_failure_silently(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(upgrade, "_cache_dir", lambda: cache_dir)

    with patch.object(upgrade, "_resolve_latest_release", return_value=None):
        notice = upgrade.check_for_upgrade_in_background()
    assert notice is None
    # Cache file should still get written (with no latest_tag) so we don't
    # retry on every command.
    cache_file = cache_dir / "update-check.json"
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text())
    assert "checked_at" in payload


def test_perform_upgrade_reports_up_to_date(monkeypatch) -> None:
    fake_latest = upgrade.LatestRelease(
        tag=f"v{upgrade.__version__}",
        version_tuple=upgrade._parse_version(upgrade.__version__),
        asset_url="",
        asset_sha_url="",
    )
    monkeypatch.setattr(upgrade, "_resolve_latest_release", lambda repo=upgrade.DEFAULT_REPO: fake_latest)
    monkeypatch.setattr(upgrade, "_detect_asset_name", lambda: "agent-memory-macos-arm64")

    result = upgrade.perform_upgrade()

    assert result["status"] == "up-to-date"
    assert result["current_version"] == upgrade.__version__


def test_perform_upgrade_reports_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(upgrade, "_detect_asset_name", lambda: None)
    result = upgrade.perform_upgrade()
    assert result["status"] == "error"
    assert "does not publish a binary" in result["details"]


def test_perform_upgrade_reports_api_failure(monkeypatch) -> None:
    monkeypatch.setattr(upgrade, "_detect_asset_name", lambda: "agent-memory-macos-arm64")
    monkeypatch.setattr(upgrade, "_resolve_latest_release", lambda repo=upgrade.DEFAULT_REPO: None)
    result = upgrade.perform_upgrade()
    assert result["status"] == "error"
    assert "GitHub releases API" in result["details"]
