from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from agent_memory import upgrade


def test_parse_version_handles_v_prefix_and_prerelease() -> None:
    assert upgrade._parse_version("v1.2.3") == (1, 2, 3, 3, 0)
    assert upgrade._parse_version("1.2.3") == (1, 2, 3, 3, 0)
    assert upgrade._parse_version("v1.2.3-rc.1") == (1, 2, 3, 2, 1)
    assert upgrade._parse_version("1.2.3rc1") == (1, 2, 3, 2, 1)
    # Non-numeric chunks become 0 rather than raising.
    assert upgrade._parse_version("v1.2.foo") == (1, 2, 0, 3, 0)


def test_parse_version_orders_stable_after_prerelease() -> None:
    assert upgrade._parse_version("v1.2.3") > upgrade._parse_version("v1.2.3-rc.1")


def test_detect_asset_name_returns_none_on_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(upgrade.platform, "system", lambda: "Plan9")
    assert upgrade._detect_asset_name() is None


def test_detect_asset_name_macos_arm64(monkeypatch) -> None:
    monkeypatch.setattr(upgrade.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(upgrade.platform, "machine", lambda: "arm64")
    assert upgrade._detect_asset_name() == "agent-memory-macos-arm64.tar.gz"


def test_detect_asset_name_linux_x86(monkeypatch) -> None:
    monkeypatch.setattr(upgrade.platform, "system", lambda: "Linux")
    monkeypatch.setattr(upgrade.platform, "machine", lambda: "x86_64")
    assert upgrade._detect_asset_name() == "agent-memory-linux-x86_64.tar.gz"


def test_check_for_upgrade_uses_cache_when_fresh(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(upgrade, "_cache_dir", lambda: cache_dir)
    cache_file = cache_dir / "update-check.json"
    # Pretend we checked 1 hour ago and the latest tag was way newer.
    cache_file.write_text(
        json.dumps({"checked_at": time.time() - 3600, "latest_tag": "v999.0.0"})
    , encoding='utf-8')

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
    cache_file.write_text(json.dumps({"checked_at": 0, "latest_tag": "v0.0.1"}), encoding='utf-8')

    fake_latest = upgrade.LatestRelease(
        tag="v0.1.0-rc.2",
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
    payload = json.loads(cache_file.read_text(encoding='utf-8'))
    assert "checked_at" in payload


def test_perform_upgrade_reports_up_to_date(monkeypatch) -> None:
    fake_latest = upgrade.LatestRelease(
        tag="v0.1.0-rc.2",
        version_tuple=upgrade._parse_version(upgrade.__version__),
        asset_url="",
        asset_sha_url="",
    )
    monkeypatch.setattr(upgrade, "_resolve_latest_release", lambda repo=upgrade.DEFAULT_REPO: fake_latest)
    monkeypatch.setattr(upgrade, "_detect_asset_name", lambda: "agent-memory-macos-arm64.tar.gz")

    result = upgrade.perform_upgrade()

    assert result["status"] == "up-to-date"
    assert result["current_version"] == upgrade.__display_version__


def test_perform_upgrade_reports_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(upgrade, "_detect_asset_name", lambda: None)
    result = upgrade.perform_upgrade()
    assert result["status"] == "error"
    assert "does not publish a binary" in result["details"]


def test_perform_upgrade_reports_api_failure(monkeypatch) -> None:
    monkeypatch.setattr(upgrade, "_detect_asset_name", lambda: "agent-memory-macos-arm64.tar.gz")
    monkeypatch.setattr(upgrade, "_resolve_latest_release", lambda repo=upgrade.DEFAULT_REPO: None)
    result = upgrade.perform_upgrade()
    assert result["status"] == "error"
    assert "GitHub releases API" in result["details"]


def test_perform_upgrade_reports_source_install_when_running_from_console_script(
    tmp_path, monkeypatch
) -> None:
    script = tmp_path / "agent-memory"
    script.write_text(
        "#!/usr/bin/env python3\nfrom agent_memory.cli import main\nmain()\n",
        encoding="utf-8",
    )
    fake_latest = upgrade.LatestRelease(
        tag="v999.0.0",
        version_tuple=upgrade._parse_version("v999.0.0"),
        asset_url="",
        asset_sha_url="",
    )
    monkeypatch.setattr(upgrade.sys, "argv", [str(script)])
    monkeypatch.setattr(upgrade.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        upgrade,
        "_detect_asset_name",
        lambda: "agent-memory-macos-arm64.tar.gz",
    )
    monkeypatch.setattr(
        upgrade,
        "_resolve_latest_release",
        lambda repo=upgrade.DEFAULT_REPO: fake_latest,
    )

    result = upgrade.perform_upgrade()

    assert result["status"] == "error"
    assert "package manager" in result["details"]


def test_resolve_running_binary_accepts_native_binary(tmp_path, monkeypatch) -> None:
    binary = tmp_path / "agent-memory"
    binary.write_bytes(b"\x7fELFfake executable")
    monkeypatch.setattr(upgrade.sys, "argv", [str(binary)])
    monkeypatch.setattr(upgrade.shutil, "which", lambda _name: None)

    assert upgrade._resolve_running_binary_path() == binary.resolve()


def test_resolve_running_binary_rejects_console_script(tmp_path, monkeypatch) -> None:
    script = tmp_path / "agent-memory"
    script.write_text(
        "#!/usr/bin/env python3\nfrom agent_memory.cli import main\nmain()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(upgrade.sys, "argv", [str(script)])
    monkeypatch.setattr(upgrade.shutil, "which", lambda _name: None)

    assert upgrade._resolve_running_binary_path() is None


def test_resolve_running_binary_does_not_fallback_from_console_script_to_path(
    tmp_path, monkeypatch
) -> None:
    script = tmp_path / "agent-memory-script"
    script.write_text(
        "#!/usr/bin/env python3\nfrom agent_memory.cli import main\nmain()\n",
        encoding="utf-8",
    )
    binary = tmp_path / "agent-memory"
    binary.write_bytes(b"\x7fELFfake executable")
    monkeypatch.setattr(upgrade.sys, "argv", [str(script)])
    monkeypatch.setattr(upgrade.shutil, "which", lambda _name: str(binary))

    assert upgrade._resolve_running_binary_path() is None


def test_resolve_running_binary_rejects_python_module_path(tmp_path, monkeypatch) -> None:
    module = tmp_path / "cli.py"
    module.write_text("print('not a binary')\n", encoding="utf-8")
    monkeypatch.setattr(upgrade.sys, "argv", [str(module)])
    monkeypatch.setattr(upgrade.shutil, "which", lambda _name: None)

    assert upgrade._resolve_running_binary_path() is None
