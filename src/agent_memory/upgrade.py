"""Self-update support for the agent-memory binary.

Two surfaces:

1. ``agent-memory upgrade`` — explicit user command. Hits the GitHub releases
   API, compares the published latest version against the running binary's
   version, downloads the right platform binary if newer, verifies the
   sha256 checksum, and atomically replaces the running binary in place.

2. ``check_for_upgrade_in_background()`` — non-blocking staleness check that
   any command can call. Caches the result for 24h in
   ``~/.cache/agent-memory/update-check.json`` so we never block on the
   network in the hot path. Prints a one-line "new version available" notice
   on stderr when there's something newer; otherwise silent.

The model matches what `gh`, `bun`, and `uv` do — the user always has the
choice to upgrade, but they're never surprised by a stale binary either.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_memory import __display_version__, __version__
from agent_memory.versioning import version_key


GITHUB_RELEASES_API = "https://api.github.com/repos/{repo}/releases/latest"
DEFAULT_REPO = "ben1787/agent-memory"
DOWNLOAD_BASE = "https://github.com/{repo}/releases/download/{tag}/{asset}"
STALENESS_CHECK_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours
NETWORK_TIMEOUT_SECONDS = 5


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "agent-memory"


def _staleness_cache_path() -> Path:
    return _cache_dir() / "update-check.json"


def _detect_asset_name() -> str | None:
    """Return the release asset filename for the current platform, or None
    if the current platform is not in the supported matrix."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "agent-memory-macos-arm64"
        # Intel macOS is not built in the release matrix (GitHub retired the
        # free macos-13 runner). Fall through to None — the upgrade command
        # will report no matching asset.
    elif system == "Linux":
        if machine in ("x86_64", "amd64"):
            return "agent-memory-linux-x86_64"
        if machine in ("arm64", "aarch64"):
            return "agent-memory-linux-arm64"
    elif system == "Windows":
        if machine in ("x86_64", "amd64"):
            return "agent-memory-windows-x86_64.exe"
    return None


def _http_get(url: str, *, accept: str | None = None) -> bytes:
    request = urllib.request.Request(url)
    if accept:
        request.add_header("Accept", accept)
    request.add_header("User-Agent", f"agent-memory/{__display_version__}")
    with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        return response.read()


def _parse_version(tag: str) -> tuple[int, ...]:
    """Parse release tags and package versions into a comparable tuple."""
    return version_key(tag)


@dataclass(slots=True)
class LatestRelease:
    tag: str
    version_tuple: tuple[int, ...]
    asset_url: str
    asset_sha_url: str


def _resolve_latest_release(repo: str = DEFAULT_REPO) -> LatestRelease | None:
    """Hit the GitHub API and resolve the latest release tag + asset URLs.

    Returns None if the platform is unsupported, the API call fails, or the
    response can't be parsed. Never raises — caller should treat None as
    "couldn't check" and move on.
    """
    asset = _detect_asset_name()
    if asset is None:
        return None
    try:
        body = _http_get(GITHUB_RELEASES_API.format(repo=repo), accept="application/vnd.github+json")
        payload = json.loads(body)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    tag = payload.get("tag_name") if isinstance(payload, dict) else None
    if not isinstance(tag, str) or not tag:
        return None
    return LatestRelease(
        tag=tag,
        version_tuple=_parse_version(tag),
        asset_url=DOWNLOAD_BASE.format(repo=repo, tag=tag, asset=asset),
        asset_sha_url=DOWNLOAD_BASE.format(repo=repo, tag=tag, asset=f"{asset}.sha256"),
    )


def check_for_upgrade_in_background(*, repo: str = DEFAULT_REPO) -> str | None:
    """Cached staleness check. Returns a notice string if a newer version is
    available, else None.

    Reads/writes ``~/.cache/agent-memory/update-check.json`` to throttle
    network calls to once per 24 hours. Failures (network down, API rate
    limited, etc.) are silent — we never block the actual command on this.
    """
    cache_path = _staleness_cache_path()
    now = time.time()

    cached: dict[str, Any] | None = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            cached = None

    last_checked = float(cached.get("checked_at", 0)) if cached else 0.0
    if now - last_checked < STALENESS_CHECK_INTERVAL_SECONDS and cached:
        # Use the cached result instead of hitting the network.
        latest_tag = cached.get("latest_tag")
        if isinstance(latest_tag, str) and latest_tag:
            current = _parse_version(__version__)
            if _parse_version(latest_tag) > current:
                return f"agent-memory {latest_tag} is available (you have {__display_version__}). Run `agent-memory upgrade` to update."
        return None

    # Cache miss or stale → refresh.
    latest = _resolve_latest_release(repo=repo)
    cache_payload: dict[str, Any] = {"checked_at": now}
    if latest is not None:
        cache_payload["latest_tag"] = latest.tag
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_payload), encoding='utf-8')
    except OSError:
        pass

    if latest is None:
        return None
    if latest.version_tuple <= _parse_version(__version__):
        return None
    return f"agent-memory {latest.tag} is available (you have {__display_version__}). Run `agent-memory upgrade` to update."


def perform_upgrade(*, repo: str = DEFAULT_REPO) -> dict[str, Any]:
    """Download the latest binary, verify checksum, and replace the running
    binary atomically. Returns a status dict for the CLI to render.

    The "running binary" is detected via ``sys.argv[0]`` plus a few fallbacks.
    If we can't locate it (e.g. running from a Python module rather than the
    PyInstaller binary), we report an explanatory error instead of crashing.
    """
    asset = _detect_asset_name()
    if asset is None:
        return {
            "status": "error",
            "details": (
                f"agent-memory does not publish a binary for {platform.system()} "
                f"{platform.machine()}. Use the source install (`uv tool install agent-memory`) instead."
            ),
        }

    latest = _resolve_latest_release(repo=repo)
    if latest is None:
        return {
            "status": "error",
            "details": "Could not reach the GitHub releases API to resolve the latest version.",
        }

    current_tuple = _parse_version(__version__)
    if latest.version_tuple <= current_tuple:
        return {
            "status": "up-to-date",
            "current_version": __display_version__,
            "latest_version": latest.tag,
            "details": f"Already on the latest version ({__display_version__}).",
        }

    target_path = _resolve_running_binary_path()
    if target_path is None:
        return {
            "status": "error",
            "details": (
                "Could not locate the running agent-memory binary on disk. This usually "
                "means agent-memory was invoked as a Python module rather than the standalone "
                "binary. Use your package manager to upgrade instead (e.g. `brew upgrade agent-memory` "
                "or `uv tool install --reinstall --force agent-memory`)."
            ),
        }

    # Download binary + checksum into a tempdir, verify, atomic move.
    tmpdir = Path(tempfile.mkdtemp(prefix="agent-memory-upgrade."))
    try:
        new_binary = tmpdir / asset
        sha_file = tmpdir / f"{asset}.sha256"
        try:
            new_binary.write_bytes(_http_get(latest.asset_url))
            sha_file.write_bytes(_http_get(latest.asset_sha_url))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            return {
                "status": "error",
                "details": f"Failed to download {latest.tag}: {exc}",
            }

        expected_sha = sha_file.read_text(encoding='utf-8').split()[0].strip()
        actual_sha = _sha256(new_binary)
        if expected_sha.lower() != actual_sha.lower():
            return {
                "status": "error",
                "details": (
                    f"Checksum mismatch on {asset}: expected {expected_sha}, got {actual_sha}. "
                    "Refusing to install. Try again later — this could be a transient mirror "
                    "issue or a serious tampering signal."
                ),
            }

        # Atomic replace. On Windows, the running binary may be locked; we
        # work around it by writing the new binary alongside and renaming.
        try:
            new_binary.chmod(0o755)
            shutil.move(str(new_binary), str(target_path))
        except OSError as exc:
            return {
                "status": "error",
                "details": (
                    f"Failed to install new binary at {target_path}: {exc}. "
                    f"You may need elevated permissions, or to remove the existing binary first."
                ),
            }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Refresh the staleness cache so we don't immediately re-prompt.
    try:
        cache_path = _staleness_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"checked_at": time.time(), "latest_tag": latest.tag}), encoding='utf-8')
    except OSError:
        pass

    return {
        "status": "upgraded",
        "previous_version": __display_version__,
        "new_version": latest.tag,
        "binary_path": str(target_path),
        "details": f"Upgraded agent-memory from {__display_version__} to {latest.tag}.",
    }


def _resolve_running_binary_path() -> Path | None:
    """Best-effort lookup of the on-disk path of the currently-running binary."""
    candidate = sys.argv[0] if sys.argv else None
    if candidate:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path.resolve()
    # PyInstaller sets sys.executable to the bundled binary path.
    if hasattr(sys, "frozen") and getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()
    # Last resort: look for `agent-memory` on PATH.
    on_path = shutil.which("agent-memory")
    if on_path:
        return Path(on_path).resolve()
    return None


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
