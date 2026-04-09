from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


if sys.platform.startswith("win"):
    pytest.skip("plugin wrapper tests require POSIX shell semantics", allow_module_level=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_SH = REPO_ROOT / "plugins" / "agent-memory" / "scripts" / "common.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_plugin_wrapper_reinstalls_when_pinned_version_changes(tmp_path: Path) -> None:
    if not (shutil.which("curl") or shutil.which("wget")):
        pytest.skip("plugin wrapper download helper requires curl or wget")

    plugin_root = tmp_path / "plugin-root"
    plugin_root.mkdir()
    (plugin_root / "release-version.txt").write_text("v0.2.0\n", encoding="utf-8")

    plugin_data = tmp_path / "plugin-data"
    (plugin_data / "bin").mkdir(parents=True)

    # Stale previously-installed binary.
    _write_executable(
        plugin_data / "bin" / "agent-memory",
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        '  echo "agent-memory 0.1.0"\n'
        "  exit 0\n"
        "fi\n"
        'echo "stale binary"\n',
    )

    installer = tmp_path / "fake-installer.sh"
    _write_executable(
        installer,
        "#!/bin/sh\n"
        "set -eu\n"
        'version=""\n'
        'install_dir=""\n'
        'libexec_dir=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    --version) version="$2"; shift 2 ;;\n'
        '    --install-dir) install_dir="$2"; shift 2 ;;\n'
        '    --libexec-dir) libexec_dir="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        '[ -n "$version" ]\n'
        '[ -n "$install_dir" ]\n'
        '[ -n "$libexec_dir" ]\n'
        'mkdir -p "$install_dir" "$libexec_dir"\n'
        'cat > "$install_dir/agent-memory" <<EOF\n'
        '#!/bin/sh\n'
        'if [ "\\$1" = "--version" ]; then\n'
        '  echo "agent-memory ${version#v}"\n'
        "  exit 0\n"
        "fi\n"
        'echo "fresh binary ${version#v}"\n'
        "EOF\n"
        'chmod +x "$install_dir/agent-memory"\n',
    )

    env = os.environ | {
        "CLAUDE_PLUGIN_ROOT": str(plugin_root),
        "CLAUDE_PLUGIN_DATA": str(plugin_data),
        "AGENT_MEMORY_INSTALLER_URL": installer.resolve().as_uri(),
    }
    result = subprocess.run(
        [
            "/bin/sh",
            "-lc",
            f'. "{COMMON_SH}"\n'
            'plugin_init "$0"\n'
            "ensure_agent_memory_installed\n"
            '"$AGENT_MEMORY_REAL_BIN" --version\n',
        ],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )

    assert result.stdout.strip() == "agent-memory 0.2.0"
