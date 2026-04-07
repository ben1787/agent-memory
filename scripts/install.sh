#!/usr/bin/env sh
set -eu

PACKAGE_SOURCE="${1:-.}"

if ! command -v uv >/dev/null 2>&1; then
  echo "agent-memory install requires \`uv\` on PATH." >&2
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "Installing agent-memory from: $PACKAGE_SOURCE"
uv tool install --reinstall "$PACKAGE_SOURCE"
uv tool update-shell

cat <<'EOF'

agent-memory is installed.

If this is your first install on this machine:
- restart your shell, terminal, or agent session so PATH picks up ~/.local/bin

Then initialize a repo:
- cd /path/to/repo
- agent-memory init

For Codex:
- `agent-memory init` adds repo-local `.codex/` wiring and trusts the repo in `~/.codex/config.toml`
- start a fresh interactive Codex session rooted in that repo after init
- use `agent-memory uninstall --remove-store` when you want to reset the repo to a clean pre-install state

EOF
