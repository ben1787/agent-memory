#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"

plugin_init "$0"

payload_file=$(mktemp)
cleanup() {
    rm -f "$payload_file"
}
trap cleanup EXIT INT TERM HUP

cat > "$payload_file"

project_root="$(find_initialized_project_root "$PWD" || true)"
if [ -z "$project_root" ]; then
    printf '{}'
    exit 0
fi

export AGENT_MEMORY_PROJECT_ROOT="$project_root"

if ! ensure_agent_memory_installed >/dev/null 2>&1; then
    printf '{}'
    exit 0
fi

if ! "$AGENT_MEMORY_REAL_BIN" _hook claude-user-prompt-submit <"$payload_file"; then
    printf '{}'
fi
