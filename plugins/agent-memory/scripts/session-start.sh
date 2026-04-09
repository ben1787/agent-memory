#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/common.sh"

plugin_init "$0"

ensure_agent_memory_installed >/dev/null 2>&1 || true
