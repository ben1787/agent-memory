#!/usr/bin/env sh

PLUGIN_ID_FALLBACK="agent-memory-agent-memory-plugins"
INSTALLER_URL_FALLBACK="https://raw.githubusercontent.com/ben1787/agent-memory/main/install.sh"
REPO_FALLBACK="ben1787/agent-memory"

plugin_init() {
    script_path="$1"
    script_dir=$(CDPATH= cd -- "$(dirname -- "$script_path")" && pwd)

    if [ -z "${CLAUDE_PLUGIN_ROOT:-}" ]; then
        CLAUDE_PLUGIN_ROOT=$(CDPATH= cd -- "$script_dir/.." && pwd)
    fi
    if [ -z "${CLAUDE_PLUGIN_DATA:-}" ]; then
        CLAUDE_PLUGIN_DATA="${HOME}/.claude/plugins/data/${PLUGIN_ID_FALLBACK}"
    fi

    AGENT_MEMORY_REAL_BIN="${CLAUDE_PLUGIN_DATA}/bin/agent-memory"
    AGENT_MEMORY_LIBEXEC_DIR="${CLAUDE_PLUGIN_DATA}/libexec"
    AGENT_MEMORY_INSTALL_LOCK="${CLAUDE_PLUGIN_DATA}/.install-lock"

    export CLAUDE_PLUGIN_ROOT
    export CLAUDE_PLUGIN_DATA
    export AGENT_MEMORY_REAL_BIN
    export AGENT_MEMORY_LIBEXEC_DIR
    export AGENT_MEMORY_INSTALL_LOCK
}

default_release_version() {
    version_file="${CLAUDE_PLUGIN_ROOT}/release-version.txt"
    if [ -f "$version_file" ]; then
        tr -d '\r' < "$version_file" | sed -n '1p'
        return 0
    fi
    return 1
}

agent_memory_ready() {
    [ -x "$AGENT_MEMORY_REAL_BIN" ] && "$AGENT_MEMORY_REAL_BIN" --version >/dev/null 2>&1
}

normalize_version() {
    value="${1:-}"
    value=$(printf '%s' "$value" | tr -d '\r' | sed 's/^agent-memory[[:space:]]\+//')
    printf '%s' "$value" | sed 's/^v//'
}

installed_agent_memory_version() {
    if ! agent_memory_ready; then
        return 1
    fi
    "$AGENT_MEMORY_REAL_BIN" --version 2>/dev/null | sed -n '1p'
}

selected_release_version() {
    selected_version="${AGENT_MEMORY_VERSION:-}"
    if [ -n "$selected_version" ]; then
        printf '%s\n' "$selected_version"
        return 0
    fi
    default_release_version
}

agent_memory_matches_selected_version() {
    desired_version="$(selected_release_version || true)"
    [ -n "$desired_version" ] || return 1

    installed_version="$(installed_agent_memory_version || true)"
    [ -n "$installed_version" ] || return 1

    [ "$(normalize_version "$installed_version")" = "$(normalize_version "$desired_version")" ]
}

download_file() {
    url="$1"
    destination="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --output "$destination" "$url"
        return 0
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -q -O "$destination" "$url"
        return 0
    fi

    echo "agent-memory plugin requires curl or wget to download the installer" >&2
    return 1
}

acquire_install_lock() {
    mkdir -p "$CLAUDE_PLUGIN_DATA"

    waited=0
    while ! mkdir "$AGENT_MEMORY_INSTALL_LOCK" 2>/dev/null; do
        waited=$((waited + 1))
        if [ "$waited" -ge 120 ]; then
            echo "timed out waiting for the agent-memory install lock" >&2
            return 1
        fi
        if agent_memory_ready; then
            return 0
        fi
        sleep 1
    done
    return 0
}

release_install_lock() {
    rmdir "$AGENT_MEMORY_INSTALL_LOCK" 2>/dev/null || true
}

ensure_agent_memory_installed() {
    if agent_memory_matches_selected_version; then
        return 0
    fi

    acquire_install_lock || return 1

    if agent_memory_matches_selected_version; then
        release_install_lock
        return 0
    fi

    tmpdir=$(mktemp -d)
    installer_path="${tmpdir}/install.sh"

    cleanup() {
        rm -rf "$tmpdir"
        release_install_lock
    }

    trap cleanup EXIT INT TERM HUP

    if ! download_file "${AGENT_MEMORY_INSTALLER_URL:-$INSTALLER_URL_FALLBACK}" "$installer_path"; then
        return 1
    fi
    chmod +x "$installer_path"

    mkdir -p "${CLAUDE_PLUGIN_DATA}/bin"

    selected_version="$(selected_release_version || true)"
    if [ -z "$selected_version" ]; then
        echo "agent-memory plugin is missing a bundled release version" >&2
        return 1
    fi

    local_tarball="${AGENT_MEMORY_LOCAL_TARBALL:-}"
    if [ -n "$local_tarball" ]; then
        if ! AGENT_MEMORY_REPO="${AGENT_MEMORY_REPO:-$REPO_FALLBACK}" \
            "$installer_path" \
            --local-tarball "$local_tarball" \
            --version "$selected_version" \
            --install-dir "${CLAUDE_PLUGIN_DATA}/bin" \
            --libexec-dir "$AGENT_MEMORY_LIBEXEC_DIR"; then
            return 1
        fi
    else
        if ! AGENT_MEMORY_REPO="${AGENT_MEMORY_REPO:-$REPO_FALLBACK}" \
            "$installer_path" \
            --version "$selected_version" \
            --install-dir "${CLAUDE_PLUGIN_DATA}/bin" \
            --libexec-dir "$AGENT_MEMORY_LIBEXEC_DIR"; then
            return 1
        fi
    fi

    if ! agent_memory_ready; then
        echo "agent-memory installer completed but the binary is still unavailable" >&2
        return 1
    fi

    trap - EXIT INT TERM HUP
    cleanup
}

find_initialized_project_root() {
    search_dir="$1"

    while [ -n "$search_dir" ]; do
        if [ -f "$search_dir/.agent-memory/config.json" ]; then
            printf '%s\n' "$search_dir"
            return 0
        fi

        parent_dir=$(dirname -- "$search_dir")
        if [ "$parent_dir" = "$search_dir" ]; then
            break
        fi
        search_dir="$parent_dir"
    done

    return 1
}
