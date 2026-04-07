#!/usr/bin/env sh
# install.sh — one-line installer for agent-memory.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/<OWNER>/agent-memory/main/installer/install.sh | sh
#
# What it does:
#   1. Detects the host platform (uname -sm)
#   2. Resolves the latest release tag from the GitHub releases API (or honors
#      $AGENT_MEMORY_VERSION if set)
#   3. Downloads the matching binary + its .sha256 checksum
#   4. Verifies the checksum
#   5. Installs the binary to ~/.local/bin/agent-memory (or $AGENT_MEMORY_INSTALL_DIR
#      if set)
#   6. Adds ~/.local/bin to your shell rc PATH if it isn't already there
#   7. Prints next steps
#
# Environment overrides:
#   AGENT_MEMORY_VERSION       Tag to install, e.g. v0.2.1. Defaults to latest.
#   AGENT_MEMORY_INSTALL_DIR   Install directory. Defaults to ~/.local/bin.
#   AGENT_MEMORY_REPO          GitHub repo path "owner/name". Defaults to the
#                              canonical project. Override for forks.

set -eu

REPO="${AGENT_MEMORY_REPO:-ben1787/agent-memory}"
INSTALL_DIR="${AGENT_MEMORY_INSTALL_DIR:-$HOME/.local/bin}"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }
info()  { printf '  %s\n' "$*"; }

die() {
    red "error: $*"
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

need_cmd uname
need_cmd mkdir
need_cmd chmod
need_cmd rm

# Prefer curl, fall back to wget. One of them is required.
DOWNLOADER=""
if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
else
    die "need either curl or wget on PATH to download the binary"
fi

download() {
    # download <url> <dest>
    if [ "$DOWNLOADER" = "curl" ]; then
        curl -fsSL --retry 3 --output "$2" "$1"
    else
        wget -q -O "$2" "$1"
    fi
}

# --- Detect platform ----------------------------------------------------------
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

case "$OS_NAME" in
    Darwin)  os_slug="macos" ;;
    Linux)   os_slug="linux" ;;
    *)       die "unsupported OS: $OS_NAME (agent-memory ships macos / linux / windows)" ;;
esac

case "$ARCH_NAME" in
    arm64|aarch64)        arch_slug="arm64" ;;
    x86_64|amd64)         arch_slug="x86_64" ;;
    *)                    die "unsupported architecture: $ARCH_NAME" ;;
esac

ASSET="agent-memory-${os_slug}-${arch_slug}"

bold "Installing agent-memory"
info "host:    ${os_slug}-${arch_slug}"
info "repo:    ${REPO}"
info "target:  ${INSTALL_DIR}/agent-memory"

# --- Resolve version ----------------------------------------------------------
if [ -n "${AGENT_MEMORY_VERSION:-}" ]; then
    VERSION="$AGENT_MEMORY_VERSION"
else
    info "resolving latest release..."
    LATEST_URL="https://api.github.com/repos/${REPO}/releases/latest"
    if [ "$DOWNLOADER" = "curl" ]; then
        LATEST_JSON="$(curl -fsSL "$LATEST_URL" || true)"
    else
        LATEST_JSON="$(wget -qO- "$LATEST_URL" || true)"
    fi
    [ -n "$LATEST_JSON" ] || die "failed to query GitHub releases API at $LATEST_URL"
    VERSION="$(printf '%s\n' "$LATEST_JSON" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)"
    [ -n "$VERSION" ] || die "could not parse latest release tag (rate limited? authentication required?)"
fi
info "version: $VERSION"

# --- Download + verify --------------------------------------------------------
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

BINARY_URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"
SHA_URL="${BINARY_URL}.sha256"

info "downloading binary from ${BINARY_URL}"
download "$BINARY_URL" "${TMPDIR}/${ASSET}"

info "downloading checksum"
download "$SHA_URL" "${TMPDIR}/${ASSET}.sha256"

info "verifying checksum"
EXPECTED_SHA="$(awk '{print $1}' "${TMPDIR}/${ASSET}.sha256")"
[ -n "$EXPECTED_SHA" ] || die "checksum file is empty"

if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_SHA="$(sha256sum "${TMPDIR}/${ASSET}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    ACTUAL_SHA="$(shasum -a 256 "${TMPDIR}/${ASSET}" | awk '{print $1}')"
else
    die "need either sha256sum or shasum to verify the download"
fi

if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    die "checksum mismatch! expected $EXPECTED_SHA, got $ACTUAL_SHA — refusing to install"
fi
green "  checksum ok"

# --- Install ------------------------------------------------------------------
mkdir -p "$INSTALL_DIR"
mv "${TMPDIR}/${ASSET}" "${INSTALL_DIR}/agent-memory"
chmod +x "${INSTALL_DIR}/agent-memory"

green "installed: ${INSTALL_DIR}/agent-memory"

# --- Ensure $INSTALL_DIR is on PATH ------------------------------------------
case ":$PATH:" in
    *":${INSTALL_DIR}:"*)
        green "PATH already contains ${INSTALL_DIR}"
        ;;
    *)
        # Choose the right shell rc to update.
        SHELL_NAME="${SHELL##*/}"
        case "$SHELL_NAME" in
            zsh)  RC_FILE="${ZDOTDIR:-$HOME}/.zshrc" ;;
            bash) RC_FILE="$HOME/.bashrc" ;;
            *)    RC_FILE="$HOME/.profile" ;;
        esac
        PATH_LINE="export PATH=\"${INSTALL_DIR}:\$PATH\""
        if [ -f "$RC_FILE" ] && grep -Fxq "$PATH_LINE" "$RC_FILE"; then
            info "PATH line already present in ${RC_FILE}"
        else
            printf '\n# added by agent-memory installer\n%s\n' "$PATH_LINE" >> "$RC_FILE"
            green "appended PATH line to ${RC_FILE}"
            info "open a new terminal, or run: source ${RC_FILE}"
        fi
        ;;
esac

# --- Verify the install -------------------------------------------------------
if "${INSTALL_DIR}/agent-memory" --version >/dev/null 2>&1; then
    green "verified: $("${INSTALL_DIR}/agent-memory" --version)"
fi

echo
bold "agent-memory installed."
info "next: cd into a project and run \`agent-memory init\`"
info "then: \`agent-memory save\`, \`agent-memory recall\`, \`agent-memory list --recent\`"
info "docs: https://github.com/${REPO}#readme"
