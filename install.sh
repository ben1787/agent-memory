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
#   AGENT_MEMORY_LIBEXEC_DIR   Bundle extraction root. Defaults to
#                              ~/.local/share/agent-memory.
#   AGENT_MEMORY_REPO          GitHub repo path "owner/name". Defaults to the
#                              canonical project. Override for forks.
#
# Flags:
#   --local-tarball <path>     Install from a local tarball instead of fetching
#                              from a GitHub release. Skips version resolution
#                              and checksum verification. Useful for development
#                              and CI smoke tests of the installer itself.
#   --version <vX.Y.Z>         Same as setting AGENT_MEMORY_VERSION.
#   --install-dir <dir>        Same as setting AGENT_MEMORY_INSTALL_DIR.
#   --libexec-dir <dir>        Same as setting AGENT_MEMORY_LIBEXEC_DIR.

set -eu

LOCAL_TARBALL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --local-tarball)
            [ $# -ge 2 ] || { echo "--local-tarball requires a path" >&2; exit 1; }
            LOCAL_TARBALL="$2"
            shift 2
            ;;
        --version)
            [ $# -ge 2 ] || { echo "--version requires a tag" >&2; exit 1; }
            AGENT_MEMORY_VERSION="$2"
            shift 2
            ;;
        --install-dir)
            [ $# -ge 2 ] || { echo "--install-dir requires a path" >&2; exit 1; }
            AGENT_MEMORY_INSTALL_DIR="$2"
            shift 2
            ;;
        --libexec-dir)
            [ $# -ge 2 ] || { echo "--libexec-dir requires a path" >&2; exit 1; }
            AGENT_MEMORY_LIBEXEC_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

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

ASSET_BASE="agent-memory-${os_slug}-${arch_slug}"
ASSET="${ASSET_BASE}.tar.gz"

# Where to extract the unpacked PyInstaller onedir bundle. The download is a
# ~180MB tarball that extracts to ~530MB. We extract once into LIBEXEC_DIR
# and symlink INSTALL_DIR/agent-memory at the bootloader inside, so cold
# start is fast (no per-call extraction) and multiple installed versions
# can coexist for `agent-memory upgrade`.
LIBEXEC_DIR="${AGENT_MEMORY_LIBEXEC_DIR:-$HOME/.local/share/agent-memory}"

bold "Installing agent-memory"
info "host:        ${os_slug}-${arch_slug}"
if [ -n "$LOCAL_TARBALL" ]; then
    info "source:      local tarball ${LOCAL_TARBALL}"
else
    info "repo:        ${REPO}"
fi
info "binary link: ${INSTALL_DIR}/agent-memory"
info "bundle dir:  ${LIBEXEC_DIR}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

if [ -n "$LOCAL_TARBALL" ]; then
    # --- Local tarball mode (dev/CI smoke test) -------------------------------
    [ -f "$LOCAL_TARBALL" ] || die "local tarball does not exist: $LOCAL_TARBALL"
    VERSION="${AGENT_MEMORY_VERSION:-local-$(date +%s)}"
    info "version:     ${VERSION} (local install)"
    cp "$LOCAL_TARBALL" "${TMPDIR}/${ASSET}"
    info "skipping checksum verification (local install mode)"
else
    # --- Resolve version --------------------------------------------------
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
    info "version:     $VERSION"

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
fi

# --- Extract bundle + install symlink ----------------------------------------
need_cmd tar

VERSION_DIR="${LIBEXEC_DIR}/${VERSION}"
mkdir -p "$LIBEXEC_DIR"

# If this exact version is already extracted, reuse it. Otherwise extract fresh.
if [ -d "$VERSION_DIR" ]; then
    info "bundle for ${VERSION} already extracted at ${VERSION_DIR}; reusing"
else
    info "extracting bundle to ${VERSION_DIR}"
    EXTRACT_TMP="$(mktemp -d)"
    tar -xzf "${TMPDIR}/${ASSET}" -C "$EXTRACT_TMP"
    if [ ! -d "${EXTRACT_TMP}/agent-memory" ]; then
        rm -rf "$EXTRACT_TMP"
        die "tarball did not contain expected agent-memory/ directory"
    fi
    # Atomic move into the version directory.
    mv "${EXTRACT_TMP}/agent-memory" "$VERSION_DIR"
    rm -rf "$EXTRACT_TMP"
fi

BUNDLE_BINARY="${VERSION_DIR}/agent-memory"
[ -x "$BUNDLE_BINARY" ] || die "extracted bundle is missing the agent-memory bootloader at ${BUNDLE_BINARY}"

mkdir -p "$INSTALL_DIR"
# Replace any existing symlink/file at the target with a fresh symlink at the
# new version's bootloader. We use a symlink so `agent-memory upgrade` can
# atomically swap versions by re-pointing the link.
rm -f "${INSTALL_DIR}/agent-memory"
ln -s "$BUNDLE_BINARY" "${INSTALL_DIR}/agent-memory"

green "installed: ${INSTALL_DIR}/agent-memory -> ${BUNDLE_BINARY}"

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
