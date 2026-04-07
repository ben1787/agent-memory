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
#   5. Picks an install directory that's ALREADY on $PATH so the binary works
#      immediately without restarting any shells. Priority:
#         /opt/homebrew/bin → /usr/local/bin → first writable dir on PATH
#         → ~/.local/bin (with rc-file edits + restart warning).
#      Override with $AGENT_MEMORY_INSTALL_DIR.
#   6. Symlinks the binary into the chosen install directory.
#   7. Prints next steps
#
# Environment overrides:
#   AGENT_MEMORY_VERSION       Tag to install, e.g. v0.2.1. Defaults to latest.
#   AGENT_MEMORY_INSTALL_DIR   Install directory. Defaults to the first dir
#                              already on $PATH that's writable without sudo
#                              (/opt/homebrew/bin on Apple Silicon, etc.),
#                              falling back to ~/.local/bin.
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

# Pick an install dir that's ALREADY on $PATH so the binary works immediately
# in every shell — current terminal, new terminals, and any subprocess shells
# spawned by AI agents that don't source rc files (e.g. `/bin/sh -c`).
#
# Priority:
#   1. AGENT_MEMORY_INSTALL_DIR (explicit override) — always wins.
#   2. /opt/homebrew/bin if it exists, is on PATH, and is user-writable
#      (Apple Silicon Homebrew default — owned by the user, no sudo needed).
#   3. /usr/local/bin if it exists, is on PATH, and is writable without sudo
#      (Intel Mac Homebrew, many Linux distros).
#   4. Any other directory on PATH that the user can write to.
#   5. Fall back to ~/.local/bin and edit shell rc files (won't work for
#      already-running agents — see warning at end of install).
pick_install_dir() {
    # Try the canonical "owned by user, on PATH by default" Homebrew prefixes
    # first. We check writability but NOT $PATH, because the install script
    # may itself be running in a stripped-PATH subprocess (e.g. piped from
    # curl into /bin/sh), while the user's actual interactive shell has
    # /opt/homebrew/bin on its PATH thanks to /etc/zprofile.
    local d
    for d in /opt/homebrew/bin /usr/local/bin; do
        if [ -d "$d" ] && [ -w "$d" ]; then
            echo "$d"
            return 0
        fi
    done
    # Otherwise walk the current $PATH for any user-writable dir we can use.
    local IFS=":"
    for d in $PATH; do
        if [ -n "$d" ] && [ -d "$d" ] && [ -w "$d" ]; then
            case "$d" in
                "$HOME/.local/bin"|"$HOME/bin") continue ;;
                /tmp*|/var/tmp*) continue ;;
            esac
            echo "$d"
            return 0
        fi
    done
    return 1
}

if [ -n "${AGENT_MEMORY_INSTALL_DIR:-}" ]; then
    INSTALL_DIR="$AGENT_MEMORY_INSTALL_DIR"
    INSTALL_DIR_ON_PATH="explicit"
elif INSTALL_DIR="$(pick_install_dir)"; then
    INSTALL_DIR_ON_PATH="yes"
else
    INSTALL_DIR="$HOME/.local/bin"
    INSTALL_DIR_ON_PATH="no"
fi

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

# Intel macOS is not in the release matrix — GitHub retired the free macos-13
# runner image so we can't build those binaries. Tell Intel-Mac users up front
# to install from source instead of failing on a 404 halfway through.
if [ "$os_slug" = "macos" ] && [ "$arch_slug" = "x86_64" ]; then
    die "Intel macOS is not a published release target. Install from source with: uv tool install git+https://github.com/ben1787/agent-memory"
fi

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
# If we picked a dir that's already on PATH (the common case on macOS thanks
# to Homebrew), we're done — agent-memory will work in every shell, including
# subprocess shells launched by AI agents.
#
# If we had to fall back to ~/.local/bin, edit every shell rc file we can
# find AND warn loudly that already-running processes (terminals, agent
# harnesses) need to be restarted.
case "$INSTALL_DIR_ON_PATH" in
    yes)
        green "installed onto existing PATH dir: ${INSTALL_DIR}"
        ;;
    explicit)
        case ":$PATH:" in
            *":${INSTALL_DIR}:"*) green "PATH already contains ${INSTALL_DIR}" ;;
            *) red "warning: ${INSTALL_DIR} is NOT on PATH — you set it explicitly via AGENT_MEMORY_INSTALL_DIR; add it to PATH yourself" ;;
        esac
        ;;
    no)
        red "warning: no writable directory on PATH was found, fell back to ${INSTALL_DIR}"
        # Edit every plausible rc file so future shells of any flavor pick it
        # up. Existing processes still need to be restarted.
        PATH_LINE="export PATH=\"${INSTALL_DIR}:\$PATH\""
        for RC_FILE in \
            "${ZDOTDIR:-$HOME}/.zshrc" \
            "${ZDOTDIR:-$HOME}/.zshenv" \
            "$HOME/.bashrc" \
            "$HOME/.bash_profile" \
            "$HOME/.profile"
        do
            [ -e "$RC_FILE" ] || [ "$RC_FILE" = "${ZDOTDIR:-$HOME}/.zshrc" ] || continue
            if [ -f "$RC_FILE" ] && grep -Fxq "$PATH_LINE" "$RC_FILE"; then
                info "PATH line already present in ${RC_FILE}"
            else
                printf '\n# added by agent-memory installer\n%s\n' "$PATH_LINE" >> "$RC_FILE"
                green "appended PATH line to ${RC_FILE}"
            fi
        done
        echo
        red "==============================================================================="
        red "  IMPORTANT: ${INSTALL_DIR} was not on PATH."
        red "  agent-memory was installed but you must RESTART any already-running"
        red "  terminals AND any AI agents (Claude Code, Codex, etc.) for them to find"
        red "  the binary. Sourcing your rc file does NOT fix existing subprocess shells."
        red "==============================================================================="
        echo
        ;;
esac

# --- Ensure Claude Code subprocess shells can find the binary ----------------
# Reason: Claude Desktop (the macOS .app) is launched by launchd with a
# stripped PATH like `/usr/bin:/bin:/usr/sbin:/sbin`. claude-code captures
# that PATH into a shell snapshot at ~/.claude/shell-snapshots/snapshot-*.sh
# and sources it on every Bash-tool invocation, so subprocess shells can't
# see anything under /opt/homebrew/bin, /usr/local/bin, ~/.local/bin, etc.
# Even moving the binary to /usr/local/bin wouldn't help — that dir isn't on
# the stripped PATH either.
#
# The only mechanism guaranteed-by-design to survive the snapshot is a
# SessionStart hook that appends to $CLAUDE_ENV_FILE. claude-code sources
# that file AFTER the snapshot on every Bash call, so it can't be clobbered.
# See: https://code.claude.com/docs/en/hooks-guide.md
#
# We only install this if ~/.claude exists (user is actually a Claude Code
# user). No-op otherwise.
install_claude_code_session_hook() {
    local claude_dir="$HOME/.claude"
    [ -d "$claude_dir" ] || return 0

    local settings_file="$claude_dir/settings.json"
    local path_prefix="${INSTALL_DIR}:\$HOME/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin"
    # Marker string embedded in the hook command so re-running the installer
    # finds and skips the existing entry instead of duplicating it. If we
    # ever need to rev the hook format, bump this tag.
    local marker="AGENT_MEMORY_INSTALLER_PATH_HOOK_v1"
    local hook_command="# ${marker}
printf 'export PATH=\"%s:\$PATH\"\\n' \"${path_prefix}\" >> \"\$CLAUDE_ENV_FILE\""

    # Prefer python3 (ships with macOS 10.15+, nearly all Linux distros).
    # Fall back to manual instructions if it's not available — we don't want
    # to depend on jq, which is not always present.
    if ! command -v python3 >/dev/null 2>&1; then
        info "python3 not found — skipping automatic Claude Code hook install"
        info "  to enable agent-memory in Claude Desktop subprocess shells, add this to ~/.claude/settings.json:"
        info '  {"hooks":{"SessionStart":[{"hooks":[{"type":"command","command":"'"$hook_command"'"}]}]}}'
        return 0
    fi

    SETTINGS_FILE="$settings_file" MARKER="$marker" HOOK_COMMAND="$hook_command" python3 <<'PY'
import json, os, sys
from pathlib import Path

settings_path = Path(os.environ["SETTINGS_FILE"])
marker = os.environ["MARKER"]
hook_command = os.environ["HOOK_COMMAND"]

if settings_path.exists():
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  warning: {settings_path} is not valid JSON ({exc}); leaving it alone", file=sys.stderr)
        sys.exit(0)
else:
    data = {}

hooks = data.setdefault("hooks", {})
session_start = hooks.setdefault("SessionStart", [])

# Idempotency: if any existing SessionStart hook already contains our marker,
# we're done. This lets users re-run the installer freely.
for group in session_start:
    for entry in group.get("hooks", []) if isinstance(group, dict) else []:
        if isinstance(entry, dict) and marker in str(entry.get("command", "")):
            print(f"  Claude Code SessionStart hook already installed in {settings_path}")
            sys.exit(0)

session_start.append({
    "hooks": [
        {
            "type": "command",
            "command": hook_command,
        }
    ]
})

# Write atomically: write to a tempfile alongside, then rename, so a crash
# mid-write never leaves the user with a broken settings.json.
tmp = settings_path.with_suffix(settings_path.suffix + ".agent-memory.tmp")
tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
tmp.replace(settings_path)
print(f"  added SessionStart PATH hook to {settings_path}")
PY
}

install_claude_code_session_hook

# --- Verify the install -------------------------------------------------------
if "${INSTALL_DIR}/agent-memory" --version >/dev/null 2>&1; then
    green "verified: $("${INSTALL_DIR}/agent-memory" --version)"
fi

echo
bold "agent-memory installed."
info "next: cd into a project and run \`agent-memory init\`"
info "then: \`agent-memory save\`, \`agent-memory recall\`, \`agent-memory list --recent\`"
info "docs: https://github.com/${REPO}#readme"
