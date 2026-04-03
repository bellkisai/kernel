#!/bin/sh
# ================================================================
#  ShrimPK Remote Installer
#  curl -fsSL https://raw.githubusercontent.com/bellkisai/kernel/master/scripts/install-remote.sh | sh
# ================================================================
#
#  Installs pre-built ShrimPK binaries from GitHub Releases.
#  No build tools required -- just curl/wget and tar/unzip.
#
#  Options:
#    --help              Show usage
#    --version VERSION   Install a specific version (e.g. v0.7.0)
#
# ================================================================

set -e

# ---- Constants ----
REPO="bellkisai/kernel"
GITHUB_API="https://api.github.com/repos/${REPO}/releases/latest"
GITHUB_DL="https://github.com/${REPO}/releases/download"
INSTALL_DIR="$HOME/.shrimpk/bin"
DAEMON_PORT=11435
BINARIES="shrimpk shrimpk-mcp shrimpk-daemon shrimpk-app"

# ---- Banner ----
banner() {
    cat <<'BANNER'

   ____  _          _           ____  _  __
  / ___|| |__  _ __(_)_ __ ___|  _ \| |/ /
  \___ \| '_ \| '__| | '_ ` _ \ |_) | ' /
   ___) | | | | |  | | | | | | |  __/| . \
  |____/|_| |_|_|  |_|_| |_| |_|_|   |_|\_\

  Push-based AI memory -- memories find you.

BANNER
}

# ---- Usage ----
usage() {
    cat <<'USAGE'
Usage:
  curl -fsSL https://raw.githubusercontent.com/bellkisai/kernel/master/scripts/install-remote.sh | sh
  curl -fsSL ... | sh -s -- --version v0.7.0

Options:
  --help              Show this message and exit
  --version VERSION   Install a specific release (default: latest)
USAGE
}

# ---- Helpers ----
info()  { printf "  %s\n" "$*"; }
step()  { printf "\n[%s] %s\n" "$1" "$2"; }
warn()  { printf "  WARNING: %s\n" "$*"; }
fail()  { printf "  ERROR: %s\n" "$*" >&2; exit 1; }

# ---- Detect OS and architecture ----
detect_platform() {
    OS_RAW="$(uname -s)"
    ARCH_RAW="$(uname -m)"

    case "$OS_RAW" in
        Linux)
            OS="linux"
            ;;
        Darwin)
            OS="darwin"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            OS="windows"
            ;;
        *)
            fail "Unsupported operating system: $OS_RAW"
            ;;
    esac

    case "$ARCH_RAW" in
        x86_64|amd64)
            ARCH="x86_64"
            ;;
        aarch64|arm64)
            ARCH="aarch64"
            ;;
        *)
            fail "Unsupported architecture: $ARCH_RAW"
            ;;
    esac

    # Validate supported combinations
    if [ "$OS" = "windows" ] && [ "$ARCH" != "x86_64" ]; then
        fail "Windows ARM builds are not available yet. Only x86_64 is supported."
    fi

    # Set archive name and extension
    if [ "$OS" = "windows" ]; then
        ARCHIVE="shrimpk-${OS}-${ARCH}.zip"
        EXT=".exe"
    else
        ARCHIVE="shrimpk-${OS}-${ARCH}.tar.gz"
        EXT=""
    fi

    info "Platform: ${OS}/${ARCH}"
    info "Archive:  ${ARCHIVE}"
}

# ---- Choose download command (curl preferred, wget fallback) ----
detect_downloader() {
    if command -v curl >/dev/null 2>&1; then
        DOWNLOADER="curl"
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER="wget"
    else
        fail "Neither curl nor wget found. Please install one and retry."
    fi
}

# Download a URL to a local file.
# $1 = URL, $2 = output path
download() {
    if [ "$DOWNLOADER" = "curl" ]; then
        curl -fSL --progress-bar -o "$2" "$1"
    else
        wget --show-progress -q -O "$2" "$1"
    fi
}

# Fetch a URL and print its body to stdout (quiet, for API calls).
# $1 = URL
fetch() {
    if [ "$DOWNLOADER" = "curl" ]; then
        curl -fsSL "$1"
    else
        wget -qO- "$1"
    fi
}

# ---- Resolve version (latest from API or user-supplied) ----
resolve_version() {
    if [ -n "$REQUESTED_VERSION" ]; then
        VERSION="$REQUESTED_VERSION"
        info "Requested version: $VERSION"
        return
    fi

    info "Fetching latest release from GitHub..."
    RELEASE_JSON="$(fetch "$GITHUB_API" 2>/dev/null)" || true

    if [ -z "$RELEASE_JSON" ]; then
        fail "Could not reach GitHub API. Check your network or specify --version."
    fi

    VERSION="$(printf '%s' "$RELEASE_JSON" | grep '"tag_name"' | sed 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')"

    if [ -z "$VERSION" ]; then
        fail "No releases found for $REPO. Specify --version to install a pre-release."
    fi

    info "Latest version: $VERSION"
}

# ---- Download and extract ----
download_and_extract() {
    URL="${GITHUB_DL}/${VERSION}/${ARCHIVE}"
    TMPDIR="$(mktemp -d)"
    TMPFILE="${TMPDIR}/${ARCHIVE}"

    info "Downloading ${URL}"
    download "$URL" "$TMPFILE"

    mkdir -p "$INSTALL_DIR"

    info "Extracting to ${INSTALL_DIR}"
    if [ "$OS" = "windows" ]; then
        # Windows: use unzip (ships with Git-for-Windows / MSYS2)
        if command -v unzip >/dev/null 2>&1; then
            unzip -oq "$TMPFILE" -d "$INSTALL_DIR"
        elif command -v 7z >/dev/null 2>&1; then
            7z x -y -o"$INSTALL_DIR" "$TMPFILE" >/dev/null
        else
            fail "Neither unzip nor 7z found. Install one and retry."
        fi
    else
        tar -xzf "$TMPFILE" -C "$INSTALL_DIR"
    fi

    # Clean up temp files
    rm -rf "$TMPDIR"

    # Make binaries executable (no-op on Windows but harmless)
    for bin in $BINARIES; do
        if [ -f "${INSTALL_DIR}/${bin}${EXT}" ]; then
            chmod +x "${INSTALL_DIR}/${bin}${EXT}" 2>/dev/null || true
        fi
    done
}

# ---- Set up PATH ----
setup_path() {
    # Export for the current session immediately
    export PATH="${INSTALL_DIR}:${PATH}"

    # On Windows (MSYS/Git Bash), also add to the Windows user PATH
    if [ "$OS" = "windows" ]; then
        WINPATH="$(cygpath -w "$INSTALL_DIR" 2>/dev/null || echo "$INSTALL_DIR")"
        powershell.exe -Command \
            "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ';${WINPATH}', 'User')" \
            2>/dev/null && info "Added to Windows user PATH: ${WINPATH}" \
                       || warn "Could not update Windows PATH. Add manually: ${WINPATH}"
        return
    fi

    # Unix: append to shell profile if not already present
    PROFILE=""
    for candidate in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile"; do
        if [ -f "$candidate" ]; then
            PROFILE="$candidate"
            break
        fi
    done
    # Default to .profile if nothing exists
    PROFILE="${PROFILE:-$HOME/.profile}"

    if grep -q 'shrimpk/bin' "$PROFILE" 2>/dev/null; then
        info "PATH already configured in ${PROFILE}"
    else
        printf '\n# ShrimPK\nexport PATH="$HOME/.shrimpk/bin:$PATH"\n' >> "$PROFILE"
        info "Added to PATH in ${PROFILE}"
    fi
}

# ---- Register MCP server with Claude Code ----
register_mcp() {
    if ! command -v claude >/dev/null 2>&1; then
        warn "'claude' CLI not found. After installing Claude Code, register manually:"
        info "  claude mcp add --transport stdio --scope user shrimpk -- ${INSTALL_DIR}/shrimpk-mcp${EXT}"
        return
    fi

    claude mcp remove shrimpk 2>/dev/null || true
    claude mcp add --transport stdio --scope user shrimpk -- "${INSTALL_DIR}/shrimpk-mcp${EXT}" 2>&1
    info "MCP server registered with Claude Code"
}

# ---- Start daemon ----
start_daemon() {
    # Kill any existing daemon
    if [ "$OS" = "windows" ]; then
        taskkill //F //IM shrimpk-daemon.exe 2>/dev/null || true
    else
        pkill -f shrimpk-daemon 2>/dev/null || true
    fi
    sleep 1

    # Launch in background
    "${INSTALL_DIR}/shrimpk-daemon${EXT}" >/dev/null 2>&1 &
    DAEMON_PID=$!
    info "Daemon starting (pid ${DAEMON_PID}) ..."

    # Wait for it to become healthy (up to 15 seconds)
    TRIES=0
    while [ "$TRIES" -lt 6 ]; do
        sleep 3
        if check_health_quiet; then
            info "Daemon healthy on http://127.0.0.1:${DAEMON_PORT}"
            return
        fi
        TRIES=$((TRIES + 1))
    done

    warn "Daemon may still be loading (first run downloads the embedding model)."
    info "Check later:  curl http://127.0.0.1:${DAEMON_PORT}/health"
}

# Silent health check -- returns 0 if healthy.
check_health_quiet() {
    if [ "$DOWNLOADER" = "curl" ]; then
        curl -s --max-time 2 "http://127.0.0.1:${DAEMON_PORT}/health" >/dev/null 2>&1
    else
        wget -q --timeout=2 -O /dev/null "http://127.0.0.1:${DAEMON_PORT}/health" 2>/dev/null
    fi
}

# ---- Install auto-start ----
install_autostart() {
    "${INSTALL_DIR}/shrimpk-daemon${EXT}" --install 2>&1 || true
    info "Auto-start configured (daemon on login)"
}

# ---- Verify installation ----
verify() {
    PASS=0
    TOTAL=3

    # 1. Binary exists
    if [ -f "${INSTALL_DIR}/shrimpk${EXT}" ]; then
        info "[ok] shrimpk CLI installed"
        PASS=$((PASS + 1))
    else
        info "[!!] shrimpk CLI not found at ${INSTALL_DIR}"
    fi

    # 2. Daemon health
    if check_health_quiet; then
        info "[ok] Daemon running on port ${DAEMON_PORT}"
        PASS=$((PASS + 1))
    else
        info "[..] Daemon still starting (may need model download)"
    fi

    # 3. MCP registration
    if command -v claude >/dev/null 2>&1; then
        if claude mcp list 2>/dev/null | grep -q "shrimpk"; then
            info "[ok] MCP server registered"
            PASS=$((PASS + 1))
        else
            info "[!!] MCP server not registered"
        fi
    else
        info "[--] Skipped MCP check (claude CLI not installed)"
        PASS=$((PASS + 1))
    fi

    info ""
    info "${PASS}/${TOTAL} checks passed"
}

# ---- Success message ----
print_success() {
    cat <<MSG

  =============================================
  ShrimPK installed!
  =============================================

  Daemon:  http://localhost:${DAEMON_PORT}
  CLI:     shrimpk store / echo / stats
  Data:    ~/.shrimpk-kernel/

  Restart your terminal, then try:
    shrimpk status

MSG
}

# ---- Parse arguments ----
parse_args() {
    REQUESTED_VERSION=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --help|-h)
                banner
                usage
                exit 0
                ;;
            --version|-v)
                if [ -z "$2" ] || [ "$(printf '%.1s' "$2")" = "-" ]; then
                    fail "--version requires a value (e.g. --version v0.7.0)"
                fi
                REQUESTED_VERSION="$2"
                shift 2
                ;;
            *)
                fail "Unknown option: $1  (use --help for usage)"
                ;;
        esac
    done
}

# ---- Main ----
main() {
    parse_args "$@"
    banner

    step "1/7" "Detecting platform"
    detect_platform
    detect_downloader

    step "2/7" "Resolving version"
    resolve_version

    step "3/7" "Downloading and extracting"
    download_and_extract

    step "4/7" "Configuring PATH"
    setup_path

    step "5/7" "Registering MCP server"
    register_mcp

    step "6/7" "Starting daemon"
    start_daemon
    install_autostart

    step "7/7" "Verifying installation"
    verify

    print_success
}

main "$@"
