#!/bin/bash
# ============================================================
# ShrimPK Full Installer
# One command. Everything works.
# ============================================================
#
# Usage: bash scripts/install.sh
#
# What this does:
# 1. Builds all 4 release binaries
# 2. Copies them to ~/.shrimpk/bin/ and adds to PATH
# 3. Registers MCP server globally with Claude Code
# 4. Starts the daemon
# 5. Installs auto-start (daemon on login)
# 6. Verifies everything works

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$HOME/.shrimpk/bin"
SETTINGS_FILE="$HOME/.claude/settings.json"

echo ""
echo "  🦐 ShrimPK Installer"
echo "  ====================="
echo "  Push-based AI memory where memories find you."
echo ""

# ---- Step 1: Build release binaries ----
echo "[1/6] Building release binaries..."
cd "$KERNEL_DIR"
cargo build --release -p shrimpk-cli -p shrimpk-mcp -p shrimpk-daemon -p shrimpk-tray 2>&1 | tail -3
echo ""

# ---- Step 2: Install to PATH ----
echo "[2/6] Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OS" == "Windows_NT" ]]; then
    EXT=".exe"
else
    EXT=""
fi

cp "$KERNEL_DIR/target/release/shrimpk${EXT}" "$INSTALL_DIR/"
cp "$KERNEL_DIR/target/release/shrimpk-mcp${EXT}" "$INSTALL_DIR/"
cp "$KERNEL_DIR/target/release/shrimpk-daemon${EXT}" "$INSTALL_DIR/"
cp "$KERNEL_DIR/target/release/shrimpk-tray${EXT}" "$INSTALL_DIR/"

echo "  Copied 4 binaries to $INSTALL_DIR"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    # Detect shell profile
    if [ -f "$HOME/.bashrc" ]; then
        PROFILE="$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        PROFILE="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        PROFILE="$HOME/.bash_profile"
    elif [ -f "$HOME/.profile" ]; then
        PROFILE="$HOME/.profile"
    else
        PROFILE="$HOME/.bashrc"
    fi

    # Check if already in profile
    if ! grep -q "shrimpk/bin" "$PROFILE" 2>/dev/null; then
        echo "" >> "$PROFILE"
        echo "# ShrimPK" >> "$PROFILE"
        echo "export PATH=\"\$HOME/.shrimpk/bin:\$PATH\"" >> "$PROFILE"
        echo "  Added to PATH in $PROFILE"
    fi

    # Also set for current session
    export PATH="$INSTALL_DIR:$PATH"

    # Windows: also add to user PATH via PowerShell
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OS" == "Windows_NT" ]]; then
        WINPATH=$(cygpath -w "$INSTALL_DIR")
        powershell.exe -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';$WINPATH', 'User')" 2>/dev/null && \
            echo "  Added to Windows user PATH: $WINPATH" || \
            echo "  WARNING: Could not add to Windows PATH. Add manually: $WINPATH"
    fi
fi
echo ""

# ---- Step 3: Register MCP server ----
echo "[3/6] Registering MCP server with Claude Code..."
if command -v claude &> /dev/null; then
    claude mcp remove shrimpk 2>/dev/null || true
    claude mcp add --transport stdio --scope user shrimpk -- "$INSTALL_DIR/shrimpk-mcp${EXT}" 2>&1
    echo "  MCP server registered globally"
else
    echo "  WARNING: 'claude' CLI not found. Register manually after installing Claude Code:"
    echo "    claude mcp add --transport stdio --scope user shrimpk -- $INSTALL_DIR/shrimpk-mcp${EXT}"
fi
echo ""

# ---- Step 4: Start daemon ----
echo "[4/6] Starting daemon..."
# Kill any existing daemon
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OS" == "Windows_NT" ]]; then
    taskkill //F //IM shrimpk-daemon.exe 2>/dev/null || true
else
    pkill -f shrimpk-daemon 2>/dev/null || true
fi
sleep 1

# Start daemon in background
"$INSTALL_DIR/shrimpk-daemon${EXT}" &
DAEMON_PID=$!
sleep 5

# Check if running
if curl -s --max-time 2 http://127.0.0.1:11435/health > /dev/null 2>&1; then
    echo "  Daemon running on localhost:11435 (pid $DAEMON_PID)"
else
    echo "  WARNING: Daemon may still be loading (first run downloads 23MB model)"
    echo "  Check with: curl localhost:11435/health"
fi
echo ""

# ---- Step 5: Install auto-start ----
echo "[5/6] Installing auto-start (daemon on login)..."
"$INSTALL_DIR/shrimpk-daemon${EXT}" --install 2>&1
echo ""

# ---- Step 6: Verify ----
echo "[6/6] Verifying installation..."
echo ""

# Check shrimpk command
if command -v shrimpk &> /dev/null || [ -f "$INSTALL_DIR/shrimpk${EXT}" ]; then
    echo "  ✓ shrimpk CLI installed"
else
    echo "  ✗ shrimpk CLI not found"
fi

# Check daemon
if curl -s --max-time 2 http://127.0.0.1:11435/health > /dev/null 2>&1; then
    HEALTH=$(curl -s http://127.0.0.1:11435/health)
    MEMORIES=$(echo "$HEALTH" | python3 -c "import json,sys; print(json.load(sys.stdin).get('memories',0))" 2>/dev/null || echo "?")
    echo "  ✓ Daemon running (${MEMORIES} memories)"
else
    echo "  ⏳ Daemon starting (first run may take 30s for model download)"
fi

# Check MCP
if command -v claude &> /dev/null; then
    if claude mcp list 2>/dev/null | grep -q "shrimpk"; then
        echo "  ✓ MCP server registered"
    else
        echo "  ✗ MCP server not registered"
    fi
fi

echo ""
echo "  ============================================"
echo "  🦐 ShrimPK installed successfully!"
echo "  ============================================"
echo ""
echo "  Daemon:  http://localhost:11435"
echo "  CLI:     shrimpk store / echo / stats / status"
echo "  Tray:    shrimpk-tray (🦐 in taskbar)"
echo "  Data:    ~/.shrimpk-kernel/"
echo ""
echo "  NOTE: Restart your terminal for PATH to take effect."
echo "  Then try: shrimpk status"
echo ""
