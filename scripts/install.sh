#!/bin/bash
# ShrimPK installer — sets up CLI, MCP server, and auto-memory hooks
#
# Usage: bash scripts/install.sh
#
# What this does:
# 1. Builds shrimpk CLI and shrimpk-mcp in release mode
# 2. Registers shrimpk-mcp as a global MCP server with Claude Code
# 3. Configures auto-memory hooks (auto-store + auto-echo on every prompt)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SHRIMPK="$KERNEL_DIR/target/release/shrimpk"
SHRIMPK_MCP="$KERNEL_DIR/target/release/shrimpk-mcp"
HOOK_SCRIPT="$SCRIPT_DIR/shrimpk-hook.sh"
SETTINGS_FILE="$HOME/.claude/settings.json"

echo "[shrimpk] Installing ShrimPK Echo Memory..."
echo ""

# Step 1: Build release binaries
echo "[shrimpk] Step 1/3: Building release binaries..."
cd "$KERNEL_DIR"
cargo build --release -p shrimpk-cli -p shrimpk-mcp
echo "[shrimpk] Built: $SHRIMPK"
echo "[shrimpk] Built: $SHRIMPK_MCP"
echo ""

# Step 2: Register MCP server globally with Claude Code
echo "[shrimpk] Step 2/3: Registering MCP server with Claude Code..."
if command -v claude &> /dev/null; then
    # Remove existing registration if any (idempotent)
    claude mcp remove shrimpk 2>/dev/null || true
    claude mcp add --transport stdio --scope user shrimpk -- "$SHRIMPK_MCP"
    echo "[shrimpk] MCP server registered globally (works from any directory)"
else
    echo "[shrimpk] WARNING: 'claude' CLI not found. Register manually:"
    echo "  claude mcp add --transport stdio --scope user shrimpk -- $SHRIMPK_MCP"
fi
echo ""

# Step 3: Configure auto-memory hooks
echo "[shrimpk] Step 3/3: Configuring auto-memory hooks..."

# Ensure settings directory exists
mkdir -p "$(dirname "$SETTINGS_FILE")"

# Read existing settings, merge hooks, write back (NEVER overwrite)
if [ -f "$SETTINGS_FILE" ]; then
    # Check if hooks already configured
    if grep -q "shrimpk-hook" "$SETTINGS_FILE" 2>/dev/null; then
        echo "[shrimpk] Auto-memory hooks already configured."
    else
        # Use python to safely merge JSON (append to existing, never overwrite)
        python3 -c "
import json, sys

settings_path = '$SETTINGS_FILE'
hook_script = '$HOOK_SCRIPT'

with open(settings_path) as f:
    settings = json.load(f)

# Initialize hooks structure if not present
if 'hooks' not in settings:
    settings['hooks'] = {}
if 'UserPromptSubmit' not in settings['hooks']:
    settings['hooks']['UserPromptSubmit'] = []

# Add shrimpk hook (check for duplicates)
hook_entry = {
    'matcher': '',
    'command': f'bash {hook_script} \"\$PROMPT\"'
}

# Only add if not already present
existing_commands = [h.get('command', '') for h in settings['hooks']['UserPromptSubmit']]
if hook_entry['command'] not in existing_commands:
    settings['hooks']['UserPromptSubmit'].append(hook_entry)

with open(settings_path, 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')

print('[shrimpk] Auto-memory hooks configured in', settings_path)
" 2>&1
    fi
else
    # Create new settings file with hooks
    python3 -c "
import json

settings = {
    'hooks': {
        'UserPromptSubmit': [{
            'matcher': '',
            'command': 'bash $HOOK_SCRIPT \"\$PROMPT\"'
        }]
    }
}

with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')

print('[shrimpk] Created settings with auto-memory hooks')
" 2>&1
fi

echo ""
echo "[shrimpk] Installation complete!"
echo ""
echo "  CLI:         $SHRIMPK"
echo "  MCP server:  $SHRIMPK_MCP (registered globally)"
echo "  Hook script: $HOOK_SCRIPT"
echo "  Data dir:    ~/.shrimpk-kernel/"
echo ""
echo "  ShrimPK is now active. Every Claude Code conversation"
echo "  automatically stores context and recalls relevant memories."
echo "  The user never needs to think about memory — it just works."
echo ""
echo "  Test it: start a new Claude Code session and have a normal"
echo "  conversation. Memories from this session will appear in the next."
