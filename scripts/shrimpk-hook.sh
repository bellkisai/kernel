#!/bin/bash
# ShrimPK auto-memory hook for Claude Code
# Runs on every user prompt submission via UserPromptSubmit hook
#
# 1. Auto-store: captures every message (ShrimPK classifies + decays)
# 2. Auto-echo: surfaces relevant memories as context
#
# Usage: shrimpk-hook.sh "$PROMPT"

SHRIMPK="C:/Users/lior1/bellkis/kernel/target/release/shrimpk"
PROMPT="$1"

# Skip empty prompts
if [ -z "$PROMPT" ]; then
    exit 0
fi

# Auto-store: fire-and-forget (don't block prompt)
"$SHRIMPK" store --quiet "$PROMPT" --source auto 2>/dev/null &

# Auto-echo: surface relevant memories (output becomes context)
RESULTS=$("$SHRIMPK" echo --json "$PROMPT" 2>/dev/null)

# Only output if there are results (not empty array)
if [ "$RESULTS" != "[]" ] && [ -n "$RESULTS" ]; then
    echo "[Echo Memory] Relevant context from previous conversations:"
    echo "$RESULTS" | python3 -c "
import json, sys
try:
    results = json.load(sys.stdin)
    for r in results[:5]:
        print(f'  - {r[\"content\"]} (similarity: {r[\"similarity\"]:.0%})')
except:
    pass
" 2>/dev/null
fi
