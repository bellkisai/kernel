#!/bin/bash
# ShrimPK auto-memory hook for Claude Code
# Uses the daemon HTTP API — no process spawning, no model loading.
#
# Usage: shrimpk-hook.sh "$PROMPT"

DAEMON="http://127.0.0.1:11435"
PROMPT="$1"

# Skip empty prompts
if [ -z "$PROMPT" ]; then
    exit 0
fi

# Auth header (if SHRIMPK_AUTH_TOKEN is set)
AUTH_HEADER=""
if [ -n "$SHRIMPK_AUTH_TOKEN" ]; then
    AUTH_HEADER="-H \"Authorization: Bearer $SHRIMPK_AUTH_TOKEN\""
fi

# Check if daemon is running (fast TCP probe)
if ! curl -s --max-time 0.2 $AUTH_HEADER "$DAEMON/health" > /dev/null 2>&1; then
    exit 0
fi

# Auto-store: fire-and-forget via daemon HTTP (instant, ~1ms)
curl -s -X POST $AUTH_HEADER "$DAEMON/api/store" \
    -H "Content-Type: application/json" \
    -d "{\"text\":$(echo "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),\"source\":\"auto\"}" \
    > /dev/null 2>&1 &

# Auto-echo: surface relevant memories via daemon HTTP (instant, ~5ms)
RESULTS=$(curl -s -X POST $AUTH_HEADER "$DAEMON/api/echo" \
    -H "Content-Type: application/json" \
    -d "{\"query\":$(echo "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),\"max_results\":5}" \
    2>/dev/null)

# Parse and output if there are results
COUNT=$(echo "$RESULTS" | python3 -c "import json,sys; print(json.load(sys.stdin).get('count',0))" 2>/dev/null)
if [ "$COUNT" != "0" ] && [ -n "$COUNT" ]; then
    echo "[Echo Memory] Relevant context from previous conversations:"
    echo "$RESULTS" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for r in data.get('results', [])[:5]:
        sim = r.get('similarity', 0)
        print(f'  - {r[\"content\"]} (similarity: {sim:.0%})')
except:
    pass
" 2>/dev/null
fi
