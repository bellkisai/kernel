#!/usr/bin/env bash
# KS77 QA Preflight — run BEFORE any Playwright tests
# Exits non-zero if any check fails.

DAEMON_URL="http://localhost:11435"
VITE_URL="http://localhost:5173"
REQUIRED_VERSION="0.7.0"
MIN_MEMORIES=10
TIMEOUT=5

pass=0
fail=0

ok() {
  echo "[PASS] $1: $2"
  ((pass++)) || true
}

ko() {
  echo "[FAIL] $1: $2 — $3"
  ((fail++)) || true
}

echo "=== KS77 QA Preflight ==="
echo ""

# PF-01: Daemon health
HEALTH=$(curl -s "$DAEMON_URL/health" --max-time $TIMEOUT 2>/dev/null || echo '{}')
VERSION=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version',''))" 2>/dev/null || echo "")
MEMORIES=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('memories',0))" 2>/dev/null || echo "0")

if [ -z "$VERSION" ]; then
  ko "PF-01a" "Daemon reachable" "not responding at $DAEMON_URL"
  echo "ABORT: Daemon is down."; exit 1
fi
ok "PF-01a" "Daemon reachable"

if [ "$VERSION" != "$REQUIRED_VERSION" ]; then
  ko "PF-01b" "Daemon version" "got $VERSION, need $REQUIRED_VERSION"
  echo "ABORT: Wrong daemon version."; exit 1
fi
ok "PF-01b" "Daemon version $REQUIRED_VERSION"

if [ "$MEMORIES" -lt "$MIN_MEMORIES" ]; then
  ko "PF-01c" "Memory count" "got $MEMORIES, need >=$MIN_MEMORIES"
  echo "ABORT: Not enough memories."; exit 1
fi
ok "PF-01c" "Daemon has $MEMORIES memories (>=$MIN_MEMORIES)"

# PF-02: Overview endpoint
OV_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$DAEMON_URL/api/graph/overview" \
  -H "Content-Type: application/json" \
  -d '{"min_members":3,"max_clusters":5}' \
  --max-time $TIMEOUT 2>/dev/null || echo "000")
if [ "$OV_STATUS" = "200" ]; then
  ok "PF-02" "Overview endpoint"
else
  ko "PF-02" "Overview endpoint" "HTTP $OV_STATUS"
  echo "ABORT: Graph API unavailable."; exit 1
fi

# PF-03: Echo endpoint
ECHO_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$DAEMON_URL/api/echo" \
  -H "Content-Type: application/json" \
  -d '{"query":"test","max_results":1}' \
  --max-time $TIMEOUT 2>/dev/null || echo "000")
if [ "$ECHO_STATUS" = "200" ]; then
  ok "PF-03" "Echo endpoint"
else
  ko "PF-03" "Echo endpoint" "HTTP $ECHO_STATUS"
  echo "ABORT: Echo API unavailable."; exit 1
fi

# PF-04: Memory get endpoint
MEM_ID=$(curl -s -X POST "$DAEMON_URL/api/graph/overview" \
  -H "Content-Type: application/json" \
  -d '{"min_members":3,"max_clusters":1}' \
  --max-time $TIMEOUT 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['clusters'][0]['top_members'][0]['id'])" 2>/dev/null || echo "")
if [ -n "$MEM_ID" ]; then
  MG_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$DAEMON_URL/api/memory_get" \
    -H "Content-Type: application/json" \
    -d "{\"memory_id\":\"$MEM_ID\"}" \
    --max-time $TIMEOUT 2>/dev/null || echo "000")
  if [ "$MG_STATUS" = "200" ]; then
    ok "PF-04" "Memory get endpoint"
  else
    ko "PF-04" "Memory get endpoint" "HTTP $MG_STATUS"
  fi
else
  ko "PF-04" "Memory get endpoint" "Could not extract memory ID"
fi

# PF-05: Vite dev server
VITE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$VITE_URL" --max-time $TIMEOUT 2>/dev/null || echo "000")
if [ "$VITE_STATUS" = "200" ]; then
  ok "PF-05" "Vite dev server"
else
  ko "PF-05" "Vite dev server" "HTTP $VITE_STATUS"
  echo "ABORT: Frontend not running."; exit 1
fi

echo ""
echo "=== Preflight: $pass passed, $fail failed ==="

if [ "$fail" -gt 0 ]; then
  echo "WARNING: Some non-critical checks failed. Review before proceeding."
  exit 1
fi

echo "ALL CLEAR: Proceed with Playwright tests."
exit 0
