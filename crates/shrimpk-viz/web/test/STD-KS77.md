# STD-KS77: Software Test Description — Test Cases

## Preflight (GATE)

### PF-01: Daemon health
- **Precondition:** None
- **Step:** `curl -s http://localhost:11435/health --max-time 5`
- **Expected:** 200 OK, JSON with `version: "0.7.0"`, `memories >= 10`
- **Fail:** ABORT all testing. Report daemon version and memory count.

### PF-02: Overview endpoint
- **Precondition:** PF-01 passed
- **Step:** `curl -s -X POST http://localhost:11435/api/graph/overview -H "Content-Type: application/json" -d '{"min_members":3,"max_clusters":5}' --max-time 10`
- **Expected:** 200 OK, JSON with `clusters` array (length >= 1), each has `label`, `member_count`, `top_members`
- **Fail:** ABORT. Report HTTP status and response body.

### PF-03: Echo endpoint
- **Precondition:** PF-01 passed
- **Step:** `curl -s -X POST http://localhost:11435/api/echo -H "Content-Type: application/json" -d '{"query":"test","max_results":3}' --max-time 10`
- **Expected:** 200 OK, JSON with `results` array, `elapsed_ms` field
- **Fail:** ABORT. Report HTTP status.

### PF-04: Memory get endpoint
- **Precondition:** PF-02 passed (need a memory ID)
- **Step:** Extract a `memory_id` from PF-02's top_members, then `POST /api/memory_get {"memory_id":"..."}`
- **Expected:** 200 OK, JSON with `memory_id`, `content`, `category`, `labels`
- **Fail:** ABORT. Report HTTP status.

### PF-05: Vite dev server
- **Precondition:** None
- **Step:** `curl -s -o /dev/null -w "%{http_code}" http://localhost:5173`
- **Expected:** 200
- **Fail:** ABORT.

### PF-06: Playwright browser connect
- **Precondition:** PF-05 passed
- **Step:** `browser_navigate` to `http://localhost:5173`, then `browser_snapshot`
- **Expected:** Page loads, snapshot contains "ShrimPK" text
- **Fail:** ABORT.

---

## Galaxy View

### GV-01: Graph loads with clusters (GATE)
- **Precondition:** PF-06 passed
- **Step:** Wait 5 seconds after navigation for graph to load. Take screenshot. Snapshot DOM.
- **Expected:**
  - Left sidebar shows "Communities" heading
  - Sidebar has cluster buttons with labels and member counts
  - Center canvas has rendered nodes (WebGL — may not appear in DOM snapshot)
  - Top bar shows "ShrimPK", search input, zoom level "Galaxy"
  - No error banner visible
- **Fail:** BLOCK all DI, NB, DP, SR, NV tests.

### GV-02: Cluster list populated
- **Precondition:** GV-01 passed
- **Step:** Count cluster buttons in sidebar via DOM snapshot
- **Expected:** >= 3 cluster buttons, each with a label and member count badge
- **Fail:** Log count and continue.

### GV-03: Zoom controls visible
- **Precondition:** GV-01 passed
- **Step:** Look for ZoomIn/ZoomOut buttons in DOM snapshot (canvas overlay area)
- **Expected:** Two buttons with "Zoom in" and "Zoom out" tooltips present
- **Fail:** Log and continue.

### GV-04: No error banner
- **Precondition:** GV-01 passed
- **Step:** Check DOM for error banner element
- **Expected:** No error banner visible
- **Fail:** Log error text and continue.

---

## Drill-In

### DI-01: Click cluster drills into community (GATE)
- **Precondition:** GV-01 passed
- **Step:**
  1. Identify first cluster button in sidebar
  2. Click it
  3. Wait 5 seconds for graph to update
  4. Take screenshot
  5. Snapshot DOM
- **Expected:**
  - Zoom level changes from "Galaxy" to "Cluster"
  - Sidebar header shows cluster name (not "Communities")
  - "Back to Galaxy" button appears at bottom of sidebar
  - No error banner
  - Daemon still responds to `/health` (no crash)
- **Fail:** Check daemon health. If daemon crashed, report as CRITICAL.

### DI-02: Community legend appears
- **Precondition:** DI-01 passed
- **Step:** Look for community legend overlay (bottom-left of canvas)
- **Expected:** Legend with colored dots and "Cluster N (count)" labels
- **Fail:** Log and continue.

### DI-03: Size legend appears
- **Precondition:** DI-01 passed
- **Step:** Look for size legend overlay (bottom-right of canvas)
- **Expected:** Three circles (small/medium/large) with "Low"/"Med"/"High" labels
- **Fail:** Log and continue.

---

## Neighborhood

### NB-01: Expand node shows neighbors
- **Precondition:** DI-01 passed (need a cluster view with nodes)
- **Step:**
  1. From cluster view, click a node on the canvas (or use Search to find a node)
  2. In detail panel, click "Expand neighbors"
  3. Wait 5 seconds
  4. Take screenshot
- **Expected:**
  - Zoom level changes to "Neighborhood"
  - Graph shows center node + connected neighbors
  - Edges visible between center and neighbors
  - Daemon still healthy (no crash)
- **Fail:** Check daemon health. If crashed, report CRITICAL.

### NB-02: Center node highlighted
- **Precondition:** NB-01 passed
- **Step:** Inspect graph visually (screenshot)
- **Expected:** Center node is larger/differently colored than neighbors
- **Fail:** Log and continue.

### NB-03: Hover dims non-neighbors
- **Precondition:** NB-01 passed
- **Step:** Hover over a node (if possible via Playwright)
- **Expected:** Non-connected nodes dim
- **Fail:** Log (may not be testable via Playwright on WebGL canvas).

---

## Detail Panel

### DP-01: Click node opens detail panel
- **Precondition:** DI-01 passed
- **Step:**
  1. Use search to find a memory (type query, press Enter, click result)
  2. After expand, look for detail panel on the right
  3. Take screenshot
  4. Snapshot DOM
- **Expected:**
  - Right panel slides in with memory content
  - Category badge visible (colored pill)
  - Content text visible
  - Metrics section (novelty bar, echo count, source, created date)
  - Action buttons: "Expand neighbors", "Copy ID"
- **Fail:** Log what's missing.

### DP-02: Close detail panel
- **Precondition:** DP-01 passed
- **Step:** Click the X (close) button in detail panel header
- **Expected:** Panel slides out, detail disappears
- **Fail:** Log.

### DP-03: Expand neighbors button
- **Precondition:** DP-01 passed
- **Step:** Click "Expand neighbors" button
- **Expected:** Graph transitions to neighborhood view (see NB-01 expected)
- **Fail:** Log.

### DP-04: Copy ID button
- **Precondition:** DP-01 passed
- **Step:** Click "Copy ID" button
- **Expected:** No error, button responds to click (clipboard write may not be verifiable)
- **Fail:** Log.

---

## Search

### SR-01: Search returns results
- **Precondition:** GV-01 passed
- **Step:**
  1. Click search input
  2. Type "project" (or any common word)
  3. Press Enter
  4. Wait 3 seconds
  5. Snapshot DOM
- **Expected:**
  - Dropdown appears below search input
  - Contains 1+ result items with content preview, similarity %, labels
- **Fail:** Log.

### SR-02: Click search result navigates
- **Precondition:** SR-01 passed
- **Step:** Click first search result
- **Expected:**
  - Dropdown closes
  - Search input clears
  - Graph transitions to neighborhood view for that memory
  - Daemon healthy
- **Fail:** Log.

### SR-03: Escape closes dropdown
- **Precondition:** SR-01 passed
- **Step:** Press Escape while dropdown is open
- **Expected:** Dropdown closes, search input retains text
- **Fail:** Log.

---

## Navigation

### NV-01: Back to Galaxy from cluster
- **Precondition:** DI-01 passed (in cluster view)
- **Step:** Click "Back to Galaxy" button at bottom of sidebar
- **Expected:**
  - Zoom level returns to "Galaxy"
  - Sidebar shows "Communities" heading
  - Cluster list visible
- **Fail:** Log.

### NV-02: Home button from non-galaxy
- **Precondition:** DI-01 passed
- **Step:** Click Home icon button in toolbar
- **Expected:** Returns to galaxy view (same as NV-01)
- **Fail:** Log.

### NV-03: Refresh reloads current view
- **Precondition:** GV-01 passed
- **Step:** Click Refresh (rotate) button in toolbar
- **Expected:** Graph reloads without changing zoom level. Loading spinner appears briefly.
- **Fail:** Log.

### NV-04: Zoom in/out controls
- **Precondition:** GV-01 passed
- **Step:** Click Zoom In button, take screenshot. Click Zoom Out button, take screenshot.
- **Expected:** Visible change in graph zoom level between screenshots
- **Fail:** Log.

---

## Visual Polish

### VP-01: No pure white
- **Precondition:** GV-01 passed
- **Step:** Grep all rendered CSS for `#fff`, `#ffffff`, `rgb(255,255,255)`, `white`
- **Expected:** Zero occurrences
- **Fail:** Report element + property.

### VP-02: Dark theme consistency
- **Precondition:** GV-01 passed
- **Step:** Take full-page screenshot. Visual inspection.
- **Expected:**
  - Canvas is darkest (#09090b)
  - Sidebars slightly lighter (#18181b)
  - Top bar slightly lighter (#1f1f22)
  - Text is readable (no low-contrast issues)
- **Fail:** Log.

### VP-03: Focus rings on interactive elements
- **Precondition:** GV-01 passed
- **Step:** Tab through interactive elements (toolbar buttons, sidebar items, search input)
- **Expected:** Visible indigo focus ring on each focused element
- **Fail:** Report which elements lack focus ring.

### VP-04: Legends correctly positioned
- **Precondition:** DI-01 passed (legends visible in cluster/neighborhood)
- **Step:** Screenshot. Check legend positions.
- **Expected:**
  - Size legend: bottom-right corner
  - Community legend: bottom-left corner
  - Neither overlaps sidebar or detail panel
- **Fail:** Log position issue.

### VP-05: Reduced motion respected
- **Precondition:** GV-01 passed
- **Step:** Check CSS for `@media (prefers-reduced-motion: reduce)` rule
- **Expected:** All duration tokens set to 1ms in reduced-motion media query
- **Fail:** Log.
