# STP-KS77: Software Test Plan — Design System + P0 Graph Polish

## 1. Scope

Test the KS77 viz sprint deliverables:
- Design system foundation (tokens, components, migration)
- P0: Node size by importance
- P0: Louvain community colors
- P0: Camera transitions
- Bugfixes (zoom controls, focus rings, a11y, drill-in/expand workarounds)

## 2. Environment Requirements

| Requirement | Check | Fail Action |
|-------------|-------|-------------|
| Daemon v0.7.0 on localhost:11435 | `GET /health` returns `version: "0.7.0"` | Abort — wrong daemon |
| Daemon has >=10 memories | `GET /health` returns `memories >= 10` | Abort — no test data |
| Overview endpoint available | `POST /api/graph/overview` returns 200 | Abort — API mismatch |
| Echo endpoint available | `POST /api/echo` returns 200 | Abort — API mismatch |
| Vite dev server on localhost:5173 | `GET /` returns 200 | Abort — frontend down |
| Chrome accessible via Playwright | `browser_navigate` succeeds | Abort — no browser |

**HARD RULE:** All 6 checks MUST pass before any test case runs. If ANY check fails, stop and report the failure — do NOT proceed to test cases.

## 3. Test Categories

| Category | ID Prefix | Description |
|----------|-----------|-------------|
| Preflight | PF- | Environment validation |
| Galaxy View | GV- | Initial load, cluster display |
| Drill-In | DI- | Community drill-in navigation |
| Neighborhood | NB- | Node expand, neighbor view |
| Detail Panel | DP- | Node selection, detail display |
| Search | SR- | Search input, results, navigation |
| Navigation | NV- | Back/home, refresh, zoom |
| Visual Polish | VP- | Design tokens, legends, a11y |

## 4. Execution Order

1. **Preflight (PF-01 to PF-06)** — GATE: all must pass
2. **Galaxy View (GV-01 to GV-04)** — GATE: GV-01 must pass
3. **Drill-In (DI-01 to DI-03)** — GATE: DI-01 must pass
4. **Neighborhood (NB-01 to NB-03)**
5. **Detail Panel (DP-01 to DP-04)**
6. **Search (SR-01 to SR-03)**
7. **Navigation (NV-01 to NV-04)**
8. **Visual Polish (VP-01 to VP-05)**

Each GATE test blocks all subsequent tests in that category if it fails.

## 5. Reporting

After each test case:
```
[TC-ID] [PASS|FAIL|BLOCKED] — one-line summary
  Expected: ...
  Actual: ...
  Screenshot: (if applicable)
```

Final summary:
```
PASSED: N/M
FAILED: N (list IDs)
BLOCKED: N (list IDs + blocker)
```
