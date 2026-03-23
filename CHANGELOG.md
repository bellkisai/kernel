# Changelog

All notable changes to ShrimPK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added — KS13: Provider Integration (v0.3.2)
- **Provider Scanner**: auto-detect 6 LLM providers via parallel port probing (Ollama, LM Studio, Jan.ai, vLLM, LocalAI, GPT4All)
- **Model-Name Routing**: requests routed by model name to correct provider backend
- **`shrimpk detect` CLI command**: scan and display detected providers + models
- **GET /api/detect**: daemon endpoint for re-scanning providers
- **README**: "Memory Proxy" documentation section with provider setup table
- **OpenClaw Plugin** (`integrations/openclaw/`): TypeScript plugin with `before_prompt_build` hook for automatic memory injection, zero runtime deps

### Added — KS12: OpenAI-Compatible Proxy (v0.3.2)
- **POST /v1/chat/completions**: transparent memory-injecting proxy to any backend
- **GET /v1/models**: passthrough to backend model list
- Streaming via raw byte passthrough (no SSE parsing overhead)
- `#[serde(flatten)]` preserves unknown request fields for compatibility
- Fire-and-forget user message storage via `engine.store()`
- Config: `proxy_target`, `proxy_enabled`, `proxy_max_echo_results`

### Added — KS11: Sleep Consolidation (v0.3.0 + v0.3.1)
- **Consolidator trait** with swappable backends: OllamaConsolidator, HttpConsolidator, NoopConsolidator
- **Sleep consolidation Step 5**: LLM fact extraction during 5-min background cycle
- **Split pipeline (Pipe A/B)**: child memory rescue for near-miss candidates
- **Parent-children index** on EchoStore (O(1) child lookup via HashMap)
- **Child memory creation** with embedded vectors in consolidation (v0.3.1)
- MemoryEntry: `enriched` + `parent_id` fields, backward-compatible persistence
- Bloom filter: lower match threshold (2→1), bypass for stores <50 memories
- LSH: Hamming-1 multi-probe for improved recall
- Config: `consolidation_provider`, `ollama_url`, `enrichment_model`, `max_facts_per_memory`
- FileConfig support for all consolidation fields (config.toml)
- Config show/set for consolidation in daemon, CLI, MCP
- CLI stats formatting fix (was printing raw JSON via daemon proxy)

### Added — KS10: MSI Installer (v0.2.0)
- WiX v3 MSI installer (per-user, no admin, 4 binaries + 3 scripts)
- PATH environment variable added during install
- Start Menu shortcuts (app, terminal, uninstall)
- PowerShell-based hook registration (register-hook.ps1 / unregister-hook.ps1)
- All WiX custom actions use `powershell.exe -WindowStyle Hidden` (no CMD flash)
- Auto-start daemon + tray on login via Registry Run keys

### Added — KS9: Product Polish (v0.2.0)
- **System tray** (`shrimpk-app.exe`): Win32 message pump, background stats, periodic health check
- README restyled as landing page
- Uninstaller clears autostart registry keys
- Competitive scan update (MuninnDB, memU, Hindsight, NeuralMemory)

### Added — KS7: MCP Server
- **MCP Server** (`shrimpk-mcp`): 9 tools over JSON-RPC 2.0 stdio
  - store, echo, stats, forget, dump, config_show, config_set, persist, status
  - Lazy engine init (fastembed loads on first tool call, not on handshake)
  - Auto-persist after store/forget, stdout sacred (logs to stderr)
  - Registered globally via `claude mcp add --scope user`
- CLI `--json` flag for `shrimpk echo` + `--quiet` for `shrimpk store` (hook integration)
- Echo Memory 3-layer rules in CLAUDE.md (MCP tools + auto-store + auto-echo)

### Added — KS8: HTTP Daemon (The Ollama Model)
- **HTTP Daemon** (`shrimpk-daemon`): Axum server on localhost:11435
  - 10 REST routes: health, store, echo, stats, memories, forget, config, persist, consolidate
  - Model loads ONCE, serves forever — no cold starts
  - Optional auth token (`SHRIMPK_AUTH_TOKEN` → Bearer header)
  - PID file for daemon discovery (`~/.shrimpk-kernel/daemon.pid`)
  - Background consolidation every 5 min
  - CORS headers for future web UI
  - Graceful shutdown: persists memories on Ctrl+C
- CLI auto-detects daemon via TCP probe → proxies via HTTP (~1ms)
- MCP auto-detects daemon via health check → proxies via HTTP
- Cross-platform auto-start: `--install` / `--uninstall` (Windows VBS, macOS launchd, Linux systemd)
- Hook script uses `curl` to daemon instead of spawning CLI processes

### Added — KS9: Product Polish
- **System tray icon** (`shrimpk-tray`): 🦐 shrimp in taskbar
  - Right-click: status, stats, copy port, open data dir, stop daemon, quit
- README rewrite: landing page style matching shrimpk.html
- Uninstaller stops running daemon before removing autostart

### Fixed — KS7.5: Audit Fixes
- File locking via `fs2` (exclusive write, shared read) — prevents CLI+MCP data corruption
- Background consolidation started in MCP server (every 5 min)
- Dump tool reads from in-memory store (was broken: looked for .json, engine writes .shrm)
- Category-aware decay applied in echo scoring (old memories rank lower)
- Engine init error returns proper JSON-RPC response (was silently dropped)

### Fixed — KS8/KS9: Audit Fixes
- Persist memories on graceful shutdown (was losing up to 5 min of data)
- CLI/MCP/Hook forward auth token to daemon (was silently falling back to in-process)
- PID file validated on startup (stale cleared, prevents duplicate daemons)
- PiiFilter shared in AppState (was recompiling regexes per request)
- Atomic binary write (write .tmp, rename — prevents corruption on crash)

## [0.1.0] - 2026-03-19

### Added

#### Echo Memory Engine (`shrimpk-memory`)
- Push-based associative memory: memories find you, no explicit query needed
- LSH sub-linear search (SimHash, 16 tables, 10 bits) for O(L*K*D) retrieval
- Bloom filter pre-screening for O(1) topic rejection
- Hebbian co-activation learning with exponential decay (7-day half-life)
- Two-pass ranking: similarity first, then Hebbian boost
- Category-aware adaptive decay: 6 types (Identity 365d, Preference 60d, Fact 30d, ActiveProject 14d, Default 7d, Conversation 3d)
- Sleep-inspired consolidation: prune stale memories, merge duplicates, rebuild indices
- Memory reformulation: 11 rules restructure text for +9% recall improvement
- PII detection and masking: 15 regex patterns, mask-don't-block approach
- Binary SHRM persistence format with CRC32 integrity checking
- 3.50ms P50 echo latency at 100,000 memories (release build)

#### Provider Router (`shrimpk-router`)
- Cascade routing with automatic fallback between providers
- Cost tracking per request with daily/monthly budgets
- Circuit breaker pattern (open after N failures, half-open recovery)
- Capability-based model selection (vision, function calling)
- Sensitivity-aware routing (local-only for private data)

#### Context Assembler (`shrimpk-context`)
- Token-budgeted prompt compilation with priority-based truncation
- 5-segment assembly: system prompt > echo memories > RAG > conversation > query
- Adaptive budget allocation: more echo results = more echo budget
- Sensitivity filtering for cloud vs local model contexts

#### Core Types (`shrimpk-core`)
- Unified `ShrimPKError` error framework with per-domain variants
- Auto-detect RAM tiers: minimal (8GB), standard (16GB), full (32GB), maximum (64GB+)
- Quantization modes: F32, F16, Int8, Binary
- TOML config file (`~/.shrimpk-kernel/config.toml`) with env var overrides
- Disk usage tracking and configurable limits (default: 2GB)
- Cross-platform home directory detection via `dirs` crate

#### CLI (`shrimpk-cli`)
- `shrimpk store` — store memories with PII scanning
- `shrimpk echo` — find memories that resonate with a query
- `shrimpk stats` — engine statistics with disk usage
- `shrimpk forget` — remove memories by UUID
- `shrimpk dump` — list all stored memories
- `shrimpk bench` — performance benchmark with P50/P95/P99 latencies
- `shrimpk config show|set|reset|path` — view and manage configuration
- `shrimpk status` — disk usage bar, system tier, RAM budget

#### Python Bindings (`shrimpk-python`)
- PyO3 bindings: `pip install shrimpk` via maturin
- `EchoMemory`, `EchoConfig`, `MemoryStats` Python classes
- Compatible with Python 3.8+

#### Infrastructure
- GitHub Actions CI: fmt, clippy, test (Ubuntu/macOS/Windows), doc
- Release workflow: cross-platform binary builds on tag push
- 222+ tests (unit, integration, stress, precision, scale, token efficiency)
- Apache 2.0 license

### Performance

| Metric | Value |
|--------|-------|
| Echo P50 at 100K memories | 3.50ms |
| Echo P95 at 100K memories | 6.88ms |
| Head-to-head accuracy | +38% vs plain LLM |
| Personalization rate | 100% |
| Token savings | 15.7% per request |
| Follow-up elimination | 100% |
