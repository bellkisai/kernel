# Changelog

All notable changes to ShrimPK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **MCP Server** (`shrimpk-mcp`): 9 tools over JSON-RPC 2.0 stdio for Claude Code integration
  - store, echo, stats, forget, dump, config_show, config_set, persist, status
  - Lazy engine init (fastembed loads on first tool call, not on handshake)
  - Auto-persist after store/forget operations
  - Registered globally via `claude mcp add --scope user`
- CLI `--json` flag for `shrimpk echo` (clean machine-readable output for hooks)
- Echo Memory rules in CLAUDE.md (auto-store instructions for Claude Code)
- Release workflow builds both `shrimpk` and `shrimpk-mcp` binaries

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
