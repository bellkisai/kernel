# Changelog

All notable changes to ShrimPK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.7.5] -- 2026-04-10

### Added
- **Schema-driven fact extraction** (KS67): structured extraction pipeline replacing free-form LLM output
- **Entity unification** (KS73): EntityFrame, EntityId resolution, alias tracking, supersession rewrite
- **Configurable embedding** (KS75): EmbeddingProvider trait, 10 fastembed models, OpenAI API support
- **Universal prompt** (KS76): single consolidation prompt for all reader models (no per-model tuning)
- **Temporal boost** (KS76): recency-weighted scoring for time-sensitive queries
- **Importance scoring** (KS76): 5-signal importance scoring (entity density, temporal salience, novelty, info density, user signal)
- **Design system foundation** (KS77): design tokens, component spec for viz app
- **Negative recall benchmark**: 3/3 baseline for "I don't know" scenarios
- **Abstention benchmark**: 5/5 -- engine correctly abstains when no relevant memory exists

### Changed
- **Consolidation redesign** (KS69): child memory pipeline rewrite with quality gates, dedup, soft invalidation
- **Consolidation Tier 2** (KS71): subject fix, quality gate, dedup, soft invalidation
- **Child keyword labels** (KS72): labels assigned at child creation time
- **Default enrichment model**: switched to `qwen2.5:1.5b`
- **MCP server**: now exposes 12 tools (was 9) -- added `memory_graph`, `memory_related`, `memory_get`

### Fixed
- **KU-3 recall** (KS77): knowledge update scenario now passes in seeded benchmark
- **IE-3, TR-3, ME-4, PT-3 recall** (KS68): multiple recall fixes across LME categories
- **Temporal label dedup trap** (KS77): avoid adding temporal labels to children when parent has temporal content
- **Persistence format version**: format mismatch fix for MCP store/echo

### Performance
- Seeded micro-benchmark: 19/20 (up from 55% baseline)
- Abstention: 5/5
- Negative recall: 3/3
- LME-S baseline (GPT-4o judge): 24.2% overall

## [0.7.0] — 2026-04-02

### Added
- **Speech ONNX inference:** ECAPA-TDNN + Whisper-tiny encoder fully wired with auto-download from HuggingFace Hub (~58 MB)
- **FBank preprocessing:** Pure Rust 80-bin filterbank computation for ECAPA-TDNN (`compute_fbank_flat()`)
- **Cross-modal speech recall:** `store_audio` now accepts optional `description` parameter — text description is embedded and indexed in text_lsh + bloom, enabling text queries to find speech memories
- **Auto-labeling:** Speech memories automatically labeled `memtype:audio` + Tier 1 labels from description
- **ROS2 bridge:** `shrimpk-ros2` crate with String/Image/Audio/Pose message types, replay mode, health check (13 tests)
- **Cross-modal vision recall:** `store_image` now accepts optional `description` parameter (same pattern as speech)
- **Memory Markdown export:** `shrimpk dump --format md` exports memories as Obsidian-compatible .md files with YAML frontmatter
- **Speech latency benchmark:** `echo_speech_bench.rs` for embed_pcm and store_audio profiling

### Changed
- **Speech dimension fix:** 896→640 (ECAPA-TDNN Wespeaker ResNet34 outputs 256-dim, not 512)
- **ECAPA-TDNN model:** `wespeaker-cnceleb-resnet34-LM` with FBank input (was assumed raw waveform)
- **Speech constants:** `SPEAKER_DIM=256`, `PROSODY_DIM=384`, `SPEECH_DIM=640`
- **Persistence fallback:** `unwrap_or(896)` → `unwrap_or(640)` in SHRM v2 speech section

### Fixed
- `speech_store_and_echo` test: speech memories use speech_lsh, not text_lsh — renamed to `speech_store_and_verify`
- Stale 896/512-dim references across 15 files (code, tests, docs)

## [0.6.0] — 2026-04-01

### Added
- **Label bucket architecture (ADR-015):** Semantic labels on memories for pre-filtered retrieval
- **7-dimension label taxonomy:** topic, domain, entity, action, temporal, memtype, sentiment
- **Tier 1 label generation:** Prototype cosine matching + rule-based classifier at store time (<2ms)
- **Tier 2 label enrichment:** Combined LLM prompt (facts + labels) during consolidation
- **Label inverted index:** HashMap<String, Vec<u32>> on EchoStore with incremental maintenance
- **D6 three-source candidate merge:** Labels + LSH + brute-force fallback in echo pipeline
- **Query classification:** Tier A keyword + Tier B prototype cosine for label-based pre-filtering
- **LabelPrototypes:** 37 rich prototype descriptions initialized at engine startup
- **ConsolidationOutput + LabelSet:** Structured output from combined consolidation call
- **bootstrap_labels():** Async retroactive labeling for existing stores
- **use_labels config:** Toggle for safe rollout and A/B benchmarking
- **100K benchmark suite:** echo_label_bench.rs with diverse 15-category content generator
- **Pipeline timing suite:** echo_label_tuning.rs for stage-level profiling
- **Public documentation:** 9 docs in kernel/docs/ (Architecture, Benchmarks, Integration Guide, etc.)
- **GitHub templates:** Bug report, feature request, PR template
- **SECURITY.md:** Vulnerability reporting policy
- **Branch protection:** master requires PR with 1 approval

### Changed
- **Echo pipeline:** Candidate retrieval uses label + LSH union (MIN_CANDIDATES lowered from 10 to 5)
- **Consolidator trait:** New extract_facts_and_labels() with default backward-compatible impl
- **MemoryEntry:** Added labels: Vec<String> and label_version: u8 fields
- **MemoryMeta:** Propagates labels through SHRM v2 save/load roundtrip
- **EchoConfig:** Added use_labels: bool (default true)
- **EchoEngine:** LabelPrototypes initialized at new()/load() before Mutex wrapping
- **Speech architecture:** Emotion channel dropped (Wav2Small CC-BY-NC-SA incompatible). 899-dim -> 640-dim (ECAPA-TDNN 256 + Whisper-tiny 384)
- **Repository:** Now public at github.com/bellkisai/kernel (Apache 2.0)

### Performance
- **100K retrieval:** 35% improvement with labels (P50 38.94ms -> 27.70ms)
- **Embedding floor:** ~8ms per query with BGE-small (2x slower than previous MiniLM)
- **Label overhead at store time:** ~30% increase (prototype matching per memory)
- **10K and below:** Labels add marginal overhead; LSH alone is sufficient

### Fixed
- Hardcoded developer paths in benchmarks/*.py replaced with portable os.path
- release.yml: -p shrimpk-app -> -p shrimpk-tray (correct package name)
- SECURITY.md version: 0.4.x -> 0.5.x
- Internal sprint references (KS numbers) removed from public docs

## [0.5.0] — 2026-03-29

### Added
- **Multimodal Echo Memory:** 3-channel architecture (text, vision, speech)
- **Vision channel:** CLIP ViT-B-32 via fastembed — store images, cross-modal text->image retrieval
- **Speech channel:** Architecture ready (ECAPA-TDNN 256 + Whisper-tiny 384 = 640-dim, wired in KS51)
- **QueryMode:** Text / Vision / Auto for cross-channel echo queries
- **Per-channel LSH:** Separate indices per modality (384/512/640-dim)
- **SHRM v2 persistence:** Per-channel bitmap+sparse sections, CRC32 per section, v1 backward compat
- **API endpoints:** store_image, store_audio (MCP + daemon + CLI), --modality flag on echo
- **Per-channel stats:** text_count, vision_count, speech_count in stats output
- **Feature flags:** `vision` and `speech` Cargo features for compile-time gating
- **Multimodal benchmarks:** echo_multimodal_bench.rs with 7 HARD gate tests
- **FileConfig multimodal fields:** enabled_modalities, vision/speech_embedding_dim configurable via TOML

### Changed
- **Text model upgrade:** all-MiniLM-L6-v2 -> BGE-small-EN-v1.5 (MTEB 56.3 -> 62.0, same 384-dim)
- **Embedder refactor:** Embedder -> MultiEmbedder with per-channel methods
- **RAM estimation:** stats() now accounts for vision (512-dim) and speech (640-dim) sizes
- **SPEAKER_DIM:** 192 -> 256 (Wespeaker ResNet34 ONNX output — was 512 in v0.5.0, corrected in KS51)

### Fixed
- Speech LSH rebuilt on load (was empty after restart)
- Consolidation skips non-text entries (no more "[image]" sent to LLM)
- Auto-mode dedup keeps highest final_score per memory_id
- Brute-force fallback skips empty embeddings
- Input size limits: store_image 10MB, store_audio 60s at 16kHz
- Header CRC coverage in SHRM v2 (prevents corrupted entry_count allocation)
- Unix parent dir fsync after atomic rename (ext4 durability)
- merge_near_duplicates skips empty text embeddings

### Security
- Input size limits on image/audio storage prevent OOM
- CRC32 covers header + metadata + per-channel sections

## [0.3.2]

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
- **MCP Server** (`shrimpk-mcp`): 12 tools over JSON-RPC 2.0 stdio
  - store, echo, memory_graph, memory_related, memory_get, stats, forget, dump, config_show, config_set, persist, status
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
- 263+ tests (unit, integration, stress, precision, scale, token efficiency, multimodal)
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
