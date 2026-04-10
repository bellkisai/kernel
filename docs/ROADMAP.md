# ShrimPK Roadmap

This roadmap reflects the current state of the kernel and planned directions.
Dates are aspirational. Contributions welcome -- see Contribution Opportunities below.

---

## Current State -- v0.7.5

Released April 2026. The core pipeline is stable with 11 crates + CLI, hybrid GraphRAG retrieval, and entity unification.

### What is shipped and working

**Echo pipeline (hybrid GraphRAG)**

Full retrieval chain: Bloom filter pre-screening, LSH candidate retrieval, cosine reranking,
Hebbian co-activation boosting, FSRS decay, ACT-R activation, label-based pre-filtering,
schema-driven fact extraction, entity unification with supersession, and temporal boosting.
Optional HyDE and LLM reranking via config flags.

**Configurable embedding**

EmbeddingProvider trait with 10 fastembed models (BGE-small-EN-v1.5 default, 384-dim) and
OpenAI API support. Runtime-switchable via config without restart.

**Multimodal SHRM v2 format**

Memory-mapped binary format with 3 channels: text (384-dim), vision (512-dim), speech (640-dim).
32-bit CRC per entry, atomic flush, crash recovery. Per-channel LSH indices.

**Speech pipeline (640-dim)**

ECAPA-TDNN (256-dim speaker) + Whisper-tiny encoder (384-dim prosody). ONNX inference via ort,
auto-download from HuggingFace Hub. Silero VAD gating. Feature-gated behind `--features speech`.

**Vision pipeline (512-dim)**

CLIP ViT-B/32 via fastembed. Cross-modal text-to-image retrieval. Feature-gated behind `--features vision`.

**Entity unification**

EntityFrame with EntityId resolution, alias tracking, supersession rewrite. Entities unify
across memories for consistent knowledge updates.

**Sleep consolidation**

Background LLM-driven fact extraction via Ollama. Schema-driven extraction with quality gates,
dedup, soft invalidation. Universal prompt works across all reader models.

**Importance scoring**

5-signal importance scoring: entity density, temporal salience, novelty, information density,
and user-signal weighting.

**MCP server**

`shrimpk-mcp` exposes 12 tools over stdio: `store`, `echo`, `memory_graph`, `memory_related`,
`memory_get`, `stats`, `forget`, `status`, `config_show`, `config_set`, `dump`, `persist`.
Compatible with Claude Desktop and any MCP client.

**Daemon + proxy**

`shrimpk-daemon` on `localhost:11435`. OpenAI-compatible proxy (`/v1/chat/completions`) with
transparent memory injection. Health, debug, and stats endpoints.

**System tray**

`shrimpk-tray` provides Windows system tray controls.

**CLI**

`store`, `echo`, `status`, `explore` (ratatui TUI), `detect`, `dump`, `bench`, `config`.

### Benchmarks

| Metric | Result |
|--------|--------|
| Seeded micro-benchmark | 19/20 |
| Abstention | 5/5 |
| Negative recall | 3/3 |
| LME-S baseline (GPT-4o judge) | 24.2% overall, 25.3% task-avg |
| P50 echo latency (10K) | 3.50ms |
| Test count | ~481 |

### Workspace (11 crates + CLI)

| Crate | Purpose |
|-------|---------|
| `shrimpk-core` | Types: MemoryEntry, EchoResult, EchoConfig, Modality |
| `shrimpk-memory` | Engine: EchoEngine, embedding, LSH, Bloom, Hebbian, labels, FSRS, ACT-R |
| `shrimpk-daemon` | HTTP server: axum, proxy, routes |
| `shrimpk-mcp` | MCP server (stdio): 12 tools |
| `shrimpk-context` | ContextAssembler: token-budgeted prompt compilation |
| `shrimpk-router` | CascadeRouter: provider routing |
| `shrimpk-security` | PII masking (6 categories, 14 regex patterns) |
| `shrimpk-kernel` | Facade crate re-exporting core + memory + context |
| `shrimpk-python` | PyO3 bindings (maturin) |
| `shrimpk-ros2` | ROS2 bridge (stub) |
| `shrimpk-tray` | Windows system tray (win32) |
| `cli/` | CLI binary |

---

## Upcoming

### KS78 -- Critical Fixes (April 2026)

- Persistence format version mismatch fix (Issue #16)
- Documentation sync (ROADMAP, CHANGELOG, MCP tool count)
- Design system v2 implementation

### KS79 -- Multi-Resolution Retrieval

- Hierarchical retrieval across raw memories, extracted facts, and entity summaries
- Adaptive context window based on query complexity

### KS80 -- Memory Lifecycle Improvements

- Smarter consolidation scheduling based on memory age and access patterns
- Improved supersession confidence scoring

---

## Future -- No Fixed Timeline

### Vision model upgrade (CLIP -> Nomic Embed Vision v1.5)

512 -> 768-dim. +7.8pp ImageNet zero-shot. 6x smaller model (62 MB vs 352 MB).
Breaking migration for stored vision embeddings.

### ROS2 bridge -- full implementation

`shrimpk-ros2` topics for text/image/audio store, echo publish, query service.
Target: ROS2 Jazzy via rclrs.

### Speaker upgrade (ECAPA-TDNN -> CAM++)

Lower EER at comparable model size. Blocked on Apache 2.0 ONNX availability.

### f16 quantization (SHRM v3)

~50% disk/memory reduction for vision and speech embeddings.

### Custom fine-tuned embedding model

BGE-small fine-tuned on personal memory data for improved recall.

### crates.io publish

`shrimpk-core`, `shrimpk-memory` once API stabilizes past v1.0.

### Cloud sync

Optional E2E encrypted memory sync across devices.

---

## Contribution Opportunities

### Good first issue

- **Fix vision feature flag propagation** -- forwarding `vision` feature to root `Cargo.toml`
- **CrossEncoder config in Tier 2 benchmark** -- add to standard benchmark suite

### Help wanted

- **100K latency regression** -- P50 23.79ms vs 4.0ms target. Needs LSH profiling.
- **Band-limited resampling** -- replace `resample_linear()` with `rubato` sinc resampling
- **Linux CI hardening** -- daemon startup, file locking, tray icon tests

### Research needed

- **Emotion model under permissive license** -- 3-dim A/D/V slot reserved, no Apache 2.0 model
- **LSH parameter tuning for BGE-small** -- hash count, bucket width optimization at 100K scale
- **SigLIP 2 fastembed support** -- 78.2% ImageNet zero-shot, no ONNX export yet
