# ShrimPK Roadmap

This roadmap reflects the current state of the kernel and planned directions for future releases.
Dates are aspirational. Contributions are welcome at any stage -- see the Contribution Opportunities
section for specific items you can pick up today.

---

## Current State -- v0.7.5

Released April 2026. The kernel is a mature push-based AI memory system with hybrid GraphRAG
retrieval, entity unification, configurable embedding, and universal prompt support.

### Workspace

11 crates + CLI binary:

| Crate | Purpose |
|-------|---------|
| `shrimpk-core` | Types: MemoryEntry, EchoResult, EchoConfig, Modality |
| `shrimpk-memory` | Engine: EchoEngine, embedding, LSH, Bloom, Hebbian, labels, FSRS decay, ACT-R activation |
| `shrimpk-daemon` | HTTP server: axum, proxy, routes (/health, /debug, /v1/chat/completions) |
| `shrimpk-mcp` | MCP server (stdio): 12 tools for memory management and graph navigation |
| `shrimpk-context` | ContextAssembler: token-budgeted prompt compilation |
| `shrimpk-router` | CascadeRouter: provider routing (not yet wired in daemon) |
| `shrimpk-security` | PII masking (stub -- 6 categories, 14 regex patterns) |
| `shrimpk-kernel` | Facade crate re-exporting core + memory + context |
| `shrimpk-python` | PyO3 bindings (maturin) |
| `shrimpk-ros2` | ROS2 bridge (stub) |
| `shrimpk-tray` | Windows system tray (win32) |
| `cli/` | CLI binary: store, echo, status, explore (ratatui TUI) |

### What is shipped and working

**Echo pipeline**

The full retrieval chain is operational: Bloom filter pre-screening (O(1) topic elimination),
LSH candidate retrieval (sub-linear at scale), label-based pre-filtering, cosine reranking,
Hebbian co-activation boosting, FSRS decay, ACT-R activation, temporal boost, importance
scoring, and multiplicative supersession demotion. Optional HyDE (hypothetical document
expansion) and LLM reranking are available via config flags.

**Hybrid GraphRAG (KS61-KS64)**

Full hybrid GraphRAG pipeline combining vector similarity with label-graph traversal.
Label-graph navigation enables neighborhood exploration from any memory. 14 MCP tools
support store, retrieval, graph exploration, and management operations. 517 tests cover
the complete pipeline.

**Entity unification (KS73)**

EntityFrame and EntityId-based supersession for structured entity tracking. When new
information contradicts or updates an existing entity, the old memory is superseded and
receives a multiplicative demotion penalty (default 0.40x, configurable). This prevents
stale knowledge from outranking current facts.

**Configurable embedding (KS75)**

EmbeddingProvider trait abstraction with 10 fastembed models and OpenAI API support.
Default model: BGE-small-EN-v1.5 (384-dim) via fastembed. The provider can be swapped
at configuration time without code changes.

**Universal prompt (KS76)**

One prompt template for all reader models. No per-model tuning required. Validated with
qwen2.5:1.5b (default) and qwen2.5:3b. Includes temporal boost for time-sensitive queries
and a 5-signal importance scoring system.

**Multimodal SHRM v2**

Memory-mapped binary format with 32-bit CRC per entry, atomic flush, and crash recovery.
Stores text embeddings (384-dim), optional vision embeddings (512-dim), optional speech
embeddings (640-dim), metadata, and sensitivity labels. Three-channel architecture: text
(BGE-small-EN-v1.5), vision (CLIP ViT-B/32), speech (ECAPA-TDNN 256 + Whisper-tiny 384).

**Sleep consolidation**

Background consolidation using a local LLM via Ollama with schema-driven fact extraction.
Child memory pipeline creates atomic facts from raw memories, supports supersession for
knowledge updates. Default reader model: qwen2.5:1.5b.

**MCP server (12 tools)**

`shrimpk-mcp` exposes 12 tools over stdio: `store`, `echo`, `memory_graph`,
`memory_related`, `memory_get`, `stats`, `forget`, `status`, `config_show`, `config_set`,
`dump`, `persist`. Additional multimodal tools (`store_image`, `store_audio`) available
when feature flags are enabled. Compatible with Claude Desktop and any MCP client.

**Daemon + tray**

`shrimpk-daemon` runs as a background HTTP service on `localhost:11435` with OpenAI-compatible
proxy (`/v1/chat/completions`). `shrimpk-tray` provides a system tray icon and launch/stop
controls on Windows.

### Benchmark results

| Benchmark | Score |
|-----------|-------|
| Seeded micro-benchmark | 19/20 |
| Abstention (no-answer detection) | 5/5 |
| Negative retrieval | 3/3 |
| LME-S (GPT-4o judge) | 24.2% overall, 25.3% task-avg |

**Performance (release build, i7-1165G7)**

| Metric | Result |
|--------|--------|
| P50 echo latency at 10K memories | 3.50ms |
| Store throughput | ~128 memories/sec |
| RAM (10K text memories) | ~85 MB |

### Key milestones (KS67-KS78)

| Sprint | Milestone |
|--------|-----------|
| KS67 | Schema-driven fact extraction, 80% micro-benchmark recall |
| KS68 | IE-1 + KU-1 fixed, 17/20 embedding-only, Greptile P1s resolved |
| KS69 | Consolidation redesign, child memory pipeline rewrite, 19/20 seeded |
| KS70 | 20/20 seeded, qwen2.5:1.5b default, first real consolidation validation |
| KS73 | Entity unification, EntityFrame, EntityId supersession |
| KS75 | Configurable embedding: EmbeddingProvider trait, 10 models, OpenAI API |
| KS76 | Universal prompt, temporal boost, importance scoring |
| KS77 | 19/20 seeded, 5/5 abstention, KU-3 fixed, temporal dedup trap found |
| KS78 | Multiplicative supersession demotion (0.40x default) |

---

## Next -- KS79: Multi-Resolution Retrieval

Target: Q2 2026. Focus: retrieval quality at multiple granularity levels.

Multi-resolution retrieval allows the echo pipeline to match queries against memories at
different levels of abstraction -- raw memories, consolidated facts, entity summaries, and
topic clusters. This enables both precise fact lookup and broad contextual recall within
the same query.

---

## Next -- KS80: Memory Lifecycle

Target: Q2 2026. Focus: memory aging, archival, and lifecycle management.

Formalize the memory lifecycle from creation through active use, staleness detection,
archival, and eventual pruning. Integrate FSRS scheduling data with usage patterns to
make informed retention decisions. Provide user-facing controls for lifecycle policies.

---

## Future -- No Fixed Timeline

These items are research directions or require dependencies that are not yet settled.

### ROS2 bridge production readiness

`shrimpk-ros2` exists as a stub. Production readiness requires ROS2 Jazzy integration
via `rclrs`, topic/service wiring, and latency validation within a 30 Hz camera frame
budget. The push-based architecture maps naturally to ROS2 topic publishing.

### Custom fine-tuned embedding model

A model fine-tuned specifically on personal memory data (short episodic sentences, user
preferences, recurring entities) could improve recall quality without increasing model
size. This requires a labeled dataset and an ML training pipeline.

### crates.io publish

Publishing `shrimpk-core`, `shrimpk-memory`, and (eventually) `shrimpk-ros2` to crates.io
is planned once the API stabilizes. The current pre-1.0 semver signals that breaking
changes are expected.

### Cloud sync

Optional encrypted sync of the memory store across devices. End-to-end encrypted, the
server sees only ciphertext. The key design question is key management -- the server must
never hold decryption keys.

### Vision model upgrade

Nomic Embed Vision v1.5 or SigLIP 2 as a CLIP replacement. The 512 to 768 dimension
change would be a breaking migration for stored vision embeddings. Deferred until the
user base is large enough to justify the migration complexity.

### Speaker upgrade: ECAPA-TDNN to CAM++

CAM++ (Context-Aware Masking) achieves lower equal error rate than ECAPA-TDNN on
VoxCeleb1/2. Blocked on availability of an Apache 2.0-compatible ONNX export.

---

## Contribution Opportunities

All issues below are open for contribution. The project uses Apache 2.0. Opening a
discussion issue before starting significant work is encouraged to avoid duplication.

### Good first issue

**Extend the Tier 2 benchmark with a CrossEncoder config** (difficulty: low, Rust)
The realistic Tier 2 benchmark tests four pipeline configs. Adding a CrossEncoder-only
config would complete the comparison matrix.

### Help wanted

**Linux CI hardening** (difficulty: medium, DevOps + Rust)
The kernel builds and tests pass on CI, but test coverage is lower on Linux than on the
primary Windows development machine. Contributions improving Linux CI coverage are welcome.

**100K latency profiling** (difficulty: medium, Rust + profiling)
P50 at 100K memories needs investigation. Likely causes: LSH bucket saturation with
BGE-small embedding distribution, or brute-force fallback frequency. Tools: `perf`,
`cargo flamegraph`, or the `tracing` spans in the echo path.

### Research needed

**Emotion model under permissive license** (difficulty: high, ML research)
The 3-dim arousal/dominance/valence emotion slot in the speech pipeline is reserved but
empty because all mature dimensional emotion models carry CC-BY-NC-SA-4.0 licenses.

**LSH parameter tuning for BGE-small distribution** (difficulty: high, information retrieval)
The LSH index was tuned for all-MiniLM-L6-v2 embeddings. The upgrade to BGE-small changed
the embedding distribution in ways that may require different hash count, bucket width, or
candidate list size to maintain sub-10ms P50 at 100K scale.
