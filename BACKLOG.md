# ShrimPK Backlog

Tracked items for the ShrimPK kernel project. Updated after each sprint.
Source of truth for the ShrimPK kernel project.

## Status Legend
- **DONE** — shipped and tested
- **PLANNED** — scheduled for a specific sprint
- **BACKLOG** — accepted, not yet scheduled
- **RESEARCH** — needs investigation before scheduling

---

## Sprint Roadmap (KS73-KS80)

- [x] KS73: Entity unification — EntityFrame, UUID v5, alias store, entity supersession (PR #10)
- [ ] KS74: v0.8.0-beta — recall gap fixes (NR demotion, abstention threshold), TUI dashboard, README rewrite, installer testing
- [ ] KS75: Store-time contradiction detection
- [ ] KS76: Memory import — cold start solver, 4+ parsers (Claude Code, ChatGPT, Obsidian, Mem0)
- [ ] KS77: KU-3 fix + remaining recall fixes (90% gate)
- [ ] KS78: Public launch preparation
- [ ] KS79: Context compression — LLMLingua-2 ONNX at store time
- [ ] KS80: Retroactive link re-scoring + sleep replay

---

## HIGH — Retrieval Quality

*Components exist, need wiring. Validated by academic research.*

- [ ] PPR-weighted Hebbian traversal — Personalized PageRank seeded on echo hits, weighted by edge strength x ACT-R. +20% multi-hop QA (HippoRAG, NeurIPS 2024)
- [ ] Multi-resolution retrieval fallback — memory → label cluster → community summary cascade. All three layers exist, not connected as fallback chain (RAPTOR, ICLR 2024)
- [ ] Retrieval mode parameter — expose naive/local/global/hybrid on `echo` API (LightRAG, EMNLP 2025)
- [ ] Citation-weighted memory scoring — track which injected memories LLM actually cites in response, upweight high-utility memories. Proxy already intercepts responses (RMM, ACL 2025)

## HIGH — Memory Lifecycle

- [ ] Merge operation — explicit ADD/UPDATE/DELETE/NOOP diff during consolidation. All production systems converge on merge as required (Mem0, RMM, Think-in-Memory)
- [ ] Multi-granularity storage — tag memories by scale: utterance/turn/session. +10% LME accuracy (RMM paper)
- [ ] Write-path learned filtering — decide what NOT to store before embedding. Most underresearched area per 2026 survey (arXiv 2603.07670)
- [ ] Soft-deletion compaction — GC when FSRS strength drops below threshold. Currently decay only de-ranks, never removes (MemoryBank pattern)

## HIGH — Cortex Prerequisites (blocking v0.10.0)

- [ ] Inter-layer protocol design — Soul ↔ Brain ↔ Memory API surface. Direct Rust calls vs Tokio channels vs message types
- [ ] Security model for agentic stack — data safety layer, poisoned memory detection. Distinct from command-level Brainstem
- [ ] Alpha/Beta ARC competition model — formal design doc. Async parallelism, leader election, Adaptive Resonance Theory mapping

## MEDIUM — Model & Format Upgrades

- [ ] Nomic Embed Vision v1.5 — CLIP ViT-B/32 → Nomic, +7.8pp ImageNet zero-shot, 6x smaller ONNX. Breaking: 512→768 dim migration
- [ ] f16 quantization for vision/speech — SHRM v3, ~50% disk/RAM savings, f32 promotion at query time
- [ ] Band-limited resampling — replace resample_linear() with rubato crate. Correctness bug: aliasing at 48→16kHz
- [ ] BuiltinConsolidator — bundled extraction model, zero Ollama dependency for consolidation quality
- [ ] Configurable embedding provider — EmbeddingProvider trait, 10 fastembed models + OpenAI API (KS75 — DONE)

## MEDIUM — Graph & Entity

- [ ] Retroactive link invalidation — when A supersedes B, downweight ALL B-anchored Hebbian links, not just B itself (A-MEM/Zettelkasten pattern)
- [ ] Episodic anchoring — bidirectional indices linking Hebbian edges back to source episodes (Graphiti/Zep pattern)
- [ ] Entity-cluster summaries — entity-level community nodes, not just label-level (Graphiti temporal KG)

## MEDIUM — Viz & UI Polish

*Current state: Tauri 2 + Sigma.js 3.0 + ForceAtlas2, 3-level zoom (KS65-66). Functional but early MVP.*

**Graph Polish:**
- [ ] Smooth view transitions — animated node repositioning between galaxy/cluster/neighborhood (currently hard-resets layout)
- [ ] Louvain community visualization — color nodes by community, show boundaries (graphology-communities-louvain installed, unused)
- [ ] Edge labels on hover — show typed relationship (CoActivation, WorksAt, PrefersTool, etc.)
- [ ] Temporal slider — filter graph by time range, animate memory formation over time
- [ ] Custom node shapes per category — distinct shapes for Identity/Fact/Preference/ActiveProject/Conversation
- [ ] Entity super-nodes — render EntityFrame nodes at graph level, not just label clusters
- [ ] Node size by echo frequency — proportional to retrieval count, not just importance score

**Memory Curation:**
- [ ] Inline memory edit — edit content/labels from detail panel, PATCH endpoint on daemon
- [ ] Memory merge — select 2+ nodes, merge into one (new daemon endpoint)
- [ ] Manual link creation — create Hebbian edges from graph view (new daemon endpoint)
- [ ] Retag from graph — drag-drop between clusters or multi-select retag
- [ ] Bulk operations — multi-select for delete/retag/export

**Export Formats:**
- [ ] JSON export per memory — full metadata + embeddings + graph edges
- [ ] Graph export — GraphML/GEXF for external visualization tools

## MEDIUM — Quantization (v0.8.0)

- [ ] Int8 scalar quantization (4x compression, simsimd ready)
- [ ] TurboQuant integration (turbo-quant crate, 8-10x)
- [ ] Binary + float32 rescore pipeline

## MEDIUM — Intelligence Tuning

- [ ] Full ACT-R retrieval history (Vec<u32> ring buffer)
- [ ] ACT-R activation ON by default (after benchmarking)
- [ ] Three-tier store (hot/warm/cold)
- [ ] Importance retrieval boost (A/B test, then enable)

## MEDIUM — Product & Distribution

- [ ] Memory file export as .md sidecars — per-memory files with YAML frontmatter (distinct from bulk `shrimpk dump`)
- [ ] Cloud sync — encrypted cross-device memory, E2E encrypted, server sees only ciphertext
- [ ] Managed API planning
- [ ] Revenue model implementation

## MEDIUM — Benchmarks Not Yet Running

- [ ] LoCoMo benchmark
- [ ] MemoryAgentBench (ICLR 2026) — contradiction/conflict resolution focus
- [ ] EverMemBench (2025) — entity disambiguation focus

## LOW — Backlog

- [ ] Memory as weights prototype (PyTorch via shrimpk-python)
- [ ] Cluster summary tree (MemTree pattern)
- [ ] Custom fine-tuned embedding model
- [ ] crates.io publish (after API stabilizes)
- [ ] Code signing certificate
- [ ] PostToolUse async hook
- [ ] Predictive coding layer — surprise/prediction error signal (~300 lines Rust)
- [ ] Session-level dynamics tracking (COMEDY pattern — user-bot relationship)
- [ ] Emotion channel — Apache 2.0 ONNX model needed (slot reserved in SHRM)
- [ ] CAM++ speaker model upgrade — needs Apache 2.0 ONNX verification
- [ ] SigLIP 2 vision model — needs upstream ONNX availability

## RESEARCH — Long-horizon

- [ ] Causal retrieval — retrieve by causal relevance, not just similarity (2026 survey frontier)
- [ ] Model weight printing — cross-model knowledge transfer via externalized Hebbian weights
- [ ] PyTorch cross-attention memory module — ShrimPK as transformer memory (v1.0+ ML stage)
- [ ] GAAMA paper (arXiv 2603.27910) — concept-mediated KG with 4 node types, very close to ShrimPK architecture
- [ ] Reflexion pattern — self-improvement via failure memories (Shinn et al. 2023)
- [ ] Interleaved replay during sleep consolidation — novel-familiar mixing (neuroscience pattern)
- [ ] EWC (Elastic Weight Consolidation) — prevent catastrophic forgetting in Hebbian updates (Nature Comms 2025)

---

## Sync Issues (fix before next release)

- [ ] `docs/ROADMAP.md` stale at v0.5.0 — update to reflect v0.7.5 state
- [ ] `CHANGELOG.md` stops at v0.7.0 — missing v0.7.1 through v0.7.5
- [ ] MCP tool count inconsistent across docs (12 vs 14)

---

*Last updated: 2026-04-09*
