# ShrimPK Roadmap

This roadmap reflects the current state of the kernel and planned directions for future releases.
Dates are aspirational. Contributions are welcome at any stage — see the Contribution Opportunities
section for specific items you can pick up today.

---

## Current State — v0.5.0

Released March 2026. The core pipeline is stable and benchmarked.

### What is shipped and working

**Echo pipeline**

The full retrieval chain is operational: Bloom filter pre-screening (O(1) topic elimination),
LSH candidate retrieval (sub-linear at scale), cosine reranking, Hebbian co-activation boosting,
and recency decay. Optional HyDE (hypothetical document expansion) and LLM reranking are
available via config flags.

**Text memory — BGE-small-EN-v1.5**

Primary embedding model: `BAAI/bge-small-en-v1.5` via fastembed. The pipeline achieves 84%
top-3 recall (combined HyDE + LLM reranker config) on a realistic 41-memory, 25-query benchmark
spanning five LongMemEval categories: information extraction, multi-session reasoning, temporal
reasoning, knowledge update, and preference tracking. Temporal queries hit 100% (5/5) across
all pipeline configs.

**Vision memory — CLIP ViT-B/32**

Image memories are embedded using CLIP ViT-B/32 (512-dim) via fastembed's `ClipVitB32` variant.
Cross-modal retrieval (text queries retrieving image memories) works in the same embedding space.
The vision feature is gated behind `--features vision`.

**Sleep consolidation**

A background consolidation pass runs during idle periods (configurable schedule). It uses a local
LLM via Ollama to extract atomic facts from raw memories, de-duplicate, and merge related entries.
In benchmarks, consolidation lifted top-3 recall from 72% to 76% over the baseline (no
consolidation) configuration.

**SHRM v2 storage format**

Memory-mapped binary format with 32-bit CRC per entry, atomic flush, and crash recovery. Stores
text embeddings (384-dim), optional vision embeddings (512-dim), optional speech embeddings
(896-dim field, populated from v0.6.0 onward), metadata, and sensitivity labels.

**Speech architecture (structure only)**

`shrimpk-memory/src/speech.rs` defines the full `SpeechEmbedder` struct with dimension constants
(`SPEAKER_DIM=512`, `PROSODY_DIM=384`, `SPEECH_DIM=896`), Whisper log-Mel preprocessing, and
ONNX sessions wired in v0.6.0. The 16 kHz resampler uses linear interpolation.

**MCP server**

`shrimpk-mcp` exposes nine tools over stdio: `store`, `echo`, `forget`, `stats`, `status`,
`config_show`, `config_set`, `dump`, `persist` (plus `store_image` and `store_audio` when
multimodal features are enabled). Compatible with Claude Desktop and any MCP client.

**Daemon + tray**

`shrimpk-daemon` runs as a background HTTP service on `localhost:11435`. `shrimpk-tray` provides
a system tray icon and launch/stop controls on Windows.

**Performance (release build, i7-1165G7)**

| Metric | Result |
|--------|--------|
| P50 echo latency at 10K memories | 3.50ms |
| P50 echo latency at 100K memories | 23.79ms (regression — see Known Issues) |
| Store throughput | ~128 memories/sec |
| RAM (10K text memories) | ~85 MB |

---

## v0.6.0 — Speech and Vision Upgrade

Target: Q2 2026. Focus: wire the speech ONNX models and upgrade the vision model.

### Speech: Wire ONNX models (896-dim after emotion removal)

The speech pipeline is **896-dim** (ECAPA-TDNN 512 + Whisper-tiny encoder 384). The emotion
channel (Wav2Small, CC-BY-NC-SA-4.0) was dropped as license-incompatible. Both wired models
carry permissive licenses: ECAPA-TDNN (Apache-2.0) and Whisper-tiny (MIT).

#### ECAPA-TDNN 512-dim — speaker identification

Model: `Wespeaker/wespeaker-voxceleb-ecapa-tdnn512` (`voxceleb_ECAPA512.onnx`, 24.9 MB,
Apache 2.0). Loaded via `ort` (ONNX Runtime Rust crate).

Input: 80-bin Kaldi log-Mel filterbank features, shape `(batch, frames, 80)`, 25ms frame,
10ms hop, 16 kHz.
Output: 512-dim L2-normalized speaker embedding.

#### Whisper-tiny encoder 384-dim — prosody

Model: `onnx-community/whisper-tiny` (`onnx/encoder_model.onnx`, 32.9 MB, MIT). The encoder
takes 80-bin Whisper log-Mel spectrogram, shape `(batch, 80, 3000)`, padded to 30 seconds.
Mean-pooling over the sequence dimension produces a 384-dim prosody vector.

#### Spectrogram preprocessing

Two spectrogram pipelines run in parallel:

- **Kaldi fbank** for ECAPA-TDNN: 80 Mel bins, 25ms frame, 10ms hop, 16 kHz. Implementation via
  the `mel-spec` crate (v0.3.4, MIT).
- **Whisper log-Mel** for the encoder: 80 Mel bins, N_FFT=400, hop=160 samples, normalized as
  `(log_spec + 4.0) / 4.0`. Also handled by `mel-spec`.

#### Band-limited resampling

The current `resample_linear()` stub in `speech.rs` introduces aliasing at high downsample ratios
(e.g., 48 kHz → 16 kHz). v0.6.0 replaces it with the `rubato` crate (v1.0.1), which provides
sinc-interpolation and FFT-based resamplers that are alias-free.

#### VAD gate — Silero VAD

A Voice Activity Detection pass runs before the ECAPA and Whisper sessions. Silent frames
(below a configurable threshold) are skipped entirely to avoid embedding noise as speech.
Silero VAD is loaded as a small ONNX model (~2 MB, MIT license) via a direct `ort::Session`.
The `silero-vad` crate on crates.io is GPL-2.0 and is explicitly avoided — the ONNX model
is loaded directly.

#### ort version pinning

fastembed v5.x pins `ort = "=2.0.0-rc.11"`. The speech code must use the exact same version
to avoid Cargo dependency conflicts. Do not add `ort` as a direct workspace dependency with a
different version specifier.

#### Model download on first use

Models are downloaded on first `SpeechEmbedder::from_config()` call if not already cached,
following the fastembed pattern: `hf-hub` crate + `dirs::cache_dir()/shrimpk/models/speech/`.
Total first-use download: ~60 MB (ECAPA 25 MB + Whisper encoder 33 MB + Silero VAD 2 MB).

### Vision: CLIP ViT-B/32 → Nomic Embed Vision v1.5 (512 → 768-dim)

`NomicEmbedVisionV15` is already a first-class variant in fastembed v5 (`ImageEmbeddingModel`
enum). The swap is a single-line change in `embedder.rs`. The quality improvement is substantial:
+7.8 percentage points on ImageNet zero-shot (71.0% vs 63.2%) and dramatically better cross-modal
MTEB quality (62.28 vs 43.82 for the paired text model). The q4-quantized ONNX is 62 MB vs
CLIP's unquantized 352 MB — a 6x size reduction.

The 512 → 768 dimension change is a **breaking migration** for stored vision embeddings. The
SHRM v2 format header records embedding dimensions per modality. On first launch after upgrade,
the kernel will detect the dimension mismatch, re-embed all stored vision memories, and rewrite
the store. For the v0.5.0 → v0.6.0 transition the user base is small and a hard-cut re-embed
is the correct strategy. A migration guide will be included in the release notes.

Cross-modal text queries against vision memories must use Nomic Text v1.5 with the mandatory
`search_query:` prefix. This is handled internally by the embedder — callers do not need to
add the prefix manually.

### Fix: 100K latency regression

The P50 latency at 100K memories is 23.79ms against a 4.0ms target. Investigation is required
before v0.6.0 ships. See Known Issues for details.

---

## v0.7.0 — Robotics, Speaker Upgrade, and Quantization

Target: Q3 2026. Focus: ROS2 integration, model quality improvements, and memory footprint.

### ROS2 bridge — `shrimpk-ros2` crate

A new workspace crate `crates/shrimpk-ros2` will provide a ROS2 node that exposes ShrimPK
memory over standard ROS2 topics and services.

The node subscribes to:
- `/shrimpk/store/text` (`std_msgs/String`) — text memories
- `/shrimpk/store/image` (`sensor_msgs/CompressedImage`) — visual memories via CLIP
- `/shrimpk/store/audio` (`audio_common_msgs/AudioStamped`) — speech memories

The node publishes to:
- `/shrimpk/echo` (`shrimpk_msgs/EchoResults`) — push-activated memories
- `/shrimpk/context` (`std_msgs/String`, latched) — current context string for downstream LLMs
- `/shrimpk/status` (`std_msgs/String`, JSON) — health and latency stats

A `/shrimpk/query` service (`shrimpk_msgs/EchoQuery`) supports pull-based querying for nodes
that prefer request/response semantics over the push model.

Primary integration path: `rclrs` 0.7+ with colcon on ROS2 Jazzy (Ubuntu 24.04).
Alternative: `r2r` for simpler `cargo build` integration without colcon.
Optional feature flag: `ros2-native` using `ros2-client` (pure Rust DDS, no ROS2 install needed)
for distribution to users who do not have a full ROS2 environment.

The echo latency budget is feasible: 3.50ms ShrimPK echo is well within a 30 Hz camera frame
(33ms). The full pipeline including embedding and topic publish should stay under 15–20ms.

No other push-based memory system has a ROS2 bridge. ReMEmbR (NVIDIA) is pull-based and
Python-only. `shrimpk-ros2` would be the first native-Rust, push-activated memory layer for ROS2.

### Speaker upgrade: ECAPA-TDNN → CAM++

CAM++ (Context-Aware Masking) achieves lower equal error rate than ECAPA-TDNN on VoxCeleb1/2
at comparable model size. The upgrade is a drop-in replacement at the 512-dim output level
provided an Apache 2.0-compatible ONNX export is available. If no suitable pre-built ONNX exists,
the ECAPA-TDNN model ships in v0.7.0 and CAM++ is deferred to v0.8.0.

### f16 quantization for vision and speech embeddings

Stored vision and speech embeddings currently use f32 (4 bytes/dimension). A v0.7.0 storage
format revision (SHRM v3) will store these as f16 (2 bytes/dimension) with promotion to f32
at query time. Impact: ~50% reduction in disk and memory footprint for vision/speech memories,
no measurable quality loss for cosine similarity.

SHRM v3 will include automatic migration from v2 on first launch.

---

## Future — No Fixed Timeline

These items are research directions or require dependencies that are not yet settled.

### Custom fine-tuned embedding model

The text embedding model (BGE-small) is a general-purpose model trained on web text. A model
fine-tuned specifically on personal memory data (short episodic sentences, user preferences,
recurring entities) could improve recall quality without increasing model size. This requires
a labeled dataset and an ML training pipeline — it is a research item, not an implementation task.

### crates.io publish

Publishing `shrimpk-core`, `shrimpk-memory`, and (eventually) `shrimpk-ros2` to crates.io
is planned once the API stabilizes beyond v0.6.0. The current pre-1.0 semver signals that
breaking changes are expected.

### Cloud sync

Optional encrypted sync of the memory store across devices. End-to-end encrypted, the server
sees only ciphertext. The key design question is key management — the server must never hold
decryption keys. This is a future research and design item.

### Emotion channel

The 3-dim arousal/dominance/valence emotion channel is architecturally present in `speech.rs`
(`EMOTION_DIM=3`) but has no available ONNX model under a permissive license. If a suitable
Apache 2.0 or MIT model emerges, the emotion channel can be re-enabled without a breaking change
to the storage format (the slot is reserved). Alternatively, a categorical speech emotion
recognition model (4-class: angry, happy, sad, neutral) under a permissive license could
replace the dimensional approach.

---

## Contribution Opportunities

All issues below are open for contribution. The project uses Apache 2.0. Opening a discussion
issue before starting significant work is encouraged to avoid duplication.

### Good first issue

**Fix vision feature flag propagation** (difficulty: low, Rust knowledge required)
Vision benchmarks (`echo_multimodal_bench.rs`) are blocked because
`#[cfg(feature = "vision")]` checks the root test crate's features, not `shrimpk-memory`'s.
The fix is adding a forwarding `vision` feature to the root `Cargo.toml` that enables
`shrimpk-memory/vision`. Estimated: 1–2 hours.

**Add `search_query:` prefix for cross-modal text queries** (difficulty: low, Rust)
When Nomic Embed Vision v1.5 is the active vision model (v0.6.0), text queries used in
cross-modal retrieval must be prefixed with `"search_query: "`. This should be applied
automatically in `MultiEmbedder` when the Nomic vision model is active, not pushed to callers.
Requires reading the fastembed API and adding a model-variant check.

**Extend the Tier 2 benchmark with a CrossEncoder config** (difficulty: low, Rust)
The realistic Tier 2 benchmark tests four pipeline configs (Baseline, HyDE, Reranker-LLM,
Combined). A CrossEncoder-only config was benchmarked separately and showed strong results
(2,823ms average at 100% recall on 6 regression cases). Adding it to the standard Tier 2
suite would complete the comparison matrix.

### Help wanted

**Investigate 100K latency regression** (difficulty: medium, Rust + profiling)
P50 at 100K memories is 23.79ms against a 4.0ms target. Likely causes: LSH bucket saturation
with BGE-small embedding distribution, brute-force fallback frequency, or Windows I/O interference
during the benchmark. The investigation should profile LSH hit rate, Bloom false-positive rate,
and brute-force fallback frequency at scale. Tools: `perf`, `cargo flamegraph`, or the
`tracing` spans already in the echo path. A fix might involve tuning LSH parameters
(hash count, bucket width) for the BGE-small distribution.

**Wire ECAPA-TDNN ONNX session** (difficulty: medium, Rust + ONNX Runtime)
`SpeechEmbedder::from_config()` has a clear placeholder comment for where to load
`ort::Session` instances. The ECAPA-TDNN model (`wespeaker-voxceleb-ecapa-tdnn512`,
Apache 2.0) is identified and the input format is documented in the codebase. This item
requires implementing the Kaldi fbank spectrogram using the `mel-spec` crate, loading the
model via `ort`, and running the `feats` → `embs` inference. The `ort` version must match
fastembed's pinned `=2.0.0-rc.11` exactly.

**Wire Whisper-tiny encoder ONNX session** (difficulty: medium, Rust + ONNX Runtime)
Companion to the ECAPA item above. The Whisper encoder takes `(batch, 80, 3000)` log-Mel
spectrogram and outputs `(batch, 1500, 384)` hidden states, mean-pooled to `(batch, 384)`.
Preprocessing uses the Whisper log-Mel formula implemented in `mel-spec`. Can be done in
parallel with the ECAPA item by a different contributor.

**Implement band-limited resampling with `rubato`** (difficulty: medium, Rust + DSP)
Replace `resample_linear()` in `speech.rs` with sinc or FFT-based resampling from the `rubato`
crate (v1.0.1). The current linear resampler causes aliasing at high downsample ratios and is
documented as a placeholder. The replacement should pass the existing `resample_*` unit tests
and add a new test verifying that a 1 kHz sine wave downsampled from 48 kHz to 16 kHz does not
contain aliasing artifacts above 8 kHz.

**Linux CI hardening** (difficulty: medium, DevOps + Rust)
The kernel builds and tests pass on CI for Linux and macOS, but the test coverage is lower than
on the primary Windows development machine. Specifically: daemon startup tests, tray icon tests,
and file locking tests need Linux-specific validation. Contributions improving Linux CI coverage
are welcome.

### Research needed

**Emotion model under permissive license** (difficulty: high, ML research)
The 3-dim arousal/dominance/valence emotion slot in the speech pipeline is reserved but empty
because all mature dimensional emotion models (Wav2Small, wav2vec2-large-robust) carry
CC-BY-NC-SA-4.0 licenses. Options: (1) identify an existing Apache 2.0 / MIT categorical
speech emotion model that can be exported to ONNX and mapped to a valence proxy, (2) train a
small distillation model on CC0 or public-domain audio corpora, or (3) propose an alternative
paralinguistic dimension that has available permissive models.

**LSH parameter tuning for BGE-small distribution** (difficulty: high, information retrieval)
The LSH index was tuned for `all-MiniLM-L6-v2` embeddings. The upgrade to `BGE-small-EN-v1.5`
changed the embedding distribution in ways that may require different hash count, bucket width,
or candidate list size to maintain sub-10ms P50 at 100K scale. This is an empirical research
task: vary LSH parameters, run the 100K latency benchmark, and identify the configuration that
recovers the 4.0ms target.

**CAM++ Apache 2.0 ONNX availability** (difficulty: medium, ML research)
The v0.7.0 speaker upgrade to CAM++ depends on finding or producing an Apache 2.0-compatible
ONNX export. WeSpeaker provides CAM++ checkpoints but the license status of any pre-built
ONNX exports needs verification. This research item should produce a clear verdict: model ID,
license, ONNX file location, and input/output specification.

**SigLIP 2 fastembed support** (difficulty: high, ML + Rust)
SigLIP 2 ViT-B/16 achieves 78.2% ImageNet zero-shot (vs Nomic Vision v1.5 at 71.0%) but has
no official ONNX model and no fastembed support as of March 2026. If an Apache 2.0 ONNX export
emerges, contributing a `SigLIP2VitB16` variant to fastembed and then updating ShrimPK's
vision channel would be a meaningful quality improvement.
