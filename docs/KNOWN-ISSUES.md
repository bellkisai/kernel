# Known Issues

This document lists known limitations, bugs, and architectural gaps in the current release.
It is updated alongside each release. Where an issue is also a contribution opportunity, that
is noted.

---

## Open Issues

### 1. 100K Latency Regression

**Severity:** High
**Component:** `shrimpk-memory` — echo pipeline, LSH index
**Status:** Under investigation

At 10,000 stored memories the P50 echo latency is 3.50ms (measured on i7-1165G7, release build).
At 100,000 stored memories the P50 is **23.79ms**, against a 4.0ms target. P95 is 54.58ms and
P99 is 62.59ms.

The 4.0ms target was established with the previous text embedding model (`all-MiniLM-L6-v2`).
The v0.5.0 upgrade to `BGE-small-EN-v1.5` changed the embedding distribution. LSH index
parameters (hash count, bucket width, candidate list size) were not re-tuned for the new
distribution, which is the most likely cause of the regression.

**Possible causes, in order of likelihood:**

1. LSH bucket saturation at 100K scale — the BGE-small distribution may be clustered in ways
   that cause many queries to fall through to brute-force cosine search. If the LSH hit rate
   (number of queries resolved from LSH vs brute-force fallback) drops significantly at 100K,
   this is the root cause.

2. Bloom filter false-positive rate at 100K — the Bloom filter is sized at initialization time.
   At 100K entries its false-positive rate may be high enough to pass many entries to the
   embedding comparison stage unnecessarily.

3. Windows I/O interference — the 100K benchmark requires ~13 minutes of store writes before
   the latency measurement. Background OS activity on the development machine during this period
   could inflate the result. Reproducing on a dedicated Linux machine with CPU isolation is
   needed to rule this out.

4. Memory pressure — at 100K f32 text embeddings (384-dim × 4 bytes × 100K = ~153 MB), the
   store may be evicted from the CPU cache between the write phase and the latency measurement.
   The v0.4.0 benchmark on `all-MiniLM-L6-v2` (same 384-dim) did not show this behavior,
   which makes model distribution the more likely cause.

**Update (v0.6.0 label bucket benchmark):**

Label-based pre-filtering (ADR-015) reduced 100K P50 from 38.94ms to 27.70ms (1.4x improvement).
However, profiling revealed the **embedding step alone costs ~8ms per query** with BGE-small-EN-v1.5.
This is a fixed cost independent of store size — the previous 3.50ms target was set against
`all-MiniLM-L6-v2` which embeds roughly 2x faster. The 4.0ms target is unrealistic with the
current model.

**Revised baseline:**
- Embedding cost: ~8ms (BGE-small, fixed per query)
- Retrieval at 100K with labels: ~20ms (down from ~31ms without labels)
- Retrieval at 10K: ~2-3ms (labels not needed at this scale)

**Future investigation:**
- Verify BGE-small embedding latency vs MiniLM on same hardware
- If confirmed ~2x slower: adjust target to P50 < 15ms at 100K (embed + retrieval)
- Consider embedding result caching for repeated query patterns
- Consider reverting to MiniLM if the quality delta (MTEB 56.3 vs 62.0) is not worth the latency cost

**Conclusion (v0.6.0):**

The label bucket architecture (ADR-015, KS42-KS48) delivered a **35% retrieval improvement
at 100K scale** — P50 dropped from 38.94ms to 27.70ms. The infrastructure is built, verified,
and working: label types, inverted index, D6 three-source merge, Tier 1 generation at store
time, Tier 2 LLM enrichment during consolidation, and async bootstrap for existing stores.

The remaining gap to sub-10ms is caused by two independent factors:
1. **Embedding cost (~8ms fixed):** BGE-small-EN-v1.5 embeds ~2x slower than the previous
   MiniLM model. This is a per-query floor that no retrieval optimization can address.
2. **Label granularity:** Current Tier 1 labels produce ~15 topic-level buckets (~6.6K entries
   each at 100K). Entity-level labels (`entity:rust`, `entity:tokyo`) would create 50-500
   entry buckets, enabling much tighter pre-filtering. GLiNER NER integration is blocked on
   an `ort` version conflict (gline-rs pins rc.9, fastembed needs rc.10+).

**Future work (revisit when the board is cleaner):**
- [ ] Verify MiniLM vs BGE-small embedding latency on same hardware — quantify the 2x claim
- [ ] Monitor gline-rs for ort version update — unblocks entity-level label granularity
- [ ] Evaluate embedding result caching for repeated query patterns
- [ ] Consider Matryoshka dimension reduction (384 -> 256) for faster cosine at scale
- [ ] Explore async/batched embedding to amortize model inference overhead
- [ ] Re-run 100K benchmark after entity labels are available

**Workaround:** At 10K memories, latency is well within target (~8-11ms including embedding).
At 100K, labels provide a 35% retrieval improvement. The architecture scales to finer-grained
labels when available — no structural changes needed, only richer label generation.

**Contribution opportunity:** See ROADMAP.md — "Investigate 100K latency regression" (Help wanted).

---

### 2. Speech Models Not Wired

**Severity:** Medium (feature is absent, not broken)
**Component:** `shrimpk-memory/src/speech.rs`
**Status:** Planned for v0.6.0

The `SpeechEmbedder` struct, dimension constants, and preprocessing stubs are present in the
codebase. `SPEECH_DIM = 896` (after the emotion channel removal — see issue 4 below).
The SHRM v2 format stores a `speech_embedding` field in every `MemoryEntry`, but it is always
`None` in v0.5.0.

Calling `SpeechEmbedder::embed()` returns an error:
```
Speech models not loaded. Install ONNX models or enable speech feature with model paths.
```

This is expected behavior. The ONNX sessions for ECAPA-TDNN (speaker, 256-dim) and
Whisper-tiny encoder (prosody, 384-dim) are not yet loaded. The architecture, input format
specifications, and model identifiers are fully documented in the codebase and in the
`shrimpk-memory/src/speech.rs` module docstring.

**Impact:** Audio memories cannot be stored or retrieved. The `Modality::Speech` enum variant
exists and round-trips through serialization correctly. No crash or data corruption occurs.

**Plan (v0.7.0):** KS50 will wire the ONNX sessions using `ort` (pinned to fastembed's
`=2.0.0-rc.11`), implement Whisper log-Mel preprocessing for the encoder, replace the linear
resampler with `rubato` for alias-free sample rate conversion, and add `hound` for WAV decode.
Note: Kaldi fbank (ECAPA-TDNN) preprocessing and Silero VAD are planned for a later sprint.

**Contribution opportunity:** See ROADMAP.md — "Wire ECAPA-TDNN ONNX session" and
"Wire Whisper-tiny encoder ONNX session" (Help wanted).

---

### 3. Vision Feature Flag Propagation

**Severity:** Medium (benchmarks blocked)
**Component:** Root `Cargo.toml`, workspace integration tests
**Status:** Needs fix, straightforward

Vision benchmark tests in `echo_multimodal_bench.rs` use `#[cfg(feature = "vision")]` to
gate vision-specific test cases. When running:

```bash
cargo bench --features shrimpk-memory/vision
```

the `vision` feature is enabled on the `shrimpk-memory` library crate but is not visible to
the root test crate's `cfg` directives. As a result, the `#[cfg(feature = "vision")]` blocks
in the integration tests evaluate to false and all five vision benchmarks are skipped.

**Affected benchmarks:**
- CLIP image embed latency (target: P50 < 100ms)
- Cross-modal echo latency (target: P50 < 15ms)
- Text-to-image recall accuracy
- RAM measurement (target: < 1 GB with vision model loaded)
- Mixed-modal throughput

**Fix:** Add a `vision` feature to the root workspace `Cargo.toml` that forwards to
`shrimpk-memory/vision`. Then run benchmarks with `--features vision` instead of
`--features shrimpk-memory/vision`.

```toml
# In root Cargo.toml [features]:
[features]
vision = ["shrimpk-memory/vision"]
```

This is a one-line change. The underlying vision code is correct — only the test feature
propagation is broken.

**Contribution opportunity:** See ROADMAP.md — "Fix vision feature flag propagation"
(Good first issue).

---

### 4. Emotion Channel Removed — License Incompatibility

**Severity:** Low (design decision, not a bug)
**Component:** `shrimpk-memory/src/speech.rs`
**Status:** Permanent until a permissive model is available

The original speech pipeline design included a 3-dim emotion embedding (arousal, dominance,
valence) via Wav2Small (`audeering/wav2small`). Wav2Small is CC-BY-NC-SA-4.0, which forbids
commercial use and requires ShareAlike. This is incompatible with ShrimPK's Apache 2.0 license.

The emotion channel has been removed. The wired speech pipeline is 640-dim (ECAPA-TDNN 256 +
Whisper-tiny 384) as confirmed in KS51. `SPEECH_DIM = 640` and `EMOTION_DIM` is gone.

**What this means for stored data:** The SHRM v2 format field `speech_embedding` is a variable-
length `Vec<f32>`. If a future version re-adds an emotion channel, the stored dimension will
differ from v0.6.0 speech embeddings. A migration step will be required at that point.

**No alternatives under permissive licenses exist today.** All dimensional speech emotion models
(arousal/dominance/valence) from audEERING carry non-commercial licenses. A categorical
4-class model (angry, happy, sad, neutral) under a permissive license is under evaluation as a
potential future replacement, but introduces different semantics than the dimensional space.

**Contribution opportunity:** See ROADMAP.md — "Emotion model under permissive license"
(Research needed).

---

### 5. Cross-Platform — Windows Primary, Linux/macOS Less Tested

**Severity:** Low
**Component:** Daemon, tray, file locking, IPC
**Status:** Ongoing

The primary development environment is Windows 11. The CI pipeline runs on Linux and macOS and
the core library tests pass. However, several OS-specific subsystems are less exercised on
non-Windows platforms:

- **Daemon startup and port binding** — tested on Windows. Linux behavior (systemd socket
  activation, port conflicts) is covered by basic CI but not by integration tests.
- **Tray icon** (`shrimpk-tray`) — uses `tao`/`tray-icon`. The Windows path is well-exercised.
  Linux (X11/Wayland) and macOS menu bar paths receive less manual testing.
- **File locking** — uses `fs2`. Behavior under NFS or networked filesystems is untested.
- **Keystore** — uses OS native keystores (Windows Credential Manager, macOS Keychain,
  Linux Secret Service). The Linux path requires `libsecret` installed on the system.

If you encounter a Linux or macOS-specific failure that is not covered by a CI test, opening
an issue with the OS version, distribution, and exact error message is helpful. Fixes that add
Linux-specific test coverage are welcome.

**Contribution opportunity:** See ROADMAP.md — "Linux CI hardening" (Help wanted).

---

### 6. Sleep Consolidation Requires Ollama

**Severity:** Low (feature is optional)
**Component:** `shrimpk-memory` — consolidation, `shrimpk-daemon`
**Status:** By design, alternatives under consideration

The sleep consolidation pass (background fact extraction and de-duplication) requires a locally
running Ollama instance with a compatible model loaded. The default configuration uses
`llama3.2:3b` (Q4_K_M, ~2 GB download). If Ollama is not running, consolidation is silently
skipped and the daemon continues operating in a non-consolidating mode.

**What works without Ollama:**
- Memory storage (all modalities)
- Echo retrieval (Bloom + LSH + cosine + Hebbian)
- HyDE query expansion (also uses Ollama — also skipped if unavailable)
- All MCP tools except consolidation-dependent features

**What does not work without Ollama:**
- Sleep consolidation (fact extraction, de-duplication, merging)
- LLM reranker (the `Combined` and `Reranker` pipeline configs fall back to baseline ranking)
- HyDE query expansion

**Workaround:** Install Ollama and run `ollama pull llama3.2:3b` before starting the daemon.
The installation guide in `README.md` covers this step.

**Future directions:** A built-in lightweight summarizer that does not require an external
process is on the long-term roadmap. This would likely be a small ONNX model for
sentence-level clustering and deduplication, without the full generative capability used for
fact extraction. No timeline is set.

---

## Resolved in v0.5.0

The following issues from earlier versions were fixed in v0.5.0:

- **LongMemEval baseline misses on TR-7 and MSR-2** — temporal and multi-session reasoning
  failures in the baseline config (no consolidation, no HyDE) are resolved when any augmented
  pipeline config is used (HyDE, reranker, or combined). The baseline config remains 4/6 on
  the regression diagnostic; all augmented configs reach 6/6.

- **CrossEncoder first-query cold load (16 sec)** — the cross-encoder model load on first use
  was 16,644ms due to ONNX Runtime session initialization. Subsequent queries average ~59ms.
  This is expected behavior for ONNX Runtime and is documented. A future optimization could
  pre-warm the session at daemon startup.
