# ShrimPK Architecture

ShrimPK is a push-based AI memory kernel written in pure Rust. It gives AI agents persistent, associative memory that activates automatically — the agent never has to ask "do you remember?" because relevant memories surface on their own.

This document describes the engineering behind the system: the data structures, the retrieval pipeline, the multimodal design, the persistence format, and the crate organization. It assumes familiarity with Rust and basic information retrieval concepts.

---

## Table of Contents

1. [Push vs. Pull Memory](#1-push-vs-pull-memory)
2. [The Echo Pipeline](#2-the-echo-pipeline)
3. [Multimodal Architecture](#3-multimodal-architecture)
4. [Core Data Structures](#4-core-data-structures)
5. [Persistence: The SHRM Format](#5-persistence-the-shrm-format)
6. [Sleep Consolidation](#6-sleep-consolidation)
7. [Crate Architecture](#7-crate-architecture)
8. [Performance Characteristics](#8-performance-characteristics)
9. [Configuration Reference](#9-configuration-reference)

---

## 1. Push vs. Pull Memory

Most AI memory systems are pull-based: the host application detects that memory might be relevant, constructs a search query, calls a memory API, and injects the results into the prompt. The application owns the retrieval decision.

ShrimPK is push-based. The kernel continuously monitors the conversation, scores stored memories against the current context, and injects those that exceed a relevance threshold — without the application asking. The application stores memories and processes echo results; the kernel decides what is relevant and when.

```
Pull model (Mem0, Hindsight, most RAG systems):

  Application ---"search('API key config')"---> Memory Store
  Application <---[result set]---------------- Memory Store
  Application decides: "these seem relevant, I'll inject them"


Push model (ShrimPK):

  Application ---store("Prefers FastAPI")-----> Echo Engine
                                                    |
  Context changes, query arrives                    |
                                                    v
                                            [Echo Cycle fires]
  Application <---[Activated memories]---- Echo Engine
  No search query needed.
```

This difference has practical consequences:

- **Invisible to agent logic.** An agent built on ShrimPK does not need memory-retrieval code. It receives a context window that already contains activated memories, indistinguishable from injected system prompt content.
- **Associative, not keyword-based.** Retrieval is driven by semantic embedding similarity, Hebbian co-activation, and recency, not by the agent's ability to phrase a good search query.
- **Self-reinforcing.** Every echo cycle strengthens the associations it activates. Memories that keep co-occurring accumulate Hebbian weight and surface faster on future queries.

The "invisible memory" property is the core design constraint. Every engineering decision downstream is evaluated against it.

---

## 2. The Echo Pipeline

The Echo pipeline is the kernel's central retrieval mechanism. When a query arrives, the pipeline scores all stored memories and returns those that exceed the relevance threshold, with a final score that blends semantic similarity, associative history, and recency.

```
Query text
    |
    v
[PII / sensitivity gate]  -- blocked content never reaches retrieval
    |
    v
[HyDE expansion (optional)] -- hypothetical answer improves sparse query recall
    |
    v
[Text Embedding]  -- BGE-small-EN-v1.5, 384-dim
    |
    v
[Bloom Filter]  -- O(1) skip: "no word in this query matches any stored memory"
    |                  if no fingerprint match => return empty, done
    v
[LSH Candidate Retrieval]  -- 16 tables x 10 bits: returns ~1-5% of store
    |                             fallback to brute-force if < 10 candidates
    v
[Cosine Similarity (SIMD)]  -- exact scoring on LSH candidates only
    |
    v
[Split: Pipe A / Pipe B]
    |                  \
    |                   Pipe B: near-misses with enriched children
    |                   -> "child rescue": a child fact scores >= threshold
    |                      => promotes parent into Pipe A
    v
[Merge Pipe A + promoted]
    |
    v
[Hebbian Co-activation]  -- two passes:
    |                          1. strengthen edges between all returned memories
    |                          2. boost results with high-weight neighbors
    v
[Recency Decay]  -- category-aware half-life; newer memories score higher
    |
    v
[Reranker (optional)]  -- cross-encoder ONNX (~5-15ms) or LLM reranker (~2s)
    |
    v
[EchoResult list]  -- sorted by final_score, capped at max_echo_results
```

Each stage exists for a specific reason. The following sections explain the design rationale for each one.

### 2.1 PII Filtering (at store time)

Before any memory is stored, the `PiiFilter` scans the text for sensitive patterns: API keys (`sk-`, `sk-proj-`, `sk-ant-`, `AKIA`, `gsk_`, `xai-`, `ghp_`, `glpat-`), credit card numbers, SSNs, email addresses, and phone numbers.

Detected patterns are replaced with masked tokens (e.g., `[REDACTED:ApiKey]`) in the stored content, but the embedding is generated from the *original* text so semantic meaning is preserved. The original text is never written to disk.

Content classified as `Blocked` (credentials, secrets) is rejected outright — not stored at all. This classification happens synchronously before the embed step.

The sensitivity level assigned at store time (`Public`, `Private`, `Restricted`, `Blocked`) controls whether a memory can be pushed to cloud providers or only to local models.

### 2.2 Memory Reformulation (at store time)

Before embedding, the `MemoryReformulator` applies regex-based pattern matching to rewrite natural-language text into structured forms that produce higher cosine similarity scores.

Example rewrites:

```
"I prefer FastAPI for REST APIs"  ->  "Preference: FastAPI for REST APIs"
"I use Neovim"                    ->  "Tool/technology: Neovim"
"Yesterday I switched to Neovim"  ->  "Temporal: yesterday — Yesterday I switched to Neovim"
```

Structured prefixes improve recall by approximately 9% on preference-tracking queries because the embedding space responds better to consistent patterns than to the variability of natural speech. The original text is stored for display; the reformulated text is used only for embedding.

### 2.3 Bloom Filter

The Bloom filter provides an O(1) pre-check: "can any stored memory possibly match this query?" If the answer is definitively no, the entire embedding and retrieval pipeline is skipped.

The filter is text-fingerprint-based, not embedding-based. Each memory text is tokenized into unigrams (individual words, 3+ characters) and bigrams (adjacent word pairs joined with `_`). These fingerprints are inserted into a probabilistic Bloom filter at store time.

At query time, the same fingerprint extraction is applied to the query. If fewer than 1 fingerprint from the query appears in the filter, the query cannot match any stored memory and the pipeline returns immediately.

```
Memory: "Rust is a systems programming language"
Fingerprints: ["rust", "systems", "programming", "language",
               "rust_systems", "systems_programming", "programming_language"]

Query: "chocolate cake recipe"
Fingerprints: ["chocolate", "cake", "recipe", "chocolate_cake", "cake_recipe"]
-> 0 matches -> return empty immediately
```

The filter is configured for 1% false-positive rate at 1M items, consuming approximately 1.14 MB. Bloom filters cannot remove items, so after deletions the filter is rebuilt from scratch during the next consolidation cycle. The `bloom_dirty` flag tracks whether a rebuild is needed.

The filter is bypassed for stores with fewer than 50 entries, where the overhead outweighs the benefit and false negatives from immature filter state could be harmful.

### 2.4 LSH (Locality-Sensitive Hashing)

With the Bloom filter passed, the next challenge is candidate retrieval. Brute-force cosine similarity is O(N * D) — at 100,000 memories with 384-dimensional embeddings, that is 38.4M float multiplications per query. LSH reduces this to O(L * K * D) independent of N.

ShrimPK uses random hyperplane hashing (the SimHash variant), which preserves cosine similarity in the hash space. The implementation (`crates/shrimpk-memory/src/lsh.rs`) maintains L independent hash tables, each with K random hyperplanes.

**How a single table works:**

1. Generate K unit vectors drawn from a Gaussian distribution and L2-normalize them. These are the "hyperplanes."
2. For any embedding vector `v`, compute `dot(v, h_i)` for each hyperplane `h_i`. If the dot product is >= 0, set bit `i` to 1; otherwise 0. Pack the K bits into a `u64` hash code.
3. Store the memory index in the bucket for that hash code.

Two vectors with high cosine similarity will frequently hash to the same code — they tend to fall on the same side of each hyperplane. Two unrelated vectors will hash to the same code by chance at a rate determined by K.

**Multi-probe extension:** The query hashes into each of the L tables. In addition to the exact bucket, all K Hamming-1 neighbors (buckets reachable by flipping one bit) are also retrieved. This improves recall for near-threshold pairs without requiring additional tables.

```
L = 16 tables, K = 10 bits per table:
- Each bucket holds N / 2^K = N / 1024 entries on average
- With 16 tables + multi-probe (16 * 10 = 160 neighbor lookups),
  candidate count is roughly 1-5% of N for well-distributed embeddings
- vs. brute-force: 100% of N
```

The reverse index (`reverse_index: HashMap<u32, Vec<(usize, u64)>>`) stores each entry's table/bucket locations, enabling O(L) deletion. The LSH index degrades gracefully: if fewer than 10 candidates are returned, the engine falls back to brute-force similarity on all stored embeddings.

Per-channel LSH instances are maintained with dimension-matched hyperplanes: 384-dim for text (BGE), 512-dim for vision (CLIP), 896-dim for speech.

### 2.5 Cosine Similarity (SIMD-Accelerated)

Exact cosine similarity is computed only on LSH candidates. The implementation (`crates/shrimpk-memory/src/similarity.rs`) uses the `simsimd` crate for SIMD-accelerated distance computation.

`simsimd` auto-detects the available instruction set (AVX2, NEON, etc.) at runtime and dispatches accordingly. It returns cosine *distance* (1 - similarity), which is converted to similarity. A scalar fallback using f64 accumulation is used when SIMD is unavailable.

Candidates are filtered against the configured `similarity_threshold` (default 0.7) and sorted by score descending.

### 2.6 Pipe A / Pipe B Split and Child Rescue

After similarity scoring, results are split into two pipes:

- **Pipe A:** Candidates at or above `similarity_threshold`. These proceed to Hebbian scoring.
- **Pipe B:** Near-miss candidates between `0.5 * threshold` and `threshold`. These are checked for "child rescue."

Child rescue handles the case where a memory was stored as a short conversational note (e.g., "Discussed API design with the team") but a more precise atomic fact was extracted from it during consolidation (e.g., "Prefers REST over GraphQL for public APIs"). The child may score above threshold even when the parent does not.

When a Pipe B entry has enriched children (detected via the `parent_children` reverse index in `EchoStore`), the best child similarity is computed. If any child exceeds `threshold`, the parent is promoted into the merged result set with that score.

When `child_rescue_only` is true (default), children are never inserted into the LSH or Bloom indices — they are only accessible through the parent's child list. This prevents atomic fact fragments from polluting direct candidate retrieval.

### 2.7 Hebbian Co-activation

Hebbian learning is borrowed from neuroscience: "neurons that fire together wire together." In ShrimPK, every pair of memories returned by the same echo query has their Hebbian edge strengthened proportional to the geometric mean of their similarity scores.

The Hebbian graph (`crates/shrimpk-memory/src/hebbian.rs`) is a sparse, edge-weighted graph stored as `HashMap<(u32, u32), CoActivation>`. The key invariant `a < b` ensures no duplicate edges.

**Two-pass ranking:**

Pass 1 (write) — Co-activate all pairs in the current result set:
```
strength = sqrt(sim_i * sim_j) * 0.1
```

Pass 2 (read) — Compute a Hebbian boost for each result based on the decayed weights of its edges to other results in the set:
```
boost = sum of decayed_weight(idx, other) for all other in result set
boost = min(boost, 0.4)   -- capped to prevent Hebbian dominance
```

**Typed relationships:** During sleep consolidation, the graph annotates edges with semantic relationship types: `WorksAt`, `LivesIn`, `PrefersTool`, `PartOf`, `TemporalSequence`, `Supersedes`, `Custom`. Typed edges receive a small additional boost (0.05) because they carry semantic structure beyond mere co-occurrence.

The `Supersedes` relationship deserves special mention. When a newer memory contradicts an older one (e.g., "Now uses Vim" supersedes "Uses Neovim"), the consolidator creates a `Supersedes` edge from old to new. During ranking:
- The newer memory receives an additional +0.1 boost.
- The older (superseded) memory receives a configurable demotion (`supersedes_demotion`, default 0.0).

**Exponential decay:** Every edge decays continuously according to the formula:

```
decayed_weight = weight * exp(-lambda * elapsed_seconds)
where lambda = ln(2) / half_life_seconds
```

The default half-life is 7 days (604,800 seconds). Associations fade naturally as time passes. Edges with decayed weight below `prune_threshold` (default 0.01) are removed during the periodic consolidation pass.

### 2.8 Recency Decay and Category-Aware Half-Life

Two time-based adjustments are applied independently:

**Category decay** reduces the final score of old memories. The decay multiplier is:

```
decay = exp(-age_seconds * ln(2) / half_life_seconds)
```

Half-lives are assigned by memory category, recognizing that different types of information have different natural lifetimes:

| Category | Half-Life | Rationale |
|---|---|---|
| `Identity` | 365 days | Name, location, personal facts change rarely |
| `Preference` | 60 days | Tool choices evolve slowly |
| `Fact` | 30 days | Learned knowledge stays relevant for weeks |
| `ActiveProject` | 14 days | Current work context, relevant while active |
| `Default` | 7 days | Uncategorized — conservative |
| `Conversation` | 3 days | One-off discussions fade quickly |

Category is auto-assigned at store time by the reformulator based on structural patterns in the text.

**Recency boost** gives newer memories a small additive advantage:

```
recency_boost = recency_weight / (1.0 + days_since_stored)
```

At the default `recency_weight` of 0.05: a memory stored today gets +0.05, a memory stored 7 days ago gets +0.006, 30 days ago gets +0.002. Cosine similarity dominates; recency is a tiebreaker.

The final score formula:

```
final_score = (similarity + hebbian_boost + recency_boost) * category_decay
```

### 2.9 HyDE Query Expansion (Optional)

Hypothetical Document Embeddings (HyDE) improve recall for queries where the query text is semantically distant from the stored memory text. Instead of embedding the raw query, the engine calls a local LLM (via Ollama) to generate a hypothetical answer, then embeds that.

Example:
```
Query: "what editor do I like?"
HyDE expansion: "The user prefers Neovim as their primary code editor"
-> The expanded text embeds much closer to stored preference memories
```

HyDE is opt-in (`query_expansion_enabled: true`). When the Ollama call fails or times out, the engine falls back to the original query embedding. Latency cost is approximately 100-500ms for the Ollama round-trip.

### 2.10 Reranker (Optional)

The reranker is the final stage in the pipeline, applied after Hebbian + recency scoring has produced the top-N results. It reorders those results by true semantic relevance using a more expensive model.

Two backends are available:

**Cross-encoder** (`reranker_backend: "cross_encoder"`): Uses the `jina-reranker-v1-turbo-en` ONNX model (~33MB) via fastembed. The model is a true cross-encoder — it processes the query and each candidate memory jointly, allowing attention across both, rather than comparing pre-computed embeddings. This catches cases where cosine similarity misleads (e.g., lexical overlap without semantic relevance). Latency: 5-15ms on CPU.

**LLM reranker** (`reranker_backend: "llm"`): Sends the query and top results to a local LLM via Ollama, asking it to reorder by relevance. Highest quality but approximately 2 seconds latency. Suitable for offline use cases.

Both backends are opt-in and fail gracefully — if the backend call fails, the original ordering from cosine+Hebbian scoring is preserved.

---

## 3. Multimodal Architecture

ShrimPK supports three sensory channels: Text (always enabled), Vision (feature flag `vision`), and Speech (feature flag `speech`). The architecture uses separate embedding spaces per channel, not a unified joint space.

### 3.1 Why Separate Spaces

The CLIP text encoder (used for vision retrieval) scores 43.8 on MTEB compared to BGE-small-EN-v1.5's 56.3. Using CLIP for all text would regress the established text recall accuracy. Each modality uses the best available model for its domain.

```
                        +--------------------+
                        |   MultiEmbedder    |
                        +--------------------+
                       /          |           \
              Text channel   Vision channel  Speech channel
              (always on)    (feature=vision) (feature=speech)
                 |               |                |
          BGE-small-EN      CLIP ViT-B-32    ECAPA-TDNN
          v1.5 (384-dim)    (512-dim image   (512-dim) +
                            + 512-dim text   Whisper-tiny
                            encoder for      encoder (384-dim)
                            cross-modal)
                 |               |                |
          Text LSH          Vision LSH       Speech LSH
          (384-dim)         (512-dim)        (896-dim)
                  \              |               /
                   \             |              /
                    +----[Shared Hebbian Graph]--+
                         (cross-modal co-activation)
```

The Hebbian graph is shared across all channels. When a text memory and a vision memory are returned together in an Auto-mode query, the edge between them is strengthened. Over time, the graph learns cross-modal associations: the memory of a conversation co-activates with the image that was being discussed.

### 3.2 Text Channel

- **Model:** BGE-small-EN-v1.5 via fastembed
- **Dimension:** 384
- **LSH:** 16 tables, 10 bits per table
- **Bloom:** Enabled (word-level fingerprints)
- **PII filtering:** Yes (at store time)
- **Reformulation:** Yes (structured prefix injection)

### 3.3 Vision Channel

- **Model:** CLIP ViT-B-32 (image encoder) + CLIP ViT-B-32 (text encoder)
- **Dimension:** 512
- **LSH:** 16 tables, 10 bits per table (512-dim hyperplanes)
- **Bloom:** Not used (Bloom works on word fingerprints, not visual features)
- **Cross-modal retrieval:** Text queries can retrieve image memories by embedding the query with the CLIP text encoder, which shares the same latent space as the CLIP image encoder.

Images larger than 10 MB are rejected before embedding to prevent OOM during ONNX inference.

**Cross-modal search (QueryMode::Vision):**
```
Query: "the whiteboard from Tuesday's meeting"
  -> CLIP text encoder -> 512-dim text embedding
  -> Cosine similarity against vision_embeddings of all stored images
  -> Returns: image memories that CLIP considers semantically similar
```

### 3.4 Speech Channel

- **Architecture:** Two-model stack producing a concatenated 896-dim embedding
- **Dimension:** 896 = 512 (ECAPA-TDNN) + 384 (Whisper-tiny encoder)
- **LSH:** 16 tables, 10 bits per table (896-dim hyperplanes)

The two models capture complementary aspects of speech:

| Model | Dimension | Captures |
|---|---|---|
| ECAPA-TDNN | 256 | Speaker identity — who is speaking |
| Whisper-tiny encoder | 384 | Prosody: rhythm, stress, pace |

> **Note:** An emotion channel (arousal/dominance/valence) was explored during design but dropped because all available models (e.g., Wav2Small) carry non-commercial licenses incompatible with Apache 2.0. The architecture can accommodate a third sub-embedding when a permissively licensed emotion model becomes available.

This design deliberately does not perform speech-to-text. ShrimPK speech memory captures *how* something was said — speaker identity and prosodic features — not a transcript. Transcripts are a text memory concern; the speech channel targets paralinguistic features for social and robotics applications.

The speech channel structure is fully implemented and integrated into the store/retrieval pipeline. The ONNX model sessions are loaded via configuration paths (`speaker_model_path`, `prosody_model_path`); the architecture is ready to accept real model files.

### 3.5 Query Modes

The `QueryMode` enum controls which channels a query searches:

```
QueryMode::Text    -- text channel only (default, backward compatible)
QueryMode::Vision  -- vision channel only (CLIP cross-modal)
QueryMode::Auto    -- all enabled channels; results merged by final_score,
                      deduplicated by memory_id, truncated to max_results
```

Auto mode deduplication uses the highest `final_score` across channels for any given `MemoryId` — relevant for multimodal entries that have embeddings in multiple channels.

---

## 4. Core Data Structures

### 4.1 MemoryEntry

The fundamental unit of storage. Defined in `crates/shrimpk-core/src/memory.rs`.

```rust
pub struct MemoryEntry {
    pub id: MemoryId,                         // UUID v4
    pub content: String,                      // original text
    pub masked_content: Option<String>,       // PII-masked version (if PII detected)
    pub reformulated: Option<String>,         // structured form used for embedding
    pub embedding: Vec<f32>,                  // primary embedding (text or modality-native)
    pub modality: Modality,                   // Text | Vision | Speech
    pub vision_embedding: Option<Vec<f32>>,   // CLIP 512-dim (if image stored)
    pub speech_embedding: Option<Vec<f32>>,   // concatenated 896-dim (if audio stored)
    pub source: String,                       // "conversation", "document", "manual", ...
    pub sensitivity: SensitivityLevel,        // Public | Private | Restricted | Blocked
    pub category: MemoryCategory,             // controls decay half-life
    pub created_at: DateTime<Utc>,
    pub last_echoed: Option<DateTime<Utc>>,
    pub echo_count: u32,                      // times this memory has activated
    pub enriched: bool,                       // true after consolidation extracted facts
    pub parent_id: Option<MemoryId>,          // set on child memories (consolidated facts)
}
```

The `embedding` field is always the *primary* embedding — the one indexed in the LSH and used for cosine similarity in the primary channel. For text memories, this is the BGE-small embedding of the reformulated text. For vision-only memories, it is empty (not indexed in the text channel). For multimodal memories, the text embedding is primary and the vision/speech embeddings are supplementary.

### 4.2 EchoStore

`EchoStore` (`crates/shrimpk-memory/src/store.rs`) is the in-memory vector store. It maintains three parallel structures:

```
entries: Vec<MemoryEntry>          -- the memory entries in insertion order
embeddings: Vec<Vec<f32>>          -- parallel embedding array (same index)
id_to_index: HashMap<MemoryId, usize>   -- O(1) lookup by ID
parent_children: HashMap<MemoryId, Vec<usize>>  -- O(1) child lookup for Pipe B
```

The parallel array design (separate `embeddings` vector mirroring `entries`) avoids pointer chasing during the hot brute-force path: the similarity scoring loop iterates over a dense array of `Vec<f32>` rather than dereferencing struct fields.

Deletion uses swap-remove (O(1)) followed by index fixup. After deletion, the Bloom filter `bloom_dirty` flag is set, triggering a rebuild on the next consolidation pass.

The `parent_children` reverse index is maintained incrementally on every `add()` and `remove()` call. This gives the Pipe B child-rescue pass O(1) lookup for a memory's children, avoiding a full store scan.

### 4.3 EchoEngine

`EchoEngine` (`crates/shrimpk-memory/src/echo.rs`) owns all engine state and exposes the public API. It is thread-safe and designed for concurrent use.

```
EchoEngine {
    embedder: Mutex<MultiEmbedder>       -- fastembed requires &mut self
    store: RwLock<EchoStore>             -- concurrent reads, exclusive writes
    text_lsh: Mutex<CosineHash>          -- write during store, read during echo
    vision_lsh: Option<Mutex<CosineHash>>  -- vision channel (feature gated)
    speech_lsh: Option<Mutex<CosineHash>>  -- speech channel (feature gated)
    bloom: RwLock<TopicFilter>           -- concurrent reads during echo
    bloom_dirty: Mutex<bool>
    pii_filter: PiiFilter                -- Send+Sync, no lock needed
    reformulator: MemoryReformulator     -- Send+Sync, no lock needed
    hebbian: RwLock<HebbianGraph>        -- two-pass (write then read) per echo
    config: EchoConfig
    stats: Mutex<EchoStats>
    consolidation_handle: Mutex<Option<JoinHandle<()>>>
    consolidator: Box<dyn Consolidator>
}
```

The RwLock on `store` allows multiple concurrent echo queries to read simultaneously, with exclusive access only during `store()`, `forget()`, and consolidation writes. The Mutex on `embedder` is required because fastembed's `TextEmbedding` requires mutable access during inference.

The background consolidation task runs every 5 minutes by default, spawned as a `tokio::task` that holds `Arc<EchoEngine>` references to the shared state.

### 4.4 HebbianGraph

The sparse co-activation graph (`crates/shrimpk-memory/src/hebbian.rs`):

```
HebbianGraph {
    edges: HashMap<(u32, u32), CoActivation>  -- edge storage (a < b invariant)
    adjacency: HashMap<u32, Vec<u32>>          -- per-node neighbor list
    half_life: f64                             -- decay half-life in seconds
    lambda: f64                                -- ln(2) / half_life
    prune_threshold: f64
    activation_count: u64
}

CoActivation {
    weight: f64                      -- current pre-decay weight
    last_activated: f64              -- UNIX timestamp of last strengthening
    activation_count: u32
    relationship: Option<RelationshipType>  -- typed semantic relationship
}
```

The `a < b` key invariant prevents storing both `(a, b)` and `(b, a)` for the same edge, halving storage. The adjacency index enables O(degree) lookups during the Hebbian boost pass — the number of edges examined per query result is bounded by the degree of that node, not the total graph size.

---

## 5. Persistence: The SHRM Format

ShrimPK stores memories in a custom binary format (`.shrm`). The format is designed for fast loading via memory-mapped I/O (`memmap2`) and efficient representation of sparse multimodal embeddings.

### 5.1 Format Layout (v2)

```
Offset  Size    Field
------  ----    -----
0       4       Magic: b"SHRM"
4       1       Version: u8 = 2
5       1       Flags: bit 0 = has_vision, bit 1 = has_speech
6       2       Text dim: u16 (384)
8       2       Vision dim: u16 (512 or 0)
10      2       Speech dim: u16 (896 or 0)
12      4       Entry count: u32
=== 16-byte header ===

16      4       Metadata JSON length (u32)
20      M       Metadata JSON (all entries, no embeddings)
20+M    4       Metadata CRC32

=== Text section (always present) ===
        N*D*4   Text embeddings: f32[count][text_dim], row-major
        4       Text section CRC32

=== Vision section (if has_vision flag set) ===
        ceil(N/8)   Presence bitmap: bit i = entry i has a vision embedding
        V*512*4     Vision embeddings (only entries where bit is set)
        4           Vision CRC32 (over bitmap + embeddings)

=== Speech section (if has_speech flag set) ===
        ceil(N/8)   Presence bitmap
        S*896*4     Speech embeddings (only entries where bit is set)
        4           Speech CRC32
```

### 5.2 Design Decisions

**Metadata/embedding split.** All scalar fields (content, timestamps, IDs, source, category) are serialized as a single JSON blob. All embeddings are stored in dense binary sections. This allows fast embedding loading (direct mmap into f32 slices) without parsing overhead.

**Sparse bitmap for optional channels.** Not every entry has a vision or speech embedding. Rather than padding absent embeddings with zeros (which wastes space and confuses zero-vector similarity), the format uses a presence bitmap. `ceil(N/8)` bytes encodes one bit per entry; only present entries contribute to the embedding array. Storage cost for a text-only entry in a vision-enabled store is one bit rather than 512 * 4 = 2048 bytes.

**Per-section CRC32.** Each section (metadata, text embeddings, vision embeddings, speech embeddings) has an independent CRC32 checksum. Corruption in one section can be detected and reported without invalidating the others.

**Backward compatibility.** The loader supports both v1 (text-only, 64-byte header) and v2 formats. Files are always written in v2. A v2 file with `has_vision = false` and `has_speech = false` is functionally equivalent to v1 but in the newer format.

**File locking.** The implementation uses `fs2::FileExt` for advisory file locking during save/load, preventing concurrent write corruption when multiple processes share a data directory.

---

## 6. Sleep Consolidation

Consolidation is the "cleaner shrimp" maintenance process. It runs as a background task every 5 minutes (configurable) and performs five passes over the engine state.

### 6.1 Hebbian Pruning

Edges in the Hebbian graph that have decayed below `prune_threshold` (default 0.01) are removed. This is not just housekeeping — removing stale edges prevents irrelevant historical associations from accumulating boost and crowding out current relevance.

### 6.2 Bloom Filter Rebuild

If `bloom_dirty` is set (indicating one or more memories were deleted since the last build), the Bloom filter is rebuilt from scratch by reinserting all current memory texts. Bloom filters cannot remove individual items.

### 6.3 Near-Duplicate Merging

An O(N²) pairwise comparison detects memories with cosine similarity above 0.95 and marks the lower-quality duplicate for removal. This pass is skipped for stores larger than 10,000 memories, where the cost is not justified for a background maintenance job.

### 6.4 Echo Count Decay

Memories that have an `echo_count > 0` but have not been activated in more than 30 days have their echo count decremented by 1. This prevents echo_count from permanently inflating the score of memories that were once relevant but are no longer being activated.

### 6.5 LLM Fact Extraction (Sleep Phase)

The most substantive consolidation step calls a local LLM to extract atomic facts from un-enriched memories. The `Consolidator` trait is pluggable; the default implementation calls Ollama.

```
Memory: "Discussed the new auth system with the team.
         We decided to use JWTs with a 24-hour expiry
         and refresh tokens stored in Redis."

Extracted facts:
  -> "Uses JWT for authentication"
  -> "JWT expiry is 24 hours"
  -> "Refresh tokens stored in Redis"
```

Each extracted fact becomes a child `MemoryEntry` linked to the parent via `parent_id`. The child is embedded independently and participates in Pipe B child-rescue during echo queries — it can promote the parent if it matches better than the parent's natural-language form.

This process is rate-limited to 10 enrichments per consolidation cycle to bound the LLM call budget.

**Typed relationship detection.** After fact extraction, the consolidator scans each fact with regex patterns to detect typed relationships: `WorksAt`, `LivesIn`, `PrefersTool`, `PartOf`. Detected relationships create typed Hebbian edges between the parent and child with an initial weight of 0.5.

**Supersedes detection.** The consolidator also scans for contradictions between newly extracted facts and facts already in the store. When a new fact appears to update an old one (same entity, conflicting predicate), a `Supersedes` edge is created between the old and new memories. This edge is placed at both the child level and the parent level so that Pipe A ranking (which only sees parents when `child_rescue_only` is true) will correctly demote stale facts.

**Per-channel consolidators.** Text memories use LLM-based fact extraction. Vision memories (images) are planned to use a small VLM (e.g., moondream) for description-based fact extraction. Speech memories will use a transcription model (e.g., Whisper) to produce text facts. Each channel can configure its own consolidation model independently.

---

## 7. Crate Architecture

ShrimPK is organized as a Cargo workspace with 10 crates plus a CLI. The separation follows the principle of minimal dependencies: lower-level crates know nothing about higher-level ones.

```
shrimpk-kernel/      workspace root (integration tests only)
    |
    +-- crates/
    |     shrimpk-core/       -- types, traits, config (no heavy deps)
    |     shrimpk-memory/     -- Echo pipeline, embedder, LSH, Hebbian
    |     shrimpk-context/    -- context assembly (prompt compilation)
    |     shrimpk-router/     -- provider routing, cascade, circuit breaker
    |     shrimpk-security/   -- sandboxing, permissions (planned)
    |     shrimpk-kernel/     -- kernel integration layer
    |     shrimpk-mcp/        -- MCP server (JSON-RPC 2.0 over stdio)
    |     shrimpk-daemon/     -- HTTP daemon (Axum on localhost:11435)
    |     shrimpk-tray/       -- system tray application
    |     shrimpk-python/     -- Python bindings (PyO3)
    |
    +-- cli/                  -- shrimpk CLI binary
```

### shrimpk-core

The foundation. Defines all shared types (`MemoryEntry`, `MemoryId`, `EchoConfig`, `EchoResult`, `Modality`, `QueryMode`, `SensitivityLevel`, `MemoryCategory`) and the `Consolidator` trait. Has minimal dependencies: `serde`, `chrono`, `uuid`, `thiserror`. No fastembed, no tokio.

Everything else in the workspace depends on `shrimpk-core`. Nothing in `shrimpk-core` depends on anything else in the workspace.

### shrimpk-memory

The core engine. Implements the full Echo pipeline:
- `EchoEngine` — the public API surface
- `EchoStore` — in-memory vector store
- `MultiEmbedder` — fastembed wrapper for text/vision/speech
- `CosineHash` — LSH index
- `TopicFilter` — Bloom filter
- `HebbianGraph` — co-activation graph
- `PiiFilter` — PII detection and masking
- `MemoryReformulator` — structured text rewriting
- `consolidation` — maintenance pass logic
- `persistence` — SHRM binary format reader/writer
- `reranker` — cross-encoder and LLM reranker backends
- `similarity` — SIMD-accelerated cosine similarity
- `speech` — speech embedding architecture (SpeechEmbedder)

Depends on: `shrimpk-core`, `fastembed`, `simsimd`, `bloomfilter`, `memmap2`, `crc32fast`, `tokio`.

### shrimpk-context

Builds optimal context windows for LLM calls. The `ContextAssembler` combines the system prompt, echo results, RAG chunks, and conversation history within a token budget, truncating lowest-priority sources first. Applies sensitivity filtering: `Private` memories are excluded when the target provider is cloud-hosted.

Depends on: `shrimpk-core`.

### shrimpk-router

Intelligent provider routing. The `CascadeRouter` filters available providers by capability, locality, and budget, then selects the best match by cost/quality score. Includes:
- Per-provider token usage tracking with daily/monthly budget enforcement
- Circuit breaker with configurable failure threshold and half-open recovery window
- `RouteDecision` records for audit logging

Depends on: `shrimpk-core`.

### shrimpk-security

Sandboxing, permission management, and audit logging (planned). Currently re-exports `ShrimPKError` for consumers. Full implementation targets capability-based permissions (WASM sandboxing via Wasmtime) and cryptographic audit trails.

Depends on: `shrimpk-core`.

### shrimpk-kernel

Integration layer that wires together `shrimpk-memory`, `shrimpk-context`, and `shrimpk-router` into a unified kernel interface. Entry point for applications that want the full pipeline rather than individual crates.

### shrimpk-mcp

Model Context Protocol server. Exposes Echo Memory as MCP tools (`store`, `echo`, `stats`, `forget`, `status`, `config_show`, `dump`) via JSON-RPC 2.0 over stdio. Compatible with any MCP-aware AI client.

Key design: the `EchoEngine` is lazily initialized on first tool call. The server starts in milliseconds; fastembed model loading (a few seconds) is deferred until a memory operation is actually requested.

Background consolidation is started immediately after engine initialization, running every 300 seconds.

Depends on: `shrimpk-core`, `shrimpk-memory`, `tokio`.

### shrimpk-daemon

HTTP daemon. Runs as a persistent background process on `localhost:11435`, serving the Echo Memory API over HTTP. Clients include the CLI, the system tray, and any application that prefers HTTP over stdio MCP.

Built on Axum 0.8. Features:
- Rate limiting (configurable RPS, default 100 req/s)
- Optional Bearer token authentication (`SHRIMPK_AUTH_TOKEN` env var)
- OpenAI-compatible proxy (`/v1/chat/completions`) — intercepts outbound LLM calls, injects activated memories into the system prompt, forwards to the configured backend
- CORS enabled for local web clients
- Auto-detection of running Ollama instances and local model inventory

Depends on: `shrimpk-core`, `shrimpk-memory`, `shrimpk-context`, `axum`, `tokio`.

### shrimpk-tray

System tray application. Provides a persistent icon in the OS notification area with quick access to memory stats, a "store this text" action, and daemon controls. Built to run at OS startup alongside the daemon.

### shrimpk-python

Python bindings via PyO3. Exposes the `EchoEngine` API to Python with async support. Enables use in Python-based AI frameworks, Jupyter notebooks, and scripts.

### cli

The `shrimpk` command-line binary. Provides the `store`, `echo`, `forget`, `stats`, `status`, `config`, `save`, `load`, and `consolidate` subcommands. Communicates with a running daemon over HTTP when available, or initializes an in-process engine for offline use.

---

## 8. Performance Characteristics

Benchmark conditions: release build (`opt-level = 3`, `lto = true`, `codegen-units = 1`), LSH enabled (16 tables, 10 bits), Bloom enabled.

| Operation | P50 | P95 | Notes |
|---|---|---|---|
| Echo query (41 memories) | ~8ms | ~11ms | Tier 2 realistic benchmark, BGE-small-EN-v1.5 |
| Echo query (100K, v0.5.0) | 23.79ms | 54.58ms | Known regression under investigation (see [Known Issues](KNOWN-ISSUES.md)) |
| Echo query (100K, v0.4.0) | 3.50ms | 6.88ms | Prior version, all-MiniLM-L6-v2 embeddings |
| Echo query (brute-force fallback) | ~12ms | ~25ms | When LSH returns < 10 candidates |
| Store (text) | ~8ms | ~15ms | Dominated by fastembed inference |
| Consolidation pass | ~50ms | ~200ms | Depends on store size and LLM latency |
| Save (100K memories) | ~120ms | ~200ms | Binary serialization + fsync |
| Load (100K memories) | ~40ms | ~80ms | Binary deserialization via mmap |

Memory usage at 100K memories (text-only, F32 quantization):

```
Embeddings:   100,000 * 384 * 4 bytes = 153.6 MB
Metadata:     ~50 bytes per entry     =   5.0 MB
LSH index:    ~10 bytes per entry     =   1.0 MB
Bloom filter: 1% FPR at 1M items      =   1.14 MB
Hebbian:      sparse, ~0 at rest      =   0.1 MB (grows with co-activation)
Total:                                ~ 161 MB
```

Quantization modes reduce the embedding footprint at the cost of precision:

| Mode | Bytes/vector | Memory (100K) | Quality loss |
|---|---|---|---|
| F32 (default) | 1,536 | 153.6 MB | 0% |
| F16 | 768 | 76.8 MB | ~0.1% |
| Int8 | 384 | 38.4 MB | ~1% |
| Binary | 48 | 4.8 MB | ~5% (needs reranker) |

The release profile uses whole-program LTO (`lto = true`) and single codegen unit (`codegen-units = 1`), which enables the Rust compiler to inline across crate boundaries and maximize SIMD auto-vectorization. The strip option removes debug symbols from the release binary.

---

## 9. Configuration Reference

Configuration is loaded from `~/.shrimpk/config.toml` (or the path in `SHRIMPK_CONFIG`). Environment variables override file values. Defaults apply where neither is specified.

Key configuration fields on `EchoConfig`:

```toml
[echo]
max_memories = 100000               # hard capacity limit
similarity_threshold = 0.7          # cosine similarity cutoff [0.0, 1.0]
max_echo_results = 10               # max results per echo call
ram_budget_bytes = 1073741824       # 1 GB RAM budget for embeddings
max_disk_bytes = 2147483648         # 2 GB disk limit for data directory
use_lsh = true                      # enable LSH (disable only for tiny stores)
use_bloom = true                    # enable Bloom pre-filter
quantization = "f32"                # f32 | f16 | int8 | binary
recency_weight = 0.05               # recency boost coefficient
child_rescue_only = true            # isolate child memories from direct ranking
supersedes_demotion = 0.0           # score penalty for superseded memories

[consolidation]
consolidation_provider = "ollama"   # "ollama" | "http" | "none"
ollama_url = "http://127.0.0.1:11434"
enrichment_model = "llama3.2:3b"
max_facts_per_memory = 5

[reranker]
reranker_backend = "none"           # "none" | "cross_encoder" | "llm"

[query]
query_expansion_enabled = false     # HyDE expansion via Ollama

[modalities]
enabled_modalities = ["Text"]       # ["Text"] | ["Text", "Vision"] | ["Text", "Vision", "Speech"]
vision_embedding_dim = 512
speech_embedding_dim = 896
```

The `EchoConfig` resolution chain: environment variables (`SHRIMPK_*` prefix) override config file values, which override compiled-in defaults. The `config::load()` function in `shrimpk-core` handles this chain and returns a validated `EchoConfig`.
