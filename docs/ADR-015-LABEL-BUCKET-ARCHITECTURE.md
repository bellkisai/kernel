# ADR-015: Label Bucket Architecture

**Date:** 2026-03-31
**Status:** Proposed
**Authors:** Architecture team

---

## 1. Status

Proposed.

---

## 2. Context

### The 100K Latency Problem

At 100K stored memories, the echo pipeline regresses from P50=3.50ms to P50=23.79ms --
a 6.8x degradation against the <4ms target. The regression is caused entirely by the
candidate retrieval stage.

The current pipeline uses Locality-Sensitive Hashing (LSH) to select candidate vectors
for cosine scoring. When LSH returns fewer than 10 candidates, the pipeline falls back
to brute-force cosine over all 100K embeddings. This fallback is the sole cause of the
regression -- not the cosine computation itself, not Hebbian boosting, and not reranking.

**Why LSH fails at scale:** LSH hash collisions become sparser as the store grows.
Memories distribute across more hash buckets, so fewer entries share a bucket with the
query. At 100K entries, "topic:language" queries may only have 8 relevant memories among
100K vectors -- LSH cannot reliably surface them because their hash signatures are
scattered.

### The Pipeline Is Not The Bottleneck -- The Data Is

Steps 4-8 of the echo pipeline (cosine scoring, pipe split, Hebbian co-activation,
recency decay, reranking) operate on whatever candidates step 3 provides. They are
O(k) where k = number of candidates, not O(N). The fix belongs entirely in step 3:
providing enough relevant candidates so brute-force fallback never triggers.

### The Consolidation Enrichment Opportunity

ShrimPK already runs a background consolidation process that visits every memory,
extracts facts via LLM, creates child memories, and detects supersedes relationships.
By the time a user accumulates 100K memories, consolidation has processed most of them.
This is idle enrichment capacity that can generate semantic labels at zero marginal
latency cost -- the LLM call that extracts facts can simultaneously classify labels.

Labels create an inverted index that provides O(1) pre-filtering: given query labels
like `topic:language`, look up the posting list to get ~50 candidate indices instantly,
bypassing the LSH scatter problem entirely.

---

## 3. Decision

### D1: Label Field on MemoryEntry

Add two fields to `MemoryEntry` in `shrimpk-core/src/memory.rs`:

```rust
/// A stored memory entry in the Echo Memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    // ... all existing fields unchanged ...

    /// Semantic labels for pre-filtered retrieval.
    /// Each label is a prefixed string (e.g., "topic:language", "entity:rust").
    /// Populated incrementally: Tier 1 at store time, Tier 2 during consolidation.
    #[serde(default)]
    pub labels: Vec<String>,

    /// Label enrichment version.
    /// 0 = unlabeled (legacy or new, pre-classification).
    /// 1 = keyword-only (Tier 1: YAKE + prototype matching + rules).
    /// 2 = LLM-enriched (Tier 2: GLiNER NER + Ollama classification).
    #[serde(default)]
    pub label_version: u8,
}
```

**Backward compatibility:** Both fields use `#[serde(default)]`. Deserializing a
pre-label memory entry produces `labels: vec![]` and `label_version: 0`. No migration
is required. Existing `.shrm` files load without changes because labels live in the
JSON metadata section, not the binary embedding array.

The `MemoryEntry::new()` constructor initializes both fields to their defaults. Labels
are populated after construction by the label generation pipeline (D4).

**`MemoryMeta` must also be updated** in `persistence.rs`:

```rust
pub struct MemoryMeta {
    // ... existing fields ...
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub label_version: u8,
}
```

Both `MemoryMeta::from_entry()` and `MemoryMeta::into_entry()` must propagate these
fields. Without this, labels would be lost on save/load roundtrip. (Amendment 5)

### D2: Label Taxonomy -- 7-Dimension Flat With Typed Prefixes

Every label is a `String` with a typed prefix separated by a colon. There is no
hierarchy -- labels are flat tokens looked up via exact string match in the inverted
index.

| Prefix | Purpose | Examples | Generator |
|-----------|----------------------|----------------------------------------------|--------------------------------|
| `topic:` | Subject domain | `topic:language`, `topic:career`, `topic:health` | LLM + prototype cosine matching |
| `domain:` | Life area | `domain:work`, `domain:life`, `domain:social` | Rule-based classifier |
| `entity:` | Named entities | `entity:rust`, `entity:tokyo`, `entity:figma` | GLiNER NER (gline-rs) |
| `action:` | Verb concepts | `action:learning`, `action:building` | YAKE keyword extraction |
| `temporal:` | Time signal | `temporal:current`, `temporal:recurring` | Rule-based (reformulator) |
| `modality:` | Memory type | `modality:fact`, `modality:preference` | LLM classification |
| `sentiment:` | Emotional valence | `sentiment:positive`, `sentiment:neutral` | LLM classification |

**Why flat beats hierarchical at this scale:**

Hierarchical taxonomies (e.g., `topic/programming/rust`) require tree traversal,
prefix matching, and consensus on tree structure. At the target scale of 60-120 unique
labels with 3-7 labels per memory, flat lookup is O(1) per label and the vocabulary
fits in a single HashMap. Tree overhead is not justified until the taxonomy exceeds
~1000 labels.

**Inclusive parent search:** A query for the bare prefix `topic:` returns the union of
all `topic:*` labels. This is implemented as a scan over HashMap keys with a
`starts_with` check, not a tree traversal. At 120 unique labels, the scan cost is
negligible.

**Normalization rules:**
- All labels are lowercased.
- Entity names are canonicalized: "Rust" becomes `entity:rust`.
- Labels per memory capped at 10.
- Total unique labels capped at 5000 (enforced during consolidation merge).

### D3: Inverted Index on EchoStore

Add a label inverted index to `EchoStore` in `shrimpk-memory/src/store.rs`:

```rust
use std::collections::HashMap;

pub struct EchoStore {
    entries: Vec<MemoryEntry>,
    embeddings: Vec<Vec<f32>>,
    id_to_index: HashMap<MemoryId, usize>,
    parent_children: HashMap<MemoryId, Vec<usize>>,

    /// Inverted index: label string -> sorted list of store indices.
    /// Enables O(1) candidate retrieval per label.
    /// Rebuilt from entries on load; updated incrementally on add/remove.
    label_index: HashMap<String, Vec<u32>>,
}
```

**Why `Vec<u32>` over `RoaringBitmap` at 100K:**

At 100K memories with ~5 labels per memory and ~120 unique labels, each posting list
averages ~4,200 entries. A `Vec<u32>` of 4,200 elements is 16.8 KB. The total index
across all 120 labels is ~2 MB. `RoaringBitmap` adds 10-15 MB of code via the
`roaring` crate, requires sorted insertion, and provides no measurable benefit at
this scale. The breakpoint where bitmap set operations (intersection, union) outperform
Vec iteration is ~500K entries per posting list -- well beyond the 100K target.

**Incremental update on `add()`:**

```rust
impl EchoStore {
    pub fn add(&mut self, entry: MemoryEntry) -> usize {
        let index = self.entries.len();
        // ... existing id_to_index + parent_children logic ...

        // Update label index
        for label in &entry.labels {
            self.label_index
                .entry(label.clone())
                .or_default()
                .push(index as u32);
        }

        self.entries.push(entry);
        self.embeddings.push(embedding);
        index
    }
}
```

**Incremental update on `remove()`:**

Because `EchoStore::remove()` uses swap-remove, the last element's index changes.
The label index must be patched: remove all occurrences of the deleted index, and
replace all occurrences of the old last-index with the deleted index.

```rust
impl EchoStore {
    pub fn remove(&mut self, id: &MemoryId) -> Option<MemoryEntry> {
        let index = self.id_to_index.remove(id)?;
        let last_index = self.entries.len() - 1;

        // Remove labels for the deleted entry
        if let Some(entry) = self.entries.get(index) {
            for label in &entry.labels {
                if let Some(indices) = self.label_index.get_mut(label) {
                    indices.retain(|&i| i != index as u32);
                    if indices.is_empty() {
                        self.label_index.remove(label);
                    }
                }
            }
        }

        // If swap-removed (not the last element), fix up the swapped entry's labels
        if index != last_index {
            if let Some(swapped_entry) = self.entries.get(last_index) {
                for label in &swapped_entry.labels {
                    if let Some(indices) = self.label_index.get_mut(label) {
                        for i in indices.iter_mut() {
                            if *i == last_index as u32 {
                                *i = index as u32;
                            }
                        }
                    }
                }
            }
        }

        // ... existing swap_remove + id_to_index fixup ...
    }
}
```

**Rebuild on `load()`:**

When deserializing from disk, the label index is reconstructed in a single pass:

```rust
impl EchoStore {
    pub fn rebuild_label_index(&mut self) {
        self.label_index.clear();
        for (idx, entry) in self.entries.iter().enumerate() {
            for label in &entry.labels {
                self.label_index
                    .entry(label.clone())
                    .or_default()
                    .push(idx as u32);
            }
        }
    }
}
```

At 100K entries with 5 labels each, this is 500K HashMap insertions: <10ms.

**Query API:**

```rust
impl EchoStore {
    /// Look up all store indices matching any of the given labels (OR semantics).
    /// Returns a deduplicated, sorted Vec of indices.
    pub fn query_labels(&self, labels: &[String]) -> Vec<u32> {
        let mut result: Vec<u32> = Vec::new();
        for label in labels {
            if let Some(indices) = self.label_index.get(label) {
                result.extend_from_slice(indices);
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }
}
```

### D4: Two-Tier Label Generation

Labels are generated in two tiers: synchronous at store time (immediate, rule-based)
and asynchronous during consolidation (deferred, LLM-assisted).

**Critical design principle: Implicit Label Inference.**

Labels must be assigned even when the memory text does not explicitly mention the label
keyword. For example:

| Memory text | Explicit match | Implicit labels needed |
|-------------|---------------|----------------------|
| "I'm learning Japanese on Duolingo" | "learning" -> `action:learning` | Direct match works |
| "In my history class we're talking about WW2" | None | `action:learning`, `topic:education`, `topic:history` |
| "I ran 5K this morning before work" | None | `action:exercise`, `domain:health`, `temporal:recurring` |
| "My manager Lisa asked me to lead the plugin project" | "project" -> `domain:work` | `action:leading`, `entity:lisa` |

Keyword/rule-based methods (YAKE, regex) only catch explicit mentions. Implicit labels
require **semantic understanding** of what the memory is about, not just what words
it contains. This is solved by two complementary mechanisms:

1. **Prototype cosine matching (Tier 1, at store time):** The embedding of "in my
   history class we're talking about WW2" is semantically close to the prototype
   embedding for "education, learning, studying, classroom, academic" (`topic:education`)
   even though the word "learning" never appears. Cosine similarity captures the
   semantic relationship, not the lexical overlap. This is the primary mechanism for
   implicit label inference at store time.

2. **LLM semantic reasoning (Tier 2, at consolidation):** The LLM understands that
   "history class" implies learning. The consolidation prompt explicitly instructs
   the model to infer implicit categories:

   > "Classify based on what this memory IS ABOUT, not just the words it contains.
   > A memory about attending a class implies learning. A memory about running
   > implies exercise and health. Assign labels for the underlying activity,
   > not just the surface text."

The prototype cosine matching is the key innovation for implicit labels at store time.
The prototype descriptions must be **rich and diverse** -- not just single words but
full descriptions covering synonyms, related concepts, and indirect indicators:

```rust
// Good: rich description catches implicit matches
("topic:education", "education, learning, studying, classroom, academic, class, course, lecture, homework, university, school, teaching, student"),
("action:learning", "learning, studying, practicing, training, skill development, taking a class, reading about, researching, tutorial"),
("domain:health", "health, fitness, exercise, running, gym, yoga, medical, diet, wellness, sleep, injury, recovery, sports"),

// Bad: single-word prototypes miss implicit matches
("topic:education", "education"),  // would NOT match "history class about WW2"
```

With rich prototype descriptions, the embedding for "history class about WW2" will have
high cosine similarity to the education/learning prototypes because the embedding model
(BGE-small) captures the semantic field, not just keyword overlap.

#### Tier 1 -- Synchronous at Store Time (<2ms total)

Runs inline in `EchoEngine::store()` before the entry is added to `EchoStore`. Labels
are immediately queryable.

| Method | Extracted dimensions | Estimated latency | Implicit label support |
|-------------------------|--------------------------------------|-------------------|----------------------|
| Prototype cosine match | `topic:`, `domain:`, `action:` | ~0.3ms | **YES -- primary mechanism** |
| YAKE keyword extraction | `entity:` (preliminary) | ~0.1ms | No (lexical only) |
| Rule-based classifier | `temporal:`, `domain:` | <0.01ms | No (pattern only) |

**Prototype cosine matching is now the PRIMARY Tier 1 method** (not YAKE). It runs
first because it handles both explicit and implicit labels. YAKE and rule-based
classifiers supplement it with entity names and temporal signals that prototypes
cannot capture.

**Prototype cosine matching:** At daemon startup, pre-compute embeddings for ~100
label description strings. Each description is a **rich multi-word phrase** covering
the full semantic field of the label (see examples above). At store time, compute
cosine similarity between the memory's embedding vector and all prototype vectors.
Assign labels where similarity exceeds a threshold of 0.6 (tunable). This reuses
the embedding already computed for the memory -- no extra embedding call.

```rust
/// Pre-computed prototype embeddings for label classification.
pub struct LabelPrototypes {
    /// Label text for each prototype (e.g., "topic:career").
    labels: Vec<String>,
    /// Embedding vector for each prototype's description.
    embeddings: Vec<Vec<f32>>,
    /// Cosine similarity threshold for assignment.
    threshold: f32,  // default: 0.6
}

impl LabelPrototypes {
    /// Classify a memory embedding against all prototypes.
    /// Returns labels whose prototype similarity exceeds the threshold.
    pub fn classify(&self, embedding: &[f32]) -> Vec<String> {
        self.labels.iter()
            .zip(self.embeddings.iter())
            .filter_map(|(label, proto_emb)| {
                let sim = cosine_similarity(embedding, proto_emb);
                if sim >= self.threshold {
                    Some(label.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}
```

**LabelPrototypes initialization (Amendment 1):** The `LabelPrototypes` struct is
initialized in `EchoEngine::new()` BEFORE the `MultiEmbedder` is wrapped in the
`Mutex`. The embedder computes ~100 prototype embeddings from hardcoded label
descriptions (e.g., "career and employment" for `topic:career`). This takes ~800ms
at startup. To avoid slow test initialization, prototype embeddings should be cached
to a file (`~/.shrimpk-kernel/label_prototypes.bin`) and loaded on subsequent starts.
Lazy initialization (compute on first `store()` call) is an acceptable alternative.

The `EchoEngine` gains a new field:

```rust
pub struct EchoEngine {
    // ... existing fields ...
    prototypes: LabelPrototypes,  // pre-computed at new(), immutable after init
}
```

**Rule-based classifier:** Reuses existing reformulator pattern matching to detect
temporal signals (`temporal:current`, `temporal:past`, `temporal:future`,
`temporal:recurring`) and basic domain cues (`domain:work` from keywords like "project",
"meeting", "deadline").

After Tier 1 completes, the entry's `label_version` is set to `1`.

#### Tier 2 -- Async Background Enrichment (Consolidation)

Runs during the consolidation cycle (see D7). Upgrades `label_version` from 1 to 2.

| Method | Extracted dimensions | Estimated latency |
|-------------------------|----------------------------------------------|----------------------|
| GLiNER zero-shot NER | `entity:` (precise, replaces Tier 1 guesses) | ~20-50ms per memory |
| Ollama LLM (JSON mode) | `modality:`, `sentiment:`, refined `topic:` | ~200-500ms per memory |

**GLiNER via `gline-rs`:** Zero-shot Named Entity Recognition without training data.
Extracts entity names and types from memory text. Entities are normalized to lowercase
and prefixed as `entity:name`.

**Ollama LLM classification:** Uses the same Ollama endpoint and model already
configured for fact extraction (`config.ollama_url`, `config.enrichment_model`).
The prompt provides a closed vocabulary of valid labels and requests JSON output
via Ollama's `format` parameter for grammar-constrained structural compliance.

```rust
/// LLM label classification prompt (closed vocabulary, JSON schema output).
const LABEL_CLASSIFICATION_PROMPT: &str = r#"
Classify this memory into semantic labels. Return ONLY a JSON object.

Memory: "{content}"

Choose from these label categories:
- topic: career, language, health, housing, music, technology, finance, education, travel, food, fitness, relationships, hobby
- domain: work, life, social, health, creative, finance
- modality: fact, preference, goal, habit, event, opinion, plan
- sentiment: positive, negative, neutral, mixed

Return JSON: {"topic": ["..."], "domain": ["..."], "modality": "...", "sentiment": "..."}
"#;
```

**`label_version` tracking:** After Tier 2 completes, `label_version` is set to `2`.
The consolidation loop (D7) uses this field to identify memories that have only Tier 1
labels and need Tier 2 enrichment.

### D5: Query Classification -- Three-Tier

When a query arrives, the pipeline must determine which labels to use for pre-filtering.
This classification runs before any vector operations and must be sub-millisecond.

#### Tier A -- Keyword Extraction (<0.001ms)

Pattern-match the query text against a static lookup table of known label triggers:

```rust
/// Static keyword -> label mapping for Tier A query classification.
fn classify_query_keywords(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut labels = Vec::new();

    // Topic triggers
    static TOPIC_TRIGGERS: &[(&[&str], &str)] = &[
        (&["language", "learn", "study", "duolingo"], "topic:language"),
        (&["career", "job", "work", "promotion"], "topic:career"),
        (&["health", "doctor", "medicine", "symptom"], "topic:health"),
        (&["travel", "trip", "flight", "hotel"], "topic:travel"),
        (&["code", "programming", "software", "debug"], "topic:technology"),
        // ... ~30 entries total
    ];

    for (triggers, label) in TOPIC_TRIGGERS {
        if triggers.iter().any(|t| lower.contains(t)) {
            labels.push(label.to_string());
        }
    }

    // Action triggers
    if lower.contains("learn") || lower.contains("study") {
        labels.push("action:learning".to_string());
    }
    if lower.contains("build") || lower.contains("creat") {
        labels.push("action:building".to_string());
    }
    // ... additional action patterns

    labels
}
```

This covers an estimated 60-70% of queries. When it produces at least one label, no
further classification is needed.

#### Tier B -- Prototype Cosine (~0.015ms)

When Tier A yields no labels, compute cosine similarity between the query embedding
(already computed in step 1 of the echo pipeline) and the ~100 label prototype
embeddings (the same `LabelPrototypes` struct from D4). Return labels above threshold.

The cost is ~100 dot products on 384-dim vectors. With SIMD (simsimd), this is
~0.015ms -- negligible relative to the embedding step (~8ms).

#### Tier C -- Fallback

When neither Tier A nor Tier B produces labels, no label-based pre-filtering is applied.
The query falls through to the existing LSH + brute-force pipeline. The label system
is purely additive -- it never gates or restricts the pipeline.

### D6: Candidate Merge Strategy

The candidate retrieval stage (step 3 in the echo pipeline) changes from a binary
LSH/brute-force decision to a three-source merge:

```rust
// In echo_text(), replacing the current binary LSH/brute-force choice:

// 3a. Label-based candidate retrieval
let query_labels = if self.config.use_labels {
    classify_query(&effective_query, &query_embedding, &self.prototypes)
} else {
    Vec::new()
};
let label_candidates: Vec<u32> = if !query_labels.is_empty() {
    store.query_labels(&query_labels)
} else {
    Vec::new()
};

// 3b. LSH candidate retrieval (existing)
let lsh_candidates: Vec<u32> = self.text_lsh
    .lock()
    .map_err(|e| ShrimPKError::Memory(format!("LSH lock poisoned: {e}")))?
    .query(&query_embedding);

// 3c. Merge + dedup (OR semantics)
let mut merged: Vec<u32> = Vec::with_capacity(
    label_candidates.len() + lsh_candidates.len()
);
merged.extend_from_slice(&label_candidates);
merged.extend_from_slice(&lsh_candidates);
merged.sort_unstable();
merged.dedup();

const MIN_CANDIDATES: usize = 5;  // lowered from 10

let candidates: Vec<(usize, &[f32])> = if merged.len() >= MIN_CANDIDATES {
    // Enough candidates from labels + LSH -- no brute-force needed
    tracing::debug!(
        label_candidates = label_candidates.len(),
        lsh_candidates = lsh_candidates.len(),
        merged = merged.len(),
        total = embeddings.len(),
        "Label + LSH candidate retrieval (sub-linear)"
    );
    merged.iter()
        .filter_map(|&idx| {
            let i = idx as usize;
            embeddings.get(i).map(|e| (i, e.as_slice()))
        })
        .collect()
} else {
    // Neither source provided enough -- brute-force fallback
    tracing::debug!(
        merged = merged.len(),
        total = embeddings.len(),
        "Labels + LSH returned < {} candidates, falling back to brute-force",
        MIN_CANDIDATES
    );
    embeddings.iter()
        .enumerate()
        .filter(|(_, e)| !e.is_empty())
        .map(|(i, e)| (i, e.as_slice()))
        .collect()
};
```

**Key design choices:**

1. **OR semantics, not AND.** Label candidates and LSH candidates are unioned, not
   intersected. AND is too aggressive -- a single missed label would exclude relevant
   memories. OR is safe because cosine scoring (step 5b) filters false positives.

2. **MIN_CANDIDATES lowered from 10 to 5.** With two candidate sources, the probability
   of both returning zero is much lower than either alone. A threshold of 5 is
   sufficient to avoid degenerate single-result queries while reducing brute-force
   fallback frequency.

3. **Cardinality-based threshold switching (Qdrant pattern).** In the future, when a
   label's posting list exceeds a configurable fraction of the total store (e.g., >20%),
   skip that label's candidates (they are too broad to provide meaningful filtering).
   This prevents high-cardinality labels like `domain:work` from flooding the merge
   with thousands of candidates. Not implemented in Phase 1; the threshold logic is a
   Phase 3 tuning item.

4. **`use_labels` config toggle (Amendment 2).** A `use_labels: bool` field on
   `EchoConfig` (default: `true`) allows disabling label retrieval without recompiling.
   Essential for production rollout safety and A/B benchmarking. When `false`, the
   label candidate set is empty and the pipeline degrades to pure LSH (current behavior).

### D7: Consolidation Integration -- Pass 4

The consolidation cycle gains a new pass for Tier 2 label enrichment:

| Pass | When | What | Neuroscience analog |
|------|--------------|----------------------------------------------|--------------------------|
| 1 | Store time | Tier 1 labels (YAKE + prototype + rules) | Hippocampal encoding |
| 2 | Consolidation | Fact extraction (existing) | NREM slow-wave replay |
| 3 | Consolidation | Dedup + supersedes detection (existing) | NREM stabilization |
| **4** | **Consolidation** | **Tier 2 labels (GLiNER + LLM)** | **REM labeling/tagging** |
| 5 | Periodic | Cross-corpus synthesis (future) | Multi-night integration |

**Consolidator trait extension (Amendment 3):** The `Consolidator` trait must be extended
to support the combined response:

```rust
pub struct ConsolidationOutput {
    pub facts: Vec<String>,
    pub labels: Option<LabelSet>,  // None for legacy consolidators
}

pub struct LabelSet {
    pub topic: Vec<String>,
    pub domain: Vec<String>,
    pub modality: Option<String>,
    pub sentiment: Option<String>,
}

pub trait Consolidator: Send + Sync {
    fn extract_facts(&self, text: &str, max_facts: usize) -> Vec<String>;
    fn name(&self) -> &str;

    /// Combined extraction. Default impl calls extract_facts() with labels: None.
    fn extract_facts_and_labels(&self, text: &str, max_facts: usize) -> ConsolidationOutput {
        ConsolidationOutput {
            facts: self.extract_facts(text, max_facts),
            labels: None,
        }
    }
}
```

The default implementation preserves backward compatibility with the noop consolidator.
The Ollama-backed consolidator overrides `extract_facts_and_labels()` with the combined
prompt.

**Same LLM call, zero extra round-trips:** The LLM prompt for fact extraction is
extended to also request label classification. A single Ollama call returns both facts
and labels:

```rust
/// Combined fact extraction + label classification prompt.
const COMBINED_ENRICHMENT_PROMPT: &str = r#"
Analyze this memory and return a JSON object with two keys:

Memory: "{content}"

1. "facts": Extract up to {max_facts} atomic, self-contained facts.
2. "labels": Classify into semantic labels. IMPORTANT: classify based on what this
   memory IS ABOUT, not just the words it contains. A memory about attending a class
   implies learning/education. A memory about running implies exercise/health.
   Assign labels for the underlying activity and context, not just the surface text.

   Categories:
   - topic: [career, language, health, housing, music, technology, education, history, science, finance, travel, food, fitness, relationships, hobby, ...]
   - domain: [work, life, social, health, creative, finance]
   - action: [learning, building, planning, moving, exercising, leading, deciding, buying, ...]
   - memtype: one of [fact, preference, goal, habit, event, opinion, plan]
   - sentiment: one of [positive, negative, neutral, mixed]

Return JSON:
{
  "facts": ["fact 1", "fact 2"],
  "labels": {
    "topic": ["..."],
    "domain": ["..."],
    "action": ["..."],
    "memtype": "...",
    "sentiment": "..."
  }
}
"#;
```

The consolidation function in `consolidation.rs` processes the LLM response, converts
the structured labels into prefixed strings, and sets them on the parent entry:

```rust
// In consolidation::consolidate(), after fact extraction:

if let Some(labels_json) = llm_response.labels {
    let mut prefixed: Vec<String> = Vec::new();

    for topic in &labels_json.topic {
        prefixed.push(format!("topic:{}", topic.to_lowercase()));
    }
    for domain in &labels_json.domain {
        prefixed.push(format!("domain:{}", domain.to_lowercase()));
    }
    if let Some(modality) = &labels_json.modality {
        prefixed.push(format!("modality:{}", modality.to_lowercase()));
    }
    if let Some(sentiment) = &labels_json.sentiment {
        prefixed.push(format!("sentiment:{}", sentiment.to_lowercase()));
    }

    // Merge with existing Tier 1 labels (don't replace -- augment)
    if let Some(entry) = store.entry_at_mut(idx) {
        for label in prefixed {
            if !entry.labels.contains(&label) {
                entry.labels.push(label);
            }
        }
        // Cap at 10 labels per memory
        entry.labels.truncate(10);
        entry.label_version = 2;
    }

    // Update inverted index for new labels
    // (handled by store.update_labels(idx, &new_labels))
}
```

**Retroactive bulk labeling for first deployment:** When the label system is first
deployed on an existing store with 100K memories, most have `label_version: 0`. Rather
than waiting for consolidation to process 10 per cycle, a one-time bootstrap pass runs
Tier 1 (keyword + prototype) on all entries. This is pure Rust, no LLM, and processes
100K entries in <5 seconds. The bootstrap runs once during the first `load()` after
upgrade:

```rust
impl EchoStore {
    /// One-time bootstrap: apply Tier 1 labels to all unlabeled entries.
    /// Called during load() when label_version == 0 entries are detected.
    pub fn bootstrap_tier1_labels(&mut self, prototypes: &LabelPrototypes) {
        let mut updated = 0;
        for idx in 0..self.entries.len() {
            if self.entries[idx].label_version == 0 {
                let labels = generate_tier1_labels(
                    &self.entries[idx].content,
                    &self.embeddings[idx],
                    prototypes,
                );
                if !labels.is_empty() {
                    for label in &labels {
                        self.label_index
                            .entry(label.clone())
                            .or_default()
                            .push(idx as u32);
                    }
                    self.entries[idx].labels = labels;
                    self.entries[idx].label_version = 1;
                    updated += 1;
                }
            }
        }
        tracing::info!(updated, total = self.entries.len(), "Tier 1 label bootstrap complete");
    }
}
```

**Async bootstrap (Amendment 7):** The bootstrap should run asynchronously after the
store is loaded, not blocking daemon startup. The first few echo queries before
bootstrap completes will not benefit from labels (pure LSH fallback), but the daemon
becomes available immediately. Spawn the bootstrap as a `tokio::spawn` task after
`load()` returns.
```

### D8: Persistence -- Labels in SHRM v2 JSON Metadata

Labels are persisted as part of the `MemoryEntry` struct, which is already serialized
to the JSON metadata section of the `.shrm` binary format. No binary format changes
are required.

```
+------------------+
| SHRM header      |  (magic, version, entry count, dim)
+------------------+
| Embedding array  |  (N * dim * sizeof(f32) bytes, flat)
+------------------+
| JSON metadata    |  <-- labels: Vec<String> + label_version: u8 live here
+------------------+
| CRC32 checksum   |
+------------------+
```

The `label_index` HashMap is **not** persisted. It is a derived structure, rebuilt from
entry labels during `load()` in O(N * L) time where N = entries and L = average labels
per entry. At 100K entries with 5 labels each: ~500K insertions, <10ms.

### D9: Concurrency Model

The current `EchoEngine` uses `RwLock<EchoStore>` for the store and `Mutex<CosineHash>`
for LSH. The label index lives inside `EchoStore` and inherits the same `RwLock`
protection: reads during echo hold a read lock; writes during store/consolidation hold
a write lock.

For Phase 3+ when consolidation batch-rebuilds the label index (e.g., after merging
near-duplicate labels), the index can be moved behind an `ArcSwap`:

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;

pub struct EchoStore {
    entries: Vec<MemoryEntry>,
    embeddings: Vec<Vec<f32>>,
    id_to_index: HashMap<MemoryId, usize>,
    parent_children: HashMap<MemoryId, Vec<usize>>,

    /// Phase 1: inline HashMap (simple, sufficient at 100K)
    label_index: HashMap<String, Vec<u32>>,

    // Phase 3+: ArcSwap for zero-reader-blocking batch rebuilds
    // label_index: ArcSwap<Arc<HashMap<String, Vec<u32>>>>,
}
```

**Phase 1 approach (this ADR):** The label index is a plain `HashMap` inside
`EchoStore`. All access goes through the existing `RwLock<EchoStore>`. This is correct
and sufficient: echo queries hold a read lock, store/consolidation holds a write lock.
The consolidation write lock is held for <10ms per cycle, which does not cause
observable reader starvation.

**Phase 3+ approach (future ADR):** When consolidation needs to do bulk label
operations (merge synonyms, normalize vocabulary), move the index to
`ArcSwap<Arc<HashMap<...>>>`. Consolidation builds a new HashMap, then atomically
swaps it in. Readers never block -- they hold an `Arc` to the old index until they
finish. This is the same pattern used by Qdrant's adaptive query planner.

### D10: License-Compatible Dependencies

ShrimPK is licensed Apache 2.0. All new dependencies must be compatible.

| Dependency | Purpose | License | Decision |
|----------------------|-------------------------------|-------------|--------------------------------------|
| `gline-rs` | GLiNER zero-shot NER | Apache 2.0 | **Use (Phase 2).** Verify `ort` version matches fastembed 5. (Amendment 4) |
| `yake-rust` | YAKE keyword extraction | Check | Preferred if Apache 2.0 or MIT. |
| `keyword_extraction` | YAKE/RAKE/TextRank | GPL-3.0 | **Reject.** GPL is incompatible. |
| `model2vec-rs` | Static embeddings for prototypes | MIT | **Use.** Verify no `ort` dependency or compatible version. (Amendment 4) |
| `arc-swap` | Lock-free concurrent swap | Apache 2.0/MIT | **Use (Phase 3+).** Already dual-licensed. |
| `roaring` | RoaringBitmap | Apache 2.0/MIT | **Use (Phase 4+).** Not needed at 100K. |

**YAKE fallback:** If no Apache 2.0/MIT YAKE implementation exists, hand-roll the YAKE
algorithm. The core algorithm is ~200 lines of Rust: word co-occurrence statistics,
positional weighting, and candidate scoring. The academic paper (Campos et al., 2020)
is freely available and the algorithm is not patented.

### D11: Scale Path

The label bucket architecture is designed to scale incrementally:

| Scale tier | Memory count | Index structure | Candidate merge |
|------------|-------------|-------------------------------|----------------------------------------------|
| Tier 0 | <10K | No label index needed | LSH alone is sufficient |
| **Tier 1** | **10K-100K** | **`HashMap<String, Vec<u32>>`** | **Label OR + LSH union (this ADR)** |
| Tier 2 | 100K-500K | `HashMap<String, Vec<u32>>` | + cardinality-based threshold switching |
| Tier 3 | 500K-1M | `HashMap<String, RoaringBitmap>` | Bitmap intersection for multi-label AND |
| Tier 4 | 1M+ | Per-category LSH sub-indexes | Category routing before any index lookup |

**Tier 1 (this ADR):** `Vec<u32>` posting lists, OR-only merge, inline HashMap inside
`EchoStore`. Targets P50 < 4ms at 100K memories.

**Tier 2 (future):** Same data structure, add adaptive filtering -- skip labels whose
posting list exceeds 20% of the store (too broad to be useful as a filter). This
prevents high-cardinality labels from flooding the candidate merge.

**Tier 3 (future):** Replace `Vec<u32>` with `RoaringBitmap` when posting lists grow
beyond ~50K entries. Bitmap intersection becomes faster than sorted-Vec merge at this
scale. Enables efficient multi-label AND queries.

**Tier 4 (future):** At 1M+ memories, partition the LSH index into per-category
sub-indexes (one LSH per `MemoryCategory` or per top-level `topic:` label). A query
is routed to the relevant sub-indexes, each returning candidates from a focused
partition. This is the Filtered-DiskANN pattern adapted for LSH.

---

## 4. Consequences

### Changes to the Echo Pipeline

1. **Step 3 (candidate retrieval)** gains a new label-based source alongside LSH.
   The hardcoded threshold of 10 for LSH fallback is replaced by `MIN_CANDIDATES = 5`
   applied to the merged label + LSH candidate set.

2. **Steps 4-8 are unchanged.** Cosine scoring, pipe split, Hebbian co-activation,
   recency decay, and reranking operate on the candidate set without modification.

3. **A new query classification step** runs between embedding (step 1) and candidate
   retrieval (step 3). This adds <0.02ms to query latency in the worst case (Tier B
   prototype matching). Tier A (keyword) is <0.001ms.

### Changes to the Consolidation Pipeline

1. **Consolidation gains Pass 4 (label enrichment).** This pass runs after fact
   extraction and supersedes detection, using the same LLM call. Zero additional
   round-trips.

2. **The `ConsolidationResult` struct** gains a new field: `labels_enriched: usize`
   tracking how many memories received Tier 2 labels per cycle.

3. **A one-time bootstrap** runs on first load after upgrade to apply Tier 1 labels
   to all existing entries (~5 seconds at 100K).

### Changes to the Store

1. **`EchoStore`** gains a `label_index: HashMap<String, Vec<u32>>` field and three
   new methods: `query_labels()`, `rebuild_label_index()`,
   `bootstrap_tier1_labels()`.

2. **`EchoStore::add()` and `remove()`** are extended to maintain the label index
   incrementally.

3. **`EchoStore::load()`** calls `rebuild_label_index()` after deserializing entries.

### New Dependencies

| Crate | When | Size impact |
|-------------|---------|----------------------------|
| `gline-rs` | Phase 2 | ~188 MB model (downloaded, not compiled in) |
| `model2vec-rs` | Phase 1 | ~8 MB model |
| `yake-rust` or hand-rolled | Phase 1 | Minimal |

### Migration Path

**Zero breaking changes.** Existing `.shrm` files deserialize without modification
because both new fields (`labels`, `label_version`) use `#[serde(default)]`. On first
load:

1. All entries have `labels: []` and `label_version: 0`.
2. The bootstrap pass applies Tier 1 labels in <5 seconds.
3. Subsequent consolidation cycles apply Tier 2 labels at 10 per cycle.
4. After ~10K consolidation cycles (at 5-minute intervals: ~35 days), all 100K entries
   are fully enriched.

Users upgrading from pre-label versions experience no interruption. The label system
activates incrementally as labels are generated.

---

## 5. Acceptance Criteria

### Phase 1 -- Label Infrastructure

**Implementation order within Phase 1 (Amendment 6):**
1. Add `labels` + `label_version` to `MemoryEntry` and `MemoryMeta` (no behavior change)
2. Add `label_index` to `EchoStore` with `add()/remove()` maintenance + `rebuild_label_index()`
3. Add `query_labels()` to `EchoStore`
4. Modify `echo_text()` D6 merge (initially with empty label candidates -- pure LSH, no regression)
5. Implement keyword/rule-based classifiers (Tier 1 partial)
6. Implement `LabelPrototypes` + prototype cosine matching (requires embedder at startup)
7. Wire Tier 1 into `store()` pipeline
8. Implement async bootstrap for existing stores

| Criterion | Metric | Target |
|-----------|--------|--------|
| Label field serialization roundtrip | Unit test passes | `labels` and `label_version` survive serde JSON + SHRM binary |
| Label index consistency | Property test | `label_index` matches entries at all times after add/remove sequences |
| Tier 1 label generation | Unit test | At least 2 labels generated for "I am learning Rust at work" |
| Label-aware echo retrieval | Integration test | Query "languages" returns label candidates for `topic:language` memories |
| Bootstrap performance | Benchmark | <5 seconds for 100K entries |
| Index rebuild performance | Benchmark | <10ms for 100K entries |
| No regression at 1K | Benchmark gate | P50 latency within 0.5ms of pre-label baseline |

### Phase 2 -- Background Enrichment

| Criterion | Metric | Target |
|-----------|--------|--------|
| Combined LLM prompt | Integration test | Single Ollama call returns both facts and labels |
| GLiNER entity extraction | Integration test | "I moved to Tokyo" produces `entity:tokyo` |
| `label_version` progression | Unit test | Entry advances from 0 -> 1 -> 2 correctly |
| Consolidation throughput | Benchmark | Labels add <50ms per memory to consolidation cycle |

### Phase 3 -- Tuning + Target

| Criterion | Metric | Target |
|-----------|--------|--------|
| 100K echo P50 latency | Benchmark gate | **P50 < 4ms** (down from 23.79ms) |
| 100K echo P99 latency | Benchmark gate | **P99 < 10ms** |
| Brute-force fallback rate | Telemetry | **< 5% of queries** trigger brute-force at 100K |
| Label coverage | Telemetry | **> 80% of memories** have `label_version >= 1` |
| Recall preservation | LongMemEval suite | **100% isolated, >= 88% shared** (no regression) |

---

## 6. Alternatives Considered

### A. Pure LSH Tuning (Rejected)

Increase the number of LSH hash tables and/or bits per table to improve recall at
100K. This was the first approach attempted.

**Why rejected:** LSH recall is fundamentally limited by the data distribution. When
only 8 out of 100K memories are about "language learning", no number of hash tables
reliably places them in the same bucket as the query. The problem is not hash quality
-- it is topical sparsity. Labels solve the distribution problem directly.

### B. BM25 Full-Text Index (Rejected)

Add a BM25 inverted index (term -> document frequency) alongside embeddings. This
would provide keyword-based retrieval as a complement to semantic similarity.

**Why rejected:** BM25 requires tokenization, stemming, IDF computation, and term
frequency storage. It is the right solution for document search engines but overkill
for personal memory where the vocabulary is small, queries are short, and the
embedding model already captures semantic meaning. The label index achieves the same
pre-filtering goal with 1/10th the complexity.

### C. Post-Filter Approach (Rejected)

Retrieve the top-200 results from LSH/brute-force, then filter by label match.

**Why rejected:** Post-filtering degrades recall when the relevant set is small
relative to the store. At <10% selectivity (e.g., 50 language memories out of 100K),
the top-200 LSH results may contain zero language memories, making the filter
return empty. Pre-filtering (this ADR) guarantees that label-matching memories are
included in the candidate set before scoring.

### D. Per-Category Separate HNSW Graphs (Rejected)

Partition the embedding space into 6 HNSW graphs, one per `MemoryCategory`. Route
queries to the appropriate graph.

**Why rejected:** Premature at 100K. HNSW graphs have significant memory overhead
(M * N * sizeof(pointer) per graph), and 6 graphs of 16K entries each would consume
more RAM than a single graph of 100K. Furthermore, `MemoryCategory` is too coarse
for routing -- a query about "language learning" could match memories categorized as
`ActiveProject`, `Preference`, or `Fact`. Labels provide finer-grained routing without
the overhead of multiple index structures.

---

## References

### Academic

- Campos, R. et al. (2020). "YAKE! Keyword Extraction from Single Documents using Multiple Local Features." *Information Sciences*, 509.
- Gollapudi, S. et al. (2023). "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters." *WWW 2023*.
- Kraft, P. (2024). "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data." *SIGMOD 2024*.
- Xu, Y. et al. (2025). "A-MEM: Agentic Memory for LLM Agents." *NeurIPS 2025*.
- Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
- Tononi, G. & Cirelli, C. (2003). "Sleep and synaptic homeostasis." *Sleep Medicine Reviews*, 10(1).

### Industry Systems

- Qdrant: Filterable HNSW with adaptive query planner (cardinality-based threshold switching).
- Pinecone: Roaring bitmap metadata per slab, single-stage filtered search.
- Milvus: Clustering compaction on scalar fields (25x QPS improvement at 20M vectors).
- Weaviate: ACORN two-hop expansion for filtered HNSW (default since v1.34).
- Obsidian: MetadataCache tag-to-files reverse index (instant tag filtering at 50K notes).

### Rust Crates

- `gline-rs` -- GLiNER zero-shot NER inference (Apache 2.0).
- `model2vec-rs` -- Static embeddings for prototype classification (MIT).
- `arc-swap` -- Lock-free atomic pointer swap (Apache 2.0 / MIT).
- `roaring` -- Compressed bitmap indexes (Apache 2.0 / MIT).
