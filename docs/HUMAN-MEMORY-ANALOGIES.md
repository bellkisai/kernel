# The Neuroscience Behind ShrimPK

ShrimPK did not begin with algorithms. It began with a question: how does the human brain actually work as a memory system, and why do AI tools fail so badly at replicating even the most basic properties of it?

This document walks through the neuroscience that shaped ShrimPK's design — the principles borrowed from human memory research, how they map to specific implementation decisions, and why those decisions matter for anyone building AI systems that need to remember.

---

## 1. Two Kinds of Memory

Neuroscientists distinguish at least two fundamentally different types of long-term memory.

**Episodic memory** stores specific events: *"I had coffee with Alex at 9am on a Tuesday."* It is autobiographical, time-stamped, and contextual. You remember the episode — the setting, the sequence, the sensory texture of the moment.

**Semantic memory** stores facts stripped of their episodic context: *"Alex works at Stripe."* It is distilled knowledge. You know this to be true without needing to recall which conversation taught you it, or when.

The hippocampus is the brain region most responsible for encoding episodic memories. It binds together the distributed cortical representations of an experience — the sights, sounds, words, and emotions — into a retrievable trace. Over time, and especially during sleep, the hippocampus replays these traces and gradually transfers stable knowledge to the neocortex as semantic memory, no longer needing the original episodic scaffolding.

ShrimPK models this distinction directly. A raw conversational exchange stored via `shrimpk store` is an episodic record — it captures context, source, and timestamp. The sleep consolidation process (described in section 3) runs an LLM over these raw episodes and extracts atomic facts: *"User prefers Rust over Go."* These extracted facts are semantic memories. The parent episode is marked as enriched; the child facts become independently searchable. The same raw conversation that arrived as an episode transforms, over time, into structured knowledge — just as the hippocampus gradually hands off memories to the cortex.

---

## 2. How Memory Is Consolidated: Working Memory to Long-Term Storage

The standard neuroscience model of memory consolidation runs through three stages.

**Working memory** holds a small amount of information in active maintenance — roughly seven items for a few seconds. It is the scratchpad: highly accessible but extremely volatile.

**Short-term memory** extends this window somewhat, but without consolidation, memories still decay rapidly. The critical event that converts short-term to long-term memory is encoding strength — driven by attention, emotional salience, and repetition.

**Long-term memory** is not a single storage location. It is the result of synaptic changes distributed across the cortex. The more a memory is rehearsed, the more its neural trace is strengthened, and the more resistant it becomes to forgetting.

This multi-stage model maps directly to ShrimPK's architecture. Memories are stored immediately (encoding), organized via associative graphs (working consolidation), and periodically restructured by a background process during low-activity periods (sleep consolidation). Memories that are never retrieved decay; memories that are frequently co-retrieved with other memories build strong associations and become entrenched.

The behavioral implication is significant. An AI system that only stores and retrieves — without modeling the consolidation and decay dynamics — cannot replicate even basic human-like memory behavior. It either remembers everything with equal weight (unnatural) or forgets everything (useless). The human brain occupies neither extreme.

---

## 3. Hebbian Learning: Neurons That Fire Together, Wire Together

Donald Hebb proposed in 1949 that when two neurons repeatedly activate together, the synaptic connection between them strengthens. This became one of the most important principles in neuroscience and underpins most modern theories of associative memory and learning.

The practical implication: memory is not stored in isolation. Each memory is part of a web of associations. Recalling "kitchen" makes "coffee" more accessible. Recalling "cooking" primes "recipes." The associations are learned from co-occurrence — not encoded explicitly, but built up naturally through repeated experience.

### How ShrimPK Implements This

ShrimPK's `HebbianGraph` in `crates/shrimpk-memory/src/hebbian.rs` implements exactly this principle. Every memory has an index. When two memories are returned together in the same echo query, their Hebbian edge is strengthened:

```rust
// From hebbian.rs: "neurons that fire together wire together"
let strength = (sim_i * sim_j).sqrt() * 0.1; // modest reinforcement
hebbian.co_activate(top_indices[i], top_indices[j], strength);
```

The strength of co-activation is proportional to the geometric mean of both memories' similarity scores. If two strongly-relevant memories are retrieved together, their association strengthens substantially. If one is only marginally relevant, the reinforcement is weaker.

These edges are not permanent. They decay exponentially with a half-life of seven days:

```rust
// From hebbian.rs: exponential decay since last activation
let elapsed = now - entry.last_activated;
if elapsed > 0.0 {
    entry.weight *= (-self.lambda * elapsed).exp();
}
entry.weight += strength;
```

This mirrors biological synapse dynamics. A synapse that is not used gradually weakens through a process called synaptic depression. In ShrimPK, edges that are never reactivated fade below a pruning threshold and are removed during consolidation.

### Why This Matters in Practice

Consider a user who frequently discusses their cooking setup. They have stored memories about kitchen organization, recipe management, and grocery habits. Each time they ask about one of these topics, the other related memories co-activate and their Hebbian edges strengthen. Eventually, a question about "what's for dinner" does not just retrieve the literal dinner memory — it surfaces the entire associated cluster, because that cluster has been rehearsed together enough times that the connections are strong.

This is not keyword matching. It is learned, usage-shaped association — the same mechanism the brain uses to make experienced knowledge feel effortless to access.

The Hebbian graph also supports typed relationships beyond simple co-occurrence. During sleep consolidation, extracted facts can be tagged with semantic relationship types: `WorksAt`, `LivesIn`, `PrefersTool`, `Supersedes`. A `Supersedes` edge signals that newer information replaces older information — so when both appear in a result set, the newer memory receives a boost and the superseded memory is demoted. This models the brain's ability to update beliefs rather than simply accumulating contradictory facts.

---

## 4. Sleep Consolidation: The Brain's Maintenance Cycle

One of the most striking discoveries in memory research from the late 20th century was that sleep is not passive rest — it is an active memory processing phase. During slow-wave sleep, the hippocampus replays the day's experiences at high speed, re-activating the neural traces of recent events. This process, called **hippocampal replay** or **memory consolidation during sleep**, gradually transfers episodic memories to more stable cortical representations.

Two complementary theories explain what happens:

**Hippocampal-neocortical transfer**: The hippocampus acts as a fast learner that can bind new information quickly but has limited capacity. During sleep, it offloads consolidated representations to the neocortex for long-term storage, freeing capacity for the next day's experiences.

**Synaptic homeostasis hypothesis (Tononi & Cirelli, 2003)**: Waking experience strengthens many synapses simultaneously. If unchecked, synaptic potentiation would eventually saturate the system. During sleep, the brain globally downscales synaptic weights — pruning the weakest connections while preserving the relative strength of the important ones. The net effect: important experiences are retained, noise is cleared.

### ShrimPK's Sleep Phase

ShrimPK implements a direct analog to this process. The background consolidation task in `crates/shrimpk-memory/src/consolidation.rs` runs every five minutes and performs five maintenance steps:

1. **Hebbian edge pruning** — edges whose decayed weight fell below threshold are removed (synaptic homeostasis).
2. **Bloom filter rebuild** — after deletions, the fast-path pre-screener is reconstructed.
3. **Near-duplicate merging** — pairs of memories with cosine similarity above 0.95 are merged (episodic compression into semantic knowledge).
4. **Echo count decay** — memories that haven't been retrieved in 30 days have their activity score reduced.
5. **LLM fact extraction** — un-enriched raw memories are processed by a language model to extract atomic, structured facts. These become child memories linked to the parent episode.

The near-duplicate merging step (step 3) is particularly important and directly models what sleep does. When you experience similar things across multiple days — meetings that follow a similar pattern, conversations that revisit similar topics — the brain does not store N separate episodic traces of equal strength. It gradually merges them into a single, more abstract representation. ShrimPK's consolidator does the same: two memories that are 95% or more similar in embedding space are merged, keeping only one. What was an accumulation of redundant episodes becomes a single, well-connected semantic entry.

The LLM fact extraction step (step 5) models the second half of this process: distilling episodic content into propositional semantic knowledge. A raw conversation like *"the user mentioned they recently switched from Python to Rust for their backend work"* becomes extracted facts: *"User switched to Rust"*, *"User works on backend systems"*. These facts are more precise, more independently retrievable, and correctly typed for relationship detection.

---

## 5. Forgetting and the Ebbinghaus Forgetting Curve

Hermann Ebbinghaus published his forgetting curve in 1885, and it has held up remarkably well. The core finding: memory retention decays exponentially over time. You forget most of a new experience within the first few hours. What survives 24 hours tends to survive much longer. Repeated recall substantially flattens the curve — each retrieval effectively resets the decay clock and strengthens the trace.

This exponential decay is not a flaw in human memory. It is a feature. The brain is not trying to be a perfect recording device. It is trying to maintain a model of the world weighted by what is useful. Things you encounter often and think about frequently are useful. Things you encountered once three months ago and never revisited probably are not.

### How ShrimPK Models Decay

ShrimPK implements category-aware exponential decay. Every stored memory is classified into a category when it is stored, using the `MemoryCategory` enum. Each category has a different half-life:

- Long-lived facts (names, permanent preferences): half-life measured in months
- Working project context: half-life measured in weeks
- Transient conversational context: half-life measured in days

When a memory is retrieved (echoed), its `final_score` is multiplied by a decay factor based on its age and category:

```rust
let age_secs = (now - entry.created_at).num_seconds().max(0) as f64;
let half_life = entry.category.half_life_secs();
let decay = (-age_secs * std::f64::consts::LN_2 / half_life).exp();
```

This means a conversational memory about what a user was debugging last week scores noticeably lower than the same memory would have scored yesterday — even if its semantic similarity to the query is identical. The system naturally deemphasizes stale context.

The Hebbian reinforcement mechanism acts as the counterweight. A memory that is frequently co-retrieved with other relevant memories accumulates Hebbian weight, which boosts its `final_score` independently of decay. A memory that is both recent AND frequently co-activated scores highest. This replicates the spacing effect in human memory: memories that are regularly revisited resist the forgetting curve. Memories that are stored once and never retrieved fade, as they should.

The balance is: remember what matters, forget what doesn't — and let usage patterns determine what matters.

---

## 6. Multimodal Memory: Multiple Senses, One Experience

Human memory is inherently multimodal. When you remember a conversation, you remember not just the words, but the visual setting (who was there, what room you were in), the acoustic qualities (tone of voice, whether the person seemed agitated or relaxed), and the emotional context. These are not separate memories — they are bound together into a single episodic trace by the hippocampus.

The classic demonstration is the **Proust effect**: a smell or piece of music can trigger a vivid episodic memory that semantic cues alone could not reach. The memory was encoded with multimodal binding; a non-verbal sensory cue can unlock the whole trace.

Neuroscience has mapped the distinct cortical regions involved:

- **Verbal/semantic content** is processed in the temporal and prefrontal cortex (language areas).
- **Visual content** is processed in the visual cortex (occipital lobe) and stored via visual representations.
- **Auditory/paralinguistic content** — tone, prosody, speaker identity, emotional valence — is processed in the auditory cortex and superior temporal sulcus. Crucially, the brain encodes *how* something was said separately from *what* was said.

When these streams converge in the hippocampus for encoding, they are bound into a single unified memory that can be triggered by cues from any modality.

### ShrimPK's Three-Channel Architecture

ShrimPK v0.5.0 mirrors this architecture directly with three independent embedding channels, each matching a distinct cortical processing stream:

**Text channel** (MiniLM, 384-dim): The verbal/semantic stream. What was said, written, or described in language. This is the primary channel for conversational AI and captures propositional content.

**Vision channel** (CLIP ViT-B/32, 512-dim): The visual cortex stream. Images, screenshots, and visual observations. CLIP's joint text-image embedding space enables a critical capability: you can query with text and retrieve images, or vice versa — the same way seeing a kitchen can trigger language-encoded cooking memories.

**Speech channel** (ECAPA-TDNN 512-dim + Whisper-tiny 384-dim, concatenated 896-dim total): The auditory/paralinguistic stream. This channel captures not what was said, but *how* it was said — speaker identity and prosodic patterns like rhythm and stress. The two-model architecture matches distinct processing layers of the auditory cortex:

- **ECAPA-TDNN** (512-dim): Speaker identity — analogous to voice recognition in the superior temporal gyrus.
- **Whisper-tiny encoder** (384-dim): Prosodic structure — analogous to prosodic processing in the right hemisphere, which tracks rhythm, stress, and intonation independently of word content.

> **Note:** An emotion sub-embedding (arousal/dominance/valence) was explored during design but dropped because available models carry non-commercial licenses incompatible with Apache 2.0. The architecture can accommodate a third sub-embedding when a permissively licensed emotion model becomes available — this would correspond to affective processing in the amygdala-connected auditory pathways.

Each channel has its own LSH index for sub-linear retrieval. The Hebbian graph is shared across all channels — a single association graph that links memories regardless of the modality they were encoded in. This is the key architectural decision: cross-modal associations are possible. An image of a kitchen and a text memory about cooking can develop a Hebbian association, so that retrieving one surfaces the other. This is how the Proust effect works: modality-independent associative binding.

### Cross-Modal Retrieval

The practical consequence of this architecture: you can store an image and retrieve it with a text query.

```bash
shrimpk store-image morning-photo.jpg
shrimpk echo --modality vision "where is the cup?"
# → morning-photo.jpg (similarity: 0.82)
```

CLIP embeds both text and images into a shared semantic space. A text query about *"cup"* lands near the image embedding of a photo containing a cup, because CLIP was trained on text-image pairs. ShrimPK queries the vision LSH with the CLIP-encoded text query, retrieving images that the language description matches. The human analogy: you describe something in words, and your visual memory surfaces the corresponding scene.

---

## 7. Push vs. Pull: Spontaneous Recall

Here is the most fundamental departure from how existing AI memory systems are designed.

When you are in a conversation and suddenly remember something relevant — that you had this same discussion last year, that the person you are talking with has a known preference, that a piece of information from last week directly applies — you did not issue a query. You did not search. The memory surfaced spontaneously because the current context activated the stored trace through associative pathways. This is **spontaneous recall**: the memory came to you.

The neurological mechanism is associative spreading activation. The current context activates nearby nodes in the brain's associative network. If one of those nodes has a strong enough connection to a stored memory, that memory crosses the threshold into conscious awareness. It is a push, not a pull.

Traditional RAG systems — and most AI memory tools — operate as pull systems. They are search engines. You (or an agent) explicitly construct a query and ask the memory system to retrieve relevant information. This requires knowing what you are looking for in order to find it. It fails when you do not know what you have forgotten. And it requires the agent or the user to build retrieval logic, choose when to query, and decide what to search.

ShrimPK inverts this model. Every incoming message is automatically embedded and used as an echo query against the full memory store. The system checks, on every turn, whether any stored memories activate. When they do, they are injected into the context automatically. The user and the agent do not need to do anything. Relevant memories find the current conversation, rather than the other way around.

This is why the core operation is named `echo` rather than `search`. In neuroscience terms, the incoming stimulus causes relevant stored traces to echo back — they resonate with the current context and become active. The terminology is intentional.

---

## 8. The Mantis Shrimp: Richest Sensory Memory in Nature

ShrimPK is named after the mantis shrimp, and not merely because it sounds distinctive.

The mantis shrimp (*Stomatopoda*) has one of the most extraordinary visual systems ever evolved. Where humans have three types of photoreceptor cells (corresponding to red, green, and blue), the mantis shrimp has sixteen. It can perceive ultraviolet light, infrared light, and polarized light across multiple orientations. Its compound eyes move independently, giving it an ability to analyze visual scenes through more than a dozen simultaneous processing channels.

What is remarkable about this is not just the number of channels. It is that the mantis shrimp processes this information differently from humans: rather than combining channels into a high-dimensional internal representation and then discriminating, research suggests it performs parallel channel comparison at the receptor level, making rapid categorical judgments across all channels simultaneously (Thoen et al., *Science*, 2014).

The aspiration embedded in ShrimPK's name is explicit: to be the richest sensory memory system for AI, processing information across as many modalities as possible, in parallel, without losing the associative connections that make memory useful rather than merely archival.

ShrimPK's three-channel architecture — text, vision, speech — is the beginning of this. The text channel corresponds to human semantic-verbal processing. The vision channel corresponds to visual cortex processing. The speech channel reaches for something human memory does naturally but AI systems almost never do: encoding the paralinguistic and affective dimensions of auditory experience, not just its transcribed content.

As the system develops further channels, the mantis shrimp remains the model: not richer storage, but richer sensory channels, processed in parallel, bound into a unified associative memory.

---

## Summary: Principles and Their Implementations

| Neuroscience Principle | ShrimPK Implementation |
|------------------------|------------------------|
| Episodic → Semantic consolidation | Raw memories → LLM fact extraction → typed child memories |
| Hebbian learning (co-activation strengthens synapses) | `HebbianGraph` with exponential decay, co-activation on every echo |
| Synaptic homeostasis (sleep prunes weak connections) | Background consolidation prunes Hebbian edges below threshold |
| Near-duplicate compression (episodes merge into semantics) | Cosine similarity > 0.95 → merge (consolidation pass) |
| Ebbinghaus forgetting curve | Category-aware exponential decay multiplied on every `final_score` |
| Spacing effect (retrieval resets decay) | Hebbian reinforcement boosts frequently co-retrieved memories |
| Multimodal binding (hippocampus binds sight, sound, language) | Single `HebbianGraph` shared across text, vision, speech channels |
| Spontaneous recall (spreading activation, not search) | Push-based echo: every incoming message queries all stored memories |
| Recency advantage (recent experiences more accessible) | Recency boost: `recency_weight / (1.0 + days_since_stored)` |
| Knowledge updates (new facts supersede old ones) | `Supersedes` edges in Hebbian graph: newer memory boosted, older demoted |

---

## References and Further Reading

- Hebb, D.O. (1949). *The Organization of Behavior: A Neuropsychological Theory.* Wiley.
- Ebbinghaus, H. (1885). *Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie.* Duncker & Humblot.
- McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419–457.
- Tononi, G., & Cirelli, C. (2003). Sleep and synaptic homeostasis: A hypothesis. *Brain Research Bulletin*, 62(2), 143–150.
- Squire, L.R. (2004). Memory systems of the brain: A brief history and current perspective. *Neurobiology of Learning and Memory*, 82(3), 171–177.
- Thoen, H.H., How, M.J., Chiou, T.H., & Marshall, J. (2014). A different form of color vision in mantis shrimp. *Science*, 343(6169), 411–413.
- Hasselmo, M.E. (2006). The role of acetylcholine in learning and memory. *Current Opinion in Neurobiology*, 16(6), 710–715.
- Schacter, D.L., & Addis, D.R. (2007). The cognitive neuroscience of constructive memory: Remembering the past and imagining the future. *Philosophical Transactions of the Royal Society B*, 362(1481), 773–786.
