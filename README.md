# Bellkis Kernel

The AI Operating System kernel — Echo Memory, provider routing, context assembly.

## What is this?

Bellkis Kernel is a standalone Rust library that provides:

- **Echo Memory** — push-based AI memory where memories find YOU (not the other way around)
- **Provider Router** — intelligent model routing with cascade fallback and cost optimization
- **Context Assembler** — smart prompt compilation with automatic memory integration

Unlike RAG (where you search for memories), Echo Memory **listens** to your conversation and **self-activates** when relevant context is found — like how human memory works.

## Quick Start

```rust
use bellkis_memory::EchoMemory;
use bellkis_core::EchoConfig;

let config = EchoConfig::auto_detect();
let memory = EchoMemory::new(config)?;

// Store memories
memory.store("I prefer FastAPI for REST APIs", "conversation")?;
memory.store("Python is my main language", "conversation")?;

// Memories find YOU — no explicit search needed
let echoes = memory.echo("What framework should I use for this API?", 5)?;
// Returns: FastAPI memory with 0.85 similarity, in 15ms
```

## Architecture

- **bellkis-core** — types, traits, error framework
- **bellkis-memory** — Echo Memory engine
- **bellkis-router** — provider routing and cascade logic
- **bellkis-context** — context assembly and prompt compilation
- **bellkis-security** — sandbox, permissions, PII filtering
- **bellkis-kernel** — facade crate (re-exports all)

## Performance

- 1M memories indexed in ~1.8GB RAM (f32) or ~150MB (binary quantized)
- Echo cycle: <30ms end-to-end (embedding + search + rank)
- Persistence: save/load 1M entries in <2 seconds

## License

Apache 2.0 — see [LICENSE](LICENSE)
