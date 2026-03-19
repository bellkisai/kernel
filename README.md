# ShrimPK Kernel

The AI Operating System kernel — Echo Memory, provider routing, context assembly.

## What is this?

ShrimPK Kernel is a standalone Rust library that provides:

- **Echo Memory** — push-based AI memory where memories find YOU (not the other way around)
- **Provider Router** — intelligent model routing with cascade fallback and cost optimization
- **Context Assembler** — smart prompt compilation with automatic memory integration

Unlike RAG (where you search for memories), Echo Memory **listens** to your conversation and **self-activates** when relevant context is found — like how human memory works.

## Quick Start

```rust
use shrimpk_memory::EchoMemory;
use shrimpk_core::EchoConfig;

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

- **shrimpk-core** — types, traits, error framework
- **shrimpk-memory** — Echo Memory engine
- **shrimpk-router** — provider routing and cascade logic
- **shrimpk-context** — context assembly and prompt compilation
- **shrimpk-security** — sandbox, permissions, PII filtering
- **shrimpk-kernel** — facade crate (re-exports all)

## Performance

- 1M memories indexed in ~1.8GB RAM (f32) or ~150MB (binary quantized)
- Echo cycle: <30ms end-to-end (embedding + search + rank)
- Persistence: save/load 1M entries in <2 seconds

## License

Apache 2.0 — see [LICENSE](LICENSE)
