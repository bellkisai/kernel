# ShrimPK

**Push-based AI memory where memories find you.**

[![CI](https://github.com/bellkisai/kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/bellkisai/kernel/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

ShrimPK is an AI memory kernel written in Rust. Unlike traditional RAG systems where you search for memories, Echo Memory **listens** to context and **self-activates** when relevant — like human spontaneous recall.

**3.50ms** P50 at 100K memories | **38%** more accurate than plain LLM | **15.7%** token savings

## Features

- **Echo Memory** — push-based activation, memories surface without explicit queries
- **Three-layer pipeline** — Bloom filter (O(1) rejection) + LSH (sub-linear search) + exact cosine
- **Hebbian learning** — co-activated memories strengthen connections over time
- **Category-aware decay** — 6 types from Identity (365d) to Conversation (3d)
- **Sleep consolidation** — background pruning, dedup merging, index rebuild
- **PII masking** — 15 patterns, mask-don't-block approach
- **Provider router** — cascade fallback, cost tracking, circuit breaker
- **Context assembler** — token-budgeted prompt compilation with priority truncation
- **Config system** — TOML file + env var overrides + auto-detect from RAM

## Install

### Rust library

```toml
[dependencies]
shrimpk-kernel = "0.1"
```

### Python

```bash
pip install shrimpk  # via maturin
```

### CLI

```bash
cargo install --path cli
```

## Quick Start

### Rust

```rust
use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;

let config = EchoConfig::auto_detect();
let engine = EchoEngine::new(config)?;

// Store memories
engine.store("I prefer FastAPI for REST APIs", "conversation").await?;
engine.store("Python is my main language", "conversation").await?;

// Memories find YOU
let echoes = engine.echo("What framework for this API?", 5).await?;
// Returns: FastAPI memory (similarity: 0.85) in ~3.5ms
```

### CLI

```bash
shrimpk store "I prefer FastAPI for REST APIs"
shrimpk echo "What framework for APIs?"
shrimpk stats
shrimpk config show
shrimpk status
```

### Python

```python
from shrimpk import EchoMemory, EchoConfig

config = EchoConfig.auto_detect()
mem = EchoMemory(config)
mem.store("I prefer FastAPI", source="conversation")
results = mem.echo("What framework?", max_results=5)
```

## MCP Server (Claude Code / AI Tool Integration)

ShrimPK ships an MCP server that exposes Echo Memory to any MCP-compatible AI tool.

```bash
# Register globally (works from any directory)
claude mcp add --transport stdio --scope user shrimpk -- shrimpk-mcp

# Claude Code now has 9 new tools:
# mcp__shrimpk__store, mcp__shrimpk__echo, mcp__shrimpk__stats, etc.
```

The MCP server uses the same data directory (`~/.shrimpk-kernel/`) as the CLI. Memories stored via MCP are visible in CLI and vice versa.

## Architecture

```
shrimpk-kernel          (facade — re-exports all)
  shrimpk-core          (types, config, errors, PII types)
  shrimpk-memory        (Echo Memory engine)
  shrimpk-router        (provider routing + cascade)
  shrimpk-context       (context assembly + token budgeting)
  shrimpk-security      (sandbox, permissions — stub)
  shrimpk-python        (PyO3 bindings)
  shrimpk-mcp           (MCP server — 9 tools over JSON-RPC stdio)
```

## Performance

| Metric | Value |
|--------|-------|
| Echo P50 at 100K memories | 3.50ms |
| Echo P95 at 100K memories | 6.88ms |
| Head-to-head accuracy | +38% vs plain LLM |
| Personalization rate | 100% |
| Token savings | 15.7% per request |
| Follow-up elimination | 100% |
| RAM (1M memories, f32) | ~1.8 GB |
| RAM (1M memories, binary) | ~150 MB |
| Tests | 222 passing |

## Configuration

ShrimPK auto-detects from system RAM (4 tiers: minimal/standard/full/maximum). Override via:

```bash
# Config file
shrimpk config set max_memories 500000
shrimpk config set quantization binary
shrimpk config show

# Environment variables (highest priority)
SHRIMPK_MAX_MEMORIES=500000
SHRIMPK_DATA_DIR=/custom/path
SHRIMPK_QUANTIZATION=binary
SHRIMPK_MAX_DISK_BYTES=4294967296
```

Priority: env vars > `~/.shrimpk-kernel/config.toml` > auto-detect

## License

Apache 2.0 — see [LICENSE](LICENSE)
