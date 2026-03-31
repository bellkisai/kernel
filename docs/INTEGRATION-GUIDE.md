# ShrimPK Integration Guide

ShrimPK is a push-based AI memory engine. Instead of explicitly searching for memories, memories activate themselves when relevant context appears. This guide covers every integration path from MCP to HTTP to the Rust library.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [MCP Integration (Claude, AI Agents)](#2-mcp-integration)
3. [HTTP API (Daemon)](#3-http-api)
4. [CLI](#4-cli)
5. [Rust Library](#5-rust-library)
6. [Python Bindings](#6-python-bindings)
7. [Configuration](#7-configuration)
8. [Multimodal Features](#8-multimodal-features)
9. [Memory Proxy](#9-memory-proxy)

---

## 1. Quick Start

### Prerequisites

- **Rust toolchain** (stable, 2024 edition): https://rustup.rs
- **Ollama** (optional, for sleep consolidation and HyDE query expansion): https://ollama.com
  - Pull at minimum one model: `ollama pull llama3.2:3b`

### Build from Source

```bash
git clone https://github.com/bellkisai/kernel.git
cd kernel
cargo build --release
```

This produces three binaries in `target/release/`:

| Binary | Purpose |
|--------|---------|
| `shrimpk` | CLI for interactive use |
| `shrimpk-daemon` | Background HTTP server on `localhost:11435` |
| `shrimpk-mcp` | MCP server (JSON-RPC over stdio) |

For vision support (CLIP ViT-B-32, adds ~352 MB model on first run):

```bash
cargo build --release --features vision
```

### First Run

**Start the daemon** (loads the embedding model once, then serves forever):

```bash
./target/release/shrimpk-daemon
# [shrimpk] Starting daemon v0.5.0...
# [shrimpk] Loading Echo Memory engine...
# [shrimpk] Loaded 0 memories.
# [shrimpk] Serving on http://127.0.0.1:11435
```

**Store a memory:**

```bash
./target/release/shrimpk store "I prefer FastAPI for REST APIs"
# [shrimpk] Stored memory 3a7c2f1b... (sensitivity: Public)
# [shrimpk] PII scan: no sensitive data detected
```

**Retrieve it:**

```bash
./target/release/shrimpk echo "What framework for this API?"
# [shrimpk] Echo results for "What framework for this API?" (3.5ms):
#   #1 [0.85] "I prefer FastAPI for REST APIs" (source: cli, id: 3a7c2f1b...)
```

The daemon persists memories across restarts to `~/.shrimpk-kernel/echo_store.shrm` (SHRM v2 binary format).

---

## 2. MCP Integration

### What MCP Is

The Model Context Protocol (MCP) is an open standard for connecting AI tools to external systems. ShrimPK ships an MCP server (`shrimpk-mcp`) that exposes all memory operations as callable tools. When registered with Claude Desktop or Claude Code, the AI can store and retrieve memories automatically during conversation without any user intervention.

### Registering with Claude Code

```bash
# Register globally (persists across projects)
claude mcp add --transport stdio --scope user shrimpk -- /path/to/shrimpk-mcp
```

If `shrimpk-mcp` is on your `PATH`:

```bash
claude mcp add --transport stdio --scope user shrimpk -- shrimpk-mcp
```

### Registering with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "shrimpk": {
      "command": "/path/to/shrimpk-mcp",
      "args": []
    }
  }
}
```

Config file location:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

The MCP server auto-detects a running daemon and proxies requests via HTTP. If no daemon is running, it initializes the engine in-process (slower startup, same results).

### Available MCP Tools

#### `store`

Store a memory that will persist across sessions.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | — | Text content to remember |
| `source` | string | no | `"mcp"` | Label identifying where this memory came from |

Response: Confirmation with truncated memory ID and PII scan result.

#### `echo`

Find memories that resonate with a query. Returns results ranked by cosine similarity and Hebbian association strength.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | Text to match against stored memories |
| `max_results` | integer | no | `10` | Maximum number of results to return |
| `modality` | string | no | `"text"` | Query channel: `"text"`, `"vision"` (CLIP cross-modal), or `"auto"` (all channels) |

Response: JSON array with ranked results including `memory_id`, `content`, `similarity`, `final_score`, and `source`.

#### `forget`

Remove a memory permanently by its UUID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | yes | Full UUID of the memory to remove |

#### `stats`

Return engine statistics: memory count, capacity, index size, RAM usage, disk usage, query count, and average echo latency.

No parameters required.

#### `status`

Return system status with disk usage bar, config tier, RAM budget, and quantization mode.

No parameters required.

#### `config_show`

Show the current configuration with the source of each value (`env`, `file`, or `auto`).

No parameters required.

#### `config_set`

Set a configuration value. Writes to `~/.shrimpk-kernel/config.toml`. Requires server restart to take effect.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `key` | string | yes | Config key name (see [Configuration](#7-configuration)) |
| `value` | string | yes | Value to set |

Supported keys: `max_memories`, `similarity_threshold`, `max_echo_results`, `ram_budget_bytes`, `max_disk_bytes`, `quantization`, `data_dir`, `use_lsh`, `use_bloom`, `ollama_url`, `enrichment_model`, `consolidation_provider`, `max_facts_per_memory`

#### `dump`

List all stored memories with IDs, content preview, source, echo count, and category.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | no | `50` | Maximum entries to return |

#### `persist`

Force save all in-memory data to disk immediately.

No parameters required.

#### `store_image` (vision feature required)

Store an image as a visual memory using CLIP ViT-B-32 embeddings. Enables text-to-image cross-modal retrieval.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_base64` | string | yes | — | Base64-encoded image bytes (PNG, JPEG, etc.) |
| `source` | string | no | `"mcp"` | Source label |

#### `store_audio` (speech feature required)

Store audio as a speech memory preserving paralinguistic features (tone, emotion, pace). This is NOT speech-to-text.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_base64` | string | yes | — | Base64-encoded raw PCM f32 audio (little-endian, every 4 bytes = one f32 sample) |
| `sample_rate` | integer | no | `16000` | Audio sample rate in Hz |
| `source` | string | no | `"mcp"` | Source label |

### Example: Auto-Store and Echo in Conversation

With ShrimPK registered, Claude automatically stores and retrieves context without any explicit commands:

```
User: I'm building a backend service. I've decided to use Rust with Axum.

Claude: [internally calls store({text: "User is building a backend service with Rust and Axum"})]
        That's a great choice. Axum is ergonomic and performs well under load.

--- later ---

User: What HTTP framework did I choose?

Claude: [internally calls echo({query: "HTTP framework choice backend"})]
        [receives: "User is building a backend service with Rust and Axum" (similarity: 0.91)]
        You chose Axum for your Rust backend service.
```

---

## 3. HTTP API

### Starting the Daemon

```bash
# Default port (11435)
shrimpk-daemon

# Custom port
shrimpk-daemon --port 8080

# Install as autostart service (runs on login)
shrimpk-daemon --install

# Remove autostart
shrimpk-daemon --uninstall
```

The daemon writes a PID file to `~/.shrimpk-kernel/daemon.pid` and will refuse to start if another instance is already running.

**Optional authentication:** Set `SHRIMPK_AUTH_TOKEN` before starting:

```bash
SHRIMPK_AUTH_TOKEN=my-secret-token shrimpk-daemon
```

All subsequent requests must include `Authorization: Bearer my-secret-token`.

**Rate limit:** 100 requests/second by default. Configurable via `daemon_rate_limit`.

### Endpoints

---

#### `GET /health`

Health check. Always returns 200.

```bash
curl http://localhost:11435/health
```

Response:

```json
{
  "status": "ok",
  "memories": 42,
  "uptime_secs": 3600,
  "version": "0.5.0"
}
```

---

#### `POST /api/store`

Store a text memory.

```bash
curl -X POST http://localhost:11435/api/store \
  -H "Content-Type: application/json" \
  -d '{"text": "I prefer PostgreSQL for relational data", "source": "notes"}'
```

Request body:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Text to store |
| `source` | string | no | `"daemon"` | Source label |

Response:

```json
{
  "memory_id": "3a7c2f1b-4e8d-4c9a-b1f2-0e3d5a6b7c8d",
  "sensitivity": "Public",
  "pii_matches": 0
}
```

Sensitivity values: `Public`, `Internal`, `Restricted`, `Blocked` (Blocked content is rejected and not stored).

---

#### `POST /api/echo`

Find memories matching a query.

```bash
curl -X POST http://localhost:11435/api/echo \
  -H "Content-Type: application/json" \
  -d '{"query": "Which database should I use?", "max_results": 5}'
```

Request body:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | — | Search query |
| `max_results` | integer | no | `10` | Maximum results |
| `modality` | string | no | `"text"` | `"text"`, `"vision"`, or `"auto"` |

Response:

```json
{
  "results": [
    {
      "rank": 1,
      "memory_id": "3a7c2f1b-4e8d-4c9a-b1f2-0e3d5a6b7c8d",
      "content": "I prefer PostgreSQL for relational data",
      "similarity": 0.88,
      "final_score": 0.91,
      "source": "notes"
    }
  ],
  "count": 1,
  "elapsed_ms": 3.5
}
```

`final_score` incorporates cosine similarity, Hebbian co-activation boost, and recency weighting. `similarity` is raw cosine.

---

#### `GET /api/stats`

Engine statistics.

```bash
curl http://localhost:11435/api/stats
```

Response:

```json
{
  "total_memories": 1024,
  "max_capacity": 1000000,
  "index_size_bytes": 1572864,
  "ram_usage_bytes": 4194304,
  "disk_usage_bytes": 2097152,
  "max_disk_bytes": 2147483648,
  "avg_echo_latency_ms": 3.5,
  "total_echo_queries": 500,
  "text_count": 1020,
  "vision_count": 4,
  "speech_count": 0
}
```

---

#### `GET /api/memories`

List all stored memories.

```bash
curl "http://localhost:11435/api/memories?limit=20"
```

Query parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | `50` | Maximum entries to return |

Response:

```json
{
  "memories": [
    {
      "id": "3a7c2f1b-4e8d-4c9a-b1f2-0e3d5a6b7c8d",
      "content": "I prefer PostgreSQL for relational data",
      "source": "notes",
      "echo_count": 3,
      "sensitivity": "Public",
      "category": "Preference"
    }
  ],
  "total": 1024,
  "showing": 20
}
```

---

#### `DELETE /api/memories/:id`

Remove a memory by UUID.

```bash
curl -X DELETE http://localhost:11435/api/memories/3a7c2f1b-4e8d-4c9a-b1f2-0e3d5a6b7c8d
```

Response:

```json
{
  "forgotten": "3a7c2f1b-4e8d-4c9a-b1f2-0e3d5a6b7c8d"
}
```

Returns 400 for an invalid UUID and 404 if the memory does not exist.

---

#### `GET /api/config`

Show current configuration.

```bash
curl http://localhost:11435/api/config
```

Response includes all `EchoConfig` fields as JSON.

---

#### `PUT /api/config`

Set a configuration value. Writes to `~/.shrimpk-kernel/config.toml`.

```bash
curl -X PUT http://localhost:11435/api/config \
  -H "Content-Type: application/json" \
  -d '{"key": "similarity_threshold", "value": "0.20"}'
```

Request body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | string | yes | Config key name |
| `value` | string | yes | Value as string |

Supported keys: `max_memories`, `similarity_threshold`, `max_disk_bytes`, `ollama_url`, `enrichment_model`, `consolidation_provider`, `max_facts_per_memory`, `proxy_target`, `proxy_enabled`, `proxy_max_echo_results`

Response:

```json
{
  "set": "similarity_threshold",
  "value": "0.20",
  "note": "Restart daemon for changes to take effect"
}
```

---

#### `POST /api/persist`

Force save all in-memory data to disk.

```bash
curl -X POST http://localhost:11435/api/persist
```

Response:

```json
{
  "persisted": true,
  "disk_usage_bytes": 2097152,
  "max_disk_bytes": 2147483648
}
```

---

#### `POST /api/consolidate`

Trigger memory consolidation immediately (normally runs every 5 minutes in the background). Consolidation prunes Hebbian edges, rebuilds the Bloom filter, merges duplicates, and runs LLM-based fact extraction if Ollama is configured.

```bash
curl -X POST http://localhost:11435/api/consolidate
```

Response:

```json
{
  "hebbian_edges_pruned": 12,
  "bloom_rebuilt": true,
  "duplicates_merged": 0,
  "echo_counts_decayed": 300,
  "facts_extracted": 5,
  "duration_ms": 142
}
```

---

#### `GET /api/detect`

Scan for running local LLM providers and update the internal routing table.

```bash
curl http://localhost:11435/api/detect
```

Response:

```json
{
  "providers": [
    {
      "name": "Ollama",
      "url": "http://127.0.0.1:11434",
      "port": 11434,
      "models": ["llama3.2:3b", "nomic-embed-text"],
      "is_running": true
    }
  ],
  "total_providers": 1,
  "total_models": 2
}
```

---

#### `POST /api/store_image` (vision feature required)

Store an image as a visual memory.

```bash
# Encode the image to base64 first
IMAGE_B64=$(base64 -w 0 photo.jpg)

curl -X POST http://localhost:11435/api/store_image \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"${IMAGE_B64}\", \"source\": \"camera\"}"
```

Request body:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image_base64` | string | yes | — | Base64-encoded image bytes |
| `source` | string | no | `"daemon"` | Source label |

Response:

```json
{
  "memory_id": "7f3e9b2a-1c4d-4a5e-b6f7-8d9e0a1b2c3d"
}
```

---

#### `POST /api/store_audio` (speech feature required)

Store audio as a speech memory using paralinguistic embeddings.

```bash
# Convert to raw PCM f32 LE first (e.g., with ffmpeg):
# ffmpeg -i input.mp3 -f f32le -ar 16000 -ac 1 output.pcm
AUDIO_B64=$(base64 -w 0 output.pcm)

curl -X POST http://localhost:11435/api/store_audio \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"${AUDIO_B64}\", \"sample_rate\": 16000}"
```

Request body:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio_base64` | string | yes | — | Base64-encoded raw PCM f32 little-endian bytes |
| `sample_rate` | integer | no | `16000` | Sample rate in Hz |
| `source` | string | no | `"daemon"` | Source label |

Audio length must be a multiple of 4 (each f32 sample is 4 bytes).

Response:

```json
{
  "memory_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

### OpenAI-Compatible Proxy

The daemon also exposes OpenAI-compatible endpoints that inject relevant memories into requests transparently:

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/v1/chat/completions` | Proxied chat with memory injection |
| `GET` | `/v1/models` | Model list from the proxied provider |

Point any OpenAI SDK at `http://localhost:11435` to get automatic memory injection. See [Memory Proxy](#9-memory-proxy) for setup.

---

## 4. CLI

The CLI loads the engine in-process by default. When a daemon is running on `localhost:11435`, commands proxy to it instead (sub-millisecond overhead).

### Commands

#### `store`

```bash
shrimpk store "Text to remember"
shrimpk store "Text to remember" --source my-app
shrimpk store "Text to remember" --quiet   # suppress output (for scripts)
```

#### `echo`

```bash
shrimpk echo "What framework did I choose?"
shrimpk echo "What framework did I choose?" --max-results 5
shrimpk echo "What framework did I choose?" --json          # machine-readable output
shrimpk echo "red car" --modality vision                   # vision-channel only
shrimpk echo "morning context" --modality auto             # all channels
```

#### `store-image` (vision feature required)

```bash
shrimpk store-image photo.jpg
shrimpk store-image screenshot.png --source screenshots
```

#### `store-audio` (speech feature required)

```bash
shrimpk store-audio recording.pcm --sample-rate 16000
shrimpk store-audio voice-note.pcm --source voice-notes
```

#### `stats`

```bash
shrimpk stats
# [shrimpk] Stats:
#   Memories:          1,024
#   Max capacity:      1,000,000
#   Index size:        1.5 MB
#   RAM usage:         4.0 MB
#   Disk usage:        2.0 MB / 2.0 GB
#   Config tier:       full (32GB detected)
#   Quantization:      f32
#   Embedding dim:     384
#   Threshold:         0.14
#   Echo queries:      500
#   Avg echo latency:  3.5ms
```

#### `status`

```bash
shrimpk status
# System Status:
#   Config tier:   full (32GB RAM detected)
#   Data dir:      /home/user/.shrimpk-kernel
#   Disk usage:    2.0 MB / 2.0 GB (0%) [------------------------------]
#   RAM budget:    1.8 GB
#   Quantization:  f32
#   Max memories:  1,000,000
```

#### `forget`

```bash
shrimpk forget 3a7c2f1b-4e8d-4c9a-b1f2-0e3d5a6b7c8d
```

#### `dump`

```bash
shrimpk dump
# [shrimpk] Stored memories (1,024):
#   3a7c2f1b "I prefer PostgreSQL for relational data" (source: notes, echoed: 3x, sensitivity: Public)
#   ...
```

#### `config`

```bash
shrimpk config show                                  # show all values with sources
shrimpk config set max_memories 500000               # write to config.toml
shrimpk config set similarity_threshold 0.20
shrimpk config reset                                 # reset to auto-detect defaults
shrimpk config path                                  # print config file location
```

#### `detect`

```bash
shrimpk detect
# Detected providers:
#   Ollama  http://127.0.0.1:11434  models: [llama3.2:3b]
```

---

## 5. Rust Library

### Adding to Cargo.toml

```toml
[dependencies]
shrimpk-memory = "0.5.0"
shrimpk-core = "0.5.0"
tokio = { version = "1", features = ["full"] }
```

For vision support:

```toml
[dependencies]
shrimpk-memory = { version = "0.5.0", features = ["vision"] }
```

### Basic Usage

```rust
use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Auto-detect configuration based on available RAM
    let config = EchoConfig::auto_detect();

    // Load engine (downloads embedding model on first run, ~50 MB)
    let engine = EchoEngine::load(config)?;

    // Store memories
    engine.store("I prefer Rust for systems programming", "conversation").await?;
    engine.store("PostgreSQL is my database of choice", "conversation").await?;
    engine.store("I use VS Code as my primary editor", "settings").await?;

    // Persist to disk
    engine.persist().await?;

    // Echo: find memories that resonate with a query
    let results = engine.echo("What language for the backend?", 5).await?;

    for r in &results {
        println!("[{:.2}] {} (source: {})", r.similarity, r.content, r.source);
    }
    // [0.85] I prefer Rust for systems programming (source: conversation)

    // Proper shutdown (avoids Tokio runtime nesting on drop)
    engine.shutdown().await;

    Ok(())
}
```

### Loading Persisted Memories

`EchoEngine::load` restores memories from the previous session. `EchoEngine::new` creates an empty engine without loading persisted data.

```rust
// Loads persisted memories from ~/.shrimpk-kernel/echo_store.json
let engine = EchoEngine::load(config)?;

// Creates empty engine (no restore from disk)
let engine = EchoEngine::new(config)?;
```

### Shared Engine (Arc)

The engine is `Send + Sync`. Use `Arc` to share across tasks:

```rust
use std::sync::Arc;
use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;

let config = EchoConfig::auto_detect();
let engine = Arc::new(EchoEngine::load(config)?);

// Clone the Arc for each task
let engine_clone = engine.clone();
tokio::spawn(async move {
    engine_clone.store("background task memory", "task").await.unwrap();
});

// When done with the Arc
engine.shutdown_arc().await;
```

### Configuration Options

```rust
use shrimpk_core::{EchoConfig, QuantizationMode};

// Use a preset tier
let config = EchoConfig::minimal();    // 8 GB RAM: 100K memories, binary quantization
let config = EchoConfig::standard();   // 16 GB RAM: 500K memories
let config = EchoConfig::full();       // 32 GB RAM: 1M memories (default)
let config = EchoConfig::maximum();    // 64+ GB RAM: 5M memories

// Or configure manually
let config = EchoConfig {
    max_memories: 250_000,
    similarity_threshold: 0.20,    // 0.0–1.0; lower = more results, higher = more precise
    max_echo_results: 10,
    ram_budget_bytes: 500_000_000, // 500 MB
    quantization: QuantizationMode::F16,
    use_lsh: true,                 // sub-linear retrieval
    use_bloom: true,               // O(1) pre-screening
    ollama_url: "http://127.0.0.1:11434".to_string(),
    enrichment_model: "llama3.2:3b".to_string(),
    ..EchoConfig::default()
};
```

### Working with Results

```rust
let results = engine.echo("database choice", 10).await?;

for result in &results {
    println!(
        "rank score={:.3} cosine={:.3} id={} content={} source={}",
        result.final_score,   // combined score (cosine + Hebbian + recency)
        result.similarity,    // raw cosine similarity
        result.memory_id,
        result.content,
        result.source
    );
}
```

### Forget a Memory

```rust
use shrimpk_core::MemoryId;

// memory_id comes from the store() return value or echo results
let id: MemoryId = engine.store("text", "source").await?;
engine.forget(id).await?;
engine.persist().await?;
```

### Feature Flags

| Feature | Default | What It Adds |
|---------|---------|--------------|
| `vision` | off | CLIP ViT-B-32 vision channel, `store_image()`, cross-modal echo |
| `speech` | off | Speech embedding architecture (model wiring in a future release) |

```toml
# Cargo.toml
shrimpk-memory = { version = "0.5.0", features = ["vision"] }
shrimpk-memory = { version = "0.5.0", features = ["vision", "speech"] }
```

---

## 6. Python Bindings

### Installation

```bash
pip install shrimpk
```

Or build from source (requires a Rust toolchain and `maturin`):

```bash
pip install maturin
cd crates/shrimpk-python
maturin develop --release
```

### Basic Usage

```python
from shrimpk import EchoMemory, PyEchoConfig

# Auto-detect configuration based on system RAM
config = PyEchoConfig.auto_detect()
mem = EchoMemory(config)

# Store memories
mem.store("I prefer FastAPI for REST APIs")
mem.store("PostgreSQL for the database", source="conversation")
mem.store("VS Code with the Rust Analyzer extension", source="tools")

# Persist to disk
mem.persist()

# Echo: find resonating memories
results = mem.echo("What framework for the API?", max_results=5)

for r in results:
    print(f"[{r.final_score:.3f}] {r.content} (source: {r.source})")
# [0.912] I prefer FastAPI for REST APIs (source: python)
```

### Configuration Presets

```python
from shrimpk import PyEchoConfig

config = PyEchoConfig.minimal()    # 8 GB RAM
config = PyEchoConfig.standard()   # 16 GB RAM
config = PyEchoConfig.full()       # 32 GB RAM
config = PyEchoConfig.auto_detect() # auto-select based on available RAM
```

### Available Methods

```python
mem = EchoMemory(config)         # Initialize (loads embedding model)

# Store
memory_id = mem.store(text, source=None)   # returns UUID string

# Retrieve
results = mem.echo(query, max_results=None) # returns list[PyEchoResult]

# Remove
mem.forget(memory_id)                        # memory_id is a UUID string

# Persist
mem.persist()

# Stats
stats = mem.stats()
print(stats.total_memories)
print(stats.avg_echo_latency_ms)
print(stats.total_echo_queries)
```

### Result Fields

Each `PyEchoResult` has:

```python
result.memory_id    # str — UUID
result.content      # str — stored text
result.similarity   # float — raw cosine similarity (0.0–1.0)
result.final_score  # float — combined score (cosine + Hebbian + recency)
result.source       # str — source label
```

---

## 7. Configuration

Configuration is resolved in priority order: **environment variables** > **config file** > **auto-detect defaults**.

### Config File Location

```
~/.shrimpk-kernel/config.toml
```

### Example config.toml

```toml
# How many memories to keep. Default: 1,000,000.
max_memories = 500000

# Cosine similarity threshold for echo activation.
# Lower = more results (recall). Higher = fewer, more precise results (precision).
# Default: 0.14. Recommended range: 0.10–0.25.
similarity_threshold = 0.16

# Maximum results per echo query. Default: 20.
max_echo_results = 10

# RAM budget for the in-memory index. Default: 1.8 GB.
ram_budget_bytes = 900000000

# Maximum disk usage for the data directory. Default: 2 GB.
max_disk_bytes = 2147483648

# Quantization mode: "f32", "f16", "int8", "binary"
# f32 = best quality, most RAM. binary = 32x less RAM, ~5% quality loss.
# Default: "f32" (auto-selects "binary" on minimal tier)
quantization = "f32"

# Data directory for persistence. Default: ~/.shrimpk-kernel/
data_dir = "/custom/path/to/data"

# Enable LSH for sub-linear retrieval. Default: true.
use_lsh = true

# Enable Bloom filter pre-screening. Default: true.
use_bloom = true

# Ollama configuration (for consolidation and HyDE query expansion)
ollama_url = "http://127.0.0.1:11434"
enrichment_model = "llama3.2:3b"
max_facts_per_memory = 5
consolidation_provider = "ollama"   # "ollama", "http", or "none"

# Memory proxy (OpenAI-compatible)
proxy_target = "http://127.0.0.1:11434"
proxy_enabled = true
proxy_max_echo_results = 5

# Enable HyDE query expansion (improves recall, adds ~100–500ms latency)
query_expansion_enabled = false

# Reranker backend: "none", "cross_encoder" (5–15ms), "llm" (2s)
# cross_encoder uses a local fastembed ONNX model — recommended when enabled.
reranker_backend = "none"

# Multimodal modalities. Default: ["text"].
# Add "vision" to enable CLIP image memory.
enabled_modalities = ["text"]
```

### Environment Variables

Every config key has a corresponding environment variable with the `SHRIMPK_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SHRIMPK_MAX_MEMORIES` | `1000000` | Maximum memory count |
| `SHRIMPK_SIMILARITY_THRESHOLD` | `0.14` | Echo activation threshold |
| `SHRIMPK_MAX_DISK_BYTES` | `2147483648` | Disk usage limit (bytes) |
| `SHRIMPK_RAM_BUDGET_BYTES` | `1800000000` | RAM budget (bytes) |
| `SHRIMPK_QUANTIZATION` | `f32` | Quantization mode |
| `SHRIMPK_DATA_DIR` | `~/.shrimpk-kernel` | Data directory |
| `SHRIMPK_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base URL |
| `SHRIMPK_ENRICHMENT_MODEL` | `llama3.2:3b` | Ollama model for fact extraction |
| `SHRIMPK_CONSOLIDATION_PROVIDER` | `ollama` | Consolidation backend |
| `SHRIMPK_AUTH_TOKEN` | unset | Bearer token for daemon API |

### Auto-Detect Tiers

ShrimPK selects a default config tier based on detected RAM:

| RAM | Tier | Max Memories | Quantization | RAM Budget |
|-----|------|-------------|--------------|------------|
| < 8 GB | minimal | 100,000 | binary | 50 MB |
| 8–15 GB | standard | 500,000 | f32 | 900 MB |
| 16–31 GB | full | 1,000,000 | f32 | 1.8 GB |
| 32+ GB | maximum | 5,000,000 | f32 | 9 GB |

### Quantization Modes

| Mode | Bytes per Vector | Quality Loss | Use When |
|------|-----------------|--------------|----------|
| `f32` | 384 × 4 = 1,536 B | 0% | Default, best accuracy |
| `f16` | 384 × 2 = 768 B | ~0.1% | Moderate RAM savings |
| `int8` | 384 B | ~1% | Significant RAM savings |
| `binary` | 48 B | ~5% | Extreme RAM constraint; pair with reranker |

---

## 8. Multimodal Features

### Text Channel (Always Available)

The default text channel uses `BGE-small-en-v1.5` (384 dimensions) with LSH retrieval. No additional setup is required.

### Vision Channel

Requires building with `--features vision`. Uses CLIP ViT-B-32 (512 dimensions). Downloads the model (~352 MB) on first use.

#### Build

```bash
cargo build --release --features vision
```

#### Enable in Config

```toml
# ~/.shrimpk-kernel/config.toml
enabled_modalities = ["text", "vision"]
vision_embedding_dim = 512
```

#### Store an Image via CLI

```bash
shrimpk store-image photo.jpg
shrimpk store-image screenshot.png --source work
```

#### Store an Image via HTTP

```bash
IMAGE_B64=$(base64 -w 0 photo.jpg)

curl -X POST http://localhost:11435/api/store_image \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"${IMAGE_B64}\", \"source\": \"camera\"}"
```

#### Cross-Modal Text-to-Image Echo

Once images are stored, query them with text:

```bash
# Find images matching a text description
shrimpk echo "morning coffee" --modality vision

# Search all channels (text memories + image memories, deduped by score)
shrimpk echo "what was on my desk?" --modality auto
```

Via HTTP:

```bash
curl -X POST http://localhost:11435/api/echo \
  -H "Content-Type: application/json" \
  -d '{"query": "morning coffee", "modality": "vision"}'
```

#### Store an Image via Rust

```rust
use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;

// Build with: cargo build --features vision
let config = EchoConfig {
    enabled_modalities: vec![
        shrimpk_core::Modality::Text,
        shrimpk_core::Modality::Vision,
    ],
    ..EchoConfig::auto_detect()
};
let engine = EchoEngine::load(config)?;

let image_bytes = std::fs::read("photo.jpg")?;
let id = engine.store_image(&image_bytes, "photos").await?;

// Text query against the vision channel
let results = engine.echo_with_mode(
    "morning coffee",
    5,
    shrimpk_core::QueryMode::Vision
).await?;
```

### Speech Channel (Architecture Ready)

The speech channel architecture is fully implemented (ECAPA-TDNN 512d + Whisper-tiny encoder 384d = 896d total). Model loading will be wired in a future release. Build with `--features speech` to enable the architecture; `store_audio` will function once models are wired.

The encoding format for audio is raw PCM f32 little-endian. Convert from any audio format using ffmpeg:

```bash
ffmpeg -i input.mp3 -f f32le -ar 16000 -ac 1 output.pcm
```

---

## 9. Memory Proxy

The daemon includes an OpenAI-compatible proxy. Point your existing LLM client at ShrimPK instead of your provider. ShrimPK intercepts each request, runs an echo query against the user's message, injects the top memories into the system prompt, then forwards the enriched request to the real provider.

### Quick Setup

Configure the proxy target to match your LLM provider:

```bash
# Ollama (default — works out of the box)
# shrimpk config set proxy_target http://127.0.0.1:11434

# LM Studio
shrimpk config set proxy_target http://127.0.0.1:1234

# Jan.ai
shrimpk config set proxy_target http://127.0.0.1:1337

# OpenAI
shrimpk config set proxy_target https://api.openai.com
```

Then point your client at `http://localhost:11435` instead of your provider's URL.

### Example: Python OpenAI Client

```python
from openai import OpenAI

# Point at ShrimPK instead of OpenAI
client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="not-needed-for-local"
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "What database did I choose?"}]
)
# ShrimPK injects relevant memories into the system prompt before forwarding.
print(response.choices[0].message.content)
```

### Proxy Configuration

| Config Key | Default | Description |
|------------|---------|-------------|
| `proxy_enabled` | `true` | Enable/disable the proxy |
| `proxy_target` | `http://127.0.0.1:11434` | Upstream LLM provider URL |
| `proxy_max_echo_results` | `5` | How many memories to inject per request |
| `proxy_context_window` | `8000` | Token budget for context injection |
