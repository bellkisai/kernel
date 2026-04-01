<p align="center">
  <h1 align="center">🦐 ShrimPK</h1>
  <p align="center"><strong>Push-based AI memory where memories find YOU.</strong></p>
  <p align="center">
    <a href="https://github.com/bellkisai/kernel/actions/workflows/ci.yml"><img src="https://github.com/bellkisai/kernel/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/rust-2024-orange.svg" alt="Rust">
    <img src="https://img.shields.io/badge/port-11435-10b981.svg" alt="Port">
  </p>
</p>

---

<p align="center">
  <code>3.50ms</code> echo at 100K memories&nbsp;&nbsp;•&nbsp;&nbsp;<code>+38%</code> more accurate than plain LLM&nbsp;&nbsp;•&nbsp;&nbsp;<code>15.7%</code> token savings&nbsp;&nbsp;•&nbsp;&nbsp;<code>multimodal</code> text + vision + speech
</p>

---

## The Problem

AI tools forget everything between sessions. You re-explain your stack, preferences, and project context **every single time**. Standard RAG requires you to search. You shouldn't have to.

> *"What framework did I use for the API?"*
>
> Without ShrimPK: *"I don't have access to your previous conversations."*
>
> With ShrimPK: *"You chose FastAPI for your REST API, with SQLAlchemy for type-safe ORM queries."*

## How It Works

ShrimPK inverts the memory paradigm. Instead of searching for memories, **memories find you**.

```
┌─────────────────────────────────────────────────────────┐
│  1. You Converse                                        │
│     Just talk normally. ShrimPK stores context           │
│     automatically. No "remember this" needed.           │
├─────────────────────────────────────────────────────────┤
│  2. Memories Self-Activate                              │
│     When you mention something related, stored           │
│     memories activate through Hebbian associations      │
│     and surface relevant context — automatically.       │
├─────────────────────────────────────────────────────────┤
│  3. AI Knows You                                        │
│     Your name persists for a year. Preferences for      │
│     months. Casual chats fade in days. Just like        │
│     human memory.                                       │
└─────────────────────────────────────────────────────────┘
```

## The Reef Ecosystem

In nature, cleaner shrimp maintain entire reef ecosystems — removing parasites, cleaning wounds, keeping everything healthy. ShrimPK does the same for your AI memory.

| | Role | What It Does |
|---|---|---|
| 🦐 **ShrimPK** | The Shrimp | Maintains your AI memory reef — stores, classifies, decays, consolidates |
| 🪸 **Echo Memory** | The Reef | The associative memory structure — LSH, Bloom filters, Hebbian learning |
| 🦞 **You** | The Lobster | You just talk. Memories come to you. No searching, no managing. |
| 🌊 **Push Activation** | The Current | The autonomous flow that delivers memories without being asked |

## Performance

| Metric | Value |
|--------|-------|
| Echo P50 at 100K memories | **3.50ms** |
| Echo P95 at 100K memories | 6.88ms |
| Head-to-head accuracy | **+38%** vs plain LLM |
| Personalization rate | **100%** |
| Token savings | **15.7%** per request |
| Follow-up elimination | **100%** |
| RAM (1M memories, f32) | ~1.8 GB |
| RAM (1M memories, binary) | ~150 MB |

## Multimodal Memory

ShrimPK v0.5.0 introduces a 3-channel architecture: **text**, **vision**, and **speech**. Each channel has its own embedding model, LSH index, and persistence section -- unified under a single Echo Memory engine.

```
┌─────────────────────────────────────────────────────────┐
│               ShrimPK Echo Memory                       │
├──────────────┬──────────────┬───────────────────────────┤
│  Text (384d) │ Vision (512d)│ Speech (896d)             │
│  BGE-small   │ CLIP ViT-B-32│ ECAPA-TDNN+Whisper-tiny  │
│  LSH 16×10   │ LSH 16×10   │ LSH 16×10                │
├──────────────┴──────────────┴───────────────────────────┤
│  Cross-Modal Retrieval: text query → image result       │
│  Auto-Mode: searches all channels, deduplicates         │
└─────────────────────────────────────────────────────────┘
```

### Cross-Modal Retrieval

Store an image. Query with text. ShrimPK finds it.

```bash
# Store an image
shrimpk store-image photo.jpg --tag "kitchen morning"

# Query with text — finds the image
shrimpk echo --modality vision "where's the cup?"
# → photo.jpg (similarity: 0.82) — CLIP matched "cup" to image content

# Auto mode searches all channels
shrimpk echo "what did I see this morning?"
# → text memories + image memories, deduplicated by score
```

### Enabling Modalities

Vision and speech are compile-time feature flags. Vision is enabled by default; speech is architecture-ready (models wired in a future release).

```toml
# ~/.shrimpk-kernel/config.toml
enabled_modalities = ["text", "vision"]
vision_embedding_dim = 512
speech_embedding_dim = 896
```

```bash
# Build with vision support (default)
cargo build --release --features vision

# Build with all modalities
cargo build --release --features "vision,speech"
```

### CLI

```bash
shrimpk store-image photo.jpg            # Store image via CLIP
shrimpk store-image screenshot.png --tag "bug report"
shrimpk echo --modality vision "red car"  # Vision-only search
shrimpk echo --modality auto "morning"    # Search all channels
shrimpk stats                             # Shows text_count, vision_count, speech_count
```

### API

```bash
# Store image via daemon
curl -X POST localhost:11435/api/store_image \
  -F "file=@photo.jpg" -F "tag=kitchen"

# Echo with modality
curl -X POST localhost:11435/api/echo \
  -H "Content-Type: application/json" \
  -d '{"query":"where is the cup?","modality":"vision"}'
```

### Speech Channel (KS50)

The speech channel combines two permissive-license models into a 896-dimensional embedding:
- **ECAPA-TDNN** (512d, Apache-2.0) — speaker identity
- **Whisper-tiny encoder** (384d, MIT) — prosody / rhythm / pace

Both models auto-download from HuggingFace Hub (~58 MB total) on first use and are cached locally. When enabled, `shrimpk store-audio recording.wav` works the same as image storage.

### Scaling

The multimodal engine scales from Raspberry Pi to data center. Per-channel LSH indices keep retrieval sub-linear regardless of memory count. Vision adds ~512 bytes per stored image embedding; speech adds ~896 bytes. RAM auto-detection adjusts budgets per channel.

## The Ollama Model

ShrimPK runs as a background daemon — just like Ollama. Install once, it serves every AI tool on your machine.

```
shrimpk-daemon                    ← runs on localhost:11435
  Model loads ONCE (~3s)          ← then serves forever
  Auto-consolidation              ← every 5 min
  Any client connects via HTTP    ← CLI, MCP, hooks, your app

No cold starts. No process spawning. Sub-5ms responses.
```

```bash
# Start the daemon
shrimpk-daemon

# Store a memory (via HTTP — instant)
curl -X POST localhost:11435/api/store \
  -H "Content-Type: application/json" \
  -d '{"text":"I prefer Rust for backend services","source":"cli"}'

# Echo memories (via HTTP — 3.5ms)
curl -X POST localhost:11435/api/echo \
  -H "Content-Type: application/json" \
  -d '{"query":"What language for the backend?"}'
```

## Quick Start

### Install

```bash
# Build from source
cargo build --release -p shrimpk-cli -p shrimpk-daemon

# Start daemon (runs in background)
./target/release/shrimpk-daemon

# Auto-start on login
./target/release/shrimpk-daemon --install
```

### CLI

```bash
shrimpk store "I prefer FastAPI for REST APIs"
shrimpk echo "What framework for APIs?"
shrimpk stats
shrimpk config show
shrimpk status
```

When the daemon is running, CLI commands are instant (~1ms) — they proxy to the daemon via HTTP. Without the daemon, CLI loads the engine in-process (slower but works anywhere).

### Rust Library

```rust
use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;

let config = EchoConfig::auto_detect();
let engine = EchoEngine::load(config)?;

engine.store("I prefer FastAPI for REST APIs", "conversation").await?;

let echoes = engine.echo("What framework for this API?", 5).await?;
// Returns: FastAPI memory (similarity: 0.85) in ~3.5ms
```

### Python

```python
from shrimpk import EchoMemory, EchoConfig

config = EchoConfig.auto_detect()
mem = EchoMemory(config)
mem.store("I prefer FastAPI", source="conversation")
results = mem.echo("What framework?", max_results=5)
```

### MCP Server (Claude Code)

```bash
# Register globally — works from any directory
claude mcp add --transport stdio --scope user shrimpk -- shrimpk-mcp
```

The MCP server auto-detects the daemon and proxies via HTTP. Falls back to in-process if daemon isn't running.

## HTTP API

The daemon exposes 10 REST endpoints on `localhost:11435`:

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/health` | Health check + memory count + uptime |
| `POST` | `/api/store` | Store a memory |
| `POST` | `/api/echo` | Find resonating memories |
| `GET` | `/api/stats` | Engine statistics |
| `GET` | `/api/memories` | List all memories |
| `DELETE` | `/api/memories/:id` | Forget a memory |
| `GET` | `/api/config` | Show configuration |
| `PUT` | `/api/config` | Set config value |
| `POST` | `/api/persist` | Force save to disk |
| `POST` | `/api/consolidate` | Trigger consolidation |

Optional auth: set `SHRIMPK_AUTH_TOKEN` env var → required as `Authorization: Bearer` header.

## Under the Hood

```
shrimpk-kernel          (facade — re-exports all)
  shrimpk-core          (types, config, errors, PII types)
  shrimpk-memory        (Echo Memory engine)
  shrimpk-router        (provider routing + cascade) — library, not yet wired to daemon
  shrimpk-context       (context assembly + token budgeting) — in progress
  shrimpk-security      (sandbox, permissions) — planned
  shrimpk-python        (PyO3 bindings)
  shrimpk-mcp           (MCP server — JSON-RPC stdio)
  shrimpk-daemon        (HTTP daemon — localhost:11435)
```

**Echo Memory pipeline:**
1. **Bloom filter** — O(1) topic rejection (is this query even relevant?)
2. **LSH** — sub-linear candidate retrieval (SimHash, 16 tables, 10 bits)
3. **Cosine similarity** — SIMD-accelerated exact scoring
4. **Hebbian boost** — co-activated memories get promoted
5. **Category decay** — Identity (365d) → Conversation (3d)

## Memory Proxy — Works With Any LLM

ShrimPK sits between your client and your LLM provider. Point your app at `localhost:11435` instead of your provider's port. Memory injection is automatic and invisible.

### Quick Setup

| Provider | Default Port | ShrimPK Config |
|----------|:---:|---|
| Ollama | 11434 | Default — works out of the box |
| LM Studio | 1234 | `shrimpk config set proxy_target http://127.0.0.1:1234` |
| Jan.ai | 1337 | `shrimpk config set proxy_target http://127.0.0.1:1337` |
| vLLM | 8000 | `shrimpk config set proxy_target http://127.0.0.1:8000` |
| LocalAI | 8080 | `shrimpk config set proxy_target http://127.0.0.1:8080` |
| GPT4All | 4891 | `shrimpk config set proxy_target http://127.0.0.1:4891` |
| OpenAI API | — | `shrimpk config set proxy_target https://api.openai.com` |
| xAI (Grok) | — | `shrimpk config set proxy_target https://api.x.ai` |

### How It Works

```
Client Request → ShrimPK (11435)
                    │
              ┌─────┴─────┐
              │ 1. Echo    │ ← find relevant memories (3.50ms)
              │ 2. Inject  │ ← prepend to system prompt
              │ 3. Store   │ ← save user message for future
              │ 4. Forward │ ← send to LLM provider
              └─────┬─────┘
                    │
              Provider (Ollama, LM Studio, etc.)
                    │
              Response → Client (streamed transparently)
```

### Auto-Detection

```bash
$ shrimpk detect

  Provider     Port   Status    Models
  ------------ ------ --------- ------
  Ollama       11434  RUNNING   gemma3:1b, phi4-mini
  LM Studio    1234   NOT FOUND
```

ShrimPK auto-routes requests by model name. If you have Ollama and LM Studio running, a request for `gemma3:1b` goes to Ollama and `mistral-7b` goes to LM Studio.

### Open WebUI

Settings → Connections → Ollama URL → change to `http://localhost:11435`

All conversations now have persistent memory across sessions.

### curl

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"hello"}]}'
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat with memory injection |
| GET | `/v1/models` | List available models from backend |

### Configuration

```bash
shrimpk config set proxy_target http://127.0.0.1:11434  # Backend URL
shrimpk config set proxy_enabled true                     # Enable/disable
shrimpk config set proxy_max_echo_results 5               # Memories per request
```

Or via environment variables:
```bash
SHRIMPK_PROXY_TARGET=http://127.0.0.1:11434
SHRIMPK_PROXY_ENABLED=true
```

## Configuration

ShrimPK auto-detects from system RAM. Override via config file or env vars:

```bash
# Config file (~/.shrimpk-kernel/config.toml)
shrimpk config set max_memories 500000
shrimpk config set quantization binary
shrimpk config show

# Environment variables (highest priority)
SHRIMPK_MAX_MEMORIES=500000
SHRIMPK_DATA_DIR=/custom/path
SHRIMPK_QUANTIZATION=binary
SHRIMPK_PORT=11435
SHRIMPK_AUTH_TOKEN=your-secret-token
```

Priority: env vars > config.toml > auto-detect

## Two Products

| | ShrimPK Kernel | Bellkis HUB |
|---|---|---|
| **What** | AI memory engine | AI desktop app |
| **For** | Developers embedding memory | Users wanting a local AI hub |
| **License** | Apache 2.0 | BSL 1.1 |
| **Install** | `cargo add shrimpk-kernel` | Desktop installer |

Two brands, one household. The kernel powers the hub.

## License

Apache 2.0 — see [LICENSE](LICENSE)
