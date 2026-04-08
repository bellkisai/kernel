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

The speech channel combines two permissive-license models into a 640-dimensional embedding:
- **ECAPA-TDNN** (256d, Apache-2.0) — speaker identity
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

## Install

### One-liner (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/bellkisai/kernel/master/scripts/install-remote.sh | sh
```

Installs pre-built binaries to `~/.shrimpk/bin/`, registers the MCP server, and starts the daemon.

### Docker

```bash
docker run -d --name shrimpk -p 11435:11435 -v shrimpk-data:/data bellkisai/shrimpk
```

### From source

```bash
git clone https://github.com/bellkisai/kernel.git && cd kernel
cargo build --release -p shrimpk-cli -p shrimpk-mcp -p shrimpk-daemon -p shrimpk-tray
bash scripts/install.sh   # or: powershell scripts/install.ps1
```

### GitHub Releases

Download pre-built binaries for your platform from [Releases](https://github.com/bellkisai/kernel/releases). Available for Linux (x86_64, aarch64), macOS (Apple Silicon, Intel), and Windows.

### Verify

```bash
curl http://localhost:11435/health          # daemon running?
shrimpk status                              # system overview
shrimpk store "I prefer Rust"               # store a memory
shrimpk echo "What language do I like?"     # recall it
```

## Proxy — Zero-Config Memory for Any LLM

Point your LLM client at ShrimPK instead of your provider. Every request gets transparent memory injection.

```bash
# Start with smart defaults (expands "ollama" to localhost:11434)
shrimpk-daemon --proxy-to ollama

# Now use localhost:11435 instead of localhost:11434
# Open WebUI, Chatbox, or any OpenAI-compatible client — just change the port
```

The daemon auto-detects local providers (Ollama, LM Studio, vLLM, Jan, LocalAI, GPT4All) and routes by model name. You'll see `Memories injected: N` in the daemon logs for every request.

| Provider | Default Port | Flag |
|----------|-------------|------|
| Ollama | 11434 | `--proxy-to ollama` |
| LM Studio | 1234 | `--proxy-to lmstudio` |
| vLLM | 8000 | `--proxy-to vllm` |
| Jan | 1337 | `--proxy-to jan` |
| LocalAI | 8080 | `--proxy-to localai` |
| GPT4All | 4891 | `--proxy-to gpt4all` |
| Custom | any | `--proxy-to http://host:port` |

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

### Proxy Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat with memory injection |
| GET | `/v1/models` | List available models from backend |

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
