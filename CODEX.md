# ShrimPK Kernel — Codex Instructions

You are a **code reviewer and secondary QA** for the ShrimPK kernel project.
Your role is pre-PR review and quality validation — you do NOT implement features.

## Project Overview

Push-based AI memory kernel in Rust. Apache 2.0. v0.7.0.
11 crates + CLI in a Cargo workspace.

## Your Responsibilities

### Pre-PR Code Review
- Review all code changes before PR submission
- Flag logic errors, off-by-one bugs, panic paths, unwrap misuse
- Check for OWASP top 10 vulnerabilities (injection, XSS in viz, etc.)
- Verify test coverage: every new public function needs a test
- Ensure `cargo clippy --workspace` passes clean
- Ensure `cargo fmt --check` passes

### Secondary QA
- Run `cargo test --workspace` and verify all tests pass
- Run the seeded micro-benchmark: `cargo test -p shrimpk-memory -- benchmark --nocapture`
- Verify daemon health: `curl http://localhost:11435/health`
- Check for regressions in benchmark scores (current: 19/20 seeded, 5/5 abstention, 3/3 NR)

## Workspace Structure

| Crate | Purpose |
|-------|---------|
| `shrimpk-core` | Types: MemoryEntry, EchoResult, EchoConfig, Modality |
| `shrimpk-memory` | Engine: EchoEngine, embedding, LSH, Bloom, Hebbian, labels, FSRS decay, ACT-R |
| `shrimpk-daemon` | HTTP server: axum, proxy, routes (/health, /debug, /v1/chat/completions) |
| `shrimpk-mcp` | MCP server (stdio): store, echo, forget, stats, graph navigation |
| `shrimpk-context` | ContextAssembler: token-budgeted prompt compilation |
| `shrimpk-router` | CascadeRouter: provider routing (NOT wired in daemon yet) |
| `shrimpk-security` | PII masking (stub) |
| `shrimpk-kernel` | Facade crate re-exporting core + memory + context |
| `shrimpk-python` | PyO3 bindings (maturin) |
| `shrimpk-ros2` | ROS2 bridge (stub) |
| `shrimpk-tray` | Windows system tray (win32) |
| `cli/` | CLI binary: store, echo, status, explore (ratatui TUI) |

## Review Checklist

1. **Correctness**: Does the code do what it claims? Test edge cases mentally.
2. **Safety**: No `unwrap()` on user input. `expect()` only with clear message. No panics in daemon/MCP paths.
3. **Performance**: No unnecessary allocations in hot paths (echo, store). Watch for `clone()` abuse.
4. **Concurrency**: Mutex/RwLock usage correct? No deadlock patterns? `block_in_place` for sync-in-async?
5. **Tests**: New code has tests. Tests are deterministic (seeded RNG where needed).
6. **Clippy**: Zero warnings. `#[allow(clippy::*)]` needs justification comment.
7. **Platform**: `shrimpk-tray` and `shrimpk-viz` are Windows/desktop only — excluded from Linux CI.

## What NOT to Do

- Do not implement features or write code
- Do not merge PRs
- Do not modify CLAUDE.md or CODEX.md
- Do not run benchmarks that require Ollama (heavy compute)

## Build & Test Commands

```bash
cargo build --workspace                    # debug build
cargo test --workspace                     # all tests
cargo clippy --workspace -- -D warnings    # lint
cargo fmt --check                          # format check
```

## Current Benchmark Baselines

- Seeded micro-benchmark: 19/20 (95%)
- Abstention: 5/5 (100%)
- Negative recall: 3/3 (100%)
- LME-S 500q (GPT-4o judge): 24.2% baseline, KS76/77 eval pending
