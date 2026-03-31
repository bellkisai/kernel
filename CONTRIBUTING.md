# Contributing to ShrimPK

Thanks for your interest in contributing to ShrimPK. This guide covers building, testing, code style, and the crate layout.

## Building

```bash
git clone https://github.com/bellkisai/kernel
cd kernel
cargo build --release
```

## Testing

```bash
cargo test --workspace           # Unit tests (fast, no external deps)
cargo test --workspace -- --ignored  # Integration tests (needs fastembed model download)
```

Unit tests run entirely in-memory and complete in seconds. Integration tests download the fastembed all-MiniLM-L6-v2 model on first run (~30 MB), so expect a one-time delay.

## Crate Map

| Crate | Purpose | Status |
|-------|---------|--------|
| `shrimpk-core` | Types, config, traits | Stable |
| `shrimpk-memory` | Echo Memory engine | Stable |
| `shrimpk-router` | Provider routing, cascade, cost | Library (not yet wired to daemon) |
| `shrimpk-context` | Context assembly, token budgeting | In progress (token budgeting for proxy) |
| `shrimpk-security` | Sandbox, permissions | Planned (stub) |
| `shrimpk-kernel` | Integration facade | Stable |
| `shrimpk-python` | PyO3 bindings | Exists (untested in CI) |
| `shrimpk-mcp` | MCP server (9 tools) | Stable |
| `shrimpk-daemon` | HTTP daemon + proxy | Stable |
| `shrimpk-tray` | System tray app | Stable |

## Code Style

- **Edition:** Rust 2024
- **Linting:** `cargo clippy` must be clean (zero warnings)
- **Formatting:** `cargo fmt` before committing
- **Documentation:** All public items documented with `///` comments
- **Tests:** Place tests in `#[cfg(test)]` modules at the bottom of each file

## Architecture

The architecture is documented in the [README](README.md#architecture) and inline doc comments throughout the codebase. Key design decisions are tracked as ADRs in the project documentation.

## Pull Requests

1. Fork the repo and create a feature branch
2. Ensure `cargo test --workspace` passes
3. Ensure `cargo clippy` reports zero warnings
4. Ensure `cargo fmt --check` passes
5. Write tests for new functionality
6. Keep commits focused -- one logical change per commit

## License

By contributing, you agree that your contributions will be licensed under Apache-2.0.
