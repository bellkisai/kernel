# ── Stage 1: Builder ─────────────────────────────────────────────
FROM rust:1.82-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY cli/ cli/
COPY src/ src/
RUN cargo build --release -p shrimpk-cli -p shrimpk-mcp -p shrimpk-daemon

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/shrimpk-daemon /usr/local/bin/
COPY --from=builder /src/target/release/shrimpk-mcp    /usr/local/bin/
COPY --from=builder /src/target/release/shrimpk         /usr/local/bin/

EXPOSE 11435
VOLUME /data
ENV SHRIMPK_DATA_DIR=/data

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:11435/health || exit 1

LABEL org.opencontainers.image.source="https://github.com/bellkisai/kernel"
LABEL org.opencontainers.image.description="ShrimPK AI Memory Daemon"
LABEL org.opencontainers.image.licenses="Apache-2.0"

CMD ["shrimpk-daemon", "--port", "11435"]
