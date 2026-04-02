//! HTTP client that talks to the ShrimPK daemon REST API.
//!
//! Endpoints used:
//!
//! | Method | Path | Body |
//! |--------|------|------|
//! | POST | `/api/store` | `{ "text": "...", "source": "ros2" }` |
//! | POST | `/api/store_image` | `{ "image_base64": "<b64>", "source": "ros2" }` |
//! | POST | `/api/store_audio` | `{ "audio_base64": "<b64-f32-le>", "sample_rate": 16000, "source": "ros2" }` |
//! | GET  | `/health` | — |

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde_json::json;
use tracing::{debug, warn};

const SOURCE: &str = "ros2";

/// HTTP client for the ShrimPK daemon.
///
/// All store methods are async and require a running Tokio runtime.
pub struct DaemonClient {
    /// Base URL, e.g. `http://localhost:11435`.
    url: String,
    /// Optional bearer token (unused until daemon auth ships).
    token: Option<String>,
    http: reqwest::Client,
}

impl DaemonClient {
    /// Create a new client. `url` must not have a trailing slash.
    pub fn new(url: impl Into<String>, token: Option<String>) -> Self {
        let url = url.into().trim_end_matches('/').to_owned();
        Self {
            url,
            token,
            http: reqwest::Client::new(),
        }
    }

    /// Build a POST request, attaching the bearer token when configured.
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let rb = self.http.post(format!("{}{}", self.url, path));
        if let Some(token) = &self.token {
            rb.bearer_auth(token)
        } else {
            rb
        }
    }

    /// Build a GET request, attaching the bearer token when configured.
    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        let rb = self.http.get(format!("{}{}", self.url, path));
        if let Some(token) = &self.token {
            rb.bearer_auth(token)
        } else {
            rb
        }
    }

    /// Store a text string as an Echo Memory.
    ///
    /// Calls `POST /api/store { "text": ..., "source": "ros2" }`.
    pub async fn store_text(&self, text: &str, label_hint: Option<&str>) -> Result<()> {
        let effective = match label_hint {
            Some(label) => format!("[{label}] {text}"),
            None => text.to_owned(),
        };
        debug!(len = effective.len(), "store_text → daemon");

        let resp = self
            .post("/api/store")
            .json(&json!({ "text": effective, "source": SOURCE }))
            .send()
            .await
            .context("POST /api/store failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("store_text: daemon returned {status}: {body}");
        }
        Ok(())
    }

    /// Store PNG image bytes as a vision memory.
    ///
    /// The daemon expects base64-encoded PNG bytes at
    /// `POST /api/store_image { "image_base64": ... }`.
    /// If the daemon was not compiled with `--features vision` the endpoint
    /// returns 404 and we log a warning rather than crashing.
    pub async fn store_image(&self, png_bytes: &[u8], label_hint: Option<&str>) -> Result<()> {
        let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
        let source = match label_hint {
            Some(l) => format!("ros2:{l}"),
            None => SOURCE.to_owned(),
        };
        debug!(png_len = png_bytes.len(), "store_image → daemon");

        let resp = self
            .post("/api/store_image")
            .json(&json!({ "image_base64": b64, "source": source }))
            .send()
            .await
            .context("POST /api/store_image failed")?;

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            warn!("store_image: daemon vision feature not compiled — skipping");
            return Ok(());
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("store_image: daemon returned {status}: {body}");
        }
        Ok(())
    }

    /// Store PCM f32 audio samples as a speech memory.
    ///
    /// The daemon expects base64-encoded little-endian f32 bytes at
    /// `POST /api/store_audio { "audio_base64": ..., "sample_rate": ... }`.
    /// If the daemon was not compiled with `--features speech` the endpoint
    /// returns 404 and we log a warning rather than crashing.
    pub async fn store_audio(
        &self,
        pcm_f32: &[f32],
        sample_rate: u32,
        label_hint: Option<&str>,
    ) -> Result<()> {
        // Serialize samples as little-endian f32 bytes, then base64-encode.
        let mut raw: Vec<u8> = Vec::with_capacity(pcm_f32.len() * 4);
        for &s in pcm_f32 {
            raw.extend_from_slice(&s.to_le_bytes());
        }
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
        let source = match label_hint {
            Some(l) => format!("ros2:{l}"),
            None => SOURCE.to_owned(),
        };
        debug!(samples = pcm_f32.len(), sample_rate, "store_audio → daemon");

        let resp = self
            .post("/api/store_audio")
            .json(&json!({
                "audio_base64": b64,
                "sample_rate": sample_rate,
                "source": source
            }))
            .send()
            .await
            .context("POST /api/store_audio failed")?;

        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            warn!("store_audio: daemon speech feature not compiled — skipping");
            return Ok(());
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("store_audio: daemon returned {status}: {body}");
        }
        Ok(())
    }

    /// Ping the daemon health endpoint.
    ///
    /// Returns `Ok(true)` if the daemon responds with `200 OK`.
    pub async fn health_check(&self) -> Result<bool> {
        let resp = self
            .get("/health")
            .send()
            .await
            .context("GET /health failed")?;
        Ok(resp.status().is_success())
    }
}
