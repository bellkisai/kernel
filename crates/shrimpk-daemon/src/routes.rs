//! HTTP route handlers for the ShrimPK daemon.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;
use serde_json::{Value, json};
use shrimpk_core::{MemoryId, QueryMode, config};
use std::time::Instant;

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct StoreRequest {
    pub text: String,
    #[serde(default = "default_source")]
    pub source: String,
}

fn default_source() -> String {
    "daemon".into()
}

#[derive(Deserialize)]
pub struct EchoRequest {
    pub query: String,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    /// Query mode: "text" (default), "vision", or "auto".
    #[serde(default)]
    pub modality: Option<String>,
}

#[cfg(feature = "vision")]
#[derive(Deserialize)]
pub struct StoreImageRequest {
    pub image_base64: String,
    #[serde(default = "default_source")]
    pub source: String,
}

#[cfg(feature = "speech")]
#[derive(Deserialize)]
pub struct StoreAudioRequest {
    pub audio_base64: String,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_source")]
    pub source: String,
    /// Optional text description for cross-modal text→speech recall.
    pub description: Option<String>,
}

#[cfg(feature = "speech")]
fn default_sample_rate() -> u32 {
    16000
}

fn default_max_results() -> usize {
    10
}

#[derive(Deserialize)]
pub struct ConfigSetRequest {
    pub key: String,
    pub value: String,
}

#[derive(Deserialize)]
pub struct MemoriesQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// GET /health
pub async fn health(State(state): State<AppState>) -> Json<Value> {
    let stats = state.engine.stats().await;
    let uptime = state.started_at.elapsed().as_secs();
    Json(json!({
        "status": "ok",
        "memories": stats.total_memories,
        "uptime_secs": uptime,
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// POST /api/store
pub async fn store(
    State(state): State<AppState>,
    Json(req): Json<StoreRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let pii_matches = state.pii_filter.scan(&req.text);
    let sensitivity = state.pii_filter.classify(&req.text);

    let id = state
        .engine
        .store(&req.text, &req.source)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?;

    state.engine.persist().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(json!({
        "memory_id": id.to_string(),
        "sensitivity": format!("{sensitivity:?}"),
        "pii_matches": pii_matches.len()
    })))
}

/// POST /api/echo
pub async fn echo(
    State(state): State<AppState>,
    Json(req): Json<EchoRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mode = match req.modality.as_deref().unwrap_or("text") {
        "vision" => QueryMode::Vision,
        "auto" => QueryMode::Auto,
        _ => QueryMode::Text,
    };

    let start = Instant::now();
    let results = state
        .engine
        .echo_with_mode(&req.query, req.max_results, mode)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let results_json: Vec<Value> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            json!({
                "rank": i + 1,
                "memory_id": r.memory_id.to_string(),
                "content": r.content,
                "similarity": (r.similarity * 100.0).round() / 100.0,
                "final_score": (r.final_score * 100.0).round() / 100.0,
                "source": r.source
            })
        })
        .collect();

    Ok(Json(json!({
        "results": results_json,
        "count": results.len(),
        "elapsed_ms": (elapsed_ms * 10.0).round() / 10.0
    })))
}

/// GET /api/stats
pub async fn stats(State(state): State<AppState>) -> Json<Value> {
    let s = state.engine.stats().await;
    Json(json!({
        "total_memories": s.total_memories,
        "max_capacity": s.max_capacity,
        "index_size_bytes": s.index_size_bytes,
        "ram_usage_bytes": s.ram_usage_bytes,
        "disk_usage_bytes": s.disk_usage_bytes,
        "max_disk_bytes": s.max_disk_bytes,
        "avg_echo_latency_ms": s.avg_echo_latency_ms,
        "total_echo_queries": s.total_echo_queries,
        "text_count": s.text_count,
        "vision_count": s.vision_count,
        "speech_count": s.speech_count
    }))
}

/// GET /api/memories?limit=50
pub async fn list_memories(
    State(state): State<AppState>,
    Query(params): Query<MemoriesQuery>,
) -> Json<Value> {
    let entries = state.engine.all_entry_summaries().await;
    let limited: Vec<Value> = entries
        .iter()
        .take(params.limit)
        .map(|e| {
            json!({
                "id": e.id.to_string(),
                "content": e.content,
                "source": e.source,
                "echo_count": e.echo_count,
                "sensitivity": format!("{:?}", e.sensitivity),
                "category": format!("{:?}", e.category)
            })
        })
        .collect();

    Json(json!({
        "memories": limited,
        "total": entries.len(),
        "showing": limited.len()
    }))
}

/// DELETE /api/memories/:id
pub async fn forget(
    State(state): State<AppState>,
    Path(id_str): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let uuid = uuid::Uuid::parse_str(&id_str).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("Invalid UUID: {e}")})),
        )
    })?;

    state
        .engine
        .forget(MemoryId::from_uuid(uuid))
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, Json(json!({"error": e.to_string()}))))?;

    state.engine.persist().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(json!({"forgotten": id_str})))
}

/// GET /api/config
pub async fn config_show(State(state): State<AppState>) -> Json<Value> {
    let c = &state.config;
    Json(json!({
        "max_memories": c.max_memories,
        "similarity_threshold": c.similarity_threshold,
        "max_echo_results": c.max_echo_results,
        "ram_budget_bytes": c.ram_budget_bytes,
        "data_dir": c.data_dir.to_string_lossy(),
        "quantization": c.quantization.to_string(),
        "embedding_dim": c.embedding_dim,
        "use_lsh": c.use_lsh,
        "use_bloom": c.use_bloom,
        "max_disk_bytes": c.max_disk_bytes,
        "ollama_url": c.ollama_url,
        "enrichment_model": c.enrichment_model,
        "max_facts_per_memory": c.max_facts_per_memory,
        "consolidation_provider": c.consolidation_provider,
        "proxy_target": c.proxy_target,
        "proxy_enabled": c.proxy_enabled,
        "proxy_max_echo_results": c.proxy_max_echo_results
    }))
}

/// PUT /api/config
pub async fn config_set(
    Json(req): Json<ConfigSetRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut fc = config::load_config_file()
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?
        .unwrap_or_default();

    match req.key.as_str() {
        "max_memories" => {
            fc.max_memories = Some(req.value.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid integer"})),
                )
            })?)
        }
        "similarity_threshold" => {
            fc.similarity_threshold = Some(req.value.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid float"})),
                )
            })?)
        }
        "max_disk_bytes" => {
            fc.max_disk_bytes = Some(req.value.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid integer"})),
                )
            })?)
        }
        "ollama_url" => fc.ollama_url = Some(req.value.clone()),
        "enrichment_model" => fc.enrichment_model = Some(req.value.clone()),
        "consolidation_provider" => fc.consolidation_provider = Some(req.value.clone()),
        "max_facts_per_memory" => {
            fc.max_facts_per_memory = Some(req.value.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid integer"})),
                )
            })?)
        }
        "proxy_target" => fc.proxy_target = Some(req.value.clone()),
        "proxy_enabled" => {
            fc.proxy_enabled = Some(req.value.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid boolean"})),
                )
            })?)
        }
        "proxy_max_echo_results" => {
            fc.proxy_max_echo_results = Some(req.value.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Invalid integer"})),
                )
            })?)
        }
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Unknown config key: {other}")})),
            ));
        }
    }

    config::save_config_file(&fc).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(json!({
        "set": req.key,
        "value": req.value,
        "note": "Restart daemon for changes to take effect"
    })))
}

/// POST /api/persist
pub async fn persist(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    state.engine.persist().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    let disk_usage = config::disk_usage(&state.config.data_dir).unwrap_or(0);
    Ok(Json(json!({
        "persisted": true,
        "disk_usage_bytes": disk_usage,
        "max_disk_bytes": state.config.max_disk_bytes
    })))
}

/// GET /api/detect — re-scan for local LLM providers and update routing table.
pub async fn detect_providers(
    State(state): State<AppState>,
) -> Json<Value> {
    let providers = crate::detect::detect_providers(&state.http_client).await;
    let new_routes = crate::detect::build_model_routes(&providers);
    let total_models: usize = providers.iter().map(|p| p.models.len()).sum();

    // Update the shared routing table
    {
        let mut routes = state.model_routes.write().await;
        *routes = new_routes;
    }

    let providers_json: Vec<Value> = providers
        .iter()
        .map(|p| {
            json!({
                "name": p.name,
                "url": p.url,
                "port": p.port,
                "models": p.models,
                "is_running": p.is_running
            })
        })
        .collect();

    Json(json!({
        "providers": providers_json,
        "total_providers": providers.len(),
        "total_models": total_models
    }))
}

/// POST /api/consolidate
pub async fn consolidate(State(state): State<AppState>) -> Json<Value> {
    let result = state.engine.consolidate_now().await;
    Json(json!({
        "hebbian_edges_pruned": result.hebbian_edges_pruned,
        "bloom_rebuilt": result.bloom_rebuilt,
        "duplicates_merged": result.duplicates_merged,
        "echo_counts_decayed": result.echo_counts_decayed,
        "facts_extracted": result.facts_extracted,
        "duration_ms": result.duration_ms
    }))
}

/// POST /api/store_image — store an image as a vision memory.
#[cfg(feature = "vision")]
pub async fn store_image(
    State(state): State<AppState>,
    Json(req): Json<StoreImageRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    use base64::Engine as _;

    let image_bytes = base64::engine::general_purpose::STANDARD
        .decode(&req.image_base64)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Invalid base64: {e}")})),
            )
        })?;

    let id = state
        .engine
        .store_image(&image_bytes, &req.source)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?;

    state.engine.persist().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(json!({
        "memory_id": id.to_string()
    })))
}

/// POST /api/store_audio — store audio as a speech memory.
#[cfg(feature = "speech")]
pub async fn store_audio(
    State(state): State<AppState>,
    Json(req): Json<StoreAudioRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    use base64::Engine as _;

    let raw_bytes = base64::engine::general_purpose::STANDARD
        .decode(&req.audio_base64)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!("Invalid base64: {e}")})),
            )
        })?;

    // Interpret as f32 PCM: every 4 bytes = one f32 sample (little-endian)
    if raw_bytes.len() % 4 != 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Audio data length must be a multiple of 4 (f32 PCM samples)"})),
        ));
    }
    let pcm_f32: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let id = state
        .engine
        .store_audio(&pcm_f32, req.sample_rate, &req.source, req.description.as_deref())
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?;

    state.engine.persist().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(json!({
        "memory_id": id.to_string()
    })))
}
