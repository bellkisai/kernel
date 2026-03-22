//! HTTP route handlers for the ShrimPK daemon.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;
use serde_json::{Value, json};
use shrimpk_core::{MemoryId, config};
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
    let start = Instant::now();
    let results = state
        .engine
        .echo(&req.query, req.max_results)
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
        "total_echo_queries": s.total_echo_queries
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
        "max_disk_bytes": c.max_disk_bytes
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
