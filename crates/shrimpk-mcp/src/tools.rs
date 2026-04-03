//! MCP tool definitions and handler functions.
//!
//! Each tool mirrors a `shrimpk` CLI command.

use crate::format::{detect_ram_gb, format_bytes, format_number, tier_name, truncate};
use crate::protocol::ToolDefinition;
use serde_json::{Value, json};
use shrimpk_core::{EchoConfig, MemoryId, QueryMode, config};
use shrimpk_memory::{EchoEngine, PiiFilter};
use std::sync::Arc;
use std::time::Instant;

/// Return all tool definitions (10 base + 2 multimodal).
pub fn all_tools() -> Vec<ToolDefinition> {
    #[allow(unused_mut)]
    let mut tools = vec![
        ToolDefinition {
            name: "store".into(),
            description: "Store a memory in Echo Memory. The memory will persist across sessions and automatically surface when relevant context appears.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text content to remember" },
                    "source": { "type": "string", "description": "Source label (default: 'mcp')" }
                },
                "required": ["text"]
            }),
        },
        ToolDefinition {
            name: "echo".into(),
            description: "Find memories that resonate with a query. Returns memories ranked by similarity and Hebbian association strength. Supports multimodal search via the modality parameter. Optionally filter by exact labels.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Text to find resonating memories for" },
                    "max_results": { "type": "integer", "description": "Maximum results (default: 10)", "default": 10 },
                    "modality": { "type": "string", "enum": ["text", "vision", "auto"], "description": "Query mode: 'text' (default), 'vision' (CLIP cross-modal), or 'auto' (all channels)", "default": "text" },
                    "labels": { "type": "array", "items": { "type": "string" }, "description": "Optional label filter — bypass classification, search only memories matching these labels" },
                    "at_time": { "type": "string", "description": "Optional ISO 8601 timestamp for point-in-time query. Only Hebbian edges valid at this time are used for boosting." }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "stats".into(),
            description: "Show Echo Memory engine statistics including memory count, disk usage, and performance metrics.".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDefinition {
            name: "forget".into(),
            description: "Remove a memory by its UUID.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Memory UUID to remove" }
                },
                "required": ["id"]
            }),
        },
        ToolDefinition {
            name: "dump".into(),
            description: "List all stored memories with their IDs, content, and metadata.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "description": "Max entries to return (default: 50)", "default": 50 }
                }
            }),
        },
        ToolDefinition {
            name: "config_show".into(),
            description: "Show current ShrimPK configuration with source info (auto-detect, config file, or environment variable).".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDefinition {
            name: "config_set".into(),
            description: "Set a ShrimPK configuration value. Writes to ~/.shrimpk-kernel/config.toml. Requires server restart to take effect.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "Config key (e.g., max_memories, similarity_threshold, max_disk_bytes, quantization)" },
                    "value": { "type": "string", "description": "Value to set" }
                },
                "required": ["key", "value"]
            }),
        },
        ToolDefinition {
            name: "persist".into(),
            description: "Force save all memories to disk immediately.".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDefinition {
            name: "status".into(),
            description: "Show system status: config tier, disk usage with progress bar, RAM budget, and quantization mode.".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDefinition {
            name: "memory_graph".into(),
            description: "Show the label-graph connections for a memory. Returns which labels it has and the top related memories per label, ranked by cosine similarity.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": { "type": "string", "description": "UUID of the source memory" },
                    "top_per_label": { "type": "integer", "description": "Max related memories per label (default: 3)", "default": 3 }
                },
                "required": ["memory_id"]
            }),
        },
        ToolDefinition {
            name: "memory_related".into(),
            description: "Find memories related to a source memory via shared labels. Uses cosine-only fast path (skips LSH, Bloom, Hebbian). Optionally filter to a single label.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": { "type": "string", "description": "UUID of the source memory" },
                    "label": { "type": "string", "description": "Optional: filter to a specific label (e.g. 'topic:language')" },
                    "max_results": { "type": "integer", "description": "Maximum results (default: 10)", "default": 10 }
                },
                "required": ["memory_id"]
            }),
        },
        ToolDefinition {
            name: "memory_get".into(),
            description: "Get the full content of a specific memory by ID. Use after echo/graph/related to expand a truncated result.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "memory_id": { "type": "string", "description": "UUID of the memory to retrieve" }
                },
                "required": ["memory_id"]
            }),
        },
        ToolDefinition {
            name: "entity_search".into(),
            description: "Search for memories mentioning a specific entity (person, place, tool, organization). Uses the entity knowledge graph index for fast lookup.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "entity": { "type": "string", "description": "Entity name to search for (case-insensitive)" },
                    "max_results": { "type": "integer", "description": "Maximum results (default 10)", "default": 10 }
                },
                "required": ["entity"]
            }),
        },
        ToolDefinition {
            name: "community_summaries".into(),
            description: "List per-label community summaries. These are LLM-generated summaries of memory clusters, useful for getting a high-level overview of what's stored.".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
    ];

    // Multimodal tools — conditionally included based on compile-time feature flags
    #[cfg(feature = "vision")]
    tools.push(ToolDefinition {
        name: "store_image".into(),
        description: "Store an image as a visual memory using CLIP embeddings. The image will be indexed in the vision channel for cross-modal retrieval.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "image_base64": { "type": "string", "description": "Base64-encoded image data (PNG, JPEG, etc.)" },
                "source": { "type": "string", "description": "Source label (default: 'mcp')" },
                "description": { "type": "string", "description": "Text description for cross-modal recall (e.g., 'kitchen photo', 'whiteboard diagram')" }
            },
            "required": ["image_base64"]
        }),
    });

    #[cfg(feature = "speech")]
    tools.push(ToolDefinition {
        name: "store_audio".into(),
        description: "Store audio as a speech memory preserving paralinguistic features (tone, emotion, pace). NOT speech-to-text.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "audio_base64": { "type": "string", "description": "Base64-encoded raw PCM f32 audio data" },
                "sample_rate": { "type": "integer", "description": "Audio sample rate in Hz (default: 16000)", "default": 16000 },
                "source": { "type": "string", "description": "Source label (default: 'mcp')" },
                "description": { "type": "string", "description": "Text description for cross-modal recall (e.g., 'meeting standup recording')" }
            },
            "required": ["audio_base64"]
        }),
    });

    tools
}

// ---------------------------------------------------------------------------
// Tool Handlers
// ---------------------------------------------------------------------------

pub async fn handle_store(engine: &Arc<EchoEngine>, args: &Value) -> Result<String, String> {
    let text = args["text"]
        .as_str()
        .ok_or("Missing required argument: text")?;
    let source = args["source"].as_str().unwrap_or("mcp");

    let pii_filter = PiiFilter::new();
    let pii_matches = pii_filter.scan(text);
    let sensitivity = pii_filter.classify(text);

    let id = engine
        .store(text, source)
        .await
        .map_err(|e| e.to_string())?;
    engine.persist().await.map_err(|e| e.to_string())?;

    let pii_info = if pii_matches.is_empty() {
        "no sensitive data detected".into()
    } else {
        let types: Vec<String> = pii_matches
            .iter()
            .map(|m| m.pattern_type.to_string())
            .collect();
        format!("{} match(es) [{}]", pii_matches.len(), types.join(", "))
    };

    Ok(format!(
        "Stored memory {}... (sensitivity: {sensitivity:?})\nPII scan: {pii_info}",
        &id.to_string()[..8]
    ))
}

#[cfg(feature = "vision")]
pub async fn handle_store_image(engine: &Arc<EchoEngine>, args: &Value) -> Result<String, String> {
    use base64::Engine as _;

    let image_b64 = args["image_base64"]
        .as_str()
        .ok_or("Missing required argument: image_base64")?;
    let source = args["source"].as_str().unwrap_or("mcp");

    let image_bytes = base64::engine::general_purpose::STANDARD
        .decode(image_b64)
        .map_err(|e| format!("Invalid base64: {e}"))?;

    let description = args["description"].as_str();

    let id = engine
        .store_image(&image_bytes, source, description)
        .await
        .map_err(|e| e.to_string())?;
    engine.persist().await.map_err(|e| e.to_string())?;

    Ok(serde_json::to_string_pretty(&json!({
        "memory_id": id.to_string()
    }))
    .unwrap())
}

#[cfg(feature = "speech")]
pub async fn handle_store_audio(engine: &Arc<EchoEngine>, args: &Value) -> Result<String, String> {
    use base64::Engine as _;

    let audio_b64 = args["audio_base64"]
        .as_str()
        .ok_or("Missing required argument: audio_base64")?;
    let sample_rate = args["sample_rate"].as_u64().unwrap_or(16000) as u32;
    let source = args["source"].as_str().unwrap_or("mcp");

    let raw_bytes = base64::engine::general_purpose::STANDARD
        .decode(audio_b64)
        .map_err(|e| format!("Invalid base64: {e}"))?;

    // Interpret as f32 PCM: every 4 bytes = one f32 sample
    if raw_bytes.len() % 4 != 0 {
        return Err("Audio data length must be a multiple of 4 (f32 PCM samples)".into());
    }
    let pcm_f32: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let description = args["description"].as_str();

    let id = engine
        .store_audio(&pcm_f32, sample_rate, source, description)
        .await
        .map_err(|e| e.to_string())?;
    engine.persist().await.map_err(|e| e.to_string())?;

    Ok(serde_json::to_string_pretty(&json!({
        "memory_id": id.to_string()
    }))
    .unwrap())
}

pub async fn handle_echo(engine: &Arc<EchoEngine>, args: &Value) -> Result<String, String> {
    let query = args["query"]
        .as_str()
        .ok_or("Missing required argument: query")?;
    let max_results = args["max_results"].as_u64().unwrap_or(10) as usize;

    let mode = match args["modality"].as_str().unwrap_or("text") {
        "vision" => QueryMode::Vision,
        "auto" => QueryMode::Auto,
        _ => QueryMode::Text,
    };

    // Optional label filter — bypass classification, use exact labels
    let label_filter: Option<Vec<String>> = args["labels"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    });

    // Optional point-in-time query (KS63): parse ISO 8601 → epoch seconds
    let at_time: Option<f64> = args["at_time"]
        .as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.timestamp() as f64);

    let start = Instant::now();
    let results = if let Some(at) = at_time {
        engine.echo_at(query, max_results, at).await
    } else {
        engine
            .echo_with_labels(query, max_results, mode, label_filter.as_deref())
            .await
    }
    .map_err(|e| e.to_string())?;
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
                "source": r.source,
                "labels": r.labels
            })
        })
        .collect();

    let output = json!({
        "query": query,
        "results": results_json,
        "count": results.len(),
        "elapsed_ms": (elapsed_ms * 10.0).round() / 10.0
    });

    Ok(serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".into()))
}

pub async fn handle_memory_graph(engine: &Arc<EchoEngine>, args: &Value) -> Result<String, String> {
    let id_str = args["memory_id"]
        .as_str()
        .ok_or("Missing required argument: memory_id")?;
    let uuid = uuid::Uuid::parse_str(id_str).map_err(|e| format!("Invalid UUID: {e}"))?;
    let id = MemoryId::from_uuid(uuid);
    let top_per_label = args["top_per_label"].as_u64().unwrap_or(3) as usize;

    let graph = engine
        .memory_graph(&id, top_per_label)
        .await
        .map_err(|e| e.to_string())?;

    let connections_json: Vec<Value> = graph
        .connections
        .iter()
        .map(|c| {
            json!({
                "label": c.label,
                "count": c.count,
                "top_ids": c.top_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>()
            })
        })
        .collect();

    let output = json!({
        "memory_id": graph.memory_id.to_string(),
        "content_preview": graph.content_preview,
        "labels": graph.labels,
        "connections": connections_json,
        "total_connected": graph.total_connected,
        "unique_connected": graph.unique_connected
    });

    Ok(serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".into()))
}

pub async fn handle_memory_related(
    engine: &Arc<EchoEngine>,
    args: &Value,
) -> Result<String, String> {
    let id_str = args["memory_id"]
        .as_str()
        .ok_or("Missing required argument: memory_id")?;
    let uuid = uuid::Uuid::parse_str(id_str).map_err(|e| format!("Invalid UUID: {e}"))?;
    let id = MemoryId::from_uuid(uuid);
    let label = args["label"].as_str();
    let max_results = args["max_results"].as_u64().unwrap_or(10) as usize;

    let start = Instant::now();
    let results = engine
        .memory_related(&id, label, max_results)
        .await
        .map_err(|e| e.to_string())?;
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
                "source": r.source,
                "labels": r.labels
            })
        })
        .collect();

    let output = json!({
        "source_memory_id": id_str,
        "label_filter": label,
        "results": results_json,
        "count": results.len(),
        "elapsed_ms": (elapsed_ms * 10.0).round() / 10.0
    });

    Ok(serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".into()))
}

pub async fn handle_memory_get(engine: &Arc<EchoEngine>, args: &Value) -> Result<String, String> {
    let id_str = args["memory_id"]
        .as_str()
        .ok_or("Missing required argument: memory_id")?;
    let uuid = uuid::Uuid::parse_str(id_str).map_err(|e| format!("Invalid UUID: {e}"))?;
    let id = MemoryId::from_uuid(uuid);

    let entry = engine.memory_get(&id).await.map_err(|e| e.to_string())?;

    let output = json!({
        "memory_id": entry.id.to_string(),
        "content": entry.display_content(),
        "source": entry.source,
        "modality": format!("{}", entry.modality),
        "labels": entry.labels,
        "echo_count": entry.echo_count,
        "created_at": entry.created_at.to_rfc3339(),
        "category": format!("{:?}", entry.category),
        "sensitivity": format!("{:?}", entry.sensitivity),
        "novelty_score": entry.novelty_score
    });

    Ok(serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".into()))
}

pub async fn handle_entity_search(
    engine: &Arc<EchoEngine>,
    args: &Value,
) -> Result<String, String> {
    let entity = args["entity"]
        .as_str()
        .ok_or("Missing required argument: entity")?;
    let max_results = args["max_results"].as_u64().unwrap_or(10) as usize;

    let start = Instant::now();
    let results = engine
        .entity_search(entity, max_results)
        .await
        .map_err(|e| e.to_string())?;
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
                "source": r.source,
                "labels": r.labels
            })
        })
        .collect();

    let output = json!({
        "entity": entity,
        "results": results_json,
        "count": results.len(),
        "elapsed_ms": (elapsed_ms * 10.0).round() / 10.0
    });

    Ok(serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".into()))
}

pub async fn handle_community_summaries(
    engine: &Arc<EchoEngine>,
) -> Result<String, String> {
    let summaries = engine.community_summaries().await;
    if summaries.is_empty() {
        return Ok(json!({"summaries": [], "count": 0, "note": "No community summaries yet. They are generated during consolidation for label clusters with enough members."}).to_string());
    }
    let summaries_json: Vec<Value> = summaries
        .iter()
        .map(|(label, s)| {
            json!({
                "label": label,
                "summary": s.summary,
                "member_count": s.member_count,
                "updated_at": s.updated_at.to_rfc3339()
            })
        })
        .collect();
    let output = json!({
        "summaries": summaries_json,
        "count": summaries.len()
    });
    Ok(serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".into()))
}

pub async fn handle_stats(engine: &Arc<EchoEngine>, config: &EchoConfig) -> Result<String, String> {
    let stats = engine.stats().await;
    let ram_gb = detect_ram_gb();

    let mut out = format!(
        "Memories:        {}\n\
         Max capacity:    {}\n\
         Index size:      {}\n\
         RAM usage:       {}\n\
         Disk usage:      {} / {}\n\
         Config tier:     {} ({ram_gb}GB detected)\n\
         Quantization:    {}\n\
         Embedding dim:   {}\n\
         Threshold:       {}\n\
         Echo queries:    {}\n\
         Avg echo latency: {:.1}ms",
        format_number(stats.total_memories),
        format_number(stats.max_capacity),
        format_bytes(stats.index_size_bytes),
        format_bytes(stats.ram_usage_bytes),
        format_bytes(stats.disk_usage_bytes),
        format_bytes(stats.max_disk_bytes),
        tier_name(config),
        config.quantization,
        config.embedding_dim,
        config.similarity_threshold,
        stats.total_echo_queries,
        stats.avg_echo_latency_ms,
    );

    // Per-channel breakdown
    if stats.vision_count > 0 || stats.speech_count > 0 {
        out.push_str(&format!(
            "\n\nChannels:\n  Text:    {}\n  Vision:  {}\n  Speech:  {}",
            format_number(stats.text_count),
            format_number(stats.vision_count),
            format_number(stats.speech_count),
        ));
    }

    Ok(out)
}

pub async fn handle_forget(engine: &Arc<EchoEngine>, args: &Value) -> Result<String, String> {
    let id_str = args["id"].as_str().ok_or("Missing required argument: id")?;
    let uuid = uuid::Uuid::parse_str(id_str).map_err(|e| format!("Invalid UUID: {e}"))?;
    let id = MemoryId::from_uuid(uuid);

    engine.forget(id).await.map_err(|e| e.to_string())?;
    engine.persist().await.map_err(|e| e.to_string())?;

    Ok(format!(
        "Forgotten memory {}",
        &id_str[..id_str.len().min(8)]
    ))
}

pub async fn handle_dump(
    engine: &Arc<EchoEngine>,
    _config: &EchoConfig,
    args: &Value,
) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(50) as usize;

    let entries = engine.all_entry_summaries().await;
    if entries.is_empty() {
        return Ok("No memories stored.".into());
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "Stored memories ({}, showing {}):",
        format_number(entries.len()),
        entries.len().min(limit)
    ));

    for entry in entries.iter().take(limit) {
        let id_str = entry.id.to_string();
        let id_short = &id_str[..id_str.len().min(8)];
        lines.push(format!(
            "  {} \"{}\" (source: {}, echoed: {}x, category: {:?}, importance: {:.2})",
            id_short,
            truncate(&entry.content, 50),
            entry.source,
            entry.echo_count,
            entry.category,
            entry.importance,
        ));
    }

    Ok(lines.join("\n"))
}

pub fn handle_config_show(config: &EchoConfig) -> Result<String, String> {
    let file_config = config::load_config_file().ok().flatten();
    let has_file = file_config.is_some();
    let fc = file_config.unwrap_or_default();

    let source = |env: &str, file_val: bool| -> &'static str {
        if std::env::var(env).is_ok() {
            "env"
        } else if file_val {
            "file"
        } else {
            "auto"
        }
    };

    let lines = vec![
        format!(
            "Configuration (priority: env > file > auto-detect):\nConfig file: {} {}",
            config::config_path().display(),
            if has_file { "(exists)" } else { "(not found)" }
        ),
        String::new(),
        format!("  {:25} {:>15}  {}", "Key", "Value", "Source"),
        format!("  {:25} {:>15}  {}", "---", "-----", "------"),
        format!(
            "  {:25} {:>15}  {}",
            "max_memories",
            format_number(config.max_memories),
            source("SHRIMPK_MAX_MEMORIES", fc.max_memories.is_some())
        ),
        format!(
            "  {:25} {:>15}  {}",
            "similarity_threshold",
            format!("{:.2}", config.similarity_threshold),
            source(
                "SHRIMPK_SIMILARITY_THRESHOLD",
                fc.similarity_threshold.is_some()
            )
        ),
        format!(
            "  {:25} {:>15}  {}",
            "quantization",
            config.quantization.to_string(),
            source("SHRIMPK_QUANTIZATION", fc.quantization.is_some())
        ),
        format!(
            "  {:25} {:>15}  {}",
            "max_disk_bytes",
            format_bytes(config.max_disk_bytes),
            source("SHRIMPK_MAX_DISK_BYTES", fc.max_disk_bytes.is_some())
        ),
        format!(
            "  {:25} {:>15}  {}",
            "data_dir",
            truncate(&config.data_dir.to_string_lossy(), 30),
            source("SHRIMPK_DATA_DIR", fc.data_dir.is_some())
        ),
        format!(
            "  {:25} {:>15}  {}",
            "consolidation_provider",
            &config.consolidation_provider,
            source(
                "SHRIMPK_CONSOLIDATION_PROVIDER",
                fc.consolidation_provider.is_some()
            )
        ),
        format!(
            "  {:25} {:>15}  {}",
            "ollama_url",
            truncate(&config.ollama_url, 30),
            source("SHRIMPK_OLLAMA_URL", fc.ollama_url.is_some())
        ),
        format!(
            "  {:25} {:>15}  {}",
            "enrichment_model",
            &config.enrichment_model,
            source("SHRIMPK_ENRICHMENT_MODEL", fc.enrichment_model.is_some())
        ),
        format!(
            "  {:25} {:>15}  {}",
            "max_facts_per_memory",
            config.max_facts_per_memory,
            if fc.max_facts_per_memory.is_some() {
                "file"
            } else {
                "auto"
            }
        ),
        String::new(),
        "  Intelligence Engine:".to_string(),
        format!(
            "  {:25} {:>15}  {}",
            "use_power_law_decay",
            config.use_power_law_decay,
            if fc.use_power_law_decay.is_some() {
                "file"
            } else {
                "auto"
            }
        ),
        format!(
            "  {:25} {:>15}  {}",
            "use_importance",
            config.use_importance,
            if fc.use_importance.is_some() {
                "file"
            } else {
                "auto"
            }
        ),
        format!(
            "  {:25} {:>15}  {}",
            "use_actr_activation",
            config.use_actr_activation,
            if fc.use_actr_activation.is_some() {
                "file"
            } else {
                "auto"
            }
        ),
        format!(
            "  {:25} {:>15}  {}",
            "activation_weight",
            format!("{:.2}", config.activation_weight),
            if fc.activation_weight.is_some() {
                "file"
            } else {
                "auto"
            }
        ),
        format!(
            "  {:25} {:>15}  {}",
            "importance_weight",
            format!("{:.2}", config.importance_weight),
            if fc.importance_weight.is_some() {
                "file"
            } else {
                "auto"
            }
        ),
        format!(
            "  {:25} {:>15}  {}",
            "use_full_actr_history",
            config.use_full_actr_history,
            if fc.use_full_actr_history.is_some() {
                "file"
            } else {
                "auto"
            }
        ),
    ];

    Ok(lines.join("\n"))
}

pub fn handle_config_set(args: &Value) -> Result<String, String> {
    let key = args["key"]
        .as_str()
        .ok_or("Missing required argument: key")?;
    let value = args["value"]
        .as_str()
        .ok_or("Missing required argument: value")?;

    let mut fc = config::load_config_file()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();

    match key {
        "max_memories" => fc.max_memories = Some(value.parse().map_err(|_| "Invalid integer")?),
        "similarity_threshold" => {
            fc.similarity_threshold = Some(value.parse().map_err(|_| "Invalid float")?)
        }
        "max_echo_results" => {
            fc.max_echo_results = Some(value.parse().map_err(|_| "Invalid integer")?)
        }
        "ram_budget_bytes" => {
            fc.ram_budget_bytes = Some(value.parse().map_err(|_| "Invalid integer")?)
        }
        "max_disk_bytes" => fc.max_disk_bytes = Some(value.parse().map_err(|_| "Invalid integer")?),
        "quantization" => fc.quantization = Some(value.parse().map_err(|e: String| e)?),
        "data_dir" => fc.data_dir = Some(std::path::PathBuf::from(value)),
        "use_lsh" => fc.use_lsh = Some(value.parse().map_err(|_| "Invalid boolean")?),
        "use_bloom" => fc.use_bloom = Some(value.parse().map_err(|_| "Invalid boolean")?),
        "ollama_url" => fc.ollama_url = Some(value.to_string()),
        "enrichment_model" => fc.enrichment_model = Some(value.to_string()),
        "consolidation_provider" => fc.consolidation_provider = Some(value.to_string()),
        "max_facts_per_memory" => {
            fc.max_facts_per_memory = Some(value.parse().map_err(|_| "Invalid integer")?)
        }
        "use_power_law_decay" => {
            fc.use_power_law_decay = Some(value.parse().map_err(|_| "Invalid boolean")?)
        }
        "use_importance" => {
            fc.use_importance = Some(value.parse().map_err(|_| "Invalid boolean")?)
        }
        "use_actr_activation" => {
            fc.use_actr_activation = Some(value.parse().map_err(|_| "Invalid boolean")?)
        }
        "activation_weight" => {
            fc.activation_weight = Some(value.parse().map_err(|_| "Invalid float")?)
        }
        "importance_weight" => {
            fc.importance_weight = Some(value.parse().map_err(|_| "Invalid float")?)
        }
        "use_full_actr_history" => {
            fc.use_full_actr_history = Some(value.parse().map_err(|_| "Invalid boolean")?)
        }
        other => return Err(format!("Unknown config key: \"{other}\"")),
    }

    config::save_config_file(&fc).map_err(|e| e.to_string())?;
    Ok(format!(
        "Set {key} = {value} in {}\nNote: restart shrimpk-mcp for this change to take effect.",
        config::config_path().display()
    ))
}

pub async fn handle_persist(
    engine: &Arc<EchoEngine>,
    config: &EchoConfig,
) -> Result<String, String> {
    engine.persist().await.map_err(|e| e.to_string())?;
    let disk_usage = config::disk_usage(&config.data_dir).unwrap_or(0);
    Ok(format!(
        "Persisted to disk. Disk usage: {} / {}",
        format_bytes(disk_usage),
        format_bytes(config.max_disk_bytes)
    ))
}

pub fn handle_status(config: &EchoConfig) -> Result<String, String> {
    let ram_gb = detect_ram_gb();
    let disk_usage = config::disk_usage(&config.data_dir).unwrap_or(0);
    let disk_pct = if config.max_disk_bytes > 0 {
        (disk_usage as f64 / config.max_disk_bytes as f64 * 100.0) as u64
    } else {
        0
    };

    let bar_width = 30;
    let filled = (disk_pct as usize * bar_width / 100).min(bar_width);
    let bar = format!("[{}{}]", "#".repeat(filled), "-".repeat(bar_width - filled));

    let mut out = format!(
        "System Status:\n\
         Config tier:   {} ({ram_gb}GB RAM detected)\n\
         Data dir:      {}\n\
         Disk usage:    {} / {} ({}%) {}\n\
         RAM budget:    {}\n\
         Quantization:  {}\n\
         Max memories:  {}",
        tier_name(config),
        config.data_dir.display(),
        format_bytes(disk_usage),
        format_bytes(config.max_disk_bytes),
        disk_pct,
        bar,
        format_bytes(config.ram_budget_bytes),
        config.quantization,
        format_number(config.max_memories),
    );

    if disk_pct >= 80 {
        out.push_str(
            "\n\nWARNING: Disk usage above 80%. Consider increasing max_disk_bytes or cleaning data.",
        );
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tools_returns_expected_count() {
        let count = all_tools().len();
        // Base: 14 tools (9 original + memory_graph + memory_related + memory_get + entity_search + community_summaries).
        // +1 if vision feature, +1 if speech feature.
        #[allow(unused_mut)]
        let mut expected = 14;
        #[cfg(feature = "vision")]
        {
            expected += 1;
        }
        #[cfg(feature = "speech")]
        {
            expected += 1;
        }
        assert_eq!(count, expected);
    }

    #[test]
    fn all_tools_have_required_fields() {
        for tool in all_tools() {
            assert!(!tool.name.is_empty());
            assert!(!tool.description.is_empty());
            assert!(tool.input_schema.is_object());
        }
    }

    #[test]
    fn config_set_rejects_unknown_key() {
        let args = json!({ "key": "nonexistent", "value": "123" });
        let result = handle_config_set(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown config key"));
    }

    #[test]
    fn status_returns_formatted_text() {
        let config = EchoConfig::default();
        let result = handle_status(&config).unwrap();
        assert!(result.contains("Config tier:"));
        assert!(result.contains("Disk usage:"));
    }
}
