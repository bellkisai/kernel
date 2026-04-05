//! Binary persistence for Echo Memory (SHRM v2 format).
//!
//! Supports reading both v1 (text-only) and v2 (multimodal) formats.
//! Always writes v2 on save for forward progress.
//!
//! ## v2 Binary Format Layout (current)
//! ```text
//! Offset  Size    Field
//! 0       4       Magic: b"SHRM"
//! 4       1       Version: u8 (2)
//! 5       1       Flags: bit 0=has_vision, bit 1=has_speech
//! 6       2       Text dim: u16 (384)
//! 8       2       Vision dim: u16 (512 or 0)
//! 10      2       Speech dim: u16 (640 or 0)
//! 12      4       Entry count: u32
//! ------- 16 bytes total header -------
//! 16      4       Metadata JSON length (u32)
//! 20      M       Metadata JSON bytes
//! 20+M    4       Metadata CRC32
//! ------- Text section -------
//!         N*D*4   Text embeddings: f32[count][text_dim]
//!         4       Text CRC32
//! ------- Vision section (if has_vision) -------
//!         ceil(N/8)  Bitmap: bit i = entry i has vision embedding
//!         V*512*4    Vision embeddings (only present entries)
//!         4          Vision CRC32 (over bitmap + embeddings)
//! ------- Speech section (if has_speech) -------
//!         ceil(N/8)  Bitmap
//!         S*640*4    Speech embeddings (only present entries)
//!         4          Speech CRC32 (over bitmap + embeddings)
//! ```
//!
//! ## v1 Legacy Format (read-only)
//! ```text
//! 64-byte header (magic, version u32=1, dim u32, count u64, metadata offset, CRC32)
//! Flat embedding array + trailing JSON metadata
//! ```

use chrono::{DateTime, Utc};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use shrimpk_core::{
    MemoryCategory, MemoryEntry, MemoryId, Modality, Result, SensitivityLevel, ShrimPKError,
};
use std::io::Write;
use std::path::Path;
use tracing::instrument;

use crate::store::EchoStore;

// --- Constants ---

/// Magic bytes identifying a ShrimPK Echo Memory binary file.
const MAGIC: &[u8; 4] = b"SHRM";

/// Current binary format version (multimodal: text + optional vision/speech).
const FORMAT_VERSION: u32 = 2;

/// v1 header size in bytes.
const HEADER_SIZE_V1: u64 = 64;

/// v2 header size in bytes.
///
/// ```text
/// Offset  Size    Field
/// 0       4       Magic: b"SHRM"
/// 4       1       Version: u8 (2)
/// 5       1       Flags: u8 (bit 0 = has_vision, bit 1 = has_speech)
/// 6       2       Text dim: u16 (384)
/// 8       2       Vision dim: u16 (512 or 0)
/// 10      2       Speech dim: u16 (640 or 0)
/// 12      4       Entry count: u32
/// ```
const HEADER_SIZE_V2: u64 = 16;

// v2 flag bits
const FLAG_HAS_VISION: u8 = 0b0000_0001;
const FLAG_HAS_SPEECH: u8 = 0b0000_0010;

// --- MemoryMeta ---

/// Serializable metadata for a memory entry (no embeddings).
///
/// Embeddings are stored separately in the flat binary array for performance.
/// This struct captures everything else needed to reconstruct a `MemoryEntry`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMeta {
    pub id: MemoryId,
    pub content: String,
    pub masked_content: Option<String>,
    #[serde(default)]
    pub reformulated: Option<String>,
    pub source: String,
    pub sensitivity: SensitivityLevel,
    #[serde(default)]
    pub category: MemoryCategory,
    pub created_at: DateTime<Utc>,
    pub last_echoed: Option<DateTime<Utc>>,
    pub echo_count: u32,
    #[serde(default)]
    pub enriched: bool,
    #[serde(default)]
    pub parent_id: Option<MemoryId>,
    /// Which sensory channel produced this memory (v2+).
    #[serde(default)]
    pub modality: Modality,
    /// Whether this entry has a vision embedding stored in the binary section (v2+).
    #[serde(default)]
    pub has_vision_embedding: bool,
    /// Whether this entry has a speech embedding stored in the binary section (v2+).
    #[serde(default)]
    pub has_speech_embedding: bool,
    /// Semantic labels for pre-filtered retrieval (ADR-015).
    #[serde(default)]
    pub labels: Vec<String>,
    /// Label enrichment version (0=unlabeled, 1=Tier1, 2=Tier2).
    #[serde(default)]
    pub label_version: u8,
    /// Novelty score at store time (0.0 to 1.0).
    #[serde(default)]
    pub novelty_score: f32,
    /// Multi-signal importance score (0.0-1.0).
    #[serde(default)]
    pub importance: f32,
    /// Cached ACT-R base-level activation (OL approximation).
    #[serde(default)]
    pub activation_cache: f32,
    /// When importance was last computed (for staleness detection).
    #[serde(default)]
    pub importance_computed_at: Option<DateTime<Utc>>,
    /// Retrieval timestamps as seconds since UNIX epoch (ring buffer, cap 16).
    #[serde(default)]
    pub retrieval_history_secs: Vec<u32>,
    /// Knowledge graph triples extracted during consolidation (KS61).
    #[serde(default)]
    pub triples: Vec<shrimpk_core::Triple>,
}

impl MemoryMeta {
    /// Extract metadata from a MemoryEntry (strips embeddings — they go in binary sections).
    fn from_entry(entry: &MemoryEntry) -> Self {
        Self {
            id: entry.id.clone(),
            content: entry.content.clone(),
            masked_content: entry.masked_content.clone(),
            reformulated: entry.reformulated.clone(),
            source: entry.source.clone(),
            sensitivity: entry.sensitivity,
            category: entry.category,
            created_at: entry.created_at,
            last_echoed: entry.last_echoed,
            echo_count: entry.echo_count,
            enriched: entry.enriched,
            parent_id: entry.parent_id.clone(),
            modality: entry.modality,
            has_vision_embedding: entry.vision_embedding.is_some(),
            has_speech_embedding: entry.speech_embedding.is_some(),
            labels: entry.labels.clone(),
            label_version: entry.label_version,
            novelty_score: entry.novelty_score,
            importance: entry.importance,
            activation_cache: entry.activation_cache,
            importance_computed_at: entry.importance_computed_at,
            retrieval_history_secs: entry.retrieval_history_secs.clone(),
            triples: entry.triples.clone(),
        }
    }

    /// Reconstruct a MemoryEntry by combining metadata with its text embedding.
    /// Vision and speech embeddings are attached separately after load.
    fn into_entry(self, embedding: Vec<f32>) -> MemoryEntry {
        MemoryEntry {
            id: self.id,
            content: self.content,
            masked_content: self.masked_content,
            reformulated: self.reformulated,
            embedding,
            modality: self.modality,
            vision_embedding: None,
            speech_embedding: None,
            source: self.source,
            sensitivity: self.sensitivity,
            category: self.category,
            created_at: self.created_at,
            last_echoed: self.last_echoed,
            echo_count: self.echo_count,
            enriched: self.enriched,
            parent_id: self.parent_id,
            labels: self.labels,
            label_version: self.label_version,
            novelty_score: self.novelty_score,
            importance: self.importance,
            activation_cache: self.activation_cache,
            importance_computed_at: self.importance_computed_at,
            retrieval_history_secs: self.retrieval_history_secs,
            triples: self.triples,
        }
    }
}

// --- Public API ---

/// Save an EchoStore to a binary file (SHRM v2 format).
///
/// ## v2 Layout
/// ```text
/// [16-byte header]
/// [JSON metadata + CRC32]
/// [text embeddings + CRC32]        — always present
/// [vision bitmap + embeddings + CRC32]  — if any entry has vision_embedding
/// [speech bitmap + embeddings + CRC32]  — if any entry has speech_embedding
/// ```
#[instrument(skip(store), fields(entries = store.len(), path = %path.display()))]
pub fn save_binary(store: &EchoStore, path: &Path) -> Result<()> {
    let start = std::time::Instant::now();

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Acquire exclusive file lock to prevent concurrent write corruption (F-05 fix)
    let lock_path = path.with_extension("shrm.lock");
    let lock_file = std::fs::File::create(&lock_path)?;
    lock_file
        .lock_exclusive()
        .map_err(|e| ShrimPKError::Persistence(format!("Failed to acquire write lock: {e}")))?;
    // Lock released on drop

    let entries = store.all_entries();
    let embeddings = store.all_embeddings();
    let count = entries.len();

    // Determine embedding dimension from first entry, or use default 384
    let text_dim: u16 = embeddings.first().map(|e| e.len() as u16).unwrap_or(384);

    // Scan for vision/speech presence and infer dimensions
    let has_vision = entries.iter().any(|e| e.vision_embedding.is_some());
    let has_speech = entries.iter().any(|e| e.speech_embedding.is_some());

    let vision_dim: u16 = if has_vision {
        entries
            .iter()
            .find_map(|e| e.vision_embedding.as_ref().map(|v| v.len() as u16))
            .unwrap_or(512)
    } else {
        0
    };
    let speech_dim: u16 = if has_speech {
        entries
            .iter()
            .find_map(|e| e.speech_embedding.as_ref().map(|v| v.len() as u16))
            .unwrap_or(640)
    } else {
        0
    };

    let flags: u8 =
        if has_vision { FLAG_HAS_VISION } else { 0 } | if has_speech { FLAG_HAS_SPEECH } else { 0 };

    // Atomic write: write to .tmp file, then rename (F-06 fix)
    let tmp_path = path.with_extension("shrm.tmp");
    let mut file = std::fs::File::create(&tmp_path)?;

    // --- Write v2 header (16 bytes) ---
    // Build header bytes for CRC coverage (Fix 3: header + metadata CRC).
    let mut header_bytes = Vec::with_capacity(HEADER_SIZE_V2 as usize);
    header_bytes.extend_from_slice(MAGIC); // 4 bytes
    header_bytes.push(FORMAT_VERSION as u8); // 1 byte (version = 2)
    header_bytes.push(flags); // 1 byte
    header_bytes.extend_from_slice(&text_dim.to_le_bytes()); // 2 bytes
    header_bytes.extend_from_slice(&vision_dim.to_le_bytes()); // 2 bytes
    header_bytes.extend_from_slice(&speech_dim.to_le_bytes()); // 2 bytes
    header_bytes.extend_from_slice(&(count as u32).to_le_bytes()); // 4 bytes
    // Total: 16 bytes
    file.write_all(&header_bytes)?;

    // --- Write metadata as JSON + CRC ---
    // CRC covers header_bytes + meta_json to protect against corrupted
    // entry_count or dim fields causing terabyte allocations (Fix 3).
    let metas: Vec<MemoryMeta> = entries.iter().map(MemoryMeta::from_entry).collect();
    let meta_json = serde_json::to_vec(&metas)?;
    let meta_len = meta_json.len() as u32;
    file.write_all(&meta_len.to_le_bytes())?;
    file.write_all(&meta_json)?;
    let mut meta_hasher = crc32fast::Hasher::new();
    meta_hasher.update(&header_bytes);
    meta_hasher.update(&meta_json);
    let meta_crc = meta_hasher.finalize();
    file.write_all(&meta_crc.to_le_bytes())?;

    // --- Write text embeddings section + CRC ---
    {
        let mut hasher = crc32fast::Hasher::new();
        for emb in embeddings {
            for &val in emb {
                let bytes = val.to_le_bytes();
                file.write_all(&bytes)?;
                hasher.update(&bytes);
            }
        }
        let crc = hasher.finalize();
        file.write_all(&crc.to_le_bytes())?;
    }

    // --- Write vision section (if any entry has vision_embedding) ---
    if has_vision {
        write_optional_section(&mut file, entries, vision_dim, |e| {
            e.vision_embedding.as_deref()
        })?;
    }

    // --- Write speech section (if any entry has speech_embedding) ---
    if has_speech {
        write_optional_section(&mut file, entries, speech_dim, |e| {
            e.speech_embedding.as_deref()
        })?;
    }

    // Flush and sync to disk
    file.flush()?;
    file.sync_all()?;

    // Flush and close the tmp file, then atomically rename (F-06 fix)
    drop(file);
    std::fs::rename(&tmp_path, path)
        .map_err(|e| ShrimPKError::Persistence(format!("Atomic rename failed: {e}")))?;

    // Ensure directory entry is durable on Unix (ext4 requires parent dir fsync).
    // Without this, a power loss after rename could lose the file entirely.
    // No-op on Windows (NTFS handles directory entry durability automatically).
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
    }

    let elapsed = start.elapsed();
    tracing::info!(
        entries = count,
        text_dim = text_dim,
        vision_dim = vision_dim,
        speech_dim = speech_dim,
        flags = flags,
        elapsed_ms = elapsed.as_millis(),
        "Binary store saved (SHRM v2)"
    );

    Ok(())
}

/// Write an optional embedding section (vision or speech).
///
/// Layout: bitmap (ceil(count/8) bytes) + flat f32 array (only present entries) + CRC32.
fn write_optional_section<F>(
    file: &mut std::fs::File,
    entries: &[MemoryEntry],
    dim: u16,
    get_embedding: F,
) -> Result<()>
where
    F: Fn(&MemoryEntry) -> Option<&[f32]>,
{
    let count = entries.len();
    let bitmap_len = count.div_ceil(8);
    let mut bitmap = vec![0u8; bitmap_len];

    // Build bitmap
    for (i, entry) in entries.iter().enumerate() {
        if get_embedding(entry).is_some() {
            bitmap[i / 8] |= 1 << (i % 8);
        }
    }

    let mut hasher = crc32fast::Hasher::new();

    // Write bitmap
    file.write_all(&bitmap)?;
    hasher.update(&bitmap);

    // Write embeddings for present entries only
    for entry in entries {
        if let Some(emb) = get_embedding(entry) {
            debug_assert_eq!(emb.len(), dim as usize);
            for &val in emb {
                let bytes = val.to_le_bytes();
                file.write_all(&bytes)?;
                hasher.update(&bytes);
            }
        }
    }

    let crc = hasher.finalize();
    file.write_all(&crc.to_le_bytes())?;

    Ok(())
}

/// Load an EchoStore from a binary file.
///
/// Supports both v1 (text-only, 64-byte header) and v2 (multimodal, 16-byte header).
/// Returns an empty store if the file does not exist (same as JSON behavior).
#[instrument(fields(path = %path.display()))]
pub fn load_binary(path: &Path) -> Result<EchoStore> {
    if !path.exists() {
        tracing::info!(path = %path.display(), "No binary store file found, starting empty");
        return Ok(EchoStore::new());
    }

    // Acquire shared lock for concurrent read safety (F-05 fix)
    let lock_path = path.with_extension("shrm.lock");
    let _lock_guard = std::fs::File::open(&lock_path)
        .ok()
        .and_then(|f| f.lock_shared().ok().map(|()| f));

    let start = std::time::Instant::now();
    let data = std::fs::read(path)
        .map_err(|e| ShrimPKError::Persistence(format!("Failed to read binary store: {e}")))?;

    // Need at least 5 bytes to read magic + version byte
    if data.len() < 5 {
        return Err(ShrimPKError::Persistence(format!(
            "File too small: {} bytes",
            data.len()
        )));
    }

    // Validate magic
    let magic: [u8; 4] = data[0..4]
        .try_into()
        .map_err(|_| ShrimPKError::Persistence("Failed to read magic".into()))?;
    if magic != *MAGIC {
        return Err(ShrimPKError::Persistence(format!(
            "Invalid magic bytes: expected {:?}, got {:?}",
            MAGIC, magic
        )));
    }

    // Detect version. v1 stored version as u32 LE at offset 4, so bytes [4..8] = [1,0,0,0].
    // v2 stores version as a single u8 at offset 4, so byte [4] = 2.
    // We distinguish: if byte[4]==1 AND bytes[5..8]==[0,0,0], it's v1.
    // If byte[4]==2, it's v2. Anything else is unsupported.
    let version_byte = data[4];

    if version_byte == 1 && data.len() >= 8 && data[5] == 0 && data[6] == 0 && data[7] == 0 {
        // v1 path
        let store = load_binary_v1(&data)?;
        let elapsed = start.elapsed();
        tracing::info!(
            entries = store.len(),
            elapsed_ms = elapsed.as_millis(),
            "Binary store loaded (SHRM v1 compat)"
        );
        Ok(store)
    } else if version_byte == 2 {
        // v2 path
        let store = load_binary_v2(&data)?;
        let elapsed = start.elapsed();
        tracing::info!(
            entries = store.len(),
            elapsed_ms = elapsed.as_millis(),
            "Binary store loaded (SHRM v2)"
        );
        Ok(store)
    } else {
        Err(ShrimPKError::Persistence(format!(
            "Unsupported format version: {}",
            version_byte
        )))
    }
}

/// Load a v1 binary store (text-only, 64-byte header). Backward compatibility path.
fn load_binary_v1(data: &[u8]) -> Result<EchoStore> {
    if data.len() < HEADER_SIZE_V1 as usize {
        return Err(ShrimPKError::Persistence(format!(
            "File too small for v1 header: {} bytes (need {})",
            data.len(),
            HEADER_SIZE_V1
        )));
    }

    let (_magic, _version, dim, count, metadata_offset, stored_crc) = parse_header_v1(data)?;

    // --- Read and verify embedding array ---
    let embedding_bytes = count * (dim as u64) * 4;
    let embedding_end = HEADER_SIZE_V1 + embedding_bytes;

    if (data.len() as u64) < embedding_end {
        return Err(ShrimPKError::Persistence(format!(
            "File truncated: need {} bytes for embeddings, file is {} bytes",
            embedding_end,
            data.len()
        )));
    }

    let embedding_slice = &data[HEADER_SIZE_V1 as usize..embedding_end as usize];
    let computed_crc = crc32fast::hash(embedding_slice);

    if computed_crc != stored_crc {
        return Err(ShrimPKError::Persistence(format!(
            "CRC32 mismatch: stored={stored_crc:#010x}, computed={computed_crc:#010x}. File may be corrupted."
        )));
    }

    // Parse flat f32 array
    let dim_usize = dim as usize;
    let count_usize = count as usize;
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(count_usize);

    for i in 0..count_usize {
        let mut emb = Vec::with_capacity(dim_usize);
        for j in 0..dim_usize {
            let offset = HEADER_SIZE_V1 as usize + (i * dim_usize + j) * 4;
            let bytes: [u8; 4] = data[offset..offset + 4]
                .try_into()
                .map_err(|_| ShrimPKError::Persistence("Failed to read f32 bytes".into()))?;
            emb.push(f32::from_le_bytes(bytes));
        }
        embeddings.push(emb);
    }

    // Read metadata
    let meta_offset = metadata_offset as usize;
    if meta_offset > data.len() {
        return Err(ShrimPKError::Persistence(format!(
            "Metadata offset {meta_offset} exceeds file size {}",
            data.len()
        )));
    }

    let meta_bytes = &data[meta_offset..];
    let metas: Vec<MemoryMeta> = serde_json::from_slice(meta_bytes)
        .map_err(|e| ShrimPKError::Persistence(format!("Failed to parse metadata: {e}")))?;

    if metas.len() != count_usize {
        return Err(ShrimPKError::Persistence(format!(
            "Metadata count mismatch: header says {count_usize}, metadata has {}",
            metas.len()
        )));
    }

    // Rebuild store — v1 entries are text-only (vision/speech = None)
    let mut store = EchoStore::new();
    for (meta, embedding) in metas.into_iter().zip(embeddings.into_iter()) {
        let entry = meta.into_entry(embedding);
        store.add(entry);
    }

    Ok(store)
}

/// Load a v2 binary store (multimodal, 16-byte header).
fn load_binary_v2(data: &[u8]) -> Result<EchoStore> {
    if data.len() < HEADER_SIZE_V2 as usize {
        return Err(ShrimPKError::Persistence(format!(
            "File too small for v2 header: {} bytes (need {})",
            data.len(),
            HEADER_SIZE_V2
        )));
    }

    // Parse v2 header
    let flags = data[5];
    let text_dim = u16::from_le_bytes(
        data[6..8]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read text_dim".into()))?,
    ) as usize;
    let vision_dim = u16::from_le_bytes(
        data[8..10]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read vision_dim".into()))?,
    ) as usize;
    let speech_dim = u16::from_le_bytes(
        data[10..12]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read speech_dim".into()))?,
    ) as usize;
    let count = u32::from_le_bytes(
        data[12..16]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read entry count".into()))?,
    ) as usize;

    let has_vision = flags & FLAG_HAS_VISION != 0;
    let has_speech = flags & FLAG_HAS_SPEECH != 0;

    let mut cursor = HEADER_SIZE_V2 as usize;

    // --- Read metadata JSON section ---
    if cursor + 4 > data.len() {
        return Err(ShrimPKError::Persistence(
            "Truncated metadata length".into(),
        ));
    }
    let meta_len = u32::from_le_bytes(
        data[cursor..cursor + 4]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read metadata length".into()))?,
    ) as usize;
    cursor += 4;

    if cursor + meta_len + 4 > data.len() {
        return Err(ShrimPKError::Persistence(
            "Truncated metadata section".into(),
        ));
    }
    let meta_bytes = &data[cursor..cursor + meta_len];
    cursor += meta_len;

    let stored_meta_crc = u32::from_le_bytes(
        data[cursor..cursor + 4]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read metadata CRC".into()))?,
    );
    cursor += 4;

    // CRC covers header_bytes + meta_json (Fix 3: header CRC coverage).
    // A corrupted entry_count or dim field is caught here before any
    // allocation based on those values.
    let header_bytes = &data[..HEADER_SIZE_V2 as usize];
    let mut meta_hasher = crc32fast::Hasher::new();
    meta_hasher.update(header_bytes);
    meta_hasher.update(meta_bytes);
    let computed_meta_crc = meta_hasher.finalize();
    if computed_meta_crc != stored_meta_crc {
        return Err(ShrimPKError::Persistence(format!(
            "Metadata CRC32 mismatch: stored={stored_meta_crc:#010x}, computed={computed_meta_crc:#010x}. File may be corrupted."
        )));
    }

    let metas: Vec<MemoryMeta> = serde_json::from_slice(meta_bytes)
        .map_err(|e| ShrimPKError::Persistence(format!("Failed to parse metadata: {e}")))?;

    if metas.len() != count {
        return Err(ShrimPKError::Persistence(format!(
            "Metadata count mismatch: header says {count}, metadata has {}",
            metas.len()
        )));
    }

    // --- Read text embeddings section + CRC ---
    let text_section_bytes = count * text_dim * 4;
    if cursor + text_section_bytes + 4 > data.len() {
        return Err(ShrimPKError::Persistence(
            "Truncated text embedding section".into(),
        ));
    }

    let text_slice = &data[cursor..cursor + text_section_bytes];
    cursor += text_section_bytes;

    let stored_text_crc = u32::from_le_bytes(
        data[cursor..cursor + 4]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read text CRC".into()))?,
    );
    cursor += 4;

    let computed_text_crc = crc32fast::hash(text_slice);
    if computed_text_crc != stored_text_crc {
        return Err(ShrimPKError::Persistence(format!(
            "Text CRC32 mismatch: stored={stored_text_crc:#010x}, computed={computed_text_crc:#010x}. File may be corrupted."
        )));
    }

    // Parse text embeddings
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(count);
    for i in 0..count {
        let mut emb = Vec::with_capacity(text_dim);
        for j in 0..text_dim {
            let off = (i * text_dim + j) * 4;
            let bytes: [u8; 4] = text_slice[off..off + 4]
                .try_into()
                .map_err(|_| ShrimPKError::Persistence("Failed to read f32 bytes".into()))?;
            emb.push(f32::from_le_bytes(bytes));
        }
        embeddings.push(emb);
    }

    // --- Read vision section ---
    let mut vision_embeddings: Vec<Option<Vec<f32>>> = vec![None; count];
    if has_vision {
        cursor = read_optional_section(
            data,
            cursor,
            count,
            vision_dim,
            &mut vision_embeddings,
            "vision",
        )?;
    }

    // --- Read speech section ---
    let mut speech_embeddings: Vec<Option<Vec<f32>>> = vec![None; count];
    if has_speech {
        cursor = read_optional_section(
            data,
            cursor,
            count,
            speech_dim,
            &mut speech_embeddings,
            "speech",
        )?;
    }

    // Suppress unused-variable warning for cursor after last section
    let _ = cursor;

    // --- Rebuild EchoStore ---
    let mut store = EchoStore::new();
    for (i, (meta, text_emb)) in metas.into_iter().zip(embeddings.into_iter()).enumerate() {
        let mut entry = meta.into_entry(text_emb);
        entry.vision_embedding = vision_embeddings[i].take();
        entry.speech_embedding = speech_embeddings[i].take();
        store.add(entry);
    }

    Ok(store)
}

/// Read an optional embedding section (vision or speech) from the data buffer.
///
/// Returns the new cursor position after reading the section.
fn read_optional_section(
    data: &[u8],
    mut cursor: usize,
    count: usize,
    dim: usize,
    out: &mut [Option<Vec<f32>>],
    label: &str,
) -> Result<usize> {
    let bitmap_len = count.div_ceil(8);

    if cursor + bitmap_len > data.len() {
        return Err(ShrimPKError::Persistence(format!(
            "Truncated {label} bitmap"
        )));
    }

    let bitmap = &data[cursor..cursor + bitmap_len];
    let present_count: usize = (0..count)
        .filter(|&i| bitmap[i / 8] & (1 << (i % 8)) != 0)
        .count();

    // CRC covers bitmap + embedding bytes
    let emb_bytes = present_count * dim * 4;
    let section_bytes = bitmap_len + emb_bytes;

    if cursor + section_bytes + 4 > data.len() {
        return Err(ShrimPKError::Persistence(format!(
            "Truncated {label} embedding section"
        )));
    }

    let section_slice = &data[cursor..cursor + section_bytes];
    cursor += section_bytes;

    let stored_crc = u32::from_le_bytes(
        data[cursor..cursor + 4]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence(format!("Failed to read {label} CRC")))?,
    );
    cursor += 4;

    let computed_crc = crc32fast::hash(section_slice);
    if computed_crc != stored_crc {
        return Err(ShrimPKError::Persistence(format!(
            "{label} CRC32 mismatch: stored={stored_crc:#010x}, computed={computed_crc:#010x}. File may be corrupted."
        )));
    }

    // Parse embeddings from bitmap_len offset into the section slice
    let emb_data = &section_slice[bitmap_len..];
    let mut emb_cursor = 0usize;
    for i in 0..count {
        if bitmap[i / 8] & (1 << (i % 8)) != 0 {
            let mut emb = Vec::with_capacity(dim);
            for j in 0..dim {
                let off = (emb_cursor * dim + j) * 4;
                let bytes: [u8; 4] = emb_data[off..off + 4].try_into().map_err(|_| {
                    ShrimPKError::Persistence(format!("Failed to read {label} f32 bytes"))
                })?;
                emb.push(f32::from_le_bytes(bytes));
            }
            out[i] = Some(emb);
            emb_cursor += 1;
        }
    }

    Ok(cursor)
}

/// Save community summaries to a sidecar JSON file (KS64).
///
/// Stored alongside the SHRM binary as `community_summaries.json`.
/// Avoids touching the binary format — easier to debug and version independently.
pub fn save_community_summaries(store: &EchoStore, data_dir: &Path) -> Result<()> {
    let summaries = store.all_summaries();
    if summaries.is_empty() {
        return Ok(());
    }
    let path = data_dir.join("community_summaries.json");
    let json = serde_json::to_string_pretty(summaries).map_err(|e| {
        ShrimPKError::Persistence(format!("Failed to serialize community summaries: {e}"))
    })?;
    std::fs::write(&path, json).map_err(|e| {
        ShrimPKError::Persistence(format!("Failed to write {}: {e}", path.display()))
    })?;
    tracing::debug!(path = %path.display(), count = summaries.len(), "Community summaries saved");
    Ok(())
}

/// Load community summaries from the sidecar JSON file (KS64).
///
/// Returns an empty map if the file does not exist.
pub fn load_community_summaries(store: &mut EchoStore, data_dir: &Path) -> Result<()> {
    let path = data_dir.join("community_summaries.json");
    if !path.exists() {
        return Ok(());
    }
    let json = std::fs::read_to_string(&path).map_err(|e| {
        ShrimPKError::Persistence(format!("Failed to read {}: {e}", path.display()))
    })?;
    let summaries: std::collections::HashMap<String, shrimpk_core::CommunitySummary> =
        serde_json::from_str(&json).map_err(|e| {
            ShrimPKError::Persistence(format!("Failed to parse community summaries: {e}"))
        })?;
    let count = summaries.len();
    *store.summaries_mut() = summaries;
    tracing::debug!(path = %path.display(), count, "Community summaries loaded");
    Ok(())
}

/// Validate a binary store file without loading all data.
///
/// Checks magic bytes, version, and CRC32 checksum(s).
/// Returns `Ok(true)` if valid, `Ok(false)` if not a valid binary store,
/// or an error on I/O failure.
/// Supports both v1 and v2 formats.
#[instrument(fields(path = %path.display()))]
pub fn validate_binary(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let data = std::fs::read(path).map_err(|e| {
        ShrimPKError::Persistence(format!("Failed to read file for validation: {e}"))
    })?;

    if data.len() < 5 {
        return Ok(false);
    }

    // Check magic
    if &data[0..4] != MAGIC {
        return Ok(false);
    }

    let version_byte = data[4];

    // v1 validation
    if version_byte == 1 && data.len() >= 8 && data[5] == 0 && data[6] == 0 && data[7] == 0 {
        if data.len() < HEADER_SIZE_V1 as usize {
            return Ok(false);
        }

        let (_magic, _version, dim, count, _metadata_offset, stored_crc) =
            match parse_header_v1(&data) {
                Ok(h) => h,
                Err(_) => return Ok(false),
            };

        let embedding_bytes = count * (dim as u64) * 4;
        let embedding_end = HEADER_SIZE_V1 + embedding_bytes;

        if (data.len() as u64) < embedding_end {
            return Ok(false);
        }

        let embedding_slice = &data[HEADER_SIZE_V1 as usize..embedding_end as usize];
        let computed_crc = crc32fast::hash(embedding_slice);

        return Ok(computed_crc == stored_crc);
    }

    // v2 validation — try a full load (which verifies all CRCs)
    if version_byte == 2 {
        return Ok(load_binary_v2(&data).is_ok());
    }

    // Unknown version
    Ok(false)
}

// --- Internal helpers ---

/// Parse the 64-byte v1 header, returning (magic, version, dim, count, metadata_offset, crc).
fn parse_header_v1(data: &[u8]) -> Result<([u8; 4], u32, u32, u64, u64, u32)> {
    if data.len() < HEADER_SIZE_V1 as usize {
        return Err(ShrimPKError::Persistence("Header too short".into()));
    }

    let magic: [u8; 4] = data[0..4]
        .try_into()
        .map_err(|_| ShrimPKError::Persistence("Failed to read magic".into()))?;

    let version = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read version".into()))?,
    );

    let dim = u32::from_le_bytes(
        data[8..12]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read dim".into()))?,
    );

    // Skip reserved/padding at 12..16

    let count = u64::from_le_bytes(
        data[16..24]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read count".into()))?,
    );

    let metadata_offset = u64::from_le_bytes(
        data[24..32]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read metadata offset".into()))?,
    );

    let crc = u32::from_le_bytes(
        data[32..36]
            .try_into()
            .map_err(|_| ShrimPKError::Persistence("Failed to read CRC".into()))?,
    );

    Ok((magic, version, dim, count, metadata_offset, crc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shrimpk_core::SensitivityLevel;
    use tempfile::tempdir;

    fn make_entry(content: &str, embedding: Vec<f32>) -> MemoryEntry {
        MemoryEntry::new(content.to_string(), embedding, "test".to_string())
    }

    fn make_entry_with_meta(content: &str, embedding: Vec<f32>) -> MemoryEntry {
        let mut entry = MemoryEntry::new(content.to_string(), embedding, "test-source".to_string());
        entry.sensitivity = SensitivityLevel::Private;
        entry.masked_content = Some(format!("[MASKED] {content}"));
        entry.echo_count = 5;
        entry
    }

    #[test]
    fn binary_save_load_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let e1 = make_entry("hello world", vec![1.0, 2.0, 3.0]);
        let e2 = make_entry("goodbye world", vec![4.0, 5.0, 6.0]);
        let id1 = e1.id.clone();
        let id2 = e2.id.clone();
        store.add(e1);
        store.add(e2);

        save_binary(&store, &path).expect("save should succeed");
        let loaded = load_binary(&path).expect("load should succeed");

        assert_eq!(loaded.len(), 2);

        let r1 = loaded.get(&id1).expect("should find entry 1");
        assert_eq!(r1.content, "hello world");
        assert_eq!(r1.embedding, vec![1.0, 2.0, 3.0]);

        let r2 = loaded.get(&id2).expect("should find entry 2");
        assert_eq!(r2.content, "goodbye world");
        assert_eq!(r2.embedding, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn binary_roundtrip_preserves_metadata() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let entry = make_entry_with_meta("sensitive data", vec![0.1, 0.2, 0.3]);
        let id = entry.id.clone();
        store.add(entry);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        let r = loaded.get(&id).expect("should find entry");
        assert_eq!(r.content, "sensitive data");
        assert_eq!(r.masked_content.as_deref(), Some("[MASKED] sensitive data"));
        assert_eq!(r.source, "test-source");
        assert_eq!(r.sensitivity, SensitivityLevel::Private);
        assert_eq!(r.echo_count, 5);
    }

    #[test]
    fn crc_detects_corruption() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("test", vec![1.0, 2.0, 3.0, 4.0]));
        save_binary(&store, &path).expect("save");

        // In v2 format, text embeddings follow the metadata section.
        // Find the text embedding region: skip header(16) + meta_len(4) + meta_json + meta_crc(4).
        let mut data = std::fs::read(&path).expect("read file");
        let meta_len = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let text_emb_start = 16 + 4 + meta_len + 4; // header + len + json + crc
        data[text_emb_start] ^= 0xFF; // flip bits in first embedding byte
        std::fs::write(&path, &data).expect("write corrupted");

        let result = load_binary(&path);
        assert!(result.is_err(), "Should detect corruption");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("CRC32 mismatch"),
            "Error should mention CRC: {err_msg}"
        );
    }

    #[test]
    fn crc_validation_catches_corruption() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("test", vec![1.0, 2.0, 3.0]));
        save_binary(&store, &path).expect("save");

        // File should validate clean
        assert!(
            validate_binary(&path).expect("validate"),
            "Clean file should be valid"
        );

        // Corrupt an embedding byte in the text section
        let mut data = std::fs::read(&path).expect("read");
        let meta_len = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let text_emb_start = 16 + 4 + meta_len + 4;
        data[text_emb_start + 4] ^= 0xFF;
        std::fs::write(&path, &data).expect("write corrupted");

        assert!(
            !validate_binary(&path).expect("validate"),
            "Corrupted file should be invalid"
        );
    }

    #[test]
    fn empty_store_save_load() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let store = EchoStore::new();
        save_binary(&store, &path).expect("save empty");

        let loaded = load_binary(&path).expect("load empty");
        assert!(loaded.is_empty());
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let path = Path::new("/tmp/nonexistent_binary_store_99999.shrm");
        let store = load_binary(path).expect("should return empty store");
        assert!(store.is_empty());
    }

    #[test]
    fn header_validation_wrong_magic() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("test", vec![1.0, 2.0, 3.0]));
        save_binary(&store, &path).expect("save");

        // Overwrite magic bytes
        let mut data = std::fs::read(&path).expect("read");
        data[0..4].copy_from_slice(b"XXXX");
        std::fs::write(&path, &data).expect("write");

        let result = load_binary(&path);
        assert!(result.is_err(), "Should reject wrong magic");
        assert!(result.unwrap_err().to_string().contains("Invalid magic"));
    }

    #[test]
    fn header_validation_wrong_version() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("test", vec![1.0, 2.0, 3.0]));
        save_binary(&store, &path).expect("save");

        // Overwrite version to 99
        let mut data = std::fs::read(&path).expect("read");
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        std::fs::write(&path, &data).expect("write");

        let result = load_binary(&path);
        assert!(result.is_err(), "Should reject wrong version");
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported format version")
        );
    }

    #[test]
    fn validate_nonexistent_returns_false() {
        let path = Path::new("/tmp/nope_not_here_8888.shrm");
        assert!(!validate_binary(path).expect("should not error"));
    }

    #[test]
    fn validate_too_small_returns_false() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("tiny.shrm");
        std::fs::write(&path, b"too small").expect("write");
        assert!(!validate_binary(&path).expect("should not error"));
    }

    #[test]
    fn migration_json_to_binary_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let json_path = dir.path().join("store.json");
        let bin_path = dir.path().join("store.shrm");

        // Create and save as JSON (the old format)
        let mut store = EchoStore::new();
        let entry = make_entry("legacy data", vec![7.0, 8.0, 9.0]);
        let id = entry.id.clone();
        store.add(entry);
        store.save_json(&json_path).expect("save json");

        // Load from JSON (old path)
        let loaded_json = EchoStore::load_json(&json_path).expect("load json");
        assert_eq!(loaded_json.len(), 1);
        assert_eq!(loaded_json.get(&id).unwrap().content, "legacy data");

        // Save as binary (new path)
        save_binary(&loaded_json, &bin_path).expect("save binary");

        // Load from binary
        let loaded_bin = load_binary(&bin_path).expect("load binary");
        assert_eq!(loaded_bin.len(), 1);
        let r = loaded_bin.get(&id).expect("should find entry");
        assert_eq!(r.content, "legacy data");
        assert_eq!(r.embedding, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn binary_large_embedding_dim() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        // Use 384-dim like the real model
        let dim = 384;
        let embedding: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();

        let mut store = EchoStore::new();
        let entry = make_entry("high-dim embedding", embedding.clone());
        let id = entry.id.clone();
        store.add(entry);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        let r = loaded.get(&id).expect("should find entry");
        assert_eq!(r.embedding.len(), dim);
        assert_eq!(r.embedding, embedding);
    }

    #[test]
    fn binary_multiple_entries_ordering() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let mut ids = Vec::new();
        for i in 0..10 {
            let entry = make_entry(
                &format!("entry {i}"),
                vec![i as f32, (i as f32) * 2.0, (i as f32) * 3.0],
            );
            ids.push(entry.id.clone());
            store.add(entry);
        }

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.len(), 10);
        for (i, id) in ids.iter().enumerate() {
            let r = loaded.get(id).expect("should find entry");
            assert_eq!(r.content, format!("entry {i}"));
            assert_eq!(r.embedding[0], i as f32);
        }
    }

    #[ignore = "requires fastembed model"]
    #[test]
    fn binary_1000_memories_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let dim = 384;
        let mut store = EchoStore::new();
        let mut ids = Vec::new();

        for i in 0..1000 {
            let embedding: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.0001).collect();
            let entry = make_entry(&format!("memory {i}"), embedding);
            ids.push(entry.id.clone());
            store.add(entry);
        }

        save_binary(&store, &path).expect("save 1000");
        let loaded = load_binary(&path).expect("load 1000");

        assert_eq!(loaded.len(), 1000);
        for id in &ids {
            let orig = store.get(id).unwrap();
            let loaded_entry = loaded.get(id).unwrap();
            assert_eq!(orig.content, loaded_entry.content);
            assert_eq!(orig.embedding, loaded_entry.embedding);
        }
    }

    #[ignore = "requires fastembed model — benchmark"]
    #[test]
    fn benchmark_binary_vs_json() {
        let dir = tempfile::tempdir().expect("temp dir");
        let bin_path = dir.path().join("store.shrm");
        let json_path = dir.path().join("store.json");

        let dim = 384;
        let mut store = EchoStore::new();
        for i in 0..1000 {
            let embedding: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.0001).collect();
            store.add(make_entry(&format!("memory {i}"), embedding));
        }

        // Binary save
        let start = std::time::Instant::now();
        save_binary(&store, &bin_path).expect("save binary");
        let bin_save_ms = start.elapsed().as_millis();

        // JSON save
        let start = std::time::Instant::now();
        store.save_json(&json_path).expect("save json");
        let json_save_ms = start.elapsed().as_millis();

        // Binary load
        let start = std::time::Instant::now();
        let _ = load_binary(&bin_path).expect("load binary");
        let bin_load_ms = start.elapsed().as_millis();

        // JSON load
        let start = std::time::Instant::now();
        let _ = EchoStore::load_json(&json_path).expect("load json");
        let json_load_ms = start.elapsed().as_millis();

        let bin_size = std::fs::metadata(&bin_path).unwrap().len();
        let json_size = std::fs::metadata(&json_path).unwrap().len();

        eprintln!("=== Binary vs JSON (1000 entries, dim={dim}) ===");
        eprintln!("Binary: save={bin_save_ms}ms, load={bin_load_ms}ms, size={bin_size} bytes");
        eprintln!("JSON:   save={json_save_ms}ms, load={json_load_ms}ms, size={json_size} bytes");
    }

    // ========================================================================
    // KS37: SHRM v2 multimodal persistence tests (15 new)
    // ========================================================================

    /// Helper: create entry with vision embedding.
    fn make_vision_entry(content: &str, text_emb: Vec<f32>, vision_emb: Vec<f32>) -> MemoryEntry {
        let mut entry = MemoryEntry::new_with_modality(
            content.into(),
            text_emb,
            "test".into(),
            Modality::Vision,
        );
        entry.vision_embedding = Some(vision_emb);
        entry
    }

    /// Helper: create entry with speech embedding.
    fn make_speech_entry(content: &str, text_emb: Vec<f32>, speech_emb: Vec<f32>) -> MemoryEntry {
        let mut entry = MemoryEntry::new_with_modality(
            content.into(),
            text_emb,
            "test".into(),
            Modality::Speech,
        );
        entry.speech_embedding = Some(speech_emb);
        entry
    }

    // 1. v2 header write/read roundtrip
    #[test]
    fn v2_header_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("hello", vec![1.0, 2.0, 3.0]));
        save_binary(&store, &path).expect("save");

        let data = std::fs::read(&path).expect("read");
        // Magic
        assert_eq!(&data[0..4], b"SHRM");
        // Version = 2
        assert_eq!(data[4], 2);
        // Flags = 0 (no vision, no speech)
        assert_eq!(data[5], 0);
        // Text dim = 3
        assert_eq!(u16::from_le_bytes(data[6..8].try_into().unwrap()), 3);
        // Vision dim = 0
        assert_eq!(u16::from_le_bytes(data[8..10].try_into().unwrap()), 0);
        // Speech dim = 0
        assert_eq!(u16::from_le_bytes(data[10..12].try_into().unwrap()), 0);
        // Entry count = 1
        assert_eq!(u32::from_le_bytes(data[12..16].try_into().unwrap()), 1);
    }

    // 2. v2 save/load text-only (no vision/speech)
    #[test]
    fn v2_text_only_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let e1 = make_entry("alpha", vec![1.0, 2.0, 3.0]);
        let e2 = make_entry("beta", vec![4.0, 5.0, 6.0]);
        let id1 = e1.id.clone();
        let id2 = e2.id.clone();
        store.add(e1);
        store.add(e2);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.len(), 2);
        let r1 = loaded.get(&id1).unwrap();
        assert_eq!(r1.embedding, vec![1.0, 2.0, 3.0]);
        assert!(r1.vision_embedding.is_none());
        assert!(r1.speech_embedding.is_none());

        let r2 = loaded.get(&id2).unwrap();
        assert_eq!(r2.embedding, vec![4.0, 5.0, 6.0]);
    }

    // 3. v2 save/load with vision embeddings (5 entries, 3 with vision)
    #[test]
    fn v2_vision_partial_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let mut ids = Vec::new();

        // 5 entries, indices 0,2,4 have vision
        for i in 0..5u32 {
            let text_emb = vec![i as f32; 3];
            if i % 2 == 0 {
                let vis_emb = vec![(i as f32) * 10.0; 4];
                let entry = make_vision_entry(&format!("vis-{i}"), text_emb, vis_emb);
                ids.push(entry.id.clone());
                store.add(entry);
            } else {
                let entry = make_entry(&format!("txt-{i}"), text_emb);
                ids.push(entry.id.clone());
                store.add(entry);
            }
        }

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.len(), 5);
        for (i, id) in ids.iter().enumerate() {
            let r = loaded.get(id).unwrap();
            if i % 2 == 0 {
                let expected_vis = vec![(i as f32) * 10.0; 4];
                assert_eq!(r.vision_embedding, Some(expected_vis), "entry {i} vision");
            } else {
                assert!(
                    r.vision_embedding.is_none(),
                    "entry {i} should have no vision"
                );
            }
        }
    }

    // 4. v2 save/load with speech embeddings
    #[test]
    fn v2_speech_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let speech_emb = vec![0.5; 5];
        let entry = make_speech_entry("audio note", vec![1.0, 2.0, 3.0], speech_emb.clone());
        let id = entry.id.clone();
        store.add(entry);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        let r = loaded.get(&id).unwrap();
        assert_eq!(r.speech_embedding, Some(speech_emb));
        assert!(r.vision_embedding.is_none());
    }

    // 5. v2 save/load mixed (text + vision + speech)
    #[test]
    fn v2_mixed_modalities_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();

        // Entry with both vision and speech
        let mut e1 = make_entry("multimodal", vec![1.0, 2.0]);
        e1.vision_embedding = Some(vec![10.0, 20.0, 30.0]);
        e1.speech_embedding = Some(vec![100.0, 200.0]);
        let id1 = e1.id.clone();

        // Entry with only text
        let e2 = make_entry("text only", vec![3.0, 4.0]);
        let id2 = e2.id.clone();

        // Entry with only vision
        let mut e3 = make_entry("vis only", vec![5.0, 6.0]);
        e3.vision_embedding = Some(vec![50.0, 60.0, 70.0]);
        let id3 = e3.id.clone();

        store.add(e1);
        store.add(e2);
        store.add(e3);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.len(), 3);

        let r1 = loaded.get(&id1).unwrap();
        assert_eq!(r1.vision_embedding, Some(vec![10.0, 20.0, 30.0]));
        assert_eq!(r1.speech_embedding, Some(vec![100.0, 200.0]));

        let r2 = loaded.get(&id2).unwrap();
        assert!(r2.vision_embedding.is_none());
        assert!(r2.speech_embedding.is_none());

        let r3 = loaded.get(&id3).unwrap();
        assert_eq!(r3.vision_embedding, Some(vec![50.0, 60.0, 70.0]));
        assert!(r3.speech_embedding.is_none());
    }

    // 6. v1 file loads in v2 code correctly (backward compat)
    #[test]
    fn v1_backward_compat_loads_in_v2_code() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        // Manually write a v1 file
        let mut store = EchoStore::new();
        let entry = make_entry("legacy v1", vec![7.0, 8.0, 9.0]);
        let id = entry.id.clone();
        store.add(entry);

        // Write v1 format manually
        write_v1_file(&store, &path);

        // Load with v2 code
        let loaded = load_binary(&path).expect("v1 file should load in v2 code");
        assert_eq!(loaded.len(), 1);
        let r = loaded.get(&id).unwrap();
        assert_eq!(r.content, "legacy v1");
        assert_eq!(r.embedding, vec![7.0, 8.0, 9.0]);
        assert!(r.vision_embedding.is_none());
        assert!(r.speech_embedding.is_none());
        assert_eq!(r.modality, Modality::Text);
    }

    /// Helper: write a v1-format file for backward compat testing.
    fn write_v1_file(store: &EchoStore, path: &Path) {
        use std::io::{Seek, SeekFrom, Write};

        let entries = store.all_entries();
        let embeddings = store.all_embeddings();
        let count = entries.len() as u64;
        let dim: u32 = embeddings.first().map(|e| e.len() as u32).unwrap_or(384);

        let mut file = std::fs::File::create(path).unwrap();
        let header_placeholder = [0u8; 64];
        file.write_all(&header_placeholder).unwrap();

        let mut hasher = crc32fast::Hasher::new();
        for emb in embeddings {
            for &val in emb {
                let bytes = val.to_le_bytes();
                file.write_all(&bytes).unwrap();
                hasher.update(&bytes);
            }
        }
        let crc = hasher.finalize();
        let metadata_offset = file.stream_position().unwrap();

        // Write metadata (v1-style: no modality/has_vision/has_speech fields)
        let metas: Vec<MemoryMeta> = entries.iter().map(MemoryMeta::from_entry).collect();
        let meta_json = serde_json::to_vec(&metas).unwrap();
        file.write_all(&meta_json).unwrap();

        // Write real header
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(MAGIC).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap(); // version 1
        file.write_all(&dim.to_le_bytes()).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap(); // reserved
        file.write_all(&count.to_le_bytes()).unwrap();
        file.write_all(&metadata_offset.to_le_bytes()).unwrap();
        file.write_all(&crc.to_le_bytes()).unwrap();

        file.flush().unwrap();
        file.sync_all().unwrap();
    }

    // 7. v2 file detected by version check (version == 2)
    #[test]
    fn v2_version_byte_is_2() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("x", vec![1.0]));
        save_binary(&store, &path).expect("save");

        let data = std::fs::read(&path).expect("read");
        assert_eq!(data[4], 2, "Version byte should be 2");
        // Byte 5 should NOT be 0,0,0 (which would look like v1 u32)
        // Actually for text-only flags=0 so byte[5]=0. But byte[6..8] is text_dim,
        // which for dim=1 would be [1,0] — same as v1 dim. But the key difference is
        // v1 has version as u32 at [4..8] = [1,0,0,0]. v2 has [2, flags, dim_lo, dim_hi].
        // So data[4]==2 distinguishes it.
    }

    // 8. CRC corruption in text section -> error
    #[test]
    fn v2_crc_corruption_text_section() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("test", vec![1.0, 2.0, 3.0]));
        save_binary(&store, &path).expect("save");

        let mut data = std::fs::read(&path).expect("read");
        let meta_len = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let text_start = 16 + 4 + meta_len + 4;
        data[text_start] ^= 0xFF;
        std::fs::write(&path, &data).expect("write corrupted");

        let result = load_binary(&path);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Text CRC32 mismatch")
        );
    }

    // 9. CRC corruption in vision section -> error
    #[test]
    fn v2_crc_corruption_vision_section() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let entry = make_vision_entry("vis", vec![1.0, 2.0], vec![10.0, 20.0, 30.0]);
        store.add(entry);
        save_binary(&store, &path).expect("save");

        let mut data = std::fs::read(&path).expect("read");
        // Find vision section: after header + meta + text section
        let meta_len = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let text_start = 16 + 4 + meta_len + 4;
        #[allow(clippy::identity_op)]
        let text_bytes = 1 * 2 * 4; // 1 entry * 2 dim * 4 bytes
        let vision_start = text_start + text_bytes + 4; // +4 for text CRC
        // Corrupt the bitmap byte of the vision section
        data[vision_start] ^= 0xFF;
        std::fs::write(&path, &data).expect("write corrupted");

        let result = load_binary(&path);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("vision CRC32 mismatch")
        );
    }

    // 10. Bitmap correctness: 100 entries, 30 with vision
    #[test]
    fn v2_bitmap_100_entries_30_vision() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let mut ids = Vec::new();
        let mut expected_vision: Vec<bool> = Vec::new();

        for i in 0..100u32 {
            let text_emb = vec![i as f32; 3];
            // Every 3rd entry (and a few extras) gets vision — gives ~33 entries
            // Let's use i % 10 < 3 for exactly 30
            if i % 10 < 3 {
                let vis_emb = vec![(i as f32) * 0.1; 4];
                let entry = make_vision_entry(&format!("e{i}"), text_emb, vis_emb);
                ids.push(entry.id.clone());
                store.add(entry);
                expected_vision.push(true);
            } else {
                let entry = make_entry(&format!("e{i}"), text_emb);
                ids.push(entry.id.clone());
                store.add(entry);
                expected_vision.push(false);
            }
        }

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.len(), 100);
        let mut vision_count = 0;
        for (i, id) in ids.iter().enumerate() {
            let r = loaded.get(id).unwrap();
            if expected_vision[i] {
                assert!(r.vision_embedding.is_some(), "entry {i} should have vision");
                vision_count += 1;
            } else {
                assert!(
                    r.vision_embedding.is_none(),
                    "entry {i} should NOT have vision"
                );
            }
        }
        assert_eq!(vision_count, 30);
    }

    // 11. Empty store save/load roundtrip (0 entries) — v2 specific
    #[test]
    fn v2_empty_store_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let store = EchoStore::new();
        save_binary(&store, &path).expect("save empty");

        let loaded = load_binary(&path).expect("load empty");
        assert!(loaded.is_empty());
        assert_eq!(loaded.len(), 0);

        // Verify it's a valid v2 file
        let data = std::fs::read(&path).expect("read");
        assert_eq!(data[4], 2, "Version should be 2");
    }

    // 12. Single entry roundtrip — v2
    #[test]
    fn v2_single_entry_roundtrip() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let entry = make_entry("sole memory", vec![42.0]);
        let id = entry.id.clone();
        store.add(entry);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.len(), 1);
        let r = loaded.get(&id).unwrap();
        assert_eq!(r.content, "sole memory");
        assert_eq!(r.embedding, vec![42.0]);
    }

    // 13. Modality preserved through save/load
    #[test]
    fn v2_modality_preserved() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();

        let e_text = make_entry("text", vec![1.0]);
        let id_text = e_text.id.clone();

        let e_vis = make_vision_entry("vision", vec![2.0], vec![20.0, 30.0]);
        let id_vis = e_vis.id.clone();

        let e_speech = make_speech_entry("speech", vec![3.0], vec![30.0, 40.0]);
        let id_speech = e_speech.id.clone();

        store.add(e_text);
        store.add(e_vis);
        store.add(e_speech);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.get(&id_text).unwrap().modality, Modality::Text);
        assert_eq!(loaded.get(&id_vis).unwrap().modality, Modality::Vision);
        assert_eq!(loaded.get(&id_speech).unwrap().modality, Modality::Speech);
    }

    // 14. Large batch: 1000 entries mixed modalities roundtrip
    #[test]
    fn v2_large_batch_mixed_1000() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let text_dim = 8;
        let vis_dim = 4;
        let speech_dim = 3;

        let mut store = EchoStore::new();
        let mut ids = Vec::new();

        for i in 0..1000u32 {
            let text_emb: Vec<f32> = (0..text_dim).map(|j| (i * 10 + j) as f32).collect();
            let mut entry = make_entry(&format!("m{i}"), text_emb);

            if i % 5 == 0 {
                entry.vision_embedding = Some((0..vis_dim).map(|j| (i * 100 + j) as f32).collect());
                entry.modality = Modality::Vision;
            }
            if i % 7 == 0 {
                entry.speech_embedding =
                    Some((0..speech_dim).map(|j| (i * 1000 + j) as f32).collect());
                if entry.modality == Modality::Text {
                    entry.modality = Modality::Speech;
                }
            }

            ids.push(entry.id.clone());
            store.add(entry);
        }

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert_eq!(loaded.len(), 1000);

        for (i, id) in ids.iter().enumerate() {
            let orig = store.get(id).unwrap();
            let r = loaded.get(id).unwrap();
            assert_eq!(orig.content, r.content, "content mismatch at {i}");
            assert_eq!(orig.embedding, r.embedding, "text emb mismatch at {i}");
            assert_eq!(
                orig.vision_embedding, r.vision_embedding,
                "vis emb mismatch at {i}"
            );
            assert_eq!(
                orig.speech_embedding, r.speech_embedding,
                "speech emb mismatch at {i}"
            );
            assert_eq!(orig.modality, r.modality, "modality mismatch at {i}");
        }
    }

    // 15. vision_embedding=None entries have no vision in loaded data
    #[test]
    fn v2_none_vision_stays_none() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();

        // One with vision, two without
        let e1 = make_vision_entry("with vision", vec![1.0], vec![10.0, 20.0]);
        let e2 = make_entry("no vision 1", vec![2.0]);
        let e3 = make_entry("no vision 2", vec![3.0]);
        let id1 = e1.id.clone();
        let id2 = e2.id.clone();
        let id3 = e3.id.clone();

        store.add(e1);
        store.add(e2);
        store.add(e3);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        assert!(loaded.get(&id1).unwrap().vision_embedding.is_some());
        assert!(
            loaded.get(&id2).unwrap().vision_embedding.is_none(),
            "Should not have vision"
        );
        assert!(
            loaded.get(&id3).unwrap().vision_embedding.is_none(),
            "Should not have vision"
        );
    }

    // Fix 3: Header CRC coverage — corrupted entry_count detected before allocation.
    #[test]
    fn header_corruption_detected_by_crc() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        store.add(make_entry("hello", vec![1.0, 2.0, 3.0]));
        save_binary(&store, &path).expect("save");

        // Corrupt byte 12 (first byte of entry_count in v2 header).
        let mut data = std::fs::read(&path).expect("read file");
        assert!(data.len() > 16, "v2 file must have at least 16-byte header");
        data[12] ^= 0xFF; // flip bits in entry_count
        std::fs::write(&path, &data).expect("write corrupted file");

        let result = load_binary(&path);
        assert!(
            result.is_err(),
            "Corrupted entry_count should be caught by metadata CRC"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("CRC32 mismatch") || err_msg.contains("corrupted"),
            "Error should mention CRC mismatch, got: {err_msg}"
        );
    }

    // --- Label persistence roundtrip (KS42, ADR-015) ---

    #[test]
    fn binary_roundtrip_preserves_labels() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let mut entry = make_entry("I am learning Japanese", vec![0.1, 0.2, 0.3]);
        entry.labels = vec![
            "topic:language".into(),
            "action:learning".into(),
            "entity:japanese".into(),
        ];
        entry.label_version = 1;
        let id = entry.id.clone();
        store.add(entry);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        let r = loaded.get(&id).expect("should find entry");
        assert_eq!(
            r.labels,
            vec!["topic:language", "action:learning", "entity:japanese"],
            "Labels must survive SHRM binary roundtrip"
        );
        assert_eq!(r.label_version, 1, "label_version must survive roundtrip");
    }

    #[test]
    fn binary_roundtrip_legacy_no_labels() {
        // Simulate loading a store where entries were saved without label fields.
        // The serde(default) on MemoryMeta should produce empty labels + version 0.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("store.shrm");

        let mut store = EchoStore::new();
        let entry = make_entry("legacy entry", vec![0.5, 0.6, 0.7]);
        let id = entry.id.clone();
        // labels and label_version are at their defaults (empty, 0)
        assert!(entry.labels.is_empty());
        assert_eq!(entry.label_version, 0);
        store.add(entry);

        save_binary(&store, &path).expect("save");
        let loaded = load_binary(&path).expect("load");

        let r = loaded.get(&id).expect("should find entry");
        assert!(
            r.labels.is_empty(),
            "Legacy entry should load with empty labels"
        );
        assert_eq!(
            r.label_version, 0,
            "Legacy entry should load with label_version 0"
        );
    }

    // --- KS64: community summary persistence tests ---

    #[test]
    fn community_summary_save_load_roundtrip() {
        let dir = tempdir().unwrap();
        let mut store = EchoStore::new();

        let summary = shrimpk_core::CommunitySummary {
            label: "career".to_string(),
            summary: "User works as a Rust engineer at a startup.".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            member_count: 8,
            updated_at: chrono::Utc::now(),
        };
        store.set_summary(summary);

        save_community_summaries(&store, dir.path()).expect("save");

        let mut loaded_store = EchoStore::new();
        load_community_summaries(&mut loaded_store, dir.path()).expect("load");

        let s = loaded_store
            .get_summary("career")
            .expect("should find summary after load");
        assert_eq!(s.member_count, 8);
        assert!(s.summary.contains("Rust"));
    }

    #[test]
    fn community_summary_load_missing_file() {
        let dir = tempdir().unwrap();
        let mut store = EchoStore::new();
        // Should succeed with no summaries loaded
        load_community_summaries(&mut store, dir.path()).expect("load empty");
        assert!(store.all_summaries().is_empty());
    }
}
