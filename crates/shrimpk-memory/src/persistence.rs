//! Binary persistence for Echo Memory.
//!
//! Replaces JSON persistence with a structured binary format for performance:
//! - Fixed 64-byte header with magic, version, dimensions, entry count
//! - Flat contiguous f32 embedding array (cache-friendly, mmap-ready)
//! - CRC32 checksum over embedding data for corruption detection
//! - JSON-serialized metadata appended after embeddings
//!
//! ## Binary Format Layout
//! ```text
//! Offset  Size    Field
//! 0       4       Magic: b"SHRM"
//! 4       4       Version: u32 (1)
//! 8       4       Embedding dim: u32 (384)
//! 12      4       Reserved/padding: u32 (0)
//! 16      8       Entry count: u64
//! 24      8       Metadata offset: u64 (byte position of metadata section)
//! 32      4       CRC32 checksum of embedding array
//! 36      28      Reserved (total header = 64 bytes)
//! 64      N*D*4   Embedding array: f32[count][dim] (flat, contiguous)
//! 64+N*D*4  M     Metadata: JSON-serialized Vec<MemoryMeta>
//! ```

use chrono::{DateTime, Utc};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use shrimpk_core::{MemoryCategory, MemoryEntry, MemoryId, Result, SensitivityLevel, ShrimPKError};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use tracing::instrument;

use crate::store::EchoStore;

// --- Constants ---

/// Magic bytes identifying a ShrimPK Echo Memory binary file.
const MAGIC: &[u8; 4] = b"SHRM";

/// Current binary format version.
const FORMAT_VERSION: u32 = 1;

/// Total header size in bytes.
const HEADER_SIZE: u64 = 64;

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
}

impl MemoryMeta {
    /// Extract metadata from a MemoryEntry (strips the embedding).
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
        }
    }

    /// Reconstruct a MemoryEntry by combining metadata with its embedding.
    fn into_entry(self, embedding: Vec<f32>) -> MemoryEntry {
        MemoryEntry {
            id: self.id,
            content: self.content,
            masked_content: self.masked_content,
            reformulated: self.reformulated,
            embedding,
            source: self.source,
            sensitivity: self.sensitivity,
            category: self.category,
            created_at: self.created_at,
            last_echoed: self.last_echoed,
            echo_count: self.echo_count,
        }
    }
}

// --- Public API ---

/// Save an EchoStore to a binary file.
///
/// Writes a 64-byte header, then a flat f32 embedding array, then JSON metadata.
/// The CRC32 checksum is computed over the embedding array and written into the header.
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
    let count = entries.len() as u64;

    // Determine embedding dimension from first entry, or use default 384
    let dim: u32 = embeddings.first().map(|e| e.len() as u32).unwrap_or(384);

    // Atomic write: write to .tmp file, then rename (F-06 fix)
    let tmp_path = path.with_extension("shrm.tmp");
    let mut file = std::fs::File::create(&tmp_path)?;

    // --- Write header (64 bytes) ---
    // We'll write a placeholder header first, then come back to fill CRC and metadata offset.
    let header_placeholder = [0u8; HEADER_SIZE as usize];
    file.write_all(&header_placeholder)?;

    // --- Write embedding array ---
    let mut hasher = crc32fast::Hasher::new();

    for emb in embeddings {
        for &val in emb {
            let bytes = val.to_le_bytes();
            file.write_all(&bytes)?;
            hasher.update(&bytes);
        }
    }

    let crc = hasher.finalize();
    let metadata_offset = file
        .stream_position()
        .map_err(|e| ShrimPKError::Persistence(format!("Failed to get stream position: {e}")))?;

    // --- Write metadata as JSON ---
    let metas: Vec<MemoryMeta> = entries.iter().map(MemoryMeta::from_entry).collect();
    let meta_json = serde_json::to_vec(&metas)?;
    file.write_all(&meta_json)?;

    // --- Go back and write the real header ---
    file.seek(SeekFrom::Start(0))
        .map_err(|e| ShrimPKError::Persistence(format!("Failed to seek: {e}")))?;

    // Magic (4 bytes)
    file.write_all(MAGIC)?;
    // Version (4 bytes)
    file.write_all(&FORMAT_VERSION.to_le_bytes())?;
    // Embedding dim (4 bytes)
    file.write_all(&dim.to_le_bytes())?;
    // Reserved/padding (4 bytes)
    file.write_all(&0u32.to_le_bytes())?;
    // Entry count (8 bytes)
    file.write_all(&count.to_le_bytes())?;
    // Metadata offset (8 bytes)
    file.write_all(&metadata_offset.to_le_bytes())?;
    // CRC32 (4 bytes)
    file.write_all(&crc.to_le_bytes())?;
    // Reserved (28 bytes) — already zeros from placeholder

    // Flush and sync to disk
    file.flush()?;
    file.sync_all()?;

    // Flush and close the tmp file, then atomically rename (F-06 fix)
    drop(file);
    std::fs::rename(&tmp_path, path)
        .map_err(|e| ShrimPKError::Persistence(format!("Atomic rename failed: {e}")))?;

    let elapsed = start.elapsed();
    tracing::info!(
        entries = count,
        dim = dim,
        metadata_offset = metadata_offset,
        crc = crc,
        elapsed_ms = elapsed.as_millis(),
        "Binary store saved"
    );

    Ok(())
}

/// Load an EchoStore from a binary file.
///
/// Reads and validates the header, verifies CRC32, then reconstructs
/// the store from the embedding array and JSON metadata.
///
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

    if data.len() < HEADER_SIZE as usize {
        return Err(ShrimPKError::Persistence(format!(
            "File too small for header: {} bytes (need {})",
            data.len(),
            HEADER_SIZE
        )));
    }

    // --- Parse header ---
    let (magic, version, dim, count, metadata_offset, stored_crc) = parse_header(&data)?;

    // Validate magic
    if magic != *MAGIC {
        return Err(ShrimPKError::Persistence(format!(
            "Invalid magic bytes: expected {:?}, got {:?}",
            MAGIC, magic
        )));
    }

    // Validate version
    if version != FORMAT_VERSION {
        return Err(ShrimPKError::Persistence(format!(
            "Unsupported format version: expected {FORMAT_VERSION}, got {version}"
        )));
    }

    // --- Read and verify embedding array ---
    let embedding_bytes = count * (dim as u64) * 4;
    let embedding_end = HEADER_SIZE + embedding_bytes;

    if (data.len() as u64) < embedding_end {
        return Err(ShrimPKError::Persistence(format!(
            "File truncated: need {} bytes for embeddings, file is {} bytes",
            embedding_end,
            data.len()
        )));
    }

    // CRC32 verification
    let embedding_slice = &data[HEADER_SIZE as usize..embedding_end as usize];
    let computed_crc = crc32fast::hash(embedding_slice);

    if computed_crc != stored_crc {
        return Err(ShrimPKError::Persistence(format!(
            "CRC32 mismatch: stored={stored_crc:#010x}, computed={computed_crc:#010x}. File may be corrupted."
        )));
    }

    // Parse flat f32 array into Vec<Vec<f32>>
    let dim_usize = dim as usize;
    let count_usize = count as usize;
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(count_usize);

    for i in 0..count_usize {
        let mut emb = Vec::with_capacity(dim_usize);
        for j in 0..dim_usize {
            let offset = HEADER_SIZE as usize + (i * dim_usize + j) * 4;
            let bytes: [u8; 4] = data[offset..offset + 4]
                .try_into()
                .map_err(|_| ShrimPKError::Persistence("Failed to read f32 bytes".into()))?;
            emb.push(f32::from_le_bytes(bytes));
        }
        embeddings.push(emb);
    }

    // --- Read metadata ---
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

    // --- Rebuild EchoStore ---
    let mut store = EchoStore::new();
    for (meta, embedding) in metas.into_iter().zip(embeddings.into_iter()) {
        let entry = meta.into_entry(embedding);
        store.add(entry);
    }

    let elapsed = start.elapsed();
    tracing::info!(
        entries = store.len(),
        dim = dim,
        elapsed_ms = elapsed.as_millis(),
        "Binary store loaded"
    );

    Ok(store)
}

/// Validate a binary store file without loading all data.
///
/// Checks magic bytes, version, and CRC32 checksum.
/// Returns `Ok(true)` if valid, `Ok(false)` if not a valid binary store,
/// or an error on I/O failure.
#[instrument(fields(path = %path.display()))]
pub fn validate_binary(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let data = std::fs::read(path).map_err(|e| {
        ShrimPKError::Persistence(format!("Failed to read file for validation: {e}"))
    })?;

    if data.len() < HEADER_SIZE as usize {
        return Ok(false);
    }

    let (magic, version, dim, count, _metadata_offset, stored_crc) = match parse_header(&data) {
        Ok(h) => h,
        Err(_) => return Ok(false),
    };

    // Check magic
    if magic != *MAGIC {
        return Ok(false);
    }

    // Check version
    if version != FORMAT_VERSION {
        return Ok(false);
    }

    // Check CRC32
    let embedding_bytes = count * (dim as u64) * 4;
    let embedding_end = HEADER_SIZE + embedding_bytes;

    if (data.len() as u64) < embedding_end {
        return Ok(false);
    }

    let embedding_slice = &data[HEADER_SIZE as usize..embedding_end as usize];
    let computed_crc = crc32fast::hash(embedding_slice);

    Ok(computed_crc == stored_crc)
}

// --- Internal helpers ---

/// Parse the 64-byte header, returning (magic, version, dim, count, metadata_offset, crc).
fn parse_header(data: &[u8]) -> Result<([u8; 4], u32, u32, u64, u64, u32)> {
    if data.len() < HEADER_SIZE as usize {
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

        // Corrupt a byte in the embedding section (byte 64 = first embedding byte)
        let mut data = std::fs::read(&path).expect("read file");
        data[HEADER_SIZE as usize] ^= 0xFF; // flip bits
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

        // Corrupt an embedding byte
        let mut data = std::fs::read(&path).expect("read");
        data[HEADER_SIZE as usize + 4] ^= 0xFF;
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
}
