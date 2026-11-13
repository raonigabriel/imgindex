//! USearch vector indexes: pHash (`bit[64]`, Hamming) and SigLIP 2
//! semantic (`f32[768]`, cosine) embeddings, one per unique content,
//! keyed by the first 8 bytes of the content hash. Both indexes are
//! stored in ONE file (`image_index.safegraphs`) as a tiny length-
//! prefixed container of two serialized USearch buffers — the file
//! format is ours, the index format inside is USearch's.
//!
//! Self-healing: a missing or corrupt index file yields an empty store,
//! and the enrichment passes re-hash / re-embed every content that has a
//! live file on disk. No bookkeeping tables in SQLite.

use std::path::{Path, PathBuf};

use usearch::{b1x8, Index, IndexOptions, MetricKind, ScalarKind};

const VEC_DIMENSIONS: usize = 64; // bits for B1 quantization
const SEMANTIC_DIMENSIONS: usize = 768; // f32 SigLIP 2 embeddings

/// Container magic + section layout:
/// `IMGIDXVEC1 | u64 phash_len | phash_buf | u64 semantic_len | semantic_buf`
const MAGIC: &[u8; 10] = b"IMGIDXVEC1";

/// Serialized USearch index, living next to the SQLite database.
/// Both the pHash and the semantic embeddings live in this ONE file
/// (as two sections of our container format).
pub fn index_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("safegraphs")
}

/// pHash index: Hamming over `bit[64]`.
fn new_phash_index() -> Result<Index, String> {
    let options = IndexOptions {
        dimensions: VEC_DIMENSIONS,
        metric: MetricKind::Hamming,
        quantization: ScalarKind::B1,
        ..Default::default()
    };
    let index = Index::new(&options).map_err(|e| format!("usearch init: {e}"))?;
    index.reserve(4096).map_err(|e| format!("usearch reserve: {e}"))?;
    Ok(index)
}

/// Semantic index: cosine over `f32[768]` SigLIP 2 embeddings.
fn new_semantic_index() -> Result<Index, String> {
    let options = IndexOptions {
        dimensions: SEMANTIC_DIMENSIONS,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        ..Default::default()
    };
    let index = Index::new(&options).map_err(|e| format!("usearch init: {e}"))?;
    index.reserve(4096).map_err(|e| format!("usearch reserve: {e}"))?;
    Ok(index)
}

/// The USearch key for a content hash: its first 8 bytes (16 hex chars),
/// big-endian. The mapping back is `substr(content_hash, 1, 16) = hex(key)`.
pub fn key_of(content_hash: &str) -> Result<u64, String> {
    u64::from_str_radix(
        content_hash.get(..16).ok_or("content hash too short")?,
        16,
    )
    .map_err(|e| format!("bad content hash: {e}"))
}

fn key_hex(key: u64) -> String {
    format!("{key:016x}")
}

pub fn vector_of(perceptual_hash: i64) -> [b1x8; 8] {
    let bytes = perceptual_hash.to_be_bytes();
    std::array::from_fn(|i| b1x8(bytes[i]))
}

/// Both vector indexes plus their file. `None` = that index has no
/// embeddings yet (never generated, or an empty section was stored).
pub struct VectorStore {
    path: PathBuf,
    pub phash: Option<Index>,
    pub semantic: Option<Index>,
}

impl VectorStore {
    /// Open the container file, or return an empty store when it doesn't
    /// exist. A corrupt/truncated file is treated as empty: the
    /// enrichment passes regenerate everything from live files.
    pub fn open(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self {
                path: path.to_path_buf(),
                phash: None,
                semantic: None,
            });
        }
        let bytes = std::fs::read(path).map_err(|e| format!("read index: {e}"))?;
        let (phash_buf, semantic_buf) = match parse_container(&bytes) {
            Some(sections) => sections,
            None => {
                return Ok(Self {
                    path: path.to_path_buf(),
                    phash: None,
                    semantic: None,
                });
            }
        };

        let phash = if phash_buf.is_empty() {
            None
        } else {
            let index = new_phash_index()?;
            index
                .load_from_buffer(&phash_buf)
                .map_err(|e| format!("usearch load (phash): {e}"))?;
            // A freshly loaded index has no spare capacity: adds of new
            // contents would fail. Grow it for future insertions.
            index.reserve(4096 + index.size()).map_err(|e| format!("usearch reserve: {e}"))?;
            Some(index)
        };
        let semantic = if semantic_buf.is_empty() {
            None
        } else {
            let index = new_semantic_index()?;
            index
                .load_from_buffer(&semantic_buf)
                .map_err(|e| format!("usearch load (semantic): {e}"))?;
            index.reserve(4096 + index.size()).map_err(|e| format!("usearch reserve: {e}"))?;
            Some(index)
        };
        Ok(Self {
            path: path.to_path_buf(),
            phash,
            semantic,
        })
    }

    /// Write both indexes back to the container file (atomically: temp
    /// file + rename). Absent indexes serialize as empty sections.
    pub fn save(&mut self) -> Result<(), String> {
        let mut out = Vec::with_capacity(64 * 1024);
        out.extend_from_slice(MAGIC);
        write_section(&mut out, &mut self.phash)?;
        write_section(&mut out, &mut self.semantic)?;
        let tmp = self.path.with_extension("safegraphs.tmp");
        std::fs::write(&tmp, &out).map_err(|e| format!("write index: {e}"))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("rename index: {e}"))?;
        Ok(())
    }

    fn ensure_phash(&mut self) -> &mut Index {
        if self.phash.is_none() {
            self.phash = Some(new_phash_index().expect("usearch init"));
        }
        self.phash.as_mut().expect("just created")
    }

    fn ensure_semantic(&mut self) -> &mut Index {
        if self.semantic.is_none() {
            self.semantic = Some(new_semantic_index().expect("usearch init"));
        }
        self.semantic.as_mut().expect("just created")
    }

    pub fn phash_size(&self) -> usize {
        self.phash.as_ref().map_or(0, |i| i.size())
    }

    pub fn semantic_size(&self) -> usize {
        self.semantic.as_ref().map_or(0, |i| i.size())
    }

    /// O(1) key lookup via USearch. NOTE the wrapper asymmetry for B1
    /// vectors: `add` validates the buffer in bytes (8) while `get`
    /// validates it against `dimensions()` in bits (64) — hence the
    /// 64-byte probe buffer.
    pub fn phash_has(&self, content_hash: &str) -> bool {
        let Some(index) = &self.phash else { return false };
        let Ok(key) = key_of(content_hash) else { return false };
        let mut buf = vec![b1x8(0); VEC_DIMENSIONS];
        index.get(key, &mut buf).unwrap_or(0) > 0
    }

    pub fn phash_add(&mut self, content_hash: &str, perceptual_hash: i64) -> Result<(), String> {
        let key = key_of(content_hash)?;
        self.ensure_phash()
            .add(key, &vector_of(perceptual_hash))
            .map_err(|e| format!("usearch add: {e}"))
    }

    /// KNN over the semantic index, nearest first. Returns (key_hex, distance).
    pub fn semantic_search(&self, embedding: &[f32], top: usize) -> Vec<(String, f32)> {
        let Some(index) = &self.semantic else { return Vec::new() };
        let matches = match index.search(embedding, top) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        matches
            .keys
            .iter()
            .zip(&matches.distances)
            .map(|(key, dist)| (key_hex(*key), *dist))
            .collect()
    }

    pub fn semantic_has(&self, content_hash: &str) -> bool {
        let Some(index) = &self.semantic else { return false };
        let Ok(key) = key_of(content_hash) else { return false };
        let mut buf = vec![0f32; SEMANTIC_DIMENSIONS];
        index.get(key, &mut buf).unwrap_or(0) > 0
    }

    pub fn semantic_add(&mut self, content_hash: &str, embedding: &[f32]) -> Result<(), String> {
        let key = key_of(content_hash)?;
        self.ensure_semantic()
            .add(key, embedding)
            .map_err(|e| format!("usearch add: {e}"))
    }

}

fn write_section(out: &mut Vec<u8>, index: &mut Option<Index>) -> Result<(), String> {
    let bytes = match index {
        Some(index) => {
            let tmp = std::env::temp_dir().join(format!(
                "imgindex_serialize_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            index
                .save(tmp.to_str().ok_or("non-utf8 temp path")?)
                .map_err(|e| format!("usearch save: {e}"))?;
            let bytes = std::fs::read(&tmp).map_err(|e| format!("read serialized index: {e}"))?;
            std::fs::remove_file(&tmp).ok();
            bytes
        }
        None => Vec::new(),
    };
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&bytes);
    Ok(())
}

fn parse_container(bytes: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    if bytes.len() < MAGIC.len() + 16 || &bytes[..MAGIC.len()] != MAGIC {
        return None;
    }
    let mut pos = MAGIC.len();
    let read_u64 = |at: usize| -> Option<u64> {
        Some(u64::from_le_bytes(
            bytes.get(at..at + 8)?.try_into().ok()?,
        ))
    };
    let phash_len = read_u64(pos)? as usize;
    pos += 8;
    let phash_end = pos.checked_add(phash_len)?;
    if phash_end > bytes.len() {
        return None;
    }
    let phash = bytes[pos..phash_end].to_vec();
    pos = phash_end;
    let semantic_len = read_u64(pos)? as usize;
    pos += 8;
    let semantic_end = pos.checked_add(semantic_len)?;
    if semantic_end > bytes.len() {
        return None;
    }
    let semantic = bytes[pos..semantic_end].to_vec();
    Some((phash, semantic))
}
