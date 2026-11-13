use std::collections::HashMap;
use std::path::Path;

use rusqlite::{Connection, params};

/// An observation row for the `images` append-only log: what was found on
/// disk at scan time. Perceptual hashes are NOT stored here — they live
/// exclusively in the USearch index file.
pub struct ImageRow {
    pub file_name_hash: String,
    pub path: String,
    pub file_name: String,
    pub content_hash: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub width: i64,
    pub height: i64,
    pub created_at: Option<i64>,
    pub modified_at: Option<i64>,
    pub metadata: Option<String>,
}

/// Names of every secondary index, used to drop them before a bulk load
/// and rebuild them afterwards.
pub const INDEX_NAMES: &[&str] = &[
    "idx_images_file_name_hash",
    "idx_images_file_name",
    "idx_images_content_hash",
    "idx_images_modified_at",
    "idx_images_scanned_at",
];

/// The executed schema doubles as the documentation (`--agent-help` and
/// the README quote these constants verbatim) — keep the comments in sync
/// with reality.
///
/// The database is 100% standard SQLite — plain tables only, no
/// extensions, no vector data. Perceptual hashes and SigLIP embeddings
/// live in a separate USearch index file (image_index.safegraphs) next
/// to this database; imgindex itself has no search command yet.
const IMAGES_SQL: &str = r#"
-- Append-only log of scan observations: one row per (file, scan) pair.
-- Every scan run appends a fresh row for each image it found on disk,
-- stamped with scanned_at; rows are never updated or deleted, so the
-- table is the full history across scans. A path's current state is its
-- row with the highest scanned_at.
CREATE TABLE IF NOT EXISTS images (
    -- Surrogate primary key: INTEGER PRIMARY KEY makes it the rowid
    -- alias, so the table's own B-tree is keyed on id — lookups by id
    -- are index seeks and it needs no index of its own. Auto-assigned in
    -- insert order; append inserts are therefore always sequential,
    -- whatever order the rows arrive in.
    id             INTEGER PRIMARY KEY,
    -- SHA-256 (lowercase hex) of the file's canonical absolute path.
    -- Stable identity of a path across scans: join or diff scans on it,
    -- and combine it with scanned_at for a path's history. NOT unique by
    -- design: the same file appears once per scan that saw it.
    file_name_hash TEXT    NOT NULL,
    -- Canonical absolute path as found during the scan.
    path           TEXT    NOT NULL,
    -- Base name of the file (last path component).
    file_name      TEXT    NOT NULL,
    -- SHA-256 (lowercase hex) of the file content. Equal values mean
    -- byte-identical files (duplicate content across paths/scans).
    content_hash   TEXT    NOT NULL,
    -- Mime type sniffed from the file's magic bytes, never from the
    -- extension. Only rows whose bytes proved to be real images are
    -- stored: image/png, image/jpeg, image/webp, ...
    mime_type      TEXT    NOT NULL,
    -- File size in bytes at scan time.
    size_bytes     INTEGER NOT NULL,
    -- Pixel dimensions read from the image header (cheap; no decode).
    -- NOT NULL: a file that passes the magic gate but has an unreadable
    -- header is corrupt and excluded.
    width          INTEGER NOT NULL,
    height         INTEGER NOT NULL,
    -- Creation (birth) time, unix seconds. NULL when the OS/filesystem
    -- cannot provide it (frequent on Linux without statx support).
    created_at     INTEGER,
    -- Last modification time, unix seconds, at scan time.
    modified_at    INTEGER,
    -- Unix seconds of the scan run that inserted this row, captured once
    -- before the scan starts: all rows of one execution share the exact
    -- same value, so it doubles as that scan's grouping key. Rows sharing
    -- the maximum value belong to the most recent scan.
    scanned_at     INTEGER NOT NULL,
    -- Image metadata as a JSON object, NAMESPACED BY SOURCE — not only
    -- EXIF: {"exif": {"Tag": "display value", ...}} holds EXIF from
    -- JPEG/TIFF containers and the Exif/EXIF chunks of PNG/WebP;
    -- {"png_text": {"Keyword": "text", ...}} holds PNG-native text
    -- metadata from tEXt/zTXt/iTXt/tIME chunks. Namespaces appear only
    -- when present; NULL when the file carries neither.
    metadata       TEXT
);
"#;


const INDEXES_SQL: &str = r#"
-- Path identity: joins/diffs across scans, history of a single path.
CREATE INDEX IF NOT EXISTS idx_images_file_name_hash
    ON images(file_name_hash);
-- Lookups by base name (e.g. LIKE 'IMG_20%.jpg').
CREATE INDEX IF NOT EXISTS idx_images_file_name
    ON images(file_name);
-- Duplicate-content detection (same bytes under several paths/scans).
CREATE INDEX IF NOT EXISTS idx_images_content_hash
    ON images(content_hash);
-- Time-range queries over file modification times.
CREATE INDEX IF NOT EXISTS idx_images_modified_at
    ON images(modified_at);
-- Selecting all rows of a given scan (typically the latest one).
CREATE INDEX IF NOT EXISTS idx_images_scanned_at
    ON images(scanned_at);
"#;

/// Self-contained schema briefing for AI agents / LLMs: who writes the
/// db, the commented DDL actually executed at create time, and how to
/// query and diff scans.
pub fn schema_doc() -> String {
    format!(
        r#"imgindex v{} — database schema briefing

WHAT THIS DATABASE IS
  imgindex scans a folder recursively and APPENDS the result to a single
  standard SQLite database, image_index.db, in the caller's current
  directory. Each --create run adds one row per image it found on disk,
  stamped with scanned_at (unix seconds); rows are never updated or
  deleted, so the table is the full history of all scans. Only files
  whose bytes proved to be real images are stored: empty files, non-image
  content, and undecodable files are excluded at scan time, so every row
  is a genuine image. The state of the disk at the most recent scan =
  all rows whose scanned_at equals MAX(scanned_at).
  This file is 100% standard SQLite: plain tables only, no extensions,
  readable by any sqlite3 client. width/height are read from the image
  header during the scan (cheap; no decoding).
  Perceptual hashes (64-bit DCT pHash) and SigLIP 2 semantic embeddings
  are NOT in this database: they live in a single USearch index file,
  image_index.safegraphs, next to it (readable with the `usearch`
  library; imgindex itself has no search command). The index answers
  "was this content already hashed/embedded?" via an O(1) key lookup,
  keyed by the first 8 bytes of content_hash.

IMAGES TABLE DDL (verbatim — this is what CREATE runs)
{images}
EXECUTED INDEX DDL (created after each bulk load)
{indexes}

QUERYING NOTES
  * Latest scan state:
      SELECT * FROM images
      WHERE scanned_at = (SELECT MAX(scanned_at) FROM images);
  * Latest KNOWN state of each path (including paths no longer on disk):
      SELECT path, file_name, max(scanned_at)
      FROM images GROUP BY file_name_hash;
  * Files removed since the most recent scan (present earlier, absent now):
      SELECT path, MAX(scanned_at) FROM images
      GROUP BY file_name_hash
      HAVING MAX(scanned_at) < (SELECT MAX(scanned_at) FROM images);
  * Content changed between the two most recent scans: compare
    content_hash of a file_name_hash across its two newest scanned_at
    values; content_hash equal = byte-identical.
  * Duplicate content within one scan (same bytes, several paths):
      SELECT i.content_hash, COUNT(*) c, group_concat(i.path)
      FROM images i
      WHERE i.scanned_at = (SELECT MAX(scanned_at) FROM images)
      GROUP BY i.content_hash HAVING c > 1;
  * Dimensions come straight from the image header at scan time:
      SELECT path, width, height FROM images
      WHERE scanned_at = (SELECT MAX(scanned_at) FROM images);
  * metadata is a JSON object NAMESPACED BY SOURCE — it is not only
    EXIF. "exif" holds EXIF tags (JPEG/TIFF containers and the
    Exif/EXIF chunks of PNG/WebP); "png_text" holds PNG-native text
    metadata (tEXt/zTXt/iTXt keywords plus tIME). Parse with any JSON
    library, and a namespace may be absent:
      SELECT json_extract(metadata, '$.exif.Model'),
             json_extract(metadata, '$.png_text.Title')
      FROM images WHERE metadata IS NOT NULL;
  * mime_type is sniffed from magic bytes and always image/*; do NOT
    trust the path extension over it.
  * created_at may be NULL (birth time unavailable); modified_at is the
    reliable timestamp for time ranges.
  * id is the rowid alias (auto-incrementing across appends) and is
    itself the table's B-tree key: WHERE id = ? is an index seek and no
    explicit index exists (or is needed) for it.
  * scanned_at is captured once per run, before the scan starts — all
    rows of one execution share the exact value, making it a safe
    grouping key:
      SELECT scanned_at, COUNT(*) FROM images GROUP BY scanned_at;
  * Size ranges, e.g. the largest files of the latest scan:
      SELECT path, size_bytes FROM images
      WHERE scanned_at = (SELECT MAX(scanned_at) FROM images)
        AND size_bytes > 10 * 1024 * 1024
      ORDER BY size_bytes DESC;
  * Introspection: PRAGMA table_info(images);
    SELECT sql FROM sqlite_master WHERE type='table'; / type='index';
  * Databases written with the --key option are SQLCipher-encrypted and
    need that key (PRAGMA key) — plain sqlite3 builds cannot open them.
    Unencrypted databases open with any standard SQLite client.

VECTOR INDEX LIBRARY (USearch)
  The pHash embeddings are stored and searched with USearch, the
  single-file vector-search engine from Unum:
    https://github.com/unum-cloud/USearch
  Both the Rust crate (`usearch`) and the Python library (`pip install
  usearch`) expose the same index format, so the serialized pHash index
  inside image_index.safegraphs can be read from either language.

PYTHON EXAMPLE — clustering images by pHash (KNN + near-duplicate
grouping) with the `usearch` Python library. The file below is embedded
verbatim from examples/cluster_by_phash.py in the source repository:

```python
{py_example}
```
"#,
        env!("CARGO_PKG_VERSION"),
        images = IMAGES_SQL,
        indexes = INDEXES_SQL,
        py_example = include_str!("../examples/cluster_by_phash.py"),
    )
}

/// Open the database, applying the SQLCipher key (if any) as the very
/// first statement — required for encryption to take effect.
pub fn open(path: &Path, key: Option<&str>) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    if let Some(key) = key {
        conn.pragma_update(None, "key", key)?;
    }
    Ok(conn)
}

pub fn create_table(conn: &Connection) -> rusqlite::Result<()> {
    // A larger page cache makes the bulk load (and index rebuild) cheaper.
    conn.execute_batch("PRAGMA cache_size = -32768;")?;
    conn.execute_batch(IMAGES_SQL)?;

    Ok(())
}

/// Rows already in the images table (all scans so far); decides whether
/// the next batch is loaded incrementally or with an index rebuild.
pub fn count_rows(conn: &Connection) -> rusqlite::Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
}

pub fn drop_indexes(conn: &Connection) -> rusqlite::Result<()> {
    for name in INDEX_NAMES {
        conn.execute(&format!("DROP INDEX IF EXISTS {name}"), [])?;
    }
    Ok(())
}

/// Idempotent: creates what's missing — after a rebuild-mode drop, after
/// the very first load, or as a self-heal on databases left incomplete.
pub fn create_indexes(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(INDEXES_SQL)?;
    Ok(())
}

pub fn insert_image(conn: &Connection, row: &ImageRow, scanned_at: i64) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO images
            (file_name_hash, path, file_name, content_hash, mime_type,
             size_bytes, width, height, created_at, modified_at,
             scanned_at, metadata)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            row.file_name_hash,
            row.path,
            row.file_name,
            row.content_hash,
            row.mime_type,
            row.size_bytes as i64,
            row.width,
            row.height,
            row.created_at,
            row.modified_at,
            scanned_at,
            row.metadata,
        ],
    )
}

/// Every distinct content hash referenced by any scan.
pub fn contents_all(conn: &Connection) -> rusqlite::Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT DISTINCT content_hash FROM images")?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
    Ok(rows.filter_map(Result::ok).collect())
}

/// All (content_hash, path) pairs from the log, grouped by content —
/// lets the enrichment pass find live files without querying mid-loop.
pub fn all_paths_by_content(
    conn: &Connection,
) -> rusqlite::Result<HashMap<String, Vec<String>>> {
    let mut stmt = conn.prepare(
        "SELECT content_hash, path FROM images ORDER BY scanned_at DESC",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })?;
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for row in rows {
        let (content_hash, path) = row?;
        map.entry(content_hash).or_default().push(path);
    }
    Ok(map)
}
