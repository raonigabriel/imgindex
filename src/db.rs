use std::collections::HashMap;
use std::path::Path;

use rusqlite::{Connection, params};

/// An observation row for the `images` append-only log: what was found on
/// disk at scan time. Perceptual hashes are NOT stored here — they live
/// exclusively in the USearch index file.
pub struct ImageRow {
    pub path_hash: String,
    pub path: String,
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
    "idx_images_path_hash",
    "idx_images_content_hash",
    "idx_images_modified_at",
    "idx_images_scanned_at",
];

/// The executed schema doubles as the documentation (`--skill` and the
/// README quote these constants verbatim) — keep the comments in sync
/// with reality.
///
/// The database is 100% standard SQLite — plain tables only, no
/// extensions, no vector data. Perceptual hashes and SigLIP embeddings
/// live in a separate USearch index file (image_index.safegraphs) next
/// to this database; similarity search is served by `imgindex --search`.
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
    -- Host name of the machine that performed the scan. All rows of one
    -- run share it; with scanned_at it identifies which run on which
    -- box produced a row — one database can be appended from several
    -- machines without ambiguity.
    hostname       TEXT    NOT NULL,
    -- SHA-256 (lowercase hex) of the file's canonical absolute path.
    -- Stable identity of a path across scans: join or diff scans on it,
    -- and combine it with scanned_at for a path's history. NOT unique by
    -- design: the same file appears once per scan that saw it.
    path_hash      TEXT    NOT NULL,
    -- Canonical absolute path as found during the scan.
    path           TEXT    NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_images_path_hash
    ON images(path_hash);
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

/// Self-contained SKILL.md for agents (Claude-style Agent Skill, printed
/// by `--skill`): YAML front matter (name + description drive skill
/// selection via progressive disclosure), then a markdown body with the
/// workflow, command reference, the commented DDL actually executed at
/// create time, querying recipes and gotchas. The user pipes the output
/// into their agent's skills directory, e.g.
///   imgindex --skill > ~/.claude/skills/imgindex/SKILL.md
pub fn skill_md() -> String {
    format!(
        r#"---
name: imgindex
description: >
  Indexes folders of images into an append-only SQLite database
  (image_index.db) with content hashes, header dimensions, metadata and
  per-scan host/time stamps, and optionally adds pHash and SigLIP 2
  embeddings to a USearch index (image_index.safegraphs) for duplicate
  detection and natural-language image search. Use when the user asks to
  scan or index image folders, find duplicate or near-duplicate images,
  diff what changed between scans, run text-to-image search, or query
  the image_index.db / image_index.safegraphs artifacts directly. Covers
  the imgindex CLI, the images table schema, and ready-made SQL recipes.
---

# imgindex — local image index and similarity search

Append-only local image index: one command scans a folder and appends
every image found to `image_index.db` (a 100% standard SQLite file —
plain tables, no extensions); optional enrichment passes add
perceptual-hash (pHash) and SigLIP 2 semantic embeddings to a single
USearch index file (`image_index.safegraphs`) next to it, enabling
"find me more like this" search.

## Workflow

1. **Scan** (fast pass — no pixel decoding; hashes content, reads image
   headers only):
   ```bash
   imgindex --create ~/photos
   ```
   Writes `./image_index.db` in the CURRENT directory (append-only: every
   run appends one row per image seen). Progress prints live (aggregated
   counters every 5s); the last line is a report with the average
   processing time per file.
2. **pHash embeddings** (optional — duplicate/near-duplicate detection):
   ```bash
   imgindex --generate-phash
   ```
3. **SigLIP 2 embeddings** (optional — natural-language search; the
   model downloads from Hugging Face on first use):
   ```bash
   imgindex --generate-embeddings
   ```
4. **Search** (requires step 3):
   ```bash
   imgindex --search "a small girl playing with dolls" --top 10
   ```

## Command reference

- `--create <ROOT>` — scan ROOT recursively and append to
  `./image_index.db`.
  - `-e, --extensions <LIST>` — candidates by file extension. Default =
    every format that is sniffable by magic bytes AND decodable: PNG,
    JPEG/JFIF, WebP, TIFF, GIF, BMP, ICO, PNM (PBM/PGM/PPM), QOI, DDS,
    EXR, Radiance HDR, farbfeld.
  - `--show-ignored severe|all` — per-file rejection warnings on stderr.
    `severe`: corrupt/undecodable images, unsupported formats (HEIF,
    AVIF, JPEG 2000, JPEG XL), IO errors. `all`: also plain non-image
    content and empty files. Counts are printed either way.
  - `-k, --key <KEY>` — SQLCipher encryption (AES-256) for the database.
- `--generate-phash` — compute 64-bit DCT pHash per distinct content.
- `--generate-embeddings` — compute SigLIP 2 image embeddings.
- `--search <QUERY> --top N` — natural-language image search.
- `--download-model` — prefetch the ONNX model into the Hugging Face
  cache (offline preparation).
- `--skill` — print this SKILL.md to stdout (pipe it into your agent's
  skills directory, e.g.
  `imgindex --skill > ~/.claude/skills/imgindex/SKILL.md`).

## Database schema

IMAGES TABLE DDL (verbatim — this is what CREATE runs):

```sql
{images}
```

EXECUTED INDEX DDL (created after each bulk load):

```sql
{indexes}
```

## Querying recipes

- Latest scan state (one scan = one `scanned_at` value):

  ```sql
  SELECT * FROM images
  WHERE scanned_at = (SELECT MAX(scanned_at) FROM images);
  ```

- Latest KNOWN state of each path (including paths no longer on disk):

  ```sql
  SELECT path, max(scanned_at) FROM images GROUP BY path_hash;
  ```

- Files removed since the most recent scan (present earlier, absent now):

  ```sql
  SELECT path, MAX(scanned_at) FROM images
  GROUP BY path_hash
  HAVING MAX(scanned_at) < (SELECT MAX(scanned_at) FROM images);
  ```

- Content changed between the two most recent scans: compare
  `content_hash` of a `path_hash` across its two newest `scanned_at`
  values; equal hashes = byte-identical.

- Duplicate content within one scan (same bytes, several paths):

  ```sql
  SELECT i.content_hash, COUNT(*) c, group_concat(i.path)
  FROM images i
  WHERE i.scanned_at = (SELECT MAX(scanned_at) FROM images)
  GROUP BY i.content_hash HAVING c > 1;
  ```

- Dimensions come straight from the image header at scan time:

  ```sql
  SELECT path, width, height FROM images
  WHERE scanned_at = (SELECT MAX(scanned_at) FROM images);
  ```

- Which run produced which rows (`hostname` and `scanned_at` are both
  captured once per run — together they identify the run and the box):

  ```sql
  SELECT hostname, scanned_at, COUNT(*) FROM images
  GROUP BY hostname, scanned_at;
  ```

- metadata is a JSON object NAMESPACED BY SOURCE — `exif` (JPEG/TIFF
  containers and PNG/WebP Exif/EXIF chunks) and `png_text` (tEXt/zTXt/
  iTXt/tIME chunks); a namespace may be absent:

  ```sql
  SELECT json_extract(metadata, '$.exif.Model'),
         json_extract(metadata, '$.png_text.Title')
  FROM images WHERE metadata IS NOT NULL;
  ```

- Largest files of the latest scan:

  ```sql
  SELECT path, size_bytes FROM images
  WHERE scanned_at = (SELECT MAX(scanned_at) FROM images)
    AND size_bytes > 10 * 1024 * 1024
  ORDER BY size_bytes DESC;
  ```

- Introspection: `PRAGMA table_info(images);` and
  `SELECT sql FROM sqlite_master WHERE type='table';`

## Gotchas

- The database is APPEND-ONLY: rows are never updated or deleted, so the
  table is the full history across scans. The current state of the disk
  is the set of rows with `scanned_at = MAX(scanned_at)`.
- `mime_type` is sniffed from magic bytes (never trusted from the
  extension) and is always `image/*`. Formats that are recognized but
  not decodable (HEIF/HEIC — Apple, AVIF, JPEG 2000, JPEG XL) never get
  rows; they are reported as `unsupported` rejections. TGA and SVG are
  not indexable (no sniffable magic / no raster decoder).
- `width`/`height` are read from the image HEADER during the scan —
  cheap, no decoding.
- Perceptual hashes and semantic embeddings are NOT in this database:
  they live in the USearch index file `image_index.safegraphs`, keyed by
  the first 8 bytes of `content_hash`. Both the Rust crate (`usearch`)
  and the Python library (`pip install usearch`) read the same format.
- Databases written with `--key` are SQLCipher-encrypted: plain sqlite3
  builds cannot open them without the key (`PRAGMA key`). Unencrypted
  databases open with any standard SQLite client.
- `created_at` (birth time) may be NULL; `modified_at` is the reliable
  timestamp for time ranges.
- `id` is the rowid alias: `WHERE id = ?` is an index seek; no explicit
  index exists (or is needed) for it.
- Never write to the images table from outside imgindex: the tool
  assumes its rows are observations made by scans.

## Python example — cluster near-duplicates by pHash

Reads the two artifacts `imgindex --create` / `--generate-phash` produce
in the current directory, does exact KNN with the USearch Python library
and clusters every content by Hamming distance (single-linkage).
Embedded verbatim from examples/cluster_by_phash.py in the source
repository:

```python
{py_example}
```
"#,
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

pub fn insert_image(
    conn: &Connection,
    row: &ImageRow,
    hostname: &str,
    scanned_at: i64,
) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO images
            (hostname, path_hash, path, content_hash, mime_type,
             size_bytes, width, height, created_at, modified_at,
             scanned_at, metadata)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            hostname,
            row.path_hash,
            row.path,
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
