mod db;
mod embed;
mod metadata;
mod phash;
mod scan;
mod vecsearch;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use rusqlite::Connection;
use sha2::{Digest, Sha256};

/// Build a local image index: an append-only SQLite log of every image
/// found under a folder, plus two USearch vector indexes (perceptual
/// hashes and SigLIP 2 semantic embeddings) for similarity search.
#[derive(Parser, Debug)]
#[command(name = "imgindex", version, about)]
struct Args {
    /// Append a scan of <ROOT> to ./image_index.db (no decoding)
    #[arg(long)]
    create: bool,

    /// Print a Claude/agent-compatible SKILL.md to stdout (content only)
    /// — pipe it into your agent's skills directory, e.g.
    /// imgindex --skill > ~/.claude/skills/imgindex/SKILL.md
    #[arg(long)]
    skill: bool,

    /// Compute pHash embeddings for contents not yet hashed
    #[arg(long)]
    generate_phash: bool,

    /// Compute SigLIP 2 embeddings for contents not yet embedded
    #[arg(long)]
    generate_embeddings: bool,

    /// Search by natural-language description (SigLIP 2 text encoder)
    #[arg(long, value_name = "QUERY")]
    search: Option<String>,

    /// Prefetch the ONNX model into the Hugging Face cache
    #[arg(long)]
    download_model: bool,

    /// How many results --search returns
    #[arg(long, value_name = "N", default_value_t = 20)]
    top: usize,

    /// Root folder to traverse recursively (required by --create)
    root: Option<PathBuf>,

    /// Comma-separated extensions to consider. The default list covers
    /// every format this binary can both sniff by magic and decode:
    /// PNG, JPEG/JFIF, WebP, TIFF, GIF, BMP, ICO, PNM (PBM/PGM/PPM),
    /// QOI, DDS, EXR, Radiance HDR and farbfeld. Formats without a
    /// decoder in the image crate (HEIC/HEIF, AVIF, JPEG 2000, JPEG XL,
    /// SVG) or without a sniffable magic (TGA) are deliberately absent.
    #[arg(
        short,
        long,
        value_delimiter = ',',
        default_value = "png,jpeg,jpg,jfif,webp,tiff,tif,gif,bmp,ico,pnm,pbm,pgm,ppm,qoi,dds,exr,ff,hdr",
        value_name = "LIST"
    )]
    extensions: Vec<String>,

    /// Per-file ignore warnings during --create, printed to stderr while
    /// files are rejected. `severe` shows only files that claimed to be
    /// images but failed to decode, plus open/read errors; `all` also
    /// shows plain non-image content and empty files. End-of-run counts
    /// are printed either way.
    #[arg(long, value_enum, value_name = "severe|all")]
    show_ignored: Option<ShowIgnored>,

    /// SQLCipher encryption key for the database (default: none)
    #[arg(short = 'k', long, value_name = "KEY")]
    key: Option<String>,
}

/// How much per-file detail `--create` prints for ignored files.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum ShowIgnored {
    /// Only severe rejections: image-formatted bytes that failed to
    /// decode, and open/read errors.
    Severe,
    /// Every ignored file, including benign non-image content.
    All,
}

/// Severity classes for ignored files, reported by --show-ignored.
/// BENIGN: the content simply isn't a usable image (unknown or
/// non-image magic, zero bytes). SEVERE: the bytes claimed to be an
/// image but the header can't be parsed, or the file couldn't be read.
const SEV_BENIGN: u8 = 1;
const SEV_SEVERE: u8 = 2;
/// Reporting floor when no --show-ignored level is given: no per-file
/// lines (the end-of-run counts are still printed).
const SEV_NEVER: u8 = 3;


/// One accepted observation of a file at scan time. Deliberately tiny:
/// no pixel decoding happens during --create (dimensions come from the
/// image header, which is a cheap read).
struct ScanResult {
    path_hash: String,
    path: String,
    content_hash: String,
    mime_type: String,
    size_bytes: u64,
    width: i64,
    height: i64,
    created_at: Option<i64>,
    modified_at: Option<i64>,
    metadata: Option<String>,
}

/// What the walker hands to the workers: the path plus the cheap stat
/// info, so workers only do the expensive part (content IO).
struct FileTask {
    path: PathBuf,
    size_bytes: u64,
    created_at: Option<i64>,
    modified_at: Option<i64>,
}

/// Number of bytes read up-front: enough for mime sniffing (magic bytes
/// live in the first few dozen bytes) before any hashing happens.
const HEAD_LEN: usize = 8192;

/// When a scan's batch is more than this fraction of the existing table,
/// drop the secondary indexes and rebuild them after the load: SQLite
/// builds an index by sorting over the whole table, which beats paying a
/// random-order B-tree write per row per index during the insert. Below
/// the threshold, incremental index maintenance wins because its cost
/// scales with the batch, not the table.
const REBUILD_RATIO: f64 = 0.25;

fn main() {
    // Rust ignores SIGPIPE by default; a closed pipe (e.g. `| head`) would
    // panic on write. Restore the default: die quietly like a good CLI.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let args = Args::parse();

    if args.skill {
        print!("{}", db::skill_md());
        return;
    }

    // exactly one action flag at a time
    let mut actions = 0;
    if args.create {
        actions += 1;
    }
    if args.generate_phash {
        actions += 1;
    }
    if args.generate_embeddings {
        actions += 1;
    }
    if args.download_model {
        actions += 1;
    }
    if args.search.is_some() {
        actions += 1;
    }
    if actions > 1 {
        eprintln!("error: --create, --generate-phash, --generate-embeddings, --download-model and --search are mutually exclusive");
        std::process::exit(1);
    }

    let db_path = match image_index_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    if args.generate_phash {
        if let Err(e) = run_generate_phash(&db_path, args.key.as_deref()) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.generate_embeddings {
        if let Err(e) = run_generate_embeddings(&db_path, args.key.as_deref()) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(query) = args.search.clone() {
        if let Err(e) = run_search(&db_path, args.key.as_deref(), &query, args.top) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.download_model {
        if let Err(e) = embed::ensure_models() {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        println!("model ready (Hugging Face cache)");
        return;
    }
    if !args.create {
        // No action requested: show what this tool can do.
        use clap::CommandFactory;
        let _ = Args::command().print_help();
        println!();
        return;
    }
    let Some(root) = args.root.clone() else {
        eprintln!("error: --create requires a <ROOT> folder");
        std::process::exit(1);
    };
    if let Err(e) = run_create(&args, root) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `--generate-phash`: the enrichment pass. Walks every distinct content
/// recorded by --create and makes sure each one has a perceptual-hash
/// embedding in the USearch index: the index itself answers "already
/// hashed?" via an O(1) key lookup — only misses get decoded. Contents
/// whose files are gone or unreadable are skipped and counted; a lost or
/// desynced index file simply regenerates from the live files.
fn run_generate_phash(db_path: &Path, key: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    if !db_path.exists() {
        return Err(format!(
            "no database at {} — run --create first",
            db_path.display()
        )
        .into());
    }
    let conn = db::open(db_path, key)?;
    db::create_table(&conn)?;

    let contents = db::contents_all(&conn)?;
    let mut already_hashed = 0usize;
    let mut newly_hashed = 0usize;
    let mut file_missing = 0usize;
    let mut unhashable = 0usize;

    // Preload all observed paths once: the loop then runs without
    // touching the database connection.
    let all_paths: HashMap<String, Vec<String>> = db::all_paths_by_content(&conn)?;

    let index_path = vecsearch::index_path(db_path);
    let mut store = vecsearch::VectorStore::open(&index_path)?;

    for (done, content_hash) in contents.iter().enumerate() {
        // The USearch key IS the lookup: an existing embedding means the
        // content was hashed in an earlier pass — no decode needed.
        if store.phash_has(content_hash) {
            already_hashed += 1;
            continue;
        }
        // The content may exist under several paths (duplicates) and any
        // of them may have been deleted since; find one that is alive.
        let paths = all_paths.get(content_hash).cloned().unwrap_or_default();
        let live = paths.iter().find(|p| Path::new(p).exists());
        let Some(live_path) = live else {
            file_missing += 1;
            continue;
        };
        let Some(fingerprint) = phash::analyze(Path::new(live_path)) else {
            unhashable += 1;
            continue;
        };
        store.phash_add(content_hash, fingerprint)?;
        newly_hashed += 1;

        if newly_hashed.is_multiple_of(500) {
            println!(
                "  … {}/{} contents processed ({} newly hashed)",
                done + 1,
                contents.len(),
                newly_hashed
            );
        }
    }

    if newly_hashed > 0 {
        store.save()?;
    }

    println!(
        "contents: {} total — already hashed: {}, newly hashed: {} ({} file(s) missing, {} unhashable) in {:.1?}",
        contents.len(),
        already_hashed,
        newly_hashed,
        file_missing,
        unhashable,
        started.elapsed()
    );
    println!(
        "vec: {} pHash embeddings → {}",
        store.phash_size(),
        index_path.display()
    );
    Ok(())
}

/// `--generate-embeddings`: the semantic enrichment pass. Walks every
/// distinct content recorded by --create and makes sure each one has a
/// SigLIP 2 embedding in the USearch index: the index itself answers
/// "already embedded?" via an O(1) key lookup — only misses get decoded
/// and embedded. Contents whose files are gone are skipped and counted.
fn run_generate_embeddings(db_path: &Path, key: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    if !db_path.exists() {
        return Err(format!(
            "no database at {} — run --create first",
            db_path.display()
        )
        .into());
    }
    let conn = db::open(db_path, key)?;
    db::create_table(&conn)?;

    let contents = db::contents_all(&conn)?;
    let mut already_embedded = 0usize;
    let mut newly_embedded = 0usize;
    let mut file_missing = 0usize;
    let mut failed = 0usize;

    let all_paths: HashMap<String, Vec<String>> = db::all_paths_by_content(&conn)?;
    let index_path = vecsearch::index_path(db_path);
    let mut store = vecsearch::VectorStore::open(&index_path)?;

    // Warm-up semantics: fetch ALL model files (vision + text + tokenizer)
    // so every later usage scenario works offline, not just this pass.
    embed::ensure_models().map_err(|e| e.to_string())?;
    let vision_model = embed::ensure_file(embed::VISION_FILE).map_err(|e| e.to_string())?;
    let mut embedder = embed::Embedder::new(&vision_model).map_err(|e| e.to_string())?;

    for (done, content_hash) in contents.iter().enumerate() {
        // The USearch key IS the lookup: an existing embedding means the
        // content was embedded in an earlier pass — no decode needed.
        if store.semantic_has(content_hash) {
            already_embedded += 1;
            continue;
        }
        let paths = all_paths.get(content_hash).cloned().unwrap_or_default();
        let live = paths.iter().find(|p| Path::new(p).exists());
        let Some(live_path) = live else {
            file_missing += 1;
            continue;
        };
        match embedder.embed(Path::new(live_path)) {
            Ok(embedding) => {
                store.semantic_add(content_hash, &embedding)?;
                newly_embedded += 1;
            }
            Err(_) => {
                failed += 1;
            }
        }

        if newly_embedded.is_multiple_of(250) && newly_embedded > 0 {
            println!(
                "  … {}/{} contents processed ({} newly embedded)",
                done + 1,
                contents.len(),
                newly_embedded
            );
        }
    }

    if newly_embedded > 0 {
        store.save()?;
    }

    println!(
        "embeddings: {} content(s) total — already embedded: {}, newly embedded: {} ({} file(s) missing, {} failed) in {:.1?}",
        contents.len(),
        already_embedded,
        newly_embedded,
        file_missing,
        failed,
        started.elapsed()
    );
    println!(
        "vec: {} semantic embeddings → {}",
        store.semantic_size(),
        index_path.display()
    );
    Ok(())
}

/// Print one ignored-file line when `--show-ignored` asks for its
/// severity class. stderr only: stdout stays a clean summary.
fn report_ignored(floor: &AtomicU8, severity: u8, path: &Path, class: &str, detail: &str) {
    if floor.load(Ordering::Relaxed) <= severity {
        eprintln!("ignored [{class}] {}: {detail}", path.display());
    }
}

/// The image mime carried by the bytes, or None when the content isn't a
/// decodable image. infer's magic table is authoritative; a small
/// supplement covers raster formats the image crate decodes but infer
/// doesn't table (PNM family, QOI, DDS, EXR, Radiance HDR, farbfeld).
/// TGA is deliberately absent: it has no reliable magic, so it can't be
/// sniffed — and what can't be sniffed isn't accepted.
fn sniff_mime(head: &[u8]) -> Option<String> {
    if let Some(t) = infer::get(head) {
        return t
            .mime_type()
            .starts_with("image/")
            .then(|| t.mime_type().to_string());
    }
    // PNM magic, per spec: 'P' + '1'..'6' (ASCII or binary) + whitespace.
    if head.len() >= 3
        && head[0] == b'P'
        && (b'1'..=b'6').contains(&head[1])
        && head[2].is_ascii_whitespace()
    {
        return Some("image/x-portable-anymap".into());
    }
    const SUPPLEMENT: &[(&[u8], &str)] = &[
        (b"qoif", "image/qoi"),                       // QOI
        (b"DDS ", "image/x-dds"),                     // DirectDraw Surface
        (b"\x76\x2f\x31\x01", "image/x-exr"),         // OpenEXR
        (b"#?RADIANCE", "image/vnd.radiance"),        // Radiance HDR
        (b"#?RGBE", "image/vnd.radiance"),            // Radiance HDR (alt)
        (b"farbfeld", "image/x-farbfeld"),            // farbfeld
    ];
    SUPPLEMENT
        .iter()
        .find(|(magic, _)| head.starts_with(magic))
        .map(|(_, mime)| mime.to_string())
}

/// Real image formats the decoder set cannot open. Sniffing still
/// recognizes them, so they are reported honestly instead of
/// masquerading as corruption.
const UNDECODABLE: &[(&str, &str)] = &[
    ("image/heif", "HEIF/HEIC (Apple/iPhone) — needs libheif, which this binary doesn't ship"),
    ("image/jp2", "JPEG 2000 — no decoder in the image crate"),
    ("image/avif", "AVIF — the image crate has no AVIF decoder"),
    ("image/jxl", "JPEG XL — no decoder in the image crate"),
];

/// Sniff + hash one file during --create. Returns `None` for anything
/// whose bytes don't prove it is an image (unknown magic or non-image
/// mime) and for IO errors — such files are left out. The magic check
/// and the header-dimension read both run before hashing, so a corrupt
/// file costs one head read. No decoding happens here: perceptual hashes
/// and embeddings come from the enrichment passes.
fn process_file(
    task: FileTask,
    skipped_not_image: &AtomicU64,
    ignored_floor: &AtomicU8,
) -> Option<ScanResult> {
    let FileTask { path, size_bytes, created_at, modified_at } = task;

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            report_ignored(ignored_floor, SEV_SEVERE, &path, "io", &format!("open failed: {e}"));
            return None;
        }
    };
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut head = [0u8; HEAD_LEN];
    let mut filled = 0;
    while filled < head.len() {
        match reader.read(&mut head[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) => {
                report_ignored(ignored_floor, SEV_SEVERE, &path, "io", &format!("read failed: {e}"));
                return None;
            }
        }
    }

    // Trust the bytes, not the extension: unknown content and non-image
    // types (a mislabeled video, a truncated file) are ignored outright.
    let mime_type = match sniff_mime(&head[..filled]) {
        Some(mime) => mime,
        None => {
            skipped_not_image.fetch_add(1, Ordering::Relaxed);
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("(none)");
            let detail = match infer::get(&head[..filled]) {
                Some(t) => format!(
                    "content is {}, not an image (extension .{}, {} bytes)",
                    t.mime_type(),
                    ext,
                    size_bytes
                ),
                None => format!(
                    "no known magic — not an image (extension .{}, {} bytes)",
                    ext,
                    size_bytes
                ),
            };
            report_ignored(ignored_floor, SEV_BENIGN, &path, "not-an-image", &detail);
            return None;
        }
    };

    // Recognized image formats the decoder set can't open: report
    // honestly instead of pretending the file is corrupt.
    if let Some((_, why)) = UNDECODABLE.iter().find(|(mime, _)| *mime == mime_type) {
        skipped_not_image.fetch_add(1, Ordering::Relaxed);
        report_ignored(ignored_floor, SEV_SEVERE, &path, "unsupported", why);
        return None;
    }

    // Header dimension read before hashing: a file whose magic passed
    // but whose header can't be parsed is corrupt — skip without the
    // full-content hash.
    let (width, height) = match phash::dimensions_of(&path) {
        Ok(dims) => dims,
        Err(e) => {
            skipped_not_image.fetch_add(1, Ordering::Relaxed);
            report_ignored(ignored_floor, SEV_SEVERE, &path, "corrupt", &e.to_string());
            return None;
        }
    };

    let mut hasher = Sha256::new();
    hasher.update(&head[..filled]);
    // digest 0.11 dropped std::io::Write for hashers, so stream the
    // file into the hasher explicitly instead of std::io::copy
    let mut buf = [0u8; 64 * 1024];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(e) => {
                report_ignored(ignored_floor, SEV_SEVERE, &path, "io", &format!("read failed mid-hash: {e}"));
                return None;
            }
        }
    }
    let content_hash = hex::encode(hasher.finalize());

    let path_str = path.to_string_lossy().into_owned();
    let path_hash = scan::sha256_hex(path_str.as_bytes());

    let metadata = metadata::extract(&path, &mime_type);

    Some(ScanResult {
        path_hash,
        path: path_str,
        content_hash,
        mime_type,
        size_bytes,
        width: width as i64,
        height: height as i64,
        created_at,
        modified_at,
        metadata,
    })
}

/// `--search <QUERY>`: natural-language search over the SigLIP 2
/// semantic index. The query is embedded with the SigLIP 2 text encoder
/// and KNN-ranked against the image embeddings (same 768-dim space).
/// Requires the semantic index; exits with guidance when it's missing.
fn run_search(
    db_path: &Path,
    key: Option<&str>,
    query: &str,
    top: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let index_path = vecsearch::index_path(db_path);
    if !index_path.exists() {
        return Err(format!(
            "no vector index at {} — run --generate-embeddings first",
            index_path.display()
        )
        .into());
    }
    let store = vecsearch::VectorStore::open(&index_path)?;
    if store.semantic_size() == 0 {
        return Err(format!(
            "the semantic index is empty — run --generate-embeddings first (index at {})",
            index_path.display()
        )
        .into());
    }

    let conn = db::open(db_path, key)?;
    let mut text_embedder = embed::TextEmbedder::new(
        &embed::ensure_file(embed::TEXT_FILE)?,
        &embed::ensure_file(embed::TOKENIZER_FILE)?,
    )?;
    let embedding = text_embedder.embed_text(query).map_err(|e| e.to_string())?;

    println!(
        "search: {query:?} — nearest images (semantic similarity, 1 = identical):"
    );
    for (key_hex, distance) in store.semantic_search(&embedding, top) {
        let similarity = 1.0 - distance;
        println!("  sim={similarity:<7.3} {}", list_paths(&conn, &key_hex));
    }
    Ok(())
}

/// One line listing every known path for a content-hash prefix (the
/// USearch key hex).
fn list_paths(conn: &Connection, key_hex: &str) -> String {
    let sql = format!(
        "SELECT path FROM images WHERE substr(content_hash, 1, 16) = '{key_hex}'
         ORDER BY path"
    );
    let mut stmt = conn.prepare(&sql).expect("sql");
    let paths: Vec<String> = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .expect("sql")
        .filter_map(Result::ok)
        .collect();
    if paths.len() > 3 {
        format!("{} (+{} more)", paths[..3].join(", "), paths.len() - 3)
    } else {
        paths.join(", ")
    }
}

/// The single, append-only database: `image_index.db` in the current
/// directory. Created on first use, appended to on every run.
fn image_index_path() -> std::io::Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    Ok(cwd.join("image_index.db"))
}

/// `--create`: the fast observation pass. No decoding happens here —
/// perceptual hashes and embeddings come from the enrichment passes.
fn run_create(args: &Args, root: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    // Captured once, before the walker starts: every row appended by this
    // run receives this exact value, so scanned_at doubles as the scan's
    // grouping key (rows of one execution always share it). The host name
    // is captured the same way: with scanned_at it identifies the run.
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    let hostname = scan::hostname();

    let root = root.canonicalize()?;
    if !root.is_dir() {
        return Err(format!("{} is not a directory", root.display()).into());
    }

    let exts: HashSet<String> = args
        .extensions
        .iter()
        .map(|e| e.trim_start_matches('.').to_ascii_lowercase())
        .collect();
    if exts.is_empty() {
        return Err("the extension list is empty".into());
    }

    // First feedback, before any work starts: what is being scanned.
    println!(
        "scanning {} — extensions: {}",
        root.display(),
        args.extensions.join(",")
    );

    let db_path = image_index_path()?;
    let mut conn = db::open(&db_path, args.key.as_deref())?;
    db::create_table(&conn)?;
    let existing_rows = db::count_rows(&conn)?;

    let skipped_zero_byte = Arc::new(AtomicU64::new(0));
    let skipped_not_image = Arc::new(AtomicU64::new(0));
    // Per-file ignore reporting floor (see --show-ignored).
    let ignored_floor = Arc::new(AtomicU8::new(match args.show_ignored {
        Some(ShowIgnored::All) => SEV_BENIGN,
        Some(ShowIgnored::Severe) => SEV_SEVERE,
        None => SEV_NEVER,
    }));
    // Live progress counters: files finished by the workers / handed out
    // by the walker.
    let processed = Arc::new(AtomicU64::new(0));
    let discovered = Arc::new(AtomicU64::new(0));

    let n_workers = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    // Bounded channels: the walker never runs arbitrarily far ahead of the
    // workers, and workers never queue unbounded results for the writer.
    let (task_tx, task_rx) = sync_channel::<FileTask>(256);
    let (res_tx, res_rx) = sync_channel::<ScanResult>(256);

    // Progress reporter: an aggregated line every 5s once hashing is
    // underway. Exits the moment it's stopped, so the final summary is
    // never delayed.
    let (stop_tx, stop_rx) = channel::<()>();
    let reporter = {
        let processed = Arc::clone(&processed);
        let skipped_zero_byte = Arc::clone(&skipped_zero_byte);
        let skipped_not_image = Arc::clone(&skipped_not_image);
        thread::spawn(move || {
            let started = Instant::now();
            let mut last_seen = 0u64;
            loop {
                match stop_rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                }
                let done = processed.load(Ordering::Relaxed);
                let ignored = skipped_zero_byte.load(Ordering::Relaxed)
                    + skipped_not_image.load(Ordering::Relaxed);
                println!(
                    "  … {:.0}s: {} processed, {} indexed, {} ignored (+{} in the last 5s)",
                    started.elapsed().as_secs_f32(),
                    done,
                    done - ignored,
                    ignored,
                    done - last_seen
                );
                last_seen = done;
            }
        })
    };

    // Producer: walk the tree, filter by extension, stat each match.
    // Empty files are rejected here, before any worker spends IO on them;
    // traversal errors and stat failures are silently skipped.
    let producer = {
        let root = root.clone();
        let exts = exts.clone();
        let skipped_zero_byte = Arc::clone(&skipped_zero_byte);
        let ignored_floor = Arc::clone(&ignored_floor);
        let discovered = Arc::clone(&discovered);
        thread::spawn(move || {
            for entry in walkdir::WalkDir::new(root) {
                let Ok(entry) = entry else { continue };
                if !entry.file_type().is_file() {
                    continue;
                }
                let matches = entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .is_some_and(|e| exts.contains(&e));
                if !matches {
                    continue;
                }
                let Ok(path) = entry.path().canonicalize() else {
                    continue;
                };
                let Ok(meta) = std::fs::metadata(&path) else {
                    continue;
                };
                if meta.len() == 0 {
                    skipped_zero_byte.fetch_add(1, Ordering::Relaxed);
                    report_ignored(&ignored_floor, SEV_BENIGN, &path, "empty", "zero-byte file");
                    continue;
                }
                let task = FileTask {
                    path,
                    size_bytes: meta.len(),
                    created_at: scan::unix_ts(meta.created().ok()),
                    modified_at: scan::unix_ts(meta.modified().ok()),
                };
                discovered.fetch_add(1, Ordering::Relaxed);
                if task_tx.send(task).is_err() {
                    break; // workers are gone, stop walking
                }
            }
            // Printed by the walker itself (not the reporter) so the
            // candidate count lands the moment the walk ends, not on the
            // next 5s tick — which a run can outlive.
            println!(
                "  walk complete: {} candidate file(s) — hashing …",
                discovered.load(Ordering::Relaxed)
            );
        })
    };

    // Consumers: one per CPU core, sharing the task channel.
    let task_rx = Arc::new(std::sync::Mutex::new(task_rx));
    let mut workers = Vec::with_capacity(n_workers);
    for _ in 0..n_workers {
        let task_rx = Arc::clone(&task_rx);
        let res_tx = res_tx.clone();
        let skipped_not_image = Arc::clone(&skipped_not_image);
        let ignored_floor = Arc::clone(&ignored_floor);
        let processed = Arc::clone(&processed);
        workers.push(thread::spawn(move || loop {
            let task = task_rx.lock().expect("task channel mutex poisoned").recv();
            let Ok(task) = task else { break };
            let result = process_file(task, &skipped_not_image, &ignored_floor);
            processed.fetch_add(1, Ordering::Relaxed);
            if let Some(result) = result
                && res_tx.send(result).is_err()
            {
                break; // writer is gone
            }
        }));
    }
    drop(res_tx); // writer's copy; channel closes once all workers exit

    let scanned: Vec<ScanResult> = res_rx.iter().collect();
    for worker in workers {
        worker.join().expect("worker thread panicked");
    }
    producer.join().expect("producer thread panicked");
    let _ = stop_tx.send(());
    reporter.join().expect("reporter thread panicked");

    let mut rows: Vec<db::ImageRow> = Vec::with_capacity(scanned.len());
    for r in &scanned {
        rows.push(db::ImageRow {
            path_hash: r.path_hash.clone(),
            path: r.path.clone(),
            content_hash: r.content_hash.clone(),
            mime_type: r.mime_type.clone(),
            size_bytes: r.size_bytes,
            width: r.width,
            height: r.height,
            created_at: r.created_at,
            modified_at: r.modified_at,
            metadata: r.metadata.clone(),
        });
    }
    rows.sort_unstable_by(|a, b| a.path_hash.cmp(&b.path_hash));

    // Drop + rebuild the secondary indexes when the batch is a large
    // fraction of the table; otherwise let SQLite maintain them
    // incrementally. Drop, inserts and re-create share one transaction:
    // a crash mid-scan rolls back to the previous consistent state
    // instead of leaving the tables without their indexes.
    let rebuild = existing_rows == 0
        || (rows.len() as f64) > REBUILD_RATIO * existing_rows as f64;
    {
        let tx = conn.transaction()?;
        if rebuild {
            db::drop_indexes(&tx)?;
        }
        for row in &rows {
            db::insert_image(&tx, row, &hostname, now)?;
        }
        // No-op in incremental mode; the bulk (re)build in rebuild mode.
        db::create_indexes(&tx)?;
        tx.commit()?;
    }

    let skipped_zero = skipped_zero_byte.load(Ordering::Relaxed);
    let skipped_img = skipped_not_image.load(Ordering::Relaxed);
    println!(
        "indexed {} image(s) under {} in {:.1?} ({} worker threads)",
        rows.len(),
        root.display(),
        started.elapsed(),
        n_workers
    );
    println!(
        "  ignored: {} (empty: {}, non-image or corrupt: {})",
        skipped_zero + skipped_img,
        skipped_zero,
        skipped_img
    );
    println!(
        "db: {} — appended {} row(s), table now {} row(s) [insert: {}]",
        db_path.display(),
        rows.len(),
        existing_rows + rows.len() as i64,
        if rebuild { "index rebuild" } else { "incremental" }
    );
    println!("hint: run --generate-phash and/or --generate-embeddings to enrich, then --search to query");

    // End-of-run report: raw counters plus the derived per-file cost.
    let processed_files = processed.load(Ordering::Relaxed);
    let elapsed = started.elapsed();
    let secs = elapsed.as_secs_f64();
    println!(
        "report: processed {} file(s) in {:.1?} — avg {:.2} ms/file ({:.0} file(s)/s)",
        processed_files,
        elapsed,
        if processed_files > 0 { secs * 1000.0 / processed_files as f64 } else { 0.0 },
        if secs > 0.0 { processed_files as f64 / secs } else { 0.0 },
    );
    Ok(())
}
