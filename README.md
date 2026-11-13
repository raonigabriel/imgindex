# imgindex

Almost everyone has a personal image library: phone backups, screenshots,
drawings, scans, wallpapers, memes — scattered across folders, backup
drives, and apps like Google Photos or Immich. And keeping that library
*organized* is a pain: which files are duplicates? Does that `.jpg`
actually contain JPEG data, or is it a mislabeled screenshot? Which
files appeared or vanished since last month? What did this folder look
like six months ago?

**imgindex** answers those questions. It is a local, offline,
append-only image index: one command scans a folder and appends every
image it finds to a SQLite database (with content hashes, dimensions,
timestamps and EXIF metadata), optionally adding perceptual hashes and
SigLIP 2 semantic embeddings to a USearch vector index. Run it from a
cron job and the database becomes a queryable historical record of your
visual library — duplicates, additions, removals, and (with the
embeddings) semantic search, all in plain SQL and a single binary.

**Built for AI agents first.** imgindex is designed to be driven by
coding agents — Claude, OpenCode, Codex, ZCode, Grok and others. The
SQLite database is 100% standard SQLite, the schema briefing
(`--agent-help`) is written to be pasted straight into an agent's
context, and the USearch indexes are readable from Python with one
`pip install`. Keep your library in an organized folder, point a cron
job at it, and the agent can answer questions like *"find duplicate
photos"*, *"what screenshots did I take last week?"*, *"which files
changed since my last backup?"*, *"show me images similar to this one"*
— all from the index, without opening a single image file.

## Features

- **Fast scanning** — producer/consumer pipeline, one hashing worker per
  CPU core; content never decoded during the scan (headers only).
- **Trust gates** — empty files, non-image magic, corrupt headers,
  undecodable and blank images are all excluded; every stored row is a
  genuine, analyzable image.
- **Perceptual hashes (64-bit DCT pHash)** — visual structure
  fingerprints for near-duplicate detection.
- **SigLIP 2 semantic embeddings (768-dim)** — content-level vectors
  aligned with natural-language descriptions, ready for text search.
- **Append-only history** — every scan appends observations stamped with
  `scanned_at`; compare any two scans to find added, removed or changed
  files.
- **Hugging Face cache integration** — model files download into the
  official HF cache (`HF_HOME`/`HF_HUB_CACHE` respected) via `hf-hub`,
  shared with `hf download`, transformers, and every other HF client.
- **Air-gap friendly** — `--download-model` prefetches everything so all
  passes and queries work fully offline afterwards.
- **Optional encryption** — SQLCipher-encrypted databases with `--key`.

## Architecture

```
                 ┌────────────┐   bounded    ┌─────────────────┐
   folder ──────▶│  producer  │───channel───▶│ N worker threads │───┐
   (walkdir)     │ (walk+stat)│              │ (mime+sha256)    │   │
                 └────────────┘              └──────────────────┘   │
                                                                    ▼
        ┌───────────────────────── single transaction ◀─────────────┘
        ▼
  image_index.db (SQLite, append-only)     image_index.safegraphs (pHash section) (USearch)
  └─ images: one row per (file, scan)      └─ pHash bit[64] + SigLIP f32[768]
                                           └─ semantic: SigLIP 2 f32[768]
```

- **`--create`** — the fast pass. A producer thread walks the tree and
  emits tasks (canonical path + stat info) into a bounded channel; one
  worker thread per CPU core sniffs the mime type from magic bytes and
  streams a SHA-256 of the content. No decoding. Results are buffered,
  sorted, and appended in one transaction with adaptive index
  maintenance (indexes dropped and rebuilt when the batch is large,
  incrementally maintained otherwise).
- **`--generate-phash` / `--generate-embeddings`** — enrichment passes.
  Each walks the distinct content hashes recorded by `--create`,
  re-locates a live file for each, decodes it once, and adds one
  embedding per unique content to the matching USearch index.
  Idempotent: the index itself answers "already done?" via an O(1) key
  lookup, so re-running costs almost nothing.

## The two similarity indexes — and why both

Both live in **one USearch file** (`image_index.safegraphs`):

| | pHash index | Semantic index |
|---|---|---|
| Embedding | 64-bit DCT perceptual hash | SigLIP 2 `f32[768]` |
| Distance | Hamming (bit differences) | Cosine |
| Answers | "is this a re-encode / near-copy of that?" | "what is this image *about*?" |
| Speed | instant (O(1) key lookup, tiny vectors) | fast ANN (proximity graph) |
| Blind spot | different photos of the same scene are far apart | re-encodes of the same photo score high but not 0 |

**pHash** fingerprints *visual structure*: two files within Hamming
distance ≤ 3 of each other are the same picture with trivial edits
(resize, re-compress, watermark). It detects re-uploads and stripped
copies across different files.

**SigLIP 2** captures *meaning*: same scene, same subject, same page
type — even across different resolutions, crops and dates — and because
it is a CLIP-style dual encoder, the same space also contains **text**
embeddings, which is what makes future natural-language search possible.

Use them together: pHash for exact-ish duplicate detection, SigLIP 2 for
"find me more like this". The db never stores either hash — the USearch
file is the single source for vectors, and the SQLite db stays 100%
standard SQL.

## Usage

```
imgindex --create <ROOT>              # 1. fast scan (paths + content hashes)
imgindex --generate-phash             # 2. pHash embeddings (optional)
imgindex --generate-embeddings        # 3. SigLIP 2 embeddings (optional)
imgindex --download-model             # warm-up: prefetch ONNX model (offline prep)

imgindex --search "a github repository page" --top 10
                                      # text search (after step 3)

imgindex --agent-help                 # schema DDL briefing for AI agents

  -e, --extensions <LIST>   extensions to consider
                            [default: png,jpeg,jpg,webp,tiff,gif,bmp]
  -k, --key <KEY>           SQLCipher encryption key [default: none]
  --top <N>                 result count for --search [default: 20]
```

## What pass 1 alone already gives you

`--create` writes no pixels and runs no neural network — and the first
pass is still useful on its own, because every row carries the **content
hash**, the **full path history** and **header dimensions**. Plain SQL
over `image_index.db` already answers:

- exact duplicate detection (same bytes under several paths or scans)
- what was added, removed or changed between any two scans
- the complete observation history of any path (sizes, timestamps)
- size and date ranges (largest files, everything modified this month)
- per-app breakdowns via the sniffed `mime_type`

The companion script
[examples/cluster_by_phash.py](examples/cluster_by_phash.py) extends
this offline: it loads the pHash embeddings from the USearch container,
runs KNN, and groups near-duplicate contents by Hamming distance — no
model, no downloads.

The ML passes start where that ends: `--generate-phash` (visual
fingerprints that work *across* different captures) and
`--generate-embeddings` (semantics + text search).

## `--agent-help` — a briefing for AI agents

`imgindex --agent-help` prints a self-contained schema briefing written
for LLM-based agents that need to query the database on their own:

- the **exact DDL** the tool executes (byte-for-byte the same constants
  that created the db — it cannot drift), with a comment on every column
- the index list with each index's purpose
- ready-made SQL recipes: latest-scan state, scan diffs, duplicate
  content, dimensions, `json_extract` over the metadata namespaces
- gotchas an agent would otherwise trip on (`scanned_at` grouping,
  NULL `created_at`, `id` as rowid alias, encrypted-db notes)

It also documents the USearch vector files and points to the upstream
project ([unum-cloud/USearch](https://github.com/unum-cloud/USearch)) —
the same indexes are readable from Python with the `usearch` library,
and a complete working example is **embedded in the briefing output**
(kept at [examples/cluster_by_phash.py](examples/cluster_by_phash.py)):
it loads the pHash index, demonstrates KNN search, and clusters
near-duplicate contents by Hamming distance.

Run it once next to the database and hand the output to your agent —
it has everything needed to write correct queries without reading the
imgindex source.

## How `--search` scores results

`--search` embeds your query with the SigLIP 2 **text encoder** and
ranks every indexed image embedding against it. The printed score
(`sim=`) is **cosine similarity on L2-normalized vectors**:

```text
cosine_similarity = dot(query, image)            -- both are unit vectors
sim               = 1 - cosine_similarity       -- converted for readability
```

- **sim = 1.0** — same direction: identical image (or a re-encode of it)
- **sim = 0.0** — unrelated
- results are printed **highest similarity first**

Because both vectors are L2-normalized before storage and at query
time, the score is a pure *angle* measurement: it is independent of
image resolution, aspect ratio, file size and brightness contrast, and
it is comparable across queries and across machines. Two practical
caveats:

- **relative beats absolute** — SigLIP-family embedding spaces are
  dense, so even unrelated pairs rarely score below ~0.1 similarity.
  Read the *ranking* (this image vs the others in the same result set),
  not the absolute value.
- **UI screenshots compress the scale** — the 987-screenshot desktop
  library used for testing scores everything between ~0.12 and ~0.15
  for a given query. The ordering is still meaningful: the best match
  sits at the top, and the gaps between consecutive results are what
  tell you where "similar" ends.

Contrast with the other identifiers: `content_hash` (exact bytes) and
`file_name_hash` (exact path) are equality checks — similarity is the
fuzzy, graded measure that turns "find me more like this" into a
ranked list.

## Database schema

The SQLite database holds **one plain table** — no extensions, no
virtual tables, openable with any sqlite3 client:

```sql
CREATE TABLE IF NOT EXISTS images (
    id             INTEGER PRIMARY KEY,   -- rowid alias, sequential appends
    file_name_hash TEXT    NOT NULL,      -- SHA-256 of canonical full path
    path           TEXT    NOT NULL,
    file_name      TEXT    NOT NULL,
    content_hash   TEXT    NOT NULL,      -- SHA-256 of the content
    mime_type      TEXT    NOT NULL,      -- sniffed from magic bytes
    size_bytes     INTEGER NOT NULL,
    width          INTEGER NOT NULL,      -- from the image header
    height         INTEGER NOT NULL,
    created_at     INTEGER,               -- birth time, NULL if unavailable
    modified_at    INTEGER,               -- unix seconds
    scanned_at     INTEGER NOT NULL,      -- this scan's grouping key
    metadata       TEXT                   -- JSON: exif + png_text namespaces
);
-- indexes on: file_name_hash, file_name, content_hash, modified_at,
--             scanned_at
```

- `file_name_hash` = identity of a *path* across scans (repeats once per
  scan, by design).
- `content_hash` = identity of the *bytes*; equal values mean
  byte-identical files anywhere in the collection.
- `scanned_at` is captured once per run — every row of one execution
  shares it, so it doubles as the scan's grouping key.
- `metadata` is a JSON object namespaced by source: `"exif"` (EXIF tags
  from JPEG/TIFF containers and PNG/WebP Exif chunks) and `"png_text"`
  (PNG tEXt/zTXt/iTXt/tIME chunks). Read it with `json_extract`.

## Why a file might not be in the index

A file exists on disk but produces no row only if it fails one of the
trust gates — checked in this order:

| Stage | Excluded when... | Cost |
|---|---|---|
| walker | file is **0 bytes** | one stat call |
| walker | **extension** not in `--extensions` | filtered before IO |
| worker | magic bytes are **not a known image format** (`infer`, content-based) | one 8 KB head read |
| worker | **image header is unreadable** (corrupt beyond the magic) | one header read |
| worker (enrichment) | **pixels don't decode** (truncated/corrupt body) | hash runs, embedding skipped |
| worker (enrichment) | image is **blank/near-blank** (pHash would be pure noise) | no pHash for that content |
| worker (enrichment) | image is **oversized** (> 300 MP safety ceiling) | no pHash/embedding for it |
| filesystem | **permission denied** / unreadable path | silently skipped |

A row that *is* stored can still lack a perceptual hash (undecodable or
blank content) — pHash completeness is tracked by `--generate-phash`'s
counters, and the SigLIP embedding is only stored for fully decodable
images.

## Notes

- **CPU only (for now)** — inference runs on ONNX Runtime's CPU
  execution provider, so anyone can use it. GPU execution providers
  (CUDA/DirectML/CoreML) can be enabled later via `ort` cargo features.
- **Model download** — the SigLIP 2 q4f16 ONNX (~55 MB) is fetched from
  Hugging Face into the official HF cache on first use of
  `--generate-embeddings` / `--download-model`; `--download-model`
  prefetches everything (vision + text encoder + tokenizer) for
  air-gapped use. Apache-2.0 weights.
- **No migration machinery** — after upgrading, delete the db and
  `.safegraphs` files and re-run the passes for a clean rebuild.
- **Why q4f16 and not int8** — the int8 ONNX exports use `ConvInteger`
  ops that ONNX Runtime's CPU provider does not implement; q4f16
  (4-bit weights, fp16 activations) is smaller *and* actually runs.
  Full research: [docs/embedding-model-research.md](docs/embedding-model-research.md).

## Keeping it fresh with cron

The passes are designed for unattended use: idempotent, append-only,
quiet on errors, safe to re-run. A crontab like this keeps your index
perpetually up to date:

```cron
# rescan nightly at 02:00 and refresh pHash embeddings
0  2 * * *  cd /path/to/library && imgindex --create . >> imgindex.log 2>&1
30 2 * * *  cd /path/to/library && imgindex --generate-phash >> imgindex.log 2>&1
# weekly — SigLIP embeddings (heavier; CPU inference per new content)
0  3 * * 1  cd /path/to/library && imgindex --generate-embeddings >> imgindex.log 2>&1
```

After that, asking your coding agent *"which photos disappeared since
October?"*, *"do I have duplicates of this picture?"* or *"what new
screenshots appeared this week?"* becomes a one-query job over the
database.

## Docker

The release workflow publishes a **multi-arch image** (linux/amd64 +
linux/arm64) to GHCR:

```bash
docker pull ghcr.io/OWNER/imgindex:latest
docker run --rm -v "$PWD:/data" -w /data ghcr.io/OWNER/imgindex --create /data
```

The runtime image is distroless Debian 13 (`gcr.io/distroless/cc-debian13`):
glibc + libstdc++ only — no shell, no package manager, ~20 MB. The
binary is fully self-contained apart from those two libraries. Mount a
writable directory as `/data` (the db and vector indexes are written to
the working directory); for the model cache on `--generate-embeddings` /
`--search`, also mount a volume on `/home/nonroot/.cache`.

For a Debian 13 base with a shell instead, swap the base image for
`debian:trixie-slim` and add `libstdc++6`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
