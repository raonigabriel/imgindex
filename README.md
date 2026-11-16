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
SQLite database is 100% standard SQLite, `--skill` emits a
Claude-compatible SKILL.md (pipe it into your agent's skills
directory), and the USearch indexes are readable from Python with one
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

## Installation

### Prebuilt binaries

Every GitHub release publishes two Linux executables:
`imgindex-linux-amd64` (x86_64) and `imgindex-linux-arm64` (aarch64).
Everything is linked in — SQLite/SQLCipher, OpenSSL, USearch and ONNX
Runtime — so the only runtime requirements are a system with glibc
2.38 or newer and the standard C++ runtime (`libstdc++6`, present on
essentially every desktop and server distro). The SigLIP 2 model
downloads on first use into the Hugging Face cache.

```bash
# pick the asset matching your architecture
# (uname -m: x86_64 -> amd64, aarch64 -> arm64)
V=v0.17.0
curl -fLO "https://github.com/raonigabriel/imgindex/releases/download/${V}/imgindex-linux-amd64"
chmod +x imgindex-linux-amd64
sudo mv imgindex-linux-amd64 /usr/local/bin/imgindex   # or ~/.local/bin, no sudo
imgindex --version
```

### From source with cargo

With a Rust toolchain installed, one command checks out the repo,
compiles it and installs the binary to `~/.cargo/bin`:

```bash
cargo install --git https://github.com/raonigabriel/imgindex
```

Build prerequisites (needed at build time only — none of it ships in
the binary): a stable Rust via rustup, a C/C++ toolchain
(`build-essential`), `cmake` (USearch), `perl` and `pkg-config`
(vendored OpenSSL/SQLCipher), and `libssl-dev` (ort-sys's build-time
model downloader links the system OpenSSL). The first build is the
slow one — vendored OpenSSL and SQLCipher compile from source, USearch
builds via cmake, and the prebuilt ONNX Runtime is downloaded — after
that it's a normal cargo build.

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

imgindex --search "a small girl playing with dolls" --top 10
                                      # text search (after step 3)

imgindex --skill                      # SKILL.md for agents (content only —
                                      # pipe it into your skills directory)

  -e, --extensions <LIST>   extensions to consider (default = every
                            format sniffable by magic AND decodable:
                            PNG, JPEG/JFIF, WebP, TIFF, GIF, BMP, ICO,
                            PNM, QOI, DDS, EXR, Radiance HDR, farbfeld)
  --show-ignored <LEVEL>    per-file reject warnings on stderr:
                            severe (corrupt/undecodable + IO errors)
                            or all (also non-image content, empty)
  -k, --key <KEY>           SQLCipher encryption key [default: none]
  --top <N>                 result count for --search [default: 20]
```

Every pass prints its progress live. `--create` shows the scan root up
front, an aggregated counter line every 5 seconds (`processed /
indexed / ignored`, +delta in the last 5s), the total candidate count
when the folder walk finishes, and an end-of-run report with the
average processing time per file. The enrichment passes follow the
same pattern — an up-front line with the content count, a 5-second
ticker (`done / total`, per-group counters, +delta), and an end-of-run
report with the average time per content. Model fetching is plain,
append-only text — no colors, no cursor rewriting, cron/pipe safe:
each file gets an explicit `cached` or `downloaded` verdict (with size
and duration), downloads announce themselves and print one line per
10%, and one `model location` line shows where the files live in the
Hugging Face cache. A warm-cache run logs just the three `cached`
verdicts.

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

## `--skill` — a SKILL.md for AI agents

`imgindex --skill` prints a Claude-compatible
[Agent Skill](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
to stdout — **content only**: the user is responsible for piping it into
a real file if they want to install it:

```bash
mkdir -p ~/.claude/skills/imgindex
imgindex --skill > ~/.claude/skills/imgindex/SKILL.md
```

The output follows the Agent Skills format: YAML front matter with
`name` and `description` (what the skill does + when to trigger it —
this is what the agent loads at startup), then a markdown body with
everything needed to drive imgindex and query the database on their own:

- the **workflow**: scan → pHash → embeddings → search, as numbered
  steps with exact commands
- the **command reference** (`--create` and its flags, `--show-ignored`,
  `--key`, …)
- the **exact DDL** the tool executes (byte-for-byte the same constants
  that created the db — it cannot drift), with a comment on every column
- the index list with each index's purpose
- ready-made SQL recipes: latest-scan state, scan diffs, duplicate
  content, dimensions, `json_extract` over the metadata namespaces
- gotchas an agent would otherwise trip on (append-only history,
  `scanned_at`/`hostname` grouping, NULL `created_at`, `id` as rowid
  alias, encrypted-db notes, vectors living outside SQLite)
- a complete working Python example **embedded verbatim** (kept at
  [examples/cluster_by_phash.py](examples/cluster_by_phash.py)): it
  loads the pHash index, demonstrates KNN search, and clusters
  near-duplicate contents by Hamming distance

Install it once next to the agent's other skills — it has everything
needed to write correct queries without reading the imgindex source.

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
- **Absolute scores compress on real collections** — embeddings are
  dense, so for a given query everything tends to cluster in a narrow
  similarity band. The **ranking is what matters**: the best match sits
  at the top, and the gaps between consecutive results tell you where
  "similar" ends.

Contrast with the other identifiers: `content_hash` (exact bytes) and
`path_hash` (exact path) are equality checks — similarity is the
fuzzy, graded measure that turns "find me more like this" into a
ranked list.

## Database schema

The SQLite database holds **one plain table** — no extensions, no
virtual tables, openable with any sqlite3 client:

```sql
CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY,   -- rowid alias, sequential appends
    hostname     TEXT    NOT NULL,      -- machine that performed the scan
    path_hash    TEXT    NOT NULL,      -- SHA-256 of canonical full path
    path         TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,      -- SHA-256 of the content
    mime_type    TEXT    NOT NULL,      -- sniffed from magic bytes
    size_bytes   INTEGER NOT NULL,
    width        INTEGER NOT NULL,      -- from the image header
    height       INTEGER NOT NULL,
    created_at   INTEGER,               -- birth time, NULL if unavailable
    modified_at  INTEGER,               -- unix seconds
    scanned_at   INTEGER NOT NULL,      -- this scan's grouping key
    metadata     TEXT                   -- JSON: exif + png_text namespaces
);
-- indexes on: path_hash, content_hash, modified_at, scanned_at
```

- `hostname` = the machine that performed the scan, captured once per
  run: `hostname` + `scanned_at` identifies which run on which box
  produced a row (one database can be appended from several machines).
- `path_hash` = identity of a *path* across scans (repeats once per
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
| worker | magic bytes are **not a known image format** (`infer` + a built-in magic supplement, content-based) | one 8 KB head read |
| worker | format is a **real image the decoder set lacks** — HEIF/HEIC (Apple), AVIF, JPEG 2000, JPEG XL — reported as `unsupported` | sniff only |
| worker | **image header is unreadable** (corrupt beyond the magic) | one header read |
| worker (enrichment) | **pixels don't decode** (truncated/corrupt body) | hash runs, embedding skipped |
| worker (enrichment) | image is **blank/near-blank** (pHash would be pure noise) | no pHash for that content |
| worker (enrichment) | image is **oversized** (> 300 MP safety ceiling) | no pHash/embedding for it |

Not in the default extension list on purpose: **TGA** (no reliable magic
bytes, so it cannot be sniffed — and untrustworthy bytes are never
accepted), and the decoder-less formats above (SVG, HEIC, AVIF, JPEG
2000, JPEG XL). Real HEIC/HEIF support requires the external `libheif`
C library, which would break the self-contained binary.
| filesystem | **permission denied** / unreadable path | silently skipped |

A row that *is* stored can still lack a perceptual hash (undecodable or
blank content) — pHash completeness is tracked by `--generate-phash`'s
counters, and the SigLIP embedding is only stored for fully decodable
images.

## The model files and the Hugging Face cache

The ML side — `--generate-embeddings`, `--search` and
`--download-model` — shares three files from one Hugging Face repo,
[`onnx-community/siglip2-base-patch32-256-ONNX`](https://huggingface.co/onnx-community/siglip2-base-patch32-256-ONNX)
— SigLIP 2 base, quantized to **q4f16** (4-bit weights, fp16
activations). Why q4f16 and not the smaller-looking int8 exports: int8
uses `ConvInteger` ops that ONNX Runtime's CPU provider does not
implement, so q4f16 is not just smaller, it actually runs. Full
research: [docs/embedding-model-research.md](docs/embedding-model-research.md).

| File | Size | Loaded by |
|---|---|---|
| `onnx/vision_model_q4f16.onnx` | ~56 MB | `--generate-embeddings` (image tower) |
| `onnx/text_model_q4f16.onnx` | ~423 MB | `--search` (text tower) |
| `tokenizer.json` | ~33 MB | `--search` (query tokenization) |

`--generate-embeddings` and `--download-model` deliberately fetch **all
three** (warm-up semantics), so a library indexed once can be searched
with no further downloads — `--search` itself loads only the text
tower and the tokenizer. Apache-2.0 weights. Everything lands in the
official Hugging Face cache, content-addressed and deduplicated:

```
<cache>/hub/models--onnx-community--siglip2-base-patch32-256-ONNX/
├── snapshots/<commit>/onnx/vision_model_q4f16.onnx   → ../../blobs/<etag>
├── snapshots/<commit>/onnx/text_model_q4f16.onnx     → ../../blobs/<etag>
└── snapshots/<commit>/tokenizer.json                 → ../../blobs/<etag>
```

That is the same layout every HF client uses, so imgindex, `hf
download`, `transformers` and any Python tool all read — and share —
the same copy of the model.

**Where the cache is** — resolved in this order: `HF_HUB_CACHE` →
`HUGGINGFACE_HUB_CACHE` → `$HF_HOME/hub` → `$XDG_CACHE_HOME/huggingface`
→ `~/.cache/huggingface`. Two more variables matter for special
setups: `HF_ENDPOINT` (mirror/proxy base URL instead of
huggingface.co) and `HF_TOKEN` / `HF_TOKEN_PATH` (authenticated
endpoints).

**Offline behavior** — once the three files are cached,
`--generate-embeddings` and `--search` run with the network completely
disabled, reading the cache without writing anything (the online path,
by contrast, revalidates against the Hub and writes lock files even on
a cache hit). That is what makes `--download-model` a complete air-gap
prep command.

## Notes

- **CPU only (for now)** — inference runs on ONNX Runtime's CPU
  execution provider, so anyone can use it. GPU execution providers
  (CUDA/DirectML/CoreML) can be enabled later via `ort` cargo features.
- **No migration machinery** — after upgrading, delete the db and
  `.safegraphs` files and re-run the passes for a clean rebuild.

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

Images are published for **linux/amd64** and **linux/arm64** at
`ghcr.io/raonigabriel/imgindex`. The examples below use `:latest`,
which always points at the newest release (CI keeps the alias in sync
on every version push). If you want reproducibility, pin a version
instead: `0.17` follows patch releases of this minor, `0.17.0` is
frozen to exactly this version. The container is Ubuntu 26.04 running
as **uid 1000** (the base image's default user, whose home is
`/home/ubuntu`) with nothing but `ca-certificates` and `libstdc++6`
added — the binary is self-contained.

### How the container finds things

Every `docker run` is determined by three facts. Once they're clear,
the mounts in the examples below explain themselves:

1. **The scan root is a command-line argument** (`--create <ROOT>`).
   It can be any path as long as it exists *inside the container* —
   whatever you want scanned must be bind-mounted first.
2. **All state is written to the current working directory** —
   `image_index.db` and `image_index.safegraphs` always land in the
   CWD, never next to the scanned files. The image's default CWD is
   `/data`; point `-w`/`--workdir` wherever you want the state.
3. **The model cache is `$HOME/.cache/huggingface`** — inside the
   container that is `/home/ubuntu/.cache/huggingface` (`HF_HOME` and
   `HF_HUB_CACHE` are respected). The SigLIP 2 weights total ~512 MB
   (see [the model section](#the-model-files-and-the-hugging-face-cache)
   for the file breakdown); bind-mounting your existing host cache
   there means the download happens once and is shared with host
   tools (`hf download`, transformers, Python) instead of being thrown
   away with every `--rm` container.

Putting those together, this is what each pass actually needs —
everything else in the examples below is just arranging mounts to
provide exactly these:

| Pass | Library (read-only is fine) | State = the CWD (writable) | Model cache |
|---|---|---|---|
| `--create` | yes | write | — |
| `--generate-phash` | yes — re-opens the files pass 1 recorded | write | — |
| `--generate-embeddings` | yes — same | write | read (write only when a model file is missing) |
| `--search` | no — paths come from the db | read | read (write only when a model file is missing) |

### The simple way — mount your home at the container's home

One mount makes your home the container's home: the library, the
state folder and the model cache are then all just ordinary files in
your home directory. This is the easiest setup and the recommended
one:

```bash
mkdir -p ~/imgindex

# Pass 1 — scan ~/Pictures. --create only reads the library; the
# database is written to the working directory (-w), which through the
# home mount is ~/imgindex on the host.
docker run --rm \
  -v "$HOME:/home/ubuntu" \
  -w /home/ubuntu/imgindex \
  ghcr.io/raonigabriel/imgindex:latest --create /home/ubuntu/Pictures

# Pass 3 — SigLIP 2 embeddings. The model downloads into
# ~/.cache/huggingface — which is the mounted home, so it is shared
# with host tools and survives the container being removed.
# (Pass 2, --generate-phash, needs no model; same command shape.)
docker run --rm \
  -v "$HOME:/home/ubuntu" \
  -w /home/ubuntu/imgindex \
  ghcr.io/raonigabriel/imgindex:latest --generate-embeddings

# Text search over the embeddings
docker run --rm \
  -v "$HOME:/home/ubuntu" \
  -w /home/ubuntu/imgindex \
  ghcr.io/raonigabriel/imgindex:latest --search "a small girl playing with dolls" --top 10
```

- The container runs as uid 1000 to match a typical first Linux user,
  so everything it writes belongs to you. If your host UID differs,
  add `--user "$(id -u):$(id -g)"` and make the involved folders
  writable by that uid.
- **Paths stored in the database are the container-side paths**
  (`/home/ubuntu/Pictures/...`). That is invisible as long as every
  pass over one database runs with the same mounts — but don't mix
  native and container runs on the same database: the same files seen
  through different paths are appended as different observations.

### The least-privilege alternative

Handing the container your whole home is convenient but broad. If you
prefer minimal mounts, mount only what each pass actually touches —
the library (read-only), one writable state folder, and the HF cache
for anything that runs the model:

```bash
mkdir -p ~/imgindex

# Pass 1 — a read-only library mount is enough: --create only reads it,
# and no model is involved. State goes to the default CWD, /data.
docker run --rm \
  -v "$HOME/Pictures:/library:ro" \
  -v "$HOME/imgindex:/data" \
  ghcr.io/raonigabriel/imgindex:latest --create /library

# Pass 3 — two extra mounts, both required:
#   * the HF cache, so the model is not re-downloaded (~512 MB) into
#     the ephemeral container — note it goes to the REAL cache path,
#     /home/ubuntu/.cache/huggingface, matching the container's HOME;
#   * the library again: the enrichment pass re-opens the files
#     recorded by pass 1 — without this mount nothing gets embedded.
docker run --rm \
  -v "$HOME/Pictures:/library:ro" \
  -v "$HOME/imgindex:/data" \
  -v "$HOME/.cache/huggingface:/home/ubuntu/.cache/huggingface" \
  ghcr.io/raonigabriel/imgindex:latest --generate-embeddings

# Search needs the state folder and the model (text tower + tokenizer),
# but not the library — result paths are looked up in the database.
docker run --rm \
  -v "$HOME/imgindex:/data" \
  -v "$HOME/.cache/huggingface:/home/ubuntu/.cache/huggingface" \
  ghcr.io/raonigabriel/imgindex:latest --search "a small girl playing with dolls" --top 10
```

A few properties worth knowing, all verified against the published
image:

- Keep the cache mount **writable** for normal runs: even a warm cache
  hit revalidates against the Hub and touches the cache's lock files —
  a read-only mount fails at the first model access (verified: the
  failure is exactly `Read-only file system`). A read-only cache mount
  works only in the air-gap case below.
- Once the model is cached, everything also runs with the network
  disabled (`--network none`) — cache reads with no downloads and no
  writes at all, which is the one case where a read-only cache mount
  is fine.

### Or: one folder for everything, model included

If you prefer a single self-contained directory — database, vector
index *and* model — point `HF_HOME` inside the state mount instead of
mounting the cache separately:

```bash
docker run --rm \
  -v "$HOME/Pictures:/library:ro" \
  -v "$HOME/imgindex:/data" \
  -e HF_HOME=/data/hf \
  ghcr.io/raonigabriel/imgindex:latest --generate-embeddings
```

The model then lives under `~/imgindex/hf/hub/` on the host, and the
folder is trivially movable/backable as a unit. (Use the same
`-e HF_HOME=/data/hf` on every pass that runs the model.)

## License

Apache-2.0 — see [LICENSE](LICENSE).
