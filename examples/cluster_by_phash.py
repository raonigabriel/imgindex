#!/usr/bin/env python3
"""Cluster imgindex's perceptual-hash embeddings with USearch.

Reads the two artifacts `imgindex --create` / `--generate-phash` produce
in the current directory:

  image_index.db           SQLite log: content_hash -> known paths
  image_index.safegraphs   container holding the serialized pHash index

then (1) demonstrates exact KNN with the USearch Python library and
(2) clusters every content by Hamming distance (single-linkage) so that
near-duplicate images end up in the same group.

Usage:
    pip install usearch numpy
    python3 cluster_by_phash.py [--db DB] [--index CONTAINER]
                                [--threshold D] [--knn K] [--limit N]

Distances are Hamming: 0 = re-encode of the same pixels, <=3
near-identical, 5..9 same-scene variants, >=12 visually distinct.
"""
import argparse
import os
import sqlite3
import struct
import sys
import tempfile

import numpy as np
from usearch.index import Index

MAGIC = b"IMGIDXVEC1"


def extract_phash_section(container):
    """Pull the pHash index bytes out of the imgindex container.

    Layout: b"IMGIDXVEC1" | u64 phash_len | phash | u64 sem_len | semantic.
    The semantic (SigLIP) section is ignored here.
    """
    data = open(container, "rb").read()
    if data[: len(MAGIC)] != MAGIC:
        sys.exit(f"{container}: not an imgindex container")
    (pos, plen) = (10, struct.unpack_from("<Q", data, 10)[0])
    section = data[pos + 8 : pos + 8 + plen]
    if not section:
        sys.exit("container has no pHash section — run --generate-phash first")
    return section


def load_paths(db_path):
    """content_hash -> every path ever seen carrying it (newest first)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT content_hash, path FROM images ORDER BY scanned_at DESC"
    ).fetchall()
    paths = {}
    for content_hash, path in rows:
        paths.setdefault(int(content_hash[:16], 16), []).append(path)
    conn.close()
    return paths


def main():
    ap = argparse.ArgumentParser(description="Cluster imgindex pHash embeddings with USearch")
    ap.add_argument("--db", default="image_index.db")
    ap.add_argument("--index", default="image_index.safegraphs")
    ap.add_argument("--threshold", type=int, default=12,
                    help="max Hamming distance inside a cluster")
    ap.add_argument("--knn", type=int, default=3,
                    help="neighbors shown in the KNN demo")
    args = ap.parse_args()

    paths = load_paths(args.db)
    section = extract_phash_section(args.index)

    tmp = os.path.join(tempfile.gettempdir(), "imgindex_phash.usearch")
    open(tmp, "wb").write(section)
    index = Index(ndim=64, metric="hamming")
    index.load(tmp)
    os.remove(tmp)
    print(f"loaded pHash index: {index.size} embeddings, 64-bit Hamming\n")

    # All keys the index knows (uint64), and their 8-byte vectors.
    keys = np.array(sorted(int(k) for k in index.keys), dtype=np.uint64)
    vectors = np.stack([np.frombuffer(index.get(int(k)), dtype=np.uint8) for k in keys])

    # --- exact KNN demo (USearch search, one query vector) -------------
    q = index.get(int(keys[0]))
    print(f"KNN demo — query {os.path.basename(paths[int(keys[0])][0])!r}, "
          f"top {args.knn} neighbors:")
    for match in index.search(q, args.knn + 1):
        if int(match.key) == int(keys[0]):
            continue  # the query itself
        p = paths.get(int(match.key), ["<no live path>"])[0]
        print(f"  hamming={int(match.distance):3d}  {os.path.basename(p)}")

    # --- clustering: brute-force Hamming, single-linkage ---------------
    bits = np.unpackbits(vectors, axis=1).astype(np.int8)
    n = len(keys)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.count_nonzero(bits[i] != bits[j]))
            if d <= args.threshold:
                pairs.append((i, j, d))
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = [sorted(g) for g in groups.values() if len(g) > 1]
    print(f"\nclustering: {n} contents, {len(clusters)} near-duplicate "
          f"cluster(s) at hamming <= {args.threshold}:")
    for g in clusters:
        names = [os.path.basename(paths.get(int(keys[i]), ["?"])[0]) for i in g]
        shown = ", ".join(names[:5]) + (f" (+{len(names) - 5} more)" if len(names) > 5 else "")
        print(f"  cluster ({len(g)}): {shown}")

    # smallest Hamming distance between two different contents, overall
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.count_nonzero(bits[i] != bits[j]))
            if best is None or d < best[0]:
                best = (d, i, j)
    if best:
        d, i, j = best
        print(f"\ntightest pair: hamming={d}")
        for i2 in (i, j):
            print(f"  {os.path.basename(paths.get(int(keys[i2]), ['?'])[0])}")


if __name__ == "__main__":
    main()
