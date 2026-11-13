# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Runtime image for imgindex — distroless Debian 13, non-root.
#
# Contains ONLY glibc + libstdc++ (the two shared libraries the binary
# links); the binary itself is otherwise self-contained (SQLite, SQLCipher,
# USearch and the image decoders are all compiled in). No shell, no
# package manager — attack surface is minimal.
#
# The release workflow (release.yml) expects the per-architecture binaries
# in the build context (imgindex-linux-amd64, imgindex-linux-arm64) and
# builds a single multi-arch manifest for linux/amd64 + linux/arm64.
#
# Local build (amd64):
#   cargo build --release
#   mkdir -p docker-context
#   cp target/release/imgindex docker-context/imgindex-linux-x86_64
#   cp Dockerfile docker-context/
#   docker build --build-arg TARGETARCH=amd64 docker-context -t imgindex
#
# Run: the tool writes image_index.db (+ .safegraphs vector indexes) to
# its working directory, so mount a writable volume:
#   docker run --rm -v "$PWD:/data" -w /data imgindex --create /data
# ---------------------------------------------------------------------------
FROM gcr.io/distroless/cc-debian13:nonroot
ARG TARGETARCH
COPY imgindex-linux-${TARGETARCH} /usr/local/bin/imgindex
ENTRYPOINT ["/usr/local/bin/imgindex"]
