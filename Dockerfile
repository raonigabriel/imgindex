# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# imgindex runtime image — Debian 13 (trixie) slim, runs as uid 1000.
# Expecting the per-architecture binaries in the build context
# (imgindex-linux-amd64, imgindex-linux-arm64); the release workflow
# (release.yml) prepares them.
#
# All state (database, vector indexes, model cache) is written to /data —
# mount a directory there:
#   docker run --rm -v "$PWD:/data" imgindex --create /data
# ---------------------------------------------------------------------------
FROM debian:trixie-slim

# TARGETARCH is a BuildKit predefined arg (amd64 / arm64 per --platform).
ARG TARGETARCH

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 1000 --create-home --user-group imgindex

COPY imgindex-linux-${TARGETARCH} /usr/local/bin/imgindex

ENV HF_HOME=/data/.cache/huggingface
USER 1000:1000
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/imgindex"]
