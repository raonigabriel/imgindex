# ---------------------------------------------------------------------------
# imgindex runtime image — Ubuntu 26.04, runs as uid 1000.
# Expecting the per-architecture binaries in the build context
# (imgindex-linux-amd64, imgindex-linux-arm64); the release workflow
# (release.yml) prepares them.
#
# The release build container is ALSO ubuntu:26.04 (release.yml): the
# binary's glibc symbol floor comes from the build container, so build
# and runtime bases must move together. The docker job smoke-tests the
# pushed image to keep that invariant honest.
#
# All state (database, vector indexes, model cache) is written to /data —
# mount a directory there:
#   docker run --rm -v "$PWD:/data" imgindex --create /data
# ---------------------------------------------------------------------------
FROM ubuntu:26.04

# TARGETARCH is a BuildKit predefined arg (amd64 / arm64 per --platform).
ARG TARGETARCH

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      libstdc++6 \
 && rm -rf /var/lib/apt/lists/*
# ubuntu:26.04 ships a default 'ubuntu' user with uid 1000 — reuse it.

COPY imgindex-linux-${TARGETARCH} /usr/local/bin/imgindex

USER 1000:1000
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/imgindex"]
