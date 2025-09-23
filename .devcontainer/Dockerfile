# --- stage 1: source Julia officielle
FROM julia AS jl

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    make git wget curl build-essential ca-certificates cmake pkg-config \
    python3-dev \
    gnuplot \
    graphviz \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Vérif
RUN julia --version