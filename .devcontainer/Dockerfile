# --- stage 1: source Julia officielle
FROM julia AS jl

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Dépendances utiles (ajuste selon ta base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

# Vérif
RUN julia --version