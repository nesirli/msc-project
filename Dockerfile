# check=skip=FromPlatformFlagConstDisallowed
# syntax=docker/dockerfile:1
# =============================================================================
# Klebsiella pneumoniae AMR Prediction Pipeline - Docker Image
# Multi-stage build: base + conda envs + pipeline code
# =============================================================================
#
# Build:    docker build -t kleb-amr-pipeline .
# Run:      docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results kleb-amr-pipeline
# Dev:      docker compose run --rm pipeline bash
#
# NOTE: Uses linux/amd64 to ensure bioinformatics tool compatibility
#       (SPAdes, freebayes, kraken2 have no ARM64 builds)
# =============================================================================

FROM --platform=linux/amd64 condaforge/mambaforge:24.3.0-0 AS base

LABEL maintainer="Nasir Nasirli"
LABEL description="AMR prediction pipeline for Klebsiella pneumoniae"
LABEL version="1.0"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    pigz \
    unzip \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Remove any macOS resource fork files that might cause permission errors
# (SPAdes fails when it can't chmod ._* files in conda environments)
RUN find /app -name "._*" -type f -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 1: Install Snakemake
# ---------------------------------------------------------------------------
RUN mamba install -y -c conda-forge -c bioconda \
    snakemake=7.32 \
    && mamba clean -afy

# ---------------------------------------------------------------------------
# Stage 2: Pre-create all conda environments (cached layer)
# This is the slowest step; Docker caches it unless envs/ changes
# ---------------------------------------------------------------------------
COPY envs/ /app/envs/

# Create all pipeline conda environments in advance
# This avoids Snakemake creating them at runtime (slow + needs internet)
RUN for env_file in /app/envs/*.yaml; do \
        env_name=$(basename "$env_file" .yaml); \
        echo "=== Creating environment: $env_name ==="; \
        mamba env create -f "$env_file" -n "snakemake-$env_name" || \
        echo "WARNING: Failed to create $env_name (will be created at runtime)"; \
    done && \
    mamba clean -afy && \
    find /app -name "._*" -type f -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 3: Copy pipeline code
# ---------------------------------------------------------------------------
COPY Snakefile /app/
COPY config/ /app/config/
COPY rules/ /app/rules/
COPY scripts/ /app/scripts/
COPY utils/ /app/utils/
COPY data/metadata.csv /app/data/metadata.csv
COPY pytest.ini /app/
COPY requirements-dev.txt /app/
COPY tests/ /app/tests/

# Create output directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/assemblies \
             /app/data/reference /app/results /app/logs

# ---------------------------------------------------------------------------
# Stage 4: Default config for Docker (adjusted resources)
# ---------------------------------------------------------------------------
COPY config/config.yaml /app/config/config.yaml.default

# Create a Docker-specific config overlay script
RUN cat > /app/docker-entrypoint.sh << 'ENTRYPOINT_SCRIPT'
#!/bin/bash
set -euo pipefail

# Display pipeline info
echo "============================================================"
echo "  Klebsiella pneumoniae AMR Prediction Pipeline (Docker)"
echo "============================================================"
echo "  CPUs available: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/{print $2}')"
echo "  Snakemake: $(snakemake --version)"
echo "============================================================"

# Auto-adjust threads in config if THREADS env var is set
if [ -n "${THREADS:-}" ]; then
    echo "Setting threads to $THREADS"
    sed "s/threads: .*/threads: $THREADS/" /app/config/config.yaml > /tmp/config.yaml \
        && cat /tmp/config.yaml > /app/config/config.yaml \
        && rm /tmp/config.yaml
fi

# Final cleanup: remove any remaining macOS resource fork files that could cause SPAdes errors
# This catches any files that may have been created during conda environment unpacking
find /app/.snakemake /opt/conda -name "._*" -type f -delete 2>/dev/null || true

# Execute the provided command or default to snakemake
exec "$@"
ENTRYPOINT_SCRIPT

RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["snakemake", "--use-conda", "--cores", "all", "--jobs", "4"]
