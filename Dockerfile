# Multi-stage, non-root production image for gridcast (< 400 MB target).
# Base image pinned by multi-arch manifest list SHA-256 digest (python:3.10-slim).
#
# Runtime data policy (baked operational snapshot + optional read-only mount overlay, single replica):
# - Committed data/curated, data/mix, and data/scoreboard snapshot files are copied into the
#   image so stateless deployments (such as Render free tier) boot and serve HTTP 200 on
#   /health, /ready, and /forecast immediately without external API dependency.
# - In container orchestration or CI with live/updated stores, mount them read-only at runtime
#   to cleanly overlay the baked snapshot:
#     docker run --rm -p 8000:8000 \
#       -v "$(pwd)/data/curated:/app/data/curated:ro" \
#       -v "$(pwd)/data/mix:/app/data/mix:ro" \
#       -v "$(pwd)/data/scoreboard:/app/data/scoreboard:ro" \
#       gridcast:latest
# - Deployment topology: replicas: 1 (--workers 1). ModelSnapshot reload
#   state and Prometheus metrics registry are held in-process.

FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d AS builder

WORKDIR /build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.lock ./
RUN pip install --require-hashes --no-deps -r requirements.lock

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation . \
    && rm -rf /opt/venv/lib/python3.10/site-packages/pip* \
              /opt/venv/lib/python3.10/site-packages/setuptools* \
              /opt/venv/bin/pip* \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} +


FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d AS runtime

# LightGBM requires OpenMP runtime (libgomp1) on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 gridcast \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin gridcast

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy pinned model, interval, runtime metadata, and evaluation fixture artifacts.
COPY --chown=gridcast:gridcast data/models ./data/models
COPY --chown=gridcast:gridcast data/eval ./data/eval

# Copy committed operational snapshot (curated, mix, scoreboard) so stateless
# deployments (e.g. Render free tier) serve immediately; runtime :ro mounts overlay these.
COPY --chown=gridcast:gridcast data/curated ./data/curated
COPY --chown=gridcast:gridcast data/mix ./data/mix
COPY --chown=gridcast:gridcast data/scoreboard ./data/scoreboard

# Build-time runtime integrity check: enforce LightGBM version and artifact hashes.
RUN python -c "import json, lightgbm as lgb; from pathlib import Path; r = json.loads(Path('data/models/runtime.json').read_text(encoding='utf-8')); assert lgb.__version__ == r['lightgbm_version'], f'LightGBM {lgb.__version__} != {r[\"lightgbm_version\"]}'; from gridcast.serve import verify_startup_environment; verify_startup_environment()"

USER gridcast

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; port = os.environ.get('PORT', '8000'); resp = urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3); sys.exit(0 if resp.status == 200 else 1)"]

CMD ["sh", "-c", "exec uvicorn gridcast.serve:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
