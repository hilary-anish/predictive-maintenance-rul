# ── Stage 1: Build ─────────────────────────────────────────────────
# Why multi-stage? The build stage has compilers, headers, pip cache —
# stuff needed to INSTALL packages but not to RUN them.
# The final image only copies what's needed, cutting image size ~50%.

FROM python:3.11-slim AS builder

# Why these system packages?
# - build-essential: C compiler for packages with C extensions (xgboost, numpy)
# - git: some pip packages pull from git repos during install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first (before code).
# Why? Docker caches layers. If pyproject.toml hasn't changed,
# Docker reuses the cached pip install layer — saves minutes on rebuilds.
COPY pyproject.toml .

# Install dependencies (without the project itself)
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source code
COPY src/ src/
COPY pyproject.toml .

# Install the project itself (editable, lightweight — deps already installed)
RUN pip install --no-cache-dir --no-deps -e .

# Create data directories (will be mounted as volumes in production)
RUN mkdir -p data/raw data/processed

# Why non-root user?
# Running as root inside a container is a security risk.
# If someone exploits the API, they get root access to the container.
# A non-root user limits the blast radius.
RUN useradd --create-home appuser
USER appuser

# Expose the API port
EXPOSE 8000

# Health check: Kubernetes and Docker Compose use this to know
# if the container is ready to receive traffic.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
# --host 0.0.0.0: listen on all interfaces (required inside containers)
# --workers 1: single worker for ML inference (model is in GPU memory)
CMD ["uvicorn", "pdm.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
