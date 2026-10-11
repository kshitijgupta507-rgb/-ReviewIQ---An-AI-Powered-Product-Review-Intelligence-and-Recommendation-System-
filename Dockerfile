# ─── ReviewIQ Dockerfile ──────────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean.
# Stage 1: build dependencies
# Stage 2: runtime image

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY pyproject.toml .

# Install the reviewiq package (editable not needed in container)
RUN pip install --no-cache-dir -e . --no-deps

# Create necessary directories
RUN mkdir -p data/raw data/interim data/processed data/external \
             models/classical models/embeddings models/transformers \
             logs mlruns

# Expose API port
EXPOSE 8000

# Default command: start the API
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
