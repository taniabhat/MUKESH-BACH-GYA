# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# System Dependencies
# ============================================================

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Working Directory
# ============================================================

WORKDIR /app

# ============================================================
# Application Directories
# ============================================================

RUN mkdir -p \
    /app/models \
    /app/logs \
    /tmp/research_discovery

# ============================================================
# Dependencies
# ============================================================

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

# ============================================================
# Application Source
# ============================================================

COPY . .

# ============================================================
# Runtime Environment
# ============================================================

ENV STORAGE_DIR=/tmp/research_discovery
ENV MODEL_CACHE_DIR=/app/models
ENV LOG_LEVEL=INFO
ENV DEBUG=false

# ============================================================
# Network
# ============================================================

EXPOSE 8000

# ============================================================
# Health Check
# ============================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
CMD curl -f http://localhost:8000/health || exit 1

# ============================================================
# Runtime
# ============================================================

CMD [
  "uvicorn",
  "research_discovery.api:app",
  "--host",
  "0.0.0.0",
  "--port",
  "8000",
  "--workers",
  "2"
]