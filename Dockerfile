# ChemRxn-Intelligence Production Backend Dockerfile
# Optimized for RDKit, PyTorch, and FastAPI inference

FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8001

WORKDIR /app

# Install system dependencies required by RDKit graphics and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first to keep image lightweight and fast to pull on container hosts
# (or use standard PyTorch if deploying to GPU host like RunPod/Modal)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code and trained checkpoints
COPY chemrxn/ chemrxn/
COPY configs/ configs/
COPY checkpoints/ checkpoints/
COPY reports/ reports/
COPY pyproject.toml .

# Expose API port
EXPOSE 8001

# Run healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn chemrxn.api.app:app --host 0.0.0.0 --port ${PORT}"]
