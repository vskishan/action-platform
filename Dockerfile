FROM python:3.12-slim

WORKDIR /app

# Install curl (needed for health checks).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data.
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/

EXPOSE 8000

# Health check for the backend itself.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
