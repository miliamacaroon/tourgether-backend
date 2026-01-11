FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install to a virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/dependencies -r requirements.txt

# ============================================
# Final stage - much smaller
# ============================================
FROM python:3.10-slim

# Install only runtime dependencies (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /app/dependencies /usr/local/lib/python3.10/site-packages/

# Copy application code
COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]