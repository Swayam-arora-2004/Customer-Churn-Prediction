# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/      ./src/
COPY app/      ./app/
COPY models/   ./models/
COPY data/     ./data/
COPY .env.example .env
COPY pyproject.toml .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Default command: Flask API
# Override with: docker run <image> streamlit run app/dashboard.py
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "--timeout", "120", \
     "app.app:create_app()"]
