FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and local libraries first
COPY Backend/requirements/ /app/requirements/
COPY lib-repo/ /app/lib-repo/

# Install Python dependencies using local wheels first
RUN pip install --no-cache-dir \
    --find-links=/app/lib-repo \
    numpy==1.26.4 \
    scipy==1.15.3 \
    scikit-learn==1.6.1 \
    torch==2.7.0 \
    transformers==4.51.3 \
    pandas==2.2.3 \
    sqlparse==0.5.3 \
    sqlglot==26.17.1 \
    pydantic-core==2.33.2 \
    pydantic==2.11.4 \
    langchain-core==0.3.59 \
    langchain==0.3.25 \
    langchain-community==0.3.24 \
    langchain-ollama==0.3.2 \
    PyYAML==6.0.2 \
    SQLAlchemy==2.0.40 \
    aiohttp==3.11.18 \
    chromadb==0.6.3 \
    sentence-transformers==4.1.0 \
    redis==6.0.0 \
    pypdf==5.5.0 \
    python-multipart==0.0.20 \
    faiss-cpu==1.11.0 \
    uvicorn[standard]==0.34.2 \
    fastapi==0.115.12 \
    langchain-chroma==0.2.3

FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY Backend/app/ /app/app/

# Create necessary directories
RUN mkdir -p /app/uploads /app/chroma_db && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app \
    ENVIRONMENT=development \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    PATH="/usr/local/bin:${PATH}"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 