FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy requirements first
COPY Backend/requirements/ /app/requirements/
COPY lib-repo/ /app/lib-repo/

# Install Python dependencies in specific order
RUN pip install --no-cache-dir \
    "pydantic-core==2.33.2" \
    "pydantic==2.11.4" \
    "pydantic-settings>=2.0.0" \
    "langchain-core>=0.3.52,<1.0.0" \
    "langchain>=0.2.0" \
    "langchain-community>=0.0.38" \
    "langchain-ollama>=0.3.2" \
    "ollama>=0.1.0" \
    "PyYAML>=6.0.2" \
    "SQLAlchemy>=1.4,<3.0" \
    "aiohttp>=3.8.3,<4.0.0" \
    "numpy>=1.0.0,<2.0.0" \
    "chromadb>=0.4.22" \
    "sentence-transformers>=2.2.2" \
    "redis>=5.0.1" \
    "pypdf>=3.0.0" \
    "python-multipart>=0.0.9"

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
    OLLAMA_BASE_URL=http://host.docker.internal:11434

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 