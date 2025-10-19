# Multi-stage Dockerfile for AICRA
FROM python:3.10-slim as builder

# Set build arguments
ARG PYTHON_VERSION=3.10
ARG PIP_VERSION=23.2.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aicra && useradd -r -g aicra aicra

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements/ requirements/
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==${PIP_VERSION} && \
    pip install --no-cache-dir -r requirements/base.txt

# Copy source code
COPY aicra/ aicra/
COPY tests/ tests/

# Install the package
RUN pip install --no-cache-dir -e .

# Runtime stage
FROM python:3.10-slim as runtime

# Set runtime arguments
ARG PYTHON_VERSION=3.10

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aicra && useradd -r -g aicra aicra

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create artifacts directory
RUN mkdir -p /app/artifacts && chown -R aicra:aicra /app

# Switch to non-root user
USER aicra

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import aicra; print('AICRA is healthy')" || exit 1

# Default command
CMD ["python", "-m", "aicra", "--help"]