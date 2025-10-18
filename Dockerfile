FROM python:3.11-slim

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-pinned.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-pinned.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash aicra
USER aicra

# Default command
CMD ["aicra", "--help"]
