FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install testing dependencies for CI
RUN pip install pytest pytest-asyncio pytest-cov pytest-benchmark pytest-timeout vaderSentiment beautifulsoup4 lxml

# Copy application code
COPY algotrading_agent/ ./algotrading_agent/
COPY config/ ./config/
COPY tests/ ./tests/
COPY main.py .

# Create data directory for persistence
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Expose port for health checks/monitoring
EXPOSE 8080

# Default command
CMD ["python", "main.py"]