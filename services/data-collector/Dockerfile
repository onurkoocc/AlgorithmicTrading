FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY services/data-collector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules first
COPY shared /app/shared

# Copy service code
COPY services/data-collector /app/services/data-collector

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Run the service
CMD ["python", "-u", "/app/services/data-collector/main.py"]