FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/feature-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared

COPY services/feature-engine /app/services/feature-engine

ENV PYTHONPATH=/app

CMD ["python", "-u", "/app/services/feature-engine/main.py"]