FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    libgomp1 \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

WORKDIR /app

COPY services/feature-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared /app/shared

COPY services/feature-engine /app/services/feature-engine

ENV PYTHONPATH=/app

CMD ["python", "-u", "/app/services/feature-engine/main.py"]