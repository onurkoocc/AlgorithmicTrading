#!/bin/bash

set -e

echo "Setting up Algorithmic Trading Application - Phase 1..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p config/strategies
mkdir -p config/models
mkdir -p services/data-collector/collectors
mkdir -p shared/utils
mkdir -p shared/connectors
mkdir -p shared/models
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p models/checkpoints
mkdir -p results/backtests
mkdir -p results/reports
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cat > .env << EOF
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=true
TZ=UTC
EOF
    echo "Please update .env with your API keys and configuration"
fi

# Set permissions
echo "Setting permissions..."
chmod +x scripts/*.sh 2>/dev/null || true

# Build Docker images
echo "Building Docker images..."
docker-compose build --no-cache data-collector

# Start infrastructure services
echo "Starting infrastructure services..."
docker-compose up -d questdb redis

# Wait for services to be ready with retry logic
echo "Waiting for services to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "Checking services (attempt $ATTEMPT/$MAX_ATTEMPTS)..."

    # Check if QuestDB is responding
    if curl -s http://localhost:9000/status > /dev/null 2>&1; then
        echo "✓ QuestDB is ready"
        break
    else
        echo "Waiting for QuestDB to start..."
        sleep 2
    fi
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "⚠️  QuestDB did not start in time. Checking logs..."
    docker-compose logs questdb
    echo ""
    echo "You may need to:"
    echo "1. Check if ports 9000/9009 are already in use"
    echo "2. Increase Docker memory allocation"
    echo "3. Run 'docker-compose down -v' and try again"
    exit 1
fi

# Additional wait for stability
sleep 5

# Test QuestDB connection and create tables
echo "Initializing database..."
docker-compose run --rm data-collector python -c "
import sys
sys.path.append('/app')
max_retries = 5
retry_count = 0

while retry_count < max_retries:
    try:
        from shared.connectors.questdb import QuestDBConnector
        db = QuestDBConnector()
        db.create_tables()
        print('Database tables created successfully')
        db.close()
        break
    except Exception as e:
        retry_count += 1
        if retry_count == max_retries:
            print(f'Failed to create tables after {max_retries} attempts: {e}')
            sys.exit(1)
        else:
            print(f'Connection attempt {retry_count} failed, retrying...')
            import time
            time.sleep(2)
"

# Start monitoring services
echo "Starting monitoring services..."
docker-compose up -d prometheus grafana

echo ""
echo "✅ Setup complete!"
echo ""
echo "Services status:"
docker-compose ps
echo ""
echo "Next steps:"
echo "1. Update .env file with your Binance API credentials (optional)"
echo "2. Run 'docker-compose up -d data-collector' to start data collection"
echo "3. Access Grafana at http://localhost:3000 (admin/admin)"
echo "4. Access QuestDB console at http://localhost:9000"
echo ""
echo "To download historical data:"
echo "  python scripts/download_historical.p