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
docker-compose up -d questdb redis prometheus grafana

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 15

# Test QuestDB connection and create tables
echo "Initializing database..."
docker-compose run --rm data-collector python -c "
import sys
sys.path.append('/app')
from shared.connectors.questdb import QuestDBConnector
try:
    db = QuestDBConnector()
    db.create_tables()
    print('Database tables created successfully')
    db.close()
except Exception as e:
    print(f'Failed to create tables: {e}')
    sys.exit(1)
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your Binance API credentials (optional)"
echo "2. Run 'docker-compose up -d data-collector' to start data collection"
echo "3. Access Grafana at http://localhost:3000 (admin/admin)"
echo "4. Access QuestDB console at http://localhost:9000"
echo ""
echo "To download historical data:"
echo "  python scripts/download_historical.py 30"