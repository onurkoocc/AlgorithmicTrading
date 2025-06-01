#!/bin/bash

set -e

echo "Rebuilding Feature Engine with TA-Lib support..."

echo "Stopping feature-engine service..."
docker-compose stop feature-engine

echo "Removing old feature-engine container..."
docker-compose rm -f feature-engine

echo "Building feature-engine with TA-Lib..."
docker-compose build --no-cache feature-engine

echo "Starting feature-engine service..."
docker-compose up -d feature-engine

echo "Waiting for service to initialize..."
sleep 15

echo "Checking service status..."
docker-compose ps feature-engine

echo "Checking logs for errors..."
docker-compose logs --tail=100 feature-engine | grep -E "ERROR|Failed|Exception" || echo "No errors found"

echo ""
echo "Testing TA-Lib installation inside container..."
docker-compose exec feature-engine python -c "import talib; print(f'TA-Lib version: {talib.__version__}')"

echo ""
echo "Rebuild complete!"
echo ""
echo "Monitor feature calculation:"
echo "  docker-compose logs -f feature-engine"
echo ""
echo "Test TA-Lib integration:"
echo "  python scripts/test-talib-integration.py"