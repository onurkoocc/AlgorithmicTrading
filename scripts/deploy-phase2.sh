#!/bin/bash

set -e

echo "Deploying Phase 2: Feature Engineering..."

echo "Building feature-engine service..."
docker-compose build feature-engine

echo "Starting feature-engine service..."
docker-compose up -d feature-engine

echo "Waiting for service to initialize..."
sleep 10

echo "Checking service status..."
docker-compose ps feature-engine

echo "Checking logs..."
docker-compose logs --tail=50 feature-engine

echo ""
echo "Phase 2 deployment complete!"
echo ""
echo "Monitor feature calculation:"
echo "  docker-compose logs -f feature-engine"
echo ""
echo "Check feature quality:"
echo "  docker exec -it algo-redis redis-cli get quality:BTCUSDT:1h"
echo ""
echo "Verify features in QuestDB:"
echo "  http://localhost:9000"
echo "  SELECT * FROM features_1h WHERE symbol='BTCUSDT' ORDER BY timestamp DESC LIMIT 10;"