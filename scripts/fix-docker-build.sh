#!/bin/bash

echo "Fixing Docker build issues..."

# Stop any running containers
docker-compose down

# Remove the version line from docker-compose.yml if it exists
if [ -f docker-compose.yml ]; then
    sed -i.bak '/^version:/d' docker-compose.yml
    echo "Removed obsolete version attribute from docker-compose.yml"
fi

# Ensure all required directories exist
mkdir -p services/data-collector/collectors
mkdir -p shared/utils
mkdir -p shared/connectors
mkdir -p shared/models
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p config

# Create empty __init__.py files
touch services/__init__.py
touch services/data-collector/__init__.py
touch services/data-collector/collectors/__init__.py
touch shared/__init__.py
touch shared/utils/__init__.py
touch shared/connectors/__init__.py
touch shared/models/__init__.py

echo "Directory structure fixed."
echo ""
echo "Now run:"
echo "  docker-compose build --no-cache"
echo "  docker-compose up -d"