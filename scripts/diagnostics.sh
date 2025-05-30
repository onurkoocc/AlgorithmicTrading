#!/bin/bash

echo "Running Algorithmic Trading System Diagnostics..."
echo "================================================"

# Check Docker status
echo ""
echo "1. Docker Status:"
docker version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Docker is running"
    docker info | grep -E "Server Version|Operating System|Total Memory|CPUs"
else
    echo "✗ Docker is not running or not accessible"
    exit 1
fi

# Check port availability
echo ""
echo "2. Port Availability:"
PORTS=(9000 9009 6379 3000 9090)
for port in "${PORTS[@]}"; do
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "✗ Port $port is in use by another process"
            lsof -i :$port | grep LISTEN
        else
            echo "✓ Port $port is available"
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -tuln | grep -q ":$port "; then
            echo "✗ Port $port is in use"
        else
            echo "✓ Port $port is available"
        fi
    else
        echo "⚠️  Cannot check port $port (install lsof or netstat)"
    fi
done

# Check container status
echo ""
echo "3. Container Status:"
docker-compose ps

# Check container health
echo ""
echo "4. Container Health:"
for container in algo-questdb algo-redis algo-data-collector algo-prometheus algo-grafana; do
    STATUS=$(docker inspect --format='{{.State.Status}}' $container 2>/dev/null)
    if [ $? -eq 0 ]; then
        HEALTH=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}N/A{{end}}' $container 2>/dev/null)
        echo "$container: Status=$STATUS, Health=$HEALTH"
    else
        echo "$container: Not found"
    fi
done

# Check logs for errors
echo ""
echo "5. Recent Error Logs:"
echo ""
echo "QuestDB logs:"
docker-compose logs --tail=10 questdb 2>&1 | grep -E "ERROR|WARN|Failed" || echo "No errors found"

echo ""
echo "Redis logs:"
docker-compose logs --tail=10 redis 2>&1 | grep -E "ERROR|WARN|Failed" || echo "No errors found"

echo ""
echo "Data Collector logs:"
docker-compose logs --tail=10 data-collector 2>&1 | grep -E "ERROR|WARN|Failed" || echo "No errors found"

# Test service connectivity
echo ""
echo "6. Service Connectivity:"
# Test QuestDB
curl -s http://localhost:9000/status > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ QuestDB HTTP API is accessible"
else
    echo "✗ QuestDB HTTP API is not accessible"
fi

# Test Redis
docker exec algo-redis redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Redis is responding to ping"
else
    echo "✗ Redis is not responding"
fi

# Memory check
echo ""
echo "7. Docker Resources:"
docker system df
echo ""
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo ""
echo "Diagnostics complete!"
echo ""
echo "Common fixes:"
echo "1. If ports are in use: Stop conflicting services or change ports in docker-compose.yml"
echo "2. If containers are unhealthy: Run 'docker-compose down -v' and './scripts/setup.sh'"
echo "3. If memory issues: Increase Docker Desktop memory allocation or reduce JAVA_OPTS in docker-compose.yml"