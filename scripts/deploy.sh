#!/bin/bash

set -e

ACTION=${1:-"start"}

case $ACTION in
    start)
        echo "Starting all services..."
        docker-compose up -d
        echo "Services started. Checking status..."
        sleep 5
        docker-compose ps
        ;;

    stop)
        echo "Stopping all services..."
        docker-compose down
        ;;

    restart)
        echo "Restarting all services..."
        docker-compose restart
        ;;

    logs)
        SERVICE=${2:-""}
        if [ -z "$SERVICE" ]; then
            docker-compose logs -f --tail=100
        else
            docker-compose logs -f --tail=100 $SERVICE
        fi
        ;;

    status)
        docker-compose ps
        echo ""
        echo "Health check:"
        docker-compose exec redis redis-cli get health:data-collector
        ;;

    rebuild)
        SERVICE=${2:-""}
        if [ -z "$SERVICE" ]; then
            echo "Rebuilding all services..."
            docker-compose build
        else
            echo "Rebuilding $SERVICE..."
            docker-compose build $SERVICE
        fi
        ;;

    update)
        echo "Updating services..."
        docker-compose pull
        docker-compose build
        docker-compose up -d
        ;;

    clean)
        echo "WARNING: This will remove all containers and volumes!"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            echo "Cleanup complete"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|logs|status|rebuild|update|clean} [service]"
        exit 1
        ;;
esac