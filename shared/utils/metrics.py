from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from typing import Dict, Optional
import time
from functools import wraps
from .config import Config


class MetricsCollector:
    _instance: Optional['MetricsCollector'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.config = Config()

        self.ws_messages = Counter(
            'ws_messages_total',
            'Total WebSocket messages received',
            ['symbol', 'type']
        )

        self.db_writes = Counter(
            'db_writes_total',
            'Total database writes',
            ['table', 'status']
        )

        self.processing_time = Histogram(
            'processing_duration_seconds',
            'Processing duration in seconds',
            ['operation'],
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )

        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['type']
        )

        self.queue_size = Gauge(
            'queue_size',
            'Current queue size',
            ['queue_name']
        )

        self.last_update = Gauge(
            'last_update_timestamp',
            'Timestamp of last update',
            ['symbol', 'interval']
        )

        self.system_info = Info(
            'system',
            'System information'
        )

        self.system_info.info({
            'version': '1.0.0',
            'service': 'algo-trading'
        })

        if self.config.get('monitoring.prometheus.enabled', True):
            port = self.config.get('monitoring.prometheus.port', 8000)
            start_http_server(port)

    def record_ws_message(self, symbol: str, msg_type: str):
        self.ws_messages.labels(symbol=symbol, type=msg_type).inc()

    def record_db_write(self, table: str, status: str):
        self.db_writes.labels(table=table, status=status).inc()

    def time_operation(self, operation: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    self.processing_time.labels(operation=operation).observe(duration)

            return wrapper

        return decorator

    def set_active_connections(self, conn_type: str, count: int):
        self.active_connections.labels(type=conn_type).set(count)

    def set_queue_size(self, queue_name: str, size: int):
        self.queue_size.labels(queue_name=queue_name).set(size)

    def update_timestamp(self, symbol: str, interval: str):
        self.last_update.labels(symbol=symbol, interval=interval).set(time.time())