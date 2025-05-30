import asyncio
import signal
import sys
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List
import threading
from queue import Empty
from collections import defaultdict

from shared.utils.config import Config
from shared.utils.logging import setup_logger
from shared.utils.metrics import MetricsCollector
from shared.connectors.questdb import QuestDBConnector
from shared.connectors.redis import RedisConnector
from collectors.binance import BinanceFuturesCollector


class DataCollectorService:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger(__name__)
        self.metrics = MetricsCollector()

        self.questdb = None
        self.redis = None

        self.symbols = self.config.symbols
        self.intervals = self.config.intervals

        self.collector = BinanceFuturesCollector(self.symbols, self.intervals)

        self.buffer = defaultdict(lambda: defaultdict(dict))
        self.buffer_size = self.config.get('data_collection.buffer_size', 10000)
        self.flush_interval = self.config.get('data_collection.flush_interval', 5)
        self.total_buffer_items = 0

        self.is_running = False
        self.tasks = []

    def _init_connections(self, max_retries: int = 10, retry_delay: float = 3.0):
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing connections (attempt {attempt + 1}/{max_retries})...")

                if not self.questdb:
                    self.questdb = QuestDBConnector()
                    self.logger.info("QuestDB connected")

                if not self.redis:
                    self.redis = RedisConnector()
                    if self.redis.ping():
                        self.logger.info("Redis connected")
                    else:
                        raise Exception("Redis ping failed")

                return True

            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Failed to initialize connections after all retries")
                    return False

        return False

    async def initialize(self):
        self.logger.info("Initializing data collector service")

        if not self._init_connections():
            raise Exception("Failed to initialize database connections")

        self.questdb.create_tables()

        self.questdb.wait_for_commit(3)

        await self._download_missing_historical_data()

    async def _download_missing_historical_data(self):
        self.logger.info("Checking for missing historical data")

        historical_days = self.config.get('data_collection.historical_days', 365)

        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    latest_timestamp = self.questdb.get_latest_timestamp(symbol, interval)

                    if latest_timestamp:
                        latest_dt = datetime.fromtimestamp(latest_timestamp / 1000)
                        interval_ms = self._get_interval_ms(interval)
                        start_time = int((latest_dt.timestamp() * 1000) + interval_ms)
                    else:
                        start_time = int((datetime.now() - timedelta(days=historical_days)).timestamp() * 1000)

                    end_time = int(time.time() * 1000)

                    if end_time - start_time > 60000:
                        self.logger.info(f"Downloading historical data for {symbol} {interval}")

                        klines = await self.collector.fetch_historical_klines(
                            symbol, interval, start_time, end_time
                        )

                        if klines:
                            self.logger.info(f"Downloaded {len(klines)} klines for {symbol} {interval}")

                            batch_data = [kline.to_dict() for kline in klines]
                            self.questdb.batch_write_klines(symbol, interval, batch_data)

                            self.questdb.wait_for_commit(5)

                            count = self.questdb.verify_data(symbol, interval)
                            self.logger.info(f"Verified {count} total records in database for {symbol} {interval}")

                            await asyncio.sleep(2)

                except Exception as e:
                    self.logger.error(f"Error downloading historical data for {symbol} {interval}: {e}")

    def _get_interval_ms(self, interval: str) -> int:
        interval_map = {
            '1m': 60000,
            '3m': 180000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '2h': 7200000,
            '4h': 14400000,
            '6h': 21600000,
            '8h': 28800000,
            '12h': 43200000,
            '1d': 86400000
        }
        return interval_map.get(interval, 60000)

    async def process_data(self):
        while self.is_running:
            try:
                data = self.collector.get_data(timeout=0.1)

                if data and data['type'] == 'kline':
                    kline = data['data']

                    self.buffer[kline.symbol][kline.interval][int(kline.timestamp)] = kline

                    self.total_buffer_items = sum(
                        len(timestamps)
                        for symbol_intervals in self.buffer.values()
                        for timestamps in symbol_intervals.values()
                    )

                    try:
                        self.redis.publish(f"kline:{kline.symbol}:{kline.interval}", kline.to_dict())
                    except Exception as e:
                        self.logger.warning(f"Failed to publish to Redis: {e}")

                    self.metrics.set_queue_size('buffer', self.total_buffer_items)
                    self.metrics.update_timestamp(kline.symbol, kline.interval)

                    if self.total_buffer_items >= self.buffer_size:
                        await self.flush_buffer()

            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error processing data: {e}")

            await asyncio.sleep(0.001)

    async def flush_buffer(self):
        if not self.buffer:
            return

        self.logger.debug(f"Flushing {self.total_buffer_items} klines to database")

        for symbol, intervals in self.buffer.items():
            for interval, klines_dict in intervals.items():
                if not klines_dict:
                    continue

                try:
                    sorted_klines = sorted(klines_dict.values(), key=lambda k: k.timestamp)
                    klines_data = [kline.to_dict() for kline in sorted_klines]

                    self.questdb.batch_write_klines(symbol, interval, klines_data)

                except Exception as e:
                    self.logger.error(f"Failed to write klines for {symbol} {interval}: {e}")

        self.buffer.clear()
        self.total_buffer_items = 0
        self.metrics.set_queue_size('buffer', 0)

    async def periodic_flush(self):
        while self.is_running:
            await asyncio.sleep(self.flush_interval)
            await self.flush_buffer()

    async def health_check(self):
        while self.is_running:
            try:
                redis_healthy = self.redis and self.redis.ping()
                if not redis_healthy:
                    self.logger.error("Redis health check failed")

                active_streams = len(self.collector.websockets)
                expected_streams = len(self.symbols) * len(self.intervals)

                if active_streams < expected_streams:
                    self.logger.warning(
                        f"Stream count mismatch: {active_streams}/{expected_streams}"
                    )

                for symbol in self.symbols:
                    for interval in ['1m', '1h', '1d']:
                        if interval not in self.intervals:
                            continue
                        try:
                            count = self.questdb.verify_data(symbol, interval)
                            if count > 0:
                                self.logger.debug(f"Data count for {symbol} {interval}: {count}")
                        except:
                            pass

                if self.redis:
                    self.redis.set_json('health:data-collector', {
                        'status': 'healthy' if redis_healthy else 'degraded',
                        'timestamp': time.time(),
                        'active_streams': active_streams,
                        'buffer_size': self.total_buffer_items
                    }, expire=60)

            except Exception as e:
                self.logger.error(f"Health check failed: {e}")

            await asyncio.sleep(30)

    async def start(self):
        self.is_running = True

        await self.initialize()

        self.tasks.append(asyncio.create_task(self.collector.start()))
        self.tasks.append(asyncio.create_task(self.process_data()))
        self.tasks.append(asyncio.create_task(self.periodic_flush()))
        self.tasks.append(asyncio.create_task(self.health_check()))

        self.logger.info("Data collector service started")

    async def stop(self):
        self.logger.info("Stopping data collector service")

        self.is_running = False

        await self.flush_buffer()

        await self.collector.stop()

        for task in self.tasks:
            task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)

        if self.questdb:
            self.questdb.close()
        if self.redis:
            self.redis.close()

        self.logger.info("Data collector service stopped")


async def main():
    service = DataCollectorService()

    def signal_handler(sig, frame):
        asyncio.create_task(service.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await service.start()
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())