import asyncio
import signal
import sys
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List
import threading
from queue import Empty

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

        self.questdb = QuestDBConnector()
        self.redis = RedisConnector()

        self.symbols = self.config.symbols
        self.intervals = self.config.intervals

        self.collector = BinanceFuturesCollector(self.symbols, self.intervals)

        self.buffer = []
        self.buffer_size = self.config.get('data_collection.buffer_size', 10000)
        self.flush_interval = self.config.get('data_collection.flush_interval', 5)

        self.is_running = False
        self.tasks = []

    async def initialize(self):
        self.logger.info("Initializing data collector service")

        self.questdb.create_tables()

        await self._download_missing_historical_data()

    async def _download_missing_historical_data(self):
        self.logger.info("Checking for missing historical data")

        for symbol in self.symbols:
            for interval in self.intervals:
                latest_timestamp = self.questdb.get_latest_timestamp(symbol, interval)

                if latest_timestamp:
                    latest_dt = datetime.fromtimestamp(latest_timestamp / 1_000_000)
                    start_time = int((latest_dt + timedelta(seconds=1)).timestamp() * 1000)
                else:
                    start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

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

                        await asyncio.sleep(1)

    async def process_data(self):
        while self.is_running:
            try:
                data = self.collector.get_data(timeout=0.1)

                if data and data['type'] == 'kline':
                    kline = data['data']

                    self.buffer.append(kline)

                    self.redis.publish(f"kline:{kline.symbol}:{kline.interval}", kline.to_dict())

                    self.metrics.set_queue_size('buffer', len(self.buffer))
                    self.metrics.update_timestamp(kline.symbol, kline.interval)

                    if len(self.buffer) >= self.buffer_size:
                        await self.flush_buffer()

            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error processing data: {e}")

            await asyncio.sleep(0.001)

    async def flush_buffer(self):
        if not self.buffer:
            return

        self.logger.debug(f"Flushing {len(self.buffer)} klines to database")

        klines_by_key = {}

        for kline in self.buffer:
            key = (kline.symbol, kline.interval)
            if key not in klines_by_key:
                klines_by_key[key] = []
            klines_by_key[key].append(kline.to_dict())

        for (symbol, interval), klines in klines_by_key.items():
            try:
                self.questdb.batch_write_klines(symbol, interval, klines)
            except Exception as e:
                self.logger.error(f"Failed to write klines for {symbol} {interval}: {e}")

        self.buffer.clear()
        self.metrics.set_queue_size('buffer', 0)

    async def periodic_flush(self):
        while self.is_running:
            await asyncio.sleep(self.flush_interval)
            await self.flush_buffer()

    async def health_check(self):
        while self.is_running:
            try:
                if not self.redis.ping():
                    self.logger.error("Redis health check failed")

                active_streams = len(self.collector.websockets)
                expected_streams = len(self.symbols) * len(self.intervals)

                if active_streams < expected_streams:
                    self.logger.warning(
                        f"Stream count mismatch: {active_streams}/{expected_streams}"
                    )

                self.redis.set_json('health:data-collector', {
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'active_streams': active_streams,
                    'buffer_size': len(self.buffer)
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

        self.questdb.close()
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