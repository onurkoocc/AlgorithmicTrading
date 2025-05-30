import asyncio
import websockets
import json
from typing import List, Dict, Any, Optional
import aiohttp
from datetime import datetime, timedelta
import time
from .base import BaseCollector
from shared.utils.config import Config
from shared.utils.logging import setup_logger
from shared.utils.metrics import MetricsCollector
from shared.models.market_data import Kline


class BinanceFuturesCollector(BaseCollector):
    def __init__(self, symbols: List[str], intervals: List[str]):
        super().__init__(symbols, intervals)
        self.config = Config()
        self.logger = setup_logger(__name__)
        self.metrics = MetricsCollector()

        self.ws_base_url = self.config.binance.futures_base_url
        self.rest_base_url = self.config.binance.rest_base_url

        self.websockets = {}
        self.subscriptions = {}
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        self.ping_interval = 20
        self.last_kline_times = {}

    async def connect(self):
        self.is_running = True
        self.logger.info("Starting Binance Futures collector")

    async def disconnect(self):
        self.is_running = False

        for stream_name, ws in self.websockets.items():
            try:
                await ws.close()
            except:
                pass

        self.websockets.clear()
        self.subscriptions.clear()
        self.logger.info("Stopped Binance Futures collector")

    async def subscribe(self, symbol: str, interval: str):
        stream_name = f"{symbol.lower()}@kline_{interval}"

        if stream_name not in self.subscriptions:
            self.subscriptions[stream_name] = {
                'symbol': symbol,
                'interval': interval
            }

            await self._connect_stream(stream_name)

    async def unsubscribe(self, symbol: str, interval: str):
        stream_name = f"{symbol.lower()}@kline_{interval}"

        if stream_name in self.subscriptions:
            del self.subscriptions[stream_name]

            if stream_name in self.websockets:
                try:
                    await self.websockets[stream_name].close()
                except:
                    pass
                del self.websockets[stream_name]

    async def _connect_stream(self, stream_name: str):
        url = f"{self.ws_base_url}/ws/{stream_name}"

        while self.is_running:
            try:
                ws = await websockets.connect(url)
                self.websockets[stream_name] = ws
                self.logger.info(f"Connected to stream: {stream_name}")
                self.metrics.set_active_connections('websocket', len(self.websockets))

                asyncio.create_task(self._handle_stream(stream_name, ws))
                break

            except Exception as e:
                self.logger.error(f"Failed to connect to {stream_name}: {e}")
                await asyncio.sleep(self.reconnect_delay)

    async def _handle_stream(self, stream_name: str, ws):
        ping_task = asyncio.create_task(self._ping_loop(ws))

        try:
            async for message in ws:
                if not self.is_running:
                    break

                try:
                    data = json.loads(message)
                    await self._process_message(stream_name, data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON from {stream_name}: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message from {stream_name}: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"Connection closed for {stream_name}")
        except Exception as e:
            self.logger.error(f"Stream error for {stream_name}: {e}")
        finally:
            ping_task.cancel()

            if stream_name in self.websockets:
                del self.websockets[stream_name]
                self.metrics.set_active_connections('websocket', len(self.websockets))

            if self.is_running and stream_name in self.subscriptions:
                await asyncio.sleep(self.reconnect_delay)
                await self._connect_stream(stream_name)

    async def _ping_loop(self, ws):
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await ws.ping()
            except:
                break

    async def _process_message(self, stream_name: str, data: Dict[str, Any]):
        if 'k' not in data:
            return

        kline_data = data['k']
        subscription = self.subscriptions.get(stream_name)

        if not subscription:
            return

        symbol = subscription['symbol']
        interval = subscription['interval']

        kline_key = f"{symbol}:{interval}"
        kline_timestamp = kline_data['t']

        if not kline_data['x']:
            return

        if kline_key in self.last_kline_times and self.last_kline_times[kline_key] >= kline_timestamp:
            return

        self.last_kline_times[kline_key] = kline_timestamp

        kline = Kline.from_binance_ws(
            symbol,
            interval,
            [
                kline_data['t'],
                kline_data['o'],
                kline_data['h'],
                kline_data['l'],
                kline_data['c'],
                kline_data['v'],
                kline_data['T'],
                kline_data['q'],
                kline_data['n'],
                kline_data['V'],
                kline_data['Q'],
                kline_data['x']
            ]
        )

        self.data_queue.put({
            'type': 'kline',
            'data': kline
        })

        self.metrics.record_ws_message(symbol, 'kline')

    async def fetch_historical_klines(self, symbol: str, interval: str,
                                      start_time: int, end_time: Optional[int] = None,
                                      limit: int = 1500) -> List[Kline]:
        klines = []
        current_start = start_time

        if not end_time:
            end_time = int(time.time() * 1000)

        async with aiohttp.ClientSession() as session:
            while current_start < end_time:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'limit': limit
                }

                if end_time:
                    params['endTime'] = end_time

                try:
                    async with session.get(
                            f"{self.rest_base_url}/fapi/v1/klines",
                            params=params
                    ) as response:
                        if response.status == 429:
                            retry_after = int(response.headers.get('Retry-After', 60))
                            self.logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue

                        response.raise_for_status()
                        data = await response.json()

                        if not data:
                            break

                        for kline_data in data:
                            kline = Kline.from_binance_rest(symbol, interval, kline_data)
                            klines.append(kline)

                        last_timestamp = data[-1][0]
                        if last_timestamp >= end_time or len(data) < limit:
                            break

                        current_start = last_timestamp + self._get_interval_ms(interval)

                        await asyncio.sleep(0.1)

                except aiohttp.ClientError as e:
                    self.logger.error(f"HTTP error fetching historical data: {e}")
                    await asyncio.sleep(5)
                except Exception as e:
                    self.logger.error(f"Error fetching historical data: {e}")
                    await asyncio.sleep(5)

        return klines

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
            '1d': 86400000,
            '3d': 259200000,
            '1w': 604800000,
            '1M': 2592000000
        }
        return interval_map.get(interval, 60000)

    async def start(self):
        await self.connect()

        for symbol in self.symbols:
            for interval in self.intervals:
                await self.subscribe(symbol, interval)
                await asyncio.sleep(0.1)

        self.logger.info(f"Subscribed to {len(self.subscriptions)} streams")

    async def stop(self):
        await self.disconnect()