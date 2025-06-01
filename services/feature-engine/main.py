import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np

from shared.utils.config import Config
from shared.utils.logging import setup_logger
from shared.utils.metrics import MetricsCollector
from shared.connectors.questdb import QuestDBConnector
from shared.connectors.redis import RedisConnector
from indicators.technical import TechnicalIndicatorCalculator
from indicators.custom import CustomIndicatorCalculator
from indicators.microstructure import MicrostructureCalculator
from pipelines.realtime import RealtimeFeaturePipeline
from pipelines.aggregator import MultiTimeframeAggregator
from pipelines.quality import DataQualityMonitor


class FeatureEngineService:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger(__name__)
        self.metrics = MetricsCollector()

        self.questdb = None
        self.redis = None

        self.symbols = self.config.symbols
        self.intervals = self.config.intervals

        self.technical_calculator = TechnicalIndicatorCalculator()
        self.custom_calculator = CustomIndicatorCalculator()
        self.microstructure_calculator = MicrostructureCalculator()

        self.realtime_pipeline = RealtimeFeaturePipeline()
        self.aggregator = MultiTimeframeAggregator()
        self.quality_monitor = DataQualityMonitor()

        self.feature_buffer = defaultdict(lambda: defaultdict(list))
        self.buffer_size = self.config.get('feature_engine.buffer_size', 2500)
        self.flush_interval = self.config.get('feature_engine.flush_interval', 15)

        self.is_running = False
        self.tasks = []
        self.data_initialized = False
        self.last_processed = defaultdict(lambda: defaultdict(int))

        self.query_delay = 0.5
        self.batch_delay = 2.0

    def _init_connections(self, max_retries: int = 10, retry_delay: float = 3.0):
        for attempt in range(max_retries):
            try:
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
                    time.sleep(retry_delay)
                else:
                    return False

        return False

    async def wait_for_data_initialization(self):
        self.logger.info("Waiting for data initialization...")

        timeout = 300
        start_time = time.time()

        while time.time() - start_time < timeout:
            init_status = self.redis.get_json('data_initialization_status')
            if init_status and init_status.get('status') == 'completed':
                self.logger.info("Data initialization completed")
                return True

            await asyncio.sleep(5)

        self.logger.error("Data initialization timeout")
        return False

    async def initialize(self):
        self.logger.info("Initializing feature engine service")

        if not self._init_connections():
            raise Exception("Failed to initialize database connections")

        if not await self.wait_for_data_initialization():
            raise Exception("Data initialization failed")

        self._create_feature_tables()
        await self._calculate_historical_features()

        self.data_initialized = True

    def _create_feature_tables(self):
        for interval in self.intervals:
            table_name = f"features_{interval}"

            columns = [
                "symbol SYMBOL capacity 256 CACHE",
                "open DOUBLE",
                "high DOUBLE",
                "low DOUBLE",
                "close DOUBLE",
                "volume DOUBLE"
            ]

            for period in [10, 20, 50, 100, 200]:
                columns.extend([
                    f"sma_{period} DOUBLE",
                    f"ema_{period} DOUBLE"
                ])

            columns.extend([
                "rsi_14 DOUBLE",
                "rsi_21 DOUBLE",
                "macd DOUBLE",
                "macd_signal DOUBLE",
                "macd_hist DOUBLE",
                "bb_upper DOUBLE",
                "bb_middle DOUBLE",
                "bb_lower DOUBLE",
                "bb_width DOUBLE",
                "bb_percent DOUBLE",
                "atr_14 DOUBLE",
                "atr_21 DOUBLE",
                "adx_14 DOUBLE",
                "plus_di DOUBLE",
                "minus_di DOUBLE",
                "cci_20 DOUBLE",
                "stoch_k DOUBLE",
                "stoch_d DOUBLE",
                "williams_r DOUBLE",
                "mfi_14 DOUBLE",
                "obv DOUBLE",
                "vwap DOUBLE",
                "volume_ratio DOUBLE",
                "volatility_20 DOUBLE",
                "volatility_50 DOUBLE",
                "price_position DOUBLE",
                "trend_strength DOUBLE",
                "momentum_score DOUBLE",
                "volume_momentum DOUBLE",
                "buy_pressure DOUBLE",
                "order_flow_imbalance DOUBLE",
                "spread_ratio DOUBLE",
                "price_impact DOUBLE",
                "kyle_lambda DOUBLE",
                "realized_volatility DOUBLE",
                "feature_version LONG",
                "timestamp TIMESTAMP"
            ])

            create_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns)}
                ) timestamp(timestamp) PARTITION BY DAY WAL DEDUP UPSERT KEYS(symbol, timestamp);
            """

            try:
                self.questdb.execute_query(create_query)
                self.logger.info(f"Created features table {table_name}")
            except Exception as e:
                self.logger.error(f"Failed to create table {table_name}: {e}")

    async def _calculate_historical_features(self):
        self.logger.info("Calculating historical features")

        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    await self._process_symbol_interval(symbol, interval)
                    await asyncio.sleep(self.batch_delay)
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {interval}: {e}")

    async def _process_symbol_interval(self, symbol: str, interval: str):
        table_name = f"klines_{interval}"
        features_table = f"features_{interval}"

        existing_features_query = f"""
            SELECT max(timestamp) as max_ts 
            FROM {features_table} 
            WHERE symbol = '{symbol}'
        """

        existing_result = self.questdb.execute_query(existing_features_query)
        last_feature_ts = None

        if existing_result and existing_result[0]['max_ts'] is not None:
            last_feature_ts = existing_result[0]['max_ts']
            if isinstance(last_feature_ts, str):
                last_feature_ts = int(pd.Timestamp(last_feature_ts).timestamp() * 1_000_000_000)
            else:
                last_feature_ts = int(last_feature_ts)
            self.logger.info(
                f"Found existing features for {symbol} {interval} up to {pd.Timestamp(last_feature_ts, unit='ns')}")

        min_ts, max_ts = self.questdb.get_timestamp_range(symbol, interval)
        if min_ts is None or max_ts is None:
            self.logger.warning(f"No data for {symbol} {interval}")
            return

        if last_feature_ts:
            min_ts = last_feature_ts + 1

        if min_ts >= max_ts:
            self.logger.info(f"Features already up to date for {symbol} {interval}")
            return

        chunk_size = 5000
        lookback = 200
        interval_ms = self._get_interval_ms(interval)
        processed_count = 0

        current_ts = min_ts

        while current_ts < max_ts:
            lookback_start_ts = max(min_ts - (lookback * interval_ms * 1_000_000), 0)
            chunk_end_ts = min(current_ts + (chunk_size * interval_ms * 1_000_000), max_ts)

            await asyncio.sleep(self.query_delay)

            try:
                df = self.questdb.get_klines_by_timestamp_range(
                    symbol, interval, lookback_start_ts, chunk_end_ts
                )

                if df.empty:
                    current_ts = chunk_end_ts
                    continue

                features_df = self._calculate_features(df, symbol, interval)

                if features_df.empty:
                    current_ts = chunk_end_ts
                    continue

                features_to_store = features_df[features_df.index >= pd.Timestamp(current_ts, unit='ns')]

                if not features_to_store.empty:
                    await self._store_features_batch(symbol, interval, features_to_store)
                    processed_count += len(features_to_store)

                    if processed_count % 1000 == 0:
                        self.logger.info(f"Processed {processed_count} features for {symbol} {interval}")

                current_ts = chunk_end_ts

            except Exception as e:
                self.logger.error(f"Error processing chunk for {symbol} {interval}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)
                continue

        self.logger.info(f"Completed historical features for {symbol} {interval}: {processed_count} features processed")

    def _calculate_features(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            return pd.DataFrame()

        df = df.sort_index()

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        features = pd.DataFrame(index=df.index)
        features[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']]

        try:
            tech_features = self.technical_calculator.calculate(df)
            features = features.join(tech_features, how='left')
        except Exception as e:
            self.logger.warning(f"Technical features error: {e}")

        try:
            custom_features = self.custom_calculator.calculate(df)
            features = features.join(custom_features, how='left')
        except Exception as e:
            self.logger.warning(f"Custom features error: {e}")

        try:
            micro_features = self.microstructure_calculator.calculate(df)
            features = features.join(micro_features, how='left')
        except Exception as e:
            self.logger.warning(f"Microstructure features error: {e}")

        features['symbol'] = symbol
        features['feature_version'] = 2

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna(how='all',
                                   subset=[col for col in features.columns if col not in ['symbol', 'feature_version']])

        return features

    async def _store_features_batch(self, symbol: str, interval: str, features_df: pd.DataFrame):
        if features_df.empty:
            return

        table_name = f"features_{interval}"

        features_df = features_df.copy()

        if features_df.index.name == 'timestamp':
            features_df = features_df.reset_index()
        else:
            features_df['timestamp'] = features_df.index
            features_df = features_df.reset_index(drop=True)

        records = features_df.to_dict('records')
        stored_count = 0

        for record in records:
            timestamp_val = record.get('timestamp')
            if pd.isna(timestamp_val):
                continue

            if isinstance(timestamp_val, pd.Timestamp):
                timestamp_ns = int(timestamp_val.timestamp() * 1_000_000_000)
            elif isinstance(timestamp_val, (int, float)):
                timestamp_ns = int(timestamp_val)
            else:
                timestamp_ns = int(pd.Timestamp(timestamp_val).timestamp() * 1_000_000_000)

            numeric_fields = []
            for key, value in record.items():
                if key in ['symbol', 'timestamp']:
                    continue

                if pd.isna(value):
                    continue

                if isinstance(value, bool):
                    numeric_fields.append(f"{key}={'true' if value else 'false'}")
                elif key in ['feature_version']:
                    numeric_fields.append(f"{key}={int(value)}i")
                else:
                    numeric_fields.append(f"{key}={float(value)}")

            if numeric_fields:
                line = f"{table_name},symbol={symbol} {','.join(numeric_fields)} {timestamp_ns}"
                self.questdb._send_line(line)
                stored_count += 1

        if stored_count > 0:
            self.questdb.wait_for_commit(2)
            self.logger.debug(f"Stored {stored_count} features for {symbol} {interval}")
            self.metrics.record_db_write(table_name, 'success')

    async def process_realtime_updates(self):
        channels = [f"kline:{symbol}:{interval}" for symbol in self.symbols for interval in self.intervals]
        self.redis.subscribe(channels)

        while self.is_running:
            try:
                message = self.redis.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    await self._process_kline_update(message)
            except Exception as e:
                self.logger.error(f"Realtime processing error: {e}")

            await asyncio.sleep(0.001)

    async def _process_kline_update(self, message: Dict[str, Any]):
        try:
            channel = message['channel'].decode('utf-8')
            _, symbol, interval = channel.split(':')

            kline_data = message['data']
            kline_timestamp = int(kline_data['timestamp'])

            if kline_timestamp <= self.last_processed[symbol][interval]:
                return

            lookback = self.config.get('feature_engine.lookback_periods', 300)
            df = self.questdb.get_klines_df(symbol, interval, limit=lookback)

            if df.empty or len(df) < 20:
                return

            features_df = self._calculate_features(df, symbol, interval)

            if not features_df.empty:
                latest_features = features_df.iloc[-1:]
                self.feature_buffer[symbol][interval].append(latest_features)

                self.last_processed[symbol][interval] = kline_timestamp

                total_buffer_size = sum(
                    len(features)
                    for intervals in self.feature_buffer.values()
                    for features in intervals.values()
                )

                if total_buffer_size >= self.buffer_size:
                    await self.flush_buffer()

                self.metrics.record_ws_message(symbol, 'feature_update')

        except Exception as e:
            self.logger.error(f"Error processing kline update: {e}")

    async def flush_buffer(self):
        if not self.feature_buffer:
            return

        for symbol, intervals in list(self.feature_buffer.items()):
            for interval, features_list in list(intervals.items()):
                if not features_list:
                    continue

                try:
                    combined_df = pd.concat(features_list)
                    if not combined_df.empty:
                        await self._store_features_batch(symbol, interval, combined_df)
                except Exception as e:
                    self.logger.error(f"Failed to flush features for {symbol} {interval}: {e}")

        self.feature_buffer.clear()

    async def periodic_flush(self):
        while self.is_running:
            await asyncio.sleep(self.flush_interval)
            await self.flush_buffer()

    async def quality_monitoring(self):
        while self.is_running:
            try:
                for symbol in self.symbols:
                    for interval in ['1h', '1d']:
                        try:
                            features_df = self.questdb.get_features_df(symbol, interval, limit=1000)

                            if not features_df.empty:
                                quality_report = self.quality_monitor.check_quality(features_df)

                                self.redis.set_json(
                                    f"quality:{symbol}:{interval}",
                                    quality_report,
                                    expire=3600
                                )

                                if quality_report.get('overall_quality_score', 100) < 70:
                                    self.logger.warning(
                                        f"Low quality score for {symbol} {interval}: "
                                        f"{quality_report['overall_quality_score']:.2f}"
                                    )
                        except Exception as e:
                            self.logger.warning(f"Quality check error for {symbol} {interval}: {e}")

            except Exception as e:
                self.logger.error(f"Quality monitoring error: {e}")

            await asyncio.sleep(600)

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

    async def start(self):
        self.is_running = True

        await self.initialize()

        self.tasks.append(asyncio.create_task(self.process_realtime_updates()))
        self.tasks.append(asyncio.create_task(self.periodic_flush()))
        self.tasks.append(asyncio.create_task(self.quality_monitoring()))

        self.logger.info("Feature engine service started")

    async def stop(self):
        self.logger.info("Stopping feature engine service")

        self.is_running = False

        await self.flush_buffer()

        for task in self.tasks:
            task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)

        if self.questdb:
            self.questdb.close()
        if self.redis:
            self.redis.close()

        self.logger.info("Feature engine service stopped")


async def main():
    service = FeatureEngineService()

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