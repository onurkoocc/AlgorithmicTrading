import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import json

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

        self.query_delay = 0.1
        self.batch_delay = 0.5
        self.failed_tables = set()
        self.max_table_retries = 3
        self.table_retry_count = defaultdict(int)

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

    def _to_nanoseconds_epoch(self, ts_input: Any) -> Optional[int]:
        if ts_input is None:
            return None

        try:
            if isinstance(ts_input, pd.Timestamp):
                return ts_input.value
            if isinstance(ts_input, datetime):
                if ts_input.tzinfo is not None and ts_input.tzinfo.utcoffset(ts_input) is not None:
                    return pd.Timestamp(ts_input.astimezone(timezone.utc)).value
                else:
                    return pd.Timestamp(ts_input).value
            if isinstance(ts_input, str):
                return pd.Timestamp(ts_input).value
            if isinstance(ts_input, (int, float)):
                return int(ts_input)

            self.logger.warning(f"Unhandled timestamp type for conversion: {type(ts_input)}. Value: {ts_input}")
            return None
        except Exception as e:
            self.logger.error(
                f"Error converting timestamp '{ts_input}' (type: {type(ts_input)}) to nanosecond epoch: {e}")
            return None

    async def _repair_table_if_needed(self, table_name: str, symbol: str) -> bool:
        try:
            if table_name in self.failed_tables:
                if self.table_retry_count[table_name] >= self.max_table_retries:
                    return False

                self.logger.warning(f"Attempting to repair table {table_name}")

                self.questdb.execute_query(f"VACUUM TABLE {table_name}")
                await asyncio.sleep(2)

                test_query = f"SELECT count() FROM {table_name} WHERE symbol = '{symbol}' LIMIT 1"
                self.questdb.execute_query(test_query)

                self.failed_tables.discard(table_name)
                self.logger.info(f"Table {table_name} repair successful")
                return True

        except Exception as e:
            self.logger.error(f"Failed to repair table {table_name}: {e}")
            self.table_retry_count[table_name] += 1

            if self.table_retry_count[table_name] >= self.max_table_retries:
                self.logger.error(
                    f"Table {table_name} marked as permanently failed after {self.max_table_retries} attempts")

        return False

    async def _calculate_historical_features(self):
        self.logger.info("Calculating historical features")

        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    await self._process_symbol_interval(symbol, interval)
                    await asyncio.sleep(self.batch_delay)
                except Exception as e:
                    self.logger.error(f"Error processing {symbol} {interval} during historical calculation: {e}",
                                      exc_info=True)

    async def _process_symbol_interval(self, symbol: str, interval: str):
        table_name = f"klines_{interval}"
        features_table = f"features_{interval}"

        if features_table in self.failed_tables and self.table_retry_count[features_table] >= self.max_table_retries:
            self.logger.warning(f"Skipping {symbol} {interval} - table {features_table} marked as failed")
            return

        klines_query = f"""
            SELECT min(timestamp) as min_ts, max(timestamp) as max_ts, count() as cnt
            FROM {table_name}
            WHERE symbol = '{symbol}'
        """

        klines_result = self.questdb.execute_query(klines_query)
        if not klines_result or klines_result[0]['min_ts'] is None:
            self.logger.warning(f"No klines data for {symbol} {interval}")
            return

        min_klines_ts_raw = klines_result[0]['min_ts']
        max_klines_ts_raw = klines_result[0]['max_ts']
        total_klines = klines_result[0]['cnt']

        min_klines_ts = self._to_nanoseconds_epoch(min_klines_ts_raw)
        max_klines_ts = self._to_nanoseconds_epoch(max_klines_ts_raw)

        if min_klines_ts is None or max_klines_ts is None:
            self.logger.error(
                f"Could not parse kline timestamp range for {symbol} {interval}. MinRaw: {min_klines_ts_raw}, MaxRaw: {max_klines_ts_raw}")
            return

        self.logger.info(
            f"Klines for {symbol} {interval}: {total_klines} records from {pd.Timestamp(min_klines_ts, unit='ns')} to {pd.Timestamp(max_klines_ts, unit='ns')}")

        existing_features_query = f"""
            SELECT count() as feature_count, max(timestamp) as max_ts 
            FROM {features_table} 
            WHERE symbol = '{symbol}'
        """

        existing_feature_count = 0
        last_feature_ts = None

        try:
            existing_result = self.questdb.execute_query(existing_features_query)

            if existing_result and existing_result[0]['feature_count'] > 0 and existing_result[0]['max_ts'] is not None:
                existing_feature_count = existing_result[0]['feature_count']
                last_feature_ts_raw = existing_result[0]['max_ts']
                last_feature_ts = self._to_nanoseconds_epoch(last_feature_ts_raw)

                if last_feature_ts is not None:
                    self.logger.info(
                        f"Found {existing_feature_count} existing features for {symbol} {interval} up to {pd.Timestamp(last_feature_ts, unit='ns')}")
                else:
                    self.logger.warning(
                        f"Found {existing_feature_count} existing features for {symbol} {interval} but could not parse last feature timestamp: {last_feature_ts_raw}")
                    existing_feature_count = 0

        except Exception as e:
            if "could not mmap" in str(e):
                self.failed_tables.add(features_table)
                self.logger.error(f"Memory mapping error for table {features_table}: {e}")

                if await self._repair_table_if_needed(features_table, symbol):
                    await self._process_symbol_interval(symbol, interval)
                return
            else:
                self.logger.error(f"Error checking existing features for {symbol} {interval}: {e}")
                return

        interval_ns = self._get_interval_ms(interval) * 1_000_000

        if last_feature_ts is not None and existing_feature_count > 0:
            if last_feature_ts >= (max_klines_ts - interval_ns * 5):
                self.logger.info(
                    f"Features considered up to date for {symbol} {interval}. Last feature at {pd.Timestamp(last_feature_ts, unit='ns')}, max kline at {pd.Timestamp(max_klines_ts, unit='ns')}")
                start_ts = last_feature_ts + interval_ns
                if start_ts > max_klines_ts:
                    self.logger.info(f"No new klines to process for {symbol} {interval} since last feature.")
                    return
            else:
                start_ts = last_feature_ts + interval_ns
                self.logger.info(
                    f"Resuming feature calculation for {symbol} {interval} from {pd.Timestamp(start_ts, unit='ns')}")
        else:
            self.logger.info(
                f"Calculating features from the beginning for {symbol} {interval} (Existing: {existing_feature_count}, LastTS: {pd.Timestamp(last_feature_ts, unit='ns') if last_feature_ts else 'N/A'})")
            start_ts = min_klines_ts

        if start_ts > max_klines_ts:
            self.logger.info(
                f"Start timestamp {pd.Timestamp(start_ts, unit='ns')} is after max kline timestamp {pd.Timestamp(max_klines_ts, unit='ns')} for {symbol} {interval}. Nothing to process.")
            return

        self.logger.info(
            f"Need to calculate features for {symbol} {interval} from {pd.Timestamp(start_ts, unit='ns')} to {pd.Timestamp(max_klines_ts, unit='ns')}")

        chunk_size = 5000
        lookback = 200
        processed_count = 0
        current_ts = start_ts

        while current_ts <= max_klines_ts:
            lookback_start_ts = max(min_klines_ts, current_ts - (lookback * interval_ns))
            processing_chunk_end_ts = min(current_ts + (chunk_size * interval_ns), max_klines_ts + interval_ns)
            fetch_end_ts = processing_chunk_end_ts

            await asyncio.sleep(self.query_delay)

            try:
                df = self.questdb.get_klines_by_timestamp_range(symbol, interval, lookback_start_ts, fetch_end_ts)

                if df.empty:
                    self.logger.warning(
                        f"Empty dataframe for {symbol} {interval} from {pd.Timestamp(lookback_start_ts, unit='ns')} to {pd.Timestamp(fetch_end_ts, unit='ns')}")
                    current_ts = min(current_ts + (chunk_size * interval_ns), max_klines_ts + interval_ns)
                    if current_ts > max_klines_ts:
                        break
                    continue

                self.logger.debug(
                    f"Processing {len(df)} klines for {symbol} {interval} (fetched range: {pd.Timestamp(lookback_start_ts, unit='ns')} to {pd.Timestamp(fetch_end_ts, unit='ns')})")

                features_df = self._calculate_features(df, symbol, interval)

                if features_df.empty:
                    self.logger.warning(
                        f"Empty features calculated for {symbol} {interval} for klines in range {df.index.min()} to {df.index.max() if not df.empty else 'N/A'}")
                    current_ts = min(current_ts + (chunk_size * interval_ns), max_klines_ts + interval_ns)
                    if current_ts > max_klines_ts:
                        break
                    continue

                features_to_store = features_df[features_df.index >= pd.Timestamp(current_ts, unit='ns')]
                features_to_store = features_to_store[features_to_store.index <= pd.Timestamp(max_klines_ts, unit='ns')]

                if not features_to_store.empty:
                    await self._store_features_batch(symbol, interval, features_to_store)
                    processed_count += len(features_to_store)
                    self.logger.debug(
                        f"Stored {len(features_to_store)} features for {symbol} {interval}. Total processed in this run: {processed_count}")

                    if processed_count > 0 and processed_count % 1000 == 0:
                        self.logger.info(
                            f"Processed {processed_count} features for {symbol} {interval} so far in this run.")
                else:
                    self.logger.debug(
                        f"No features to store for {symbol} {interval} for current_ts {pd.Timestamp(current_ts, unit='ns')}")

                if not features_to_store.empty:
                    last_stored_ts_in_chunk = features_to_store.index.max().value
                    current_ts = last_stored_ts_in_chunk + interval_ns
                else:
                    current_ts = min(current_ts + (chunk_size * interval_ns), max_klines_ts + interval_ns)

                if current_ts > max_klines_ts:
                    break

            except Exception as e:
                self.logger.error(
                    f"Error processing chunk for {symbol} {interval} starting {pd.Timestamp(current_ts, unit='ns')}: {e}",
                    exc_info=True)
                await asyncio.sleep(5)
                current_ts = min(current_ts + interval_ns, max_klines_ts + interval_ns)
                if current_ts > max_klines_ts:
                    break
                continue

        self.logger.info(
            f"Completed historical features for {symbol} {interval}: {processed_count} features processed in this run.")

    def _calculate_features(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            self.logger.debug(
                f"DataFrame for {symbol} {interval} is empty or too short ({len(df)} rows) for feature calculation.")
            return pd.DataFrame()

        df = df.sort_index()

        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)

        features = pd.DataFrame(index=df.index)
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in base_cols:
            if col not in df.columns:
                self.logger.error(
                    f"Missing base column '{col}' in DataFrame for {symbol} {interval}. Columns: {df.columns.tolist()}")
        features[base_cols] = df[base_cols]

        try:
            tech_features = self.technical_calculator.calculate(df)
            if not tech_features.empty:
                features = features.join(tech_features, how='left')
        except Exception as e:
            self.logger.warning(f"Technical features calculation error for {symbol} {interval}: {e}", exc_info=True)

        try:
            custom_features = self.custom_calculator.calculate(df)
            if not custom_features.empty:
                features = features.join(custom_features, how='left')
        except Exception as e:
            self.logger.warning(f"Custom features calculation error for {symbol} {interval}: {e}", exc_info=True)

        try:
            micro_features = self.microstructure_calculator.calculate(df)
            if not micro_features.empty:
                features = features.join(micro_features, how='left')
        except Exception as e:
            self.logger.warning(f"Microstructure features calculation error for {symbol} {interval}: {e}",
                                exc_info=True)

        features['symbol'] = symbol
        features['feature_version'] = 2

        features = features.replace([np.inf, -np.inf], np.nan)

        return features

    async def _store_features_batch(self, symbol: str, interval: str, features_df: pd.DataFrame):
        if features_df.empty:
            return

        table_name = f"features_{interval}"

        features_df = features_df.copy()

        if 'timestamp' not in features_df.columns:
            if features_df.index.name == 'timestamp':
                features_df.reset_index(inplace=True)
            else:
                features_df['timestamp'] = features_df.index
                features_df.reset_index(drop=True, inplace=True)

        records = features_df.to_dict('records')
        stored_count = 0
        lines_to_send = []

        for record in records:
            timestamp_val = record.get('timestamp')
            if pd.isna(timestamp_val):
                continue

            if isinstance(timestamp_val, pd.Timestamp):
                timestamp_ns = timestamp_val.value
            elif isinstance(timestamp_val, (int, float)):
                timestamp_ns = int(timestamp_val)
            else:
                timestamp_ns = pd.Timestamp(timestamp_val).value

            line_tags = f"symbol={symbol}"
            line_fields = []

            for key, value in record.items():
                if key in ['symbol', 'timestamp'] or pd.isna(value):
                    continue

                if isinstance(value, bool):
                    line_fields.append(f"{key}={'T' if value else 'F'}")
                elif isinstance(value, (int, np.integer)):
                    if key == 'feature_version':
                        line_fields.append(f"{key}={int(value)}i")
                    else:
                        line_fields.append(f"{key}={int(value)}i")
                elif isinstance(value, (float, np.floating)):
                    line_fields.append(f"{key}={float(value)}")
                elif isinstance(value, str):
                    line_fields.append(f"{key}=\"{value}\"")

            if line_fields:
                lines_to_send.append(f"{table_name},{line_tags} {','.join(line_fields)} {timestamp_ns}")
                stored_count += 1

        if lines_to_send:
            try:
                for line in lines_to_send:
                    self.questdb._send_line(line)

                self.questdb.wait_for_commit(timeout=2)
                self.logger.debug(f"Successfully stored {stored_count} features for {symbol} {interval}")
                self.metrics.record_db_write(table_name, 'success')
            except Exception as e:
                self.logger.error(f"Failed to store batch of {stored_count} features for {symbol} {interval}: {e}",
                                  exc_info=True)
                self.metrics.record_db_write(table_name, 'failure')

    async def process_realtime_updates(self):
        if not self.redis:
            self.logger.error("Redis not initialized. Cannot process realtime updates.")
            return

        channels = [f"kline:{symbol}:{interval}" for symbol in self.symbols for interval in self.intervals]
        if not channels:
            self.logger.warning("No channels to subscribe for realtime updates.")
            return

        self.redis.subscribe(channels)
        self.logger.info(f"Subscribed to Redis channels: {channels}")

        while self.is_running:
            try:
                message = self.redis.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    await self._process_kline_update(message)
            except Exception as e:
                self.logger.error(f"Realtime processing error: {e}", exc_info=True)

            await asyncio.sleep(0.001)

    async def _process_kline_update(self, message: Dict[str, Any]):
        try:
            channel_bytes = message.get('channel')
            data = message.get('data')

            if not channel_bytes or not data:
                self.logger.warning(f"Received incomplete message: {message}")
                return

            channel = channel_bytes.decode('utf-8') if isinstance(channel_bytes, bytes) else channel_bytes

            if isinstance(data, dict):
                kline_update = data
            elif isinstance(data, bytes):
                kline_update = json.loads(data.decode('utf-8'))
            elif isinstance(data, str):
                kline_update = json.loads(data)
            else:
                self.logger.error(f"Unexpected data type in message: {type(data)}")
                return

            _, symbol, interval = channel.split(':')

            kline_timestamp_ms = int(kline_update['timestamp'] * 1000)
            kline_timestamp_ns = kline_timestamp_ms * 1_000_000

            if kline_timestamp_ns <= self.last_processed[symbol][interval]:
                return

            features_table = f"features_{interval}"
            if features_table in self.failed_tables and self.table_retry_count[
                features_table] >= self.max_table_retries:
                return

            lookback = self.config.get('feature_engine.lookback_periods', 300)
            df = self.questdb.get_klines_df(symbol, interval, limit=lookback)

            if df.empty or len(df) < 20:
                self.logger.debug(
                    f"Not enough data for realtime feature calculation for {symbol} {interval} at {pd.Timestamp(kline_timestamp_ns, unit='ns')}")
                return

            features_df = self._calculate_features(df, symbol, interval)

            if not features_df.empty:
                latest_features = features_df.iloc[-1:]
                if not latest_features.empty:
                    self.feature_buffer[symbol][interval].append(latest_features)
                    self.last_processed[symbol][interval] = kline_timestamp_ns

                    total_buffer_size = sum(
                        len(features_list)
                        for intervals_map in self.feature_buffer.values()
                        for features_list in intervals_map.values()
                    )

                    if total_buffer_size >= self.buffer_size:
                        await self.flush_buffer()

                    self.metrics.record_ws_message(symbol, 'feature_update')
                else:
                    self.logger.debug(
                        f"No features calculated for the exact timestamp {pd.Timestamp(kline_timestamp_ns, unit='ns')} for {symbol} {interval}")
            else:
                self.logger.debug(
                    f"Feature calculation returned empty for realtime update {symbol} {interval} at {pd.Timestamp(kline_timestamp_ns, unit='ns')}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from kline update: {data}, Error: {e}")
        except Exception as e:
            self.logger.error(f"Error processing kline update for {channel}: {e}", exc_info=True)

    async def flush_buffer(self):
        if not self.feature_buffer:
            return

        self.logger.info(
            f"Flushing feature buffer. Current size: {sum(len(fl) for il in self.feature_buffer.values() for fl in il.values())}")
        buffer_copy = defaultdict(lambda: defaultdict(list))
        for symbol, intervals_map in self.feature_buffer.items():
            for interval, features_list in intervals_map.items():
                if features_list:
                    buffer_copy[symbol][interval] = list(features_list)

        self.feature_buffer.clear()

        for symbol, intervals_map in buffer_copy.items():
            for interval, features_list in intervals_map.items():
                if not features_list:
                    continue
                try:
                    combined_df = pd.concat(features_list)
                    if not combined_df.empty:
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        await self._store_features_batch(symbol, interval, combined_df)
                except Exception as e:
                    self.logger.error(f"Failed to flush features for {symbol} {interval}: {e}", exc_info=True)
        self.logger.info("Feature buffer flushed.")

    async def periodic_flush(self):
        while self.is_running:
            await asyncio.sleep(self.flush_interval)
            if self.feature_buffer:
                self.logger.debug(
                    f"Periodic flush triggered. Buffer items: {sum(len(fl) for il in self.feature_buffer.values() for fl in il.values())}")
            await self.flush_buffer()

    async def quality_monitoring(self):
        while self.is_running:
            await asyncio.sleep(self.config.get('feature_engine.quality_check_interval', 600))
            try:
                for symbol in self.symbols:
                    for interval in ['1h', '1d']:
                        try:
                            features_table = f"features_{interval}"
                            if features_table in self.failed_tables and self.table_retry_count[
                                features_table] >= self.max_table_retries:
                                continue

                            features_df = self.questdb.get_features_df(symbol, interval, limit=1000)

                            if not features_df.empty:
                                quality_report = self.quality_monitor.check_quality(features_df)

                                if self.redis:
                                    self.redis.set_json(
                                        f"quality:{symbol}:{interval}",
                                        quality_report,
                                        expire=int(self.config.get('feature_engine.quality_check_interval', 600) * 1.5)
                                    )

                                overall_score = quality_report.get('overall_quality_score', 100.0)
                                if overall_score < 70:
                                    self.logger.warning(
                                        f"Low data quality score for {symbol} {interval}: {overall_score:.2f}. Report: {quality_report.get('recommendations')}")
                                else:
                                    self.logger.info(f"Data quality score for {symbol} {interval}: {overall_score:.2f}")
                            else:
                                self.logger.info(f"No features data found for quality check: {symbol} {interval}")
                        except Exception as e:
                            self.logger.warning(f"Data quality check failed for {symbol} {interval}: {e}",
                                                exc_info=True)
            except Exception as e:
                self.logger.error(f"Quality monitoring loop error: {e}", exc_info=True)

    def _get_interval_ms(self, interval: str) -> int:
        val = interval[:-1]
        unit = interval[-1]
        multiplier = 1
        if unit == 'm':
            multiplier = 60 * 1000
        elif unit == 'h':
            multiplier = 60 * 60 * 1000
        elif unit == 'd':
            multiplier = 24 * 60 * 60 * 1000

        return int(val) * multiplier

    async def start(self):
        self.is_running = True

        await self.initialize()

        if self.data_initialized:
            self.tasks.append(asyncio.create_task(self.process_realtime_updates()))
            self.tasks.append(asyncio.create_task(self.periodic_flush()))
            self.tasks.append(asyncio.create_task(self.quality_monitoring()))
            self.logger.info("Feature engine service started with all tasks.")
        else:
            self.logger.error(
                "Feature engine service started but data initialization failed. Realtime processing might be impacted.")

    async def stop(self):
        self.logger.info("Stopping feature engine service...")
        self.is_running = False

        for task in self.tasks:
            if not task.done():
                task.cancel()

        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.logger.info("All async tasks cancelled.")

        await self.flush_buffer()
        self.logger.info("Final buffer flush completed.")

        if self.questdb:
            self.questdb.close()
            self.logger.info("QuestDB connection closed.")
        if self.redis:
            self.redis.close()
            self.logger.info("Redis connection closed.")

        self.logger.info("Feature engine service stopped gracefully.")


async def main():
    service = FeatureEngineService()

    loop = asyncio.get_event_loop()

    def signal_handler(sig, frame):
        print(f"Signal {sig} received, initiating shutdown...")
        if not service.is_running:
            return
        loop.create_task(service.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await service.start()
        while service.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main, stopping...")
    except Exception as e:
        service.logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        if service.is_running:
            service.logger.info("Main function exiting, ensuring service stop...")
            await service.stop()
        print("Service shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())