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

    def _to_nanoseconds_epoch(self, ts_input: Any) -> Optional[int]:
        """
        Converts various timestamp inputs to a nanosecond epoch integer.
        - Strings are parsed by pandas.Timestamp.
        - datetime.datetime and pd.Timestamp objects are converted.
        - Numeric values are assumed to be epoch microseconds (common for QuestDB raw results).
        """
        if ts_input is None:
            return None

        try:
            if isinstance(ts_input, pd.Timestamp):
                return ts_input.value  # Already in nanoseconds
            if isinstance(ts_input, datetime):
                # Ensure UTC if timezone aware, then get value
                if ts_input.tzinfo is not None and ts_input.tzinfo.utcoffset(ts_input) is not None:
                    return pd.Timestamp(ts_input.astimezone(timezone.utc)).value
                else:  # Naive or already UTC
                    return pd.Timestamp(ts_input).value
            if isinstance(ts_input, str):
                return pd.Timestamp(ts_input).value
            if isinstance(ts_input, (int, float)):
                # Assumption: numeric input from QuestDB (if not string/datetime) is epoch microseconds.
                # This is a critical assumption based on common QuestDB behavior.
                # If QuestDB could return seconds or milliseconds as numbers, this needs adjustment
                # or more sophisticated heuristics (e.g., checking magnitude).
                return int(ts_input * 1000)  # Convert microseconds to nanoseconds

            self.logger.warning(f"Unhandled timestamp type for conversion: {type(ts_input)}. Value: {ts_input}")
            return None
        except Exception as e:
            self.logger.error(
                f"Error converting timestamp '{ts_input}' (type: {type(ts_input)}) to nanosecond epoch: {e}")
            return None

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

        existing_result = self.questdb.execute_query(existing_features_query)
        existing_feature_count = 0
        last_feature_ts = None

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
                existing_feature_count = 0  # Treat as if no valid last timestamp

        interval_ns = self._get_interval_ms(interval) * 1_000_000

        if last_feature_ts is not None and existing_feature_count > 0:
            # Check if features are mostly complete and up-to-date
            # The original logic `total_klines - 200` might be too strict if klines grow much faster than features
            # A time-based check is often better. If last_feature_ts is very close to max_klines_ts.
            if last_feature_ts >= (
                    max_klines_ts - interval_ns * 5):  # If last feature is within 5 intervals of max kline
                self.logger.info(
                    f"Features considered up to date for {symbol} {interval}. Last feature at {pd.Timestamp(last_feature_ts, unit='ns')}, max kline at {pd.Timestamp(max_klines_ts, unit='ns')}")
                # Still might need to process a few recent ones
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

        # Ensure start_ts does not exceed max_klines_ts
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
            # Define the window for fetching klines: [lookback_start_ts, chunk_end_ts_exclusive)
            # lookback_start_ts ensures enough data for indicators at current_ts
            lookback_start_ts = max(min_klines_ts, current_ts - (lookback * interval_ns))
            # chunk_end_ts_exclusive is the end of the current chunk we want to process features for, plus lookback
            # The actual klines to fetch should go up to current_ts + chunk_size * interval_ns
            # to calculate features for the points in [current_ts, current_ts + chunk_size * interval_ns)

            # End of the data points for which features will be calculated in this iteration
            processing_chunk_end_ts = min(current_ts + (chunk_size * interval_ns),
                                          max_klines_ts + interval_ns)  # Inclusive end for klines

            # The actual fetch range for QuestDB needs to cover up to processing_chunk_end_ts
            # and start from lookback_start_ts to have enough history.
            fetch_end_ts = processing_chunk_end_ts

            await asyncio.sleep(self.query_delay)

            try:
                # get_klines_by_timestamp_range should fetch [lookback_start_ts, fetch_end_ts)
                # Ensure your QuestDBConnector method handles this range correctly (end usually exclusive)
                df = self.questdb.get_klines_by_timestamp_range(
                    symbol, interval, lookback_start_ts, fetch_end_ts
                )

                if df.empty:
                    self.logger.warning(
                        f"Empty dataframe for {symbol} {interval} from {pd.Timestamp(lookback_start_ts, unit='ns')} to {pd.Timestamp(fetch_end_ts, unit='ns')}")
                    current_ts = fetch_end_ts  # Advance current_ts to the end of the attempted chunk
                    if current_ts <= start_ts and fetch_end_ts > start_ts:  # Stuck at the beginning
                        current_ts = fetch_end_ts
                    else:  # Normal advancement
                        current_ts = min(current_ts + chunk_size * interval_ns, max_klines_ts + interval_ns)

                    if current_ts == processing_chunk_end_ts and current_ts > max_klines_ts:  # check if we are past the end
                        break
                    continue

                self.logger.debug(
                    f"Processing {len(df)} klines for {symbol} {interval} (fetched range: {pd.Timestamp(lookback_start_ts, unit='ns')} to {pd.Timestamp(fetch_end_ts, unit='ns')})")

                features_df = self._calculate_features(df, symbol, interval)

                if features_df.empty:
                    self.logger.warning(
                        f"Empty features calculated for {symbol} {interval} for klines in range {df.index.min()} to {df.index.max() if not df.empty else 'N/A'}")
                    current_ts = fetch_end_ts
                    if current_ts <= start_ts and fetch_end_ts > start_ts:
                        current_ts = fetch_end_ts
                    else:
                        current_ts = min(current_ts + chunk_size * interval_ns, max_klines_ts + interval_ns)
                    if current_ts == processing_chunk_end_ts and current_ts > max_klines_ts:
                        break
                    continue

                # Select features that are for the current processing window (not the lookback part)
                # Timestamps in features_df are nanosecond-based from klines.
                # current_ts is the start of the new data we want to store.
                features_to_store = features_df[features_df.index >= pd.Timestamp(current_ts, unit='ns')]

                # Filter out features beyond max_klines_ts, as klines might be fetched slightly beyond due to chunking
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
                        f"No features to store for {symbol} {interval} for current_ts {pd.Timestamp(current_ts, unit='ns')} (features_df might be too short or all NaNs for this slice)")

                # Advance current_ts to the start of the next chunk
                # This should be the timestamp of the first kline *after* the current chunk
                if not features_to_store.empty:
                    # Advance current_ts based on the last stored feature's timestamp + interval
                    # This handles cases where chunk might not be full or processing_chunk_end_ts was too far
                    last_stored_ts_in_chunk = features_to_store.index.max().value
                    current_ts = last_stored_ts_in_chunk + interval_ns
                else:
                    # If nothing was stored, advance by the chunk size to avoid getting stuck
                    current_ts = min(current_ts + (chunk_size * interval_ns), max_klines_ts + interval_ns)

                if current_ts > max_klines_ts:  # Ensure we don't loop unnecessarily if current_ts went past max_klines_ts
                    break

            except Exception as e:
                self.logger.error(
                    f"Error processing chunk for {symbol} {interval} starting {pd.Timestamp(current_ts, unit='ns')}: {e}",
                    exc_info=True)
                await asyncio.sleep(5)  # Wait before retrying or moving to next chunk
                # Advance current_ts to avoid getting stuck on a problematic chunk, or implement more robust retry
                current_ts = min(current_ts + interval_ns,
                                 max_klines_ts + interval_ns)  # Advance by at least one interval
                if current_ts > max_klines_ts:
                    break
                continue

        self.logger.info(
            f"Completed historical features for {symbol} {interval}: {processed_count} features processed in this run.")

    def _calculate_features(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if df.empty or len(df) < 20:  # Minimum length for some basic indicators
            self.logger.debug(
                f"DataFrame for {symbol} {interval} is empty or too short ({len(df)} rows) for feature calculation.")
            return pd.DataFrame()

        df = df.sort_index()

        # Ensure index is timezone-naive (assuming UTC from QuestDB)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)

        features = pd.DataFrame(index=df.index)
        # Ensure base columns are present before trying to access them
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in base_cols:
            if col not in df.columns:
                self.logger.error(
                    f"Missing base column '{col}' in DataFrame for {symbol} {interval}. Columns: {df.columns.tolist()}")
                # Fill with NaN or handle as error; for now, let it proceed, join will handle missing cols
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
        features['feature_version'] = 2  # Example version

        features = features.replace([np.inf, -np.inf], np.nan)

        # Drop rows where ALL specified feature columns (indicators + ohlcv) are NaN.
        # This means if OHLCV is present, the row is kept even if all indicators are NaN.
        indicator_subset = [col for col in features.columns if col not in ['symbol', 'feature_version', 'timestamp']]
        # However, if a row in features_df has valid OHLCV but all indicators are NaN due to insufficient lookback for that specific point,
        # it should still be kept. The original dropna might be too aggressive if it includes OHLCV.
        # Let's ensure we only drop if *all* calculable features are NaN.
        # The original dropna was:
        # features = features.dropna(how='all', subset=[col for col in features.columns if col not in ['symbol', 'feature_version']])
        # This is generally fine if OHLCV are present.

        # A check to see if any actual indicator values were produced
        # If features DataFrame (after joins) only contains OHLCV, symbol, version, and all other indicator columns are entirely NaN,
        # it might indicate an issue upstream or that no indicators could be calculated.
        # However, for partial calculations (e.g. start of series), many NaNs are expected.

        return features

    async def _store_features_batch(self, symbol: str, interval: str, features_df: pd.DataFrame):
        if features_df.empty:
            return

        table_name = f"features_{interval}"

        # Make a copy to avoid SettingWithCopyWarning if features_df is a slice
        features_df = features_df.copy()

        # Ensure 'timestamp' column exists from index for QuestDB line protocol
        if 'timestamp' not in features_df.columns:
            if features_df.index.name == 'timestamp':
                features_df.reset_index(inplace=True)
            else:  # If index is not named 'timestamp', create it from index
                features_df['timestamp'] = features_df.index
                features_df.reset_index(drop=True, inplace=True)  # remove the original index if it's not timestamp

        records = features_df.to_dict('records')
        stored_count = 0
        lines_to_send = []

        for record in records:
            timestamp_val = record.get('timestamp')
            if pd.isna(timestamp_val):  # Should not happen if index was valid
                continue

            # Ensure timestamp is nanosecond epoch integer for QuestDB
            if isinstance(timestamp_val, pd.Timestamp):
                timestamp_ns = timestamp_val.value
            elif isinstance(timestamp_val, (int, float)):  # Already epoch
                timestamp_ns = int(timestamp_val)
            else:  # Try to parse if it's a string or other type (should be pd.Timestamp by now)
                timestamp_ns = pd.Timestamp(timestamp_val).value

            line_tags = f"symbol={symbol}"
            line_fields = []

            for key, value in record.items():
                if key in ['symbol', 'timestamp'] or pd.isna(value):
                    continue

                # QuestDB line protocol field formatting
                if isinstance(value, bool):
                    line_fields.append(f"{key}={'T' if value else 'F'}")  # Or true/false
                elif isinstance(value, (int, np.integer)):
                    if key == 'feature_version':  # Explicitly an integer type in schema
                        line_fields.append(f"{key}={int(value)}i")
                    else:  # Other integers can be sent as floats or integers
                        line_fields.append(f"{key}={int(value)}i")
                elif isinstance(value, (float, np.floating)):
                    line_fields.append(f"{key}={float(value)}")
                elif isinstance(value, str):  # String fields need quotes
                    line_fields.append(f"{key}=\"{value}\"")

            if line_fields:
                lines_to_send.append(f"{table_name},{line_tags} {','.join(line_fields)} {timestamp_ns}")
                stored_count += 1

        if lines_to_send:
            try:
                # Assuming _send_line can handle a list or you have a batch send method
                for line in lines_to_send:
                    self.questdb._send_line(line)  # If _send_line sends immediately and waits for commit
                # Or, if QuestDBConnector has a batch send: self.questdb.send_lines(lines_to_send)

                self.questdb.wait_for_commit(timeout=2)  # Wait for commit after sending lines
                self.logger.debug(f"Successfully stored {stored_count} features for {symbol} {interval}")
                self.metrics.record_db_write(table_name, 'success', count=stored_count)
            except Exception as e:
                self.logger.error(f"Failed to store batch of {stored_count} features for {symbol} {interval}: {e}",
                                  exc_info=True)
                self.metrics.record_db_write(table_name, 'failure', count=stored_count)

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
            except redis.exceptions.ConnectionError as e:
                self.logger.error(f"Redis connection error in realtime processing: {e}. Attempting to reconnect...")
                self._init_connections()  # Try to re-establish connection
                if self.redis and self.redis.ping():
                    self.redis.subscribe(channels)  # Re-subscribe
                    self.logger.info("Reconnected to Redis and re-subscribed to channels.")
                else:
                    await asyncio.sleep(5)  # Wait before retrying connection
            except Exception as e:
                self.logger.error(f"Realtime processing error: {e}", exc_info=True)

            await asyncio.sleep(0.001)  # Small sleep to prevent tight loop

    async def _process_kline_update(self, message: Dict[str, Any]):
        try:
            channel_bytes = message.get('channel')
            data_bytes = message.get('data')

            if not channel_bytes or not data_bytes:
                self.logger.warning(f"Received incomplete message: {message}")
                return

            channel = channel_bytes.decode('utf-8')
            # Assuming data is JSON string, needs to be parsed
            kline_data_str = data_bytes.decode('utf-8')
            kline_update = json.loads(kline_data_str)  # Use json library

            _, symbol, interval = channel.split(':')

            # kline_update should be a dict like {'timestamp': ..., 'open': ..., ...}
            kline_timestamp_ms = int(kline_update['t'])  # Assuming 't' is kline start time in ms
            kline_timestamp_ns = kline_timestamp_ms * 1_000_000

            if kline_timestamp_ns <= self.last_processed[symbol][interval]:
                return  # Already processed or older data

            lookback = self.config.get('feature_engine.lookback_periods', 300)
            # Fetch klines ending at the current kline's timestamp to ensure it's included
            # The end timestamp for get_klines_df should be inclusive or slightly after
            df = self.questdb.get_klines_df(symbol, interval, limit=lookback, end_timestamp_ns=kline_timestamp_ns)

            if df.empty or len(df) < 20:
                self.logger.debug(
                    f"Not enough data for realtime feature calculation for {symbol} {interval} at {pd.Timestamp(kline_timestamp_ns, unit='ns')}")
                return

            features_df = self._calculate_features(df, symbol, interval)

            if not features_df.empty:
                # Get features for the specific kline that triggered the update
                latest_features = features_df[features_df.index == pd.Timestamp(kline_timestamp_ns, unit='ns')]
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
            self.logger.error(f"Failed to decode JSON from kline update: {data_bytes}, Error: {e}")
        except Exception as e:
            self.logger.error(f"Error processing kline update for {message.get('channel')}: {e}", exc_info=True)

    async def flush_buffer(self):
        if not self.feature_buffer:
            return

        self.logger.info(
            f"Flushing feature buffer. Current size: {sum(len(fl) for il in self.feature_buffer.values() for fl in il.values())}")
        # Create a copy of items to iterate over, as buffer might be modified
        buffer_copy = defaultdict(lambda: defaultdict(list))
        for symbol, intervals_map in self.feature_buffer.items():
            for interval, features_list in intervals_map.items():
                if features_list:
                    buffer_copy[symbol][interval] = list(features_list)  # Deep copy list

        self.feature_buffer.clear()  # Clear original buffer immediately

        for symbol, intervals_map in buffer_copy.items():
            for interval, features_list in intervals_map.items():
                if not features_list:
                    continue
                try:
                    combined_df = pd.concat(features_list)
                    if not combined_df.empty:
                        # Deduplicate in case same feature got added multiple times before flush
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        await self._store_features_batch(symbol, interval, combined_df)
                except Exception as e:
                    self.logger.error(f"Failed to flush features for {symbol} {interval}: {e}", exc_info=True)
                    # Optionally, add back to buffer or handle failed data
        self.logger.info("Feature buffer flushed.")

    async def periodic_flush(self):
        while self.is_running:
            await asyncio.sleep(self.flush_interval)
            if self.feature_buffer:  # Only log if there's something to potentially flush
                self.logger.debug(
                    f"Periodic flush triggered. Buffer items: {sum(len(fl) for il in self.feature_buffer.values() for fl in il.values())}")
            await self.flush_buffer()

    async def quality_monitoring(self):
        while self.is_running:
            await asyncio.sleep(self.config.get('feature_engine.quality_monitor_interval', 600))
            try:
                for symbol in self.symbols:
                    for interval in ['1h', '1d']:  # Monitor key intervals
                        try:
                            # Fetch a decent sample, e.g., last 1000 points
                            features_df = self.questdb.get_features_df(symbol, interval, limit=1000)

                            if not features_df.empty:
                                quality_report = self.quality_monitor.check_quality(features_df)

                                if self.redis:
                                    self.redis.set_json(
                                        f"quality_report:{symbol}:{interval}",
                                        quality_report,
                                        expire=int(
                                            self.config.get('feature_engine.quality_monitor_interval', 600) * 1.5)
                                        # Expire slightly longer than interval
                                    )

                                overall_score = quality_report.get('overall_quality_score', 100.0)
                                if overall_score < 70:
                                    self.logger.warning(
                                        f"Low data quality score for {symbol} {interval}: {overall_score:.2f}. Report: {quality_report.get('recommendations')}"
                                    )
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
        # Standard interval to millisecond mapping
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

        await self.initialize()  # This now includes historical calculation

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

        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.logger.info("All async tasks cancelled.")

        # Final flush of any remaining data
        await self.flush_buffer()
        self.logger.info("Final buffer flush completed.")

        if self.questdb:
            self.questdb.close()
            self.logger.info("QuestDB connection closed.")
        if self.redis:
            self.redis.close()  # Assuming redis connector has a close method
            self.logger.info("Redis connection closed.")

        self.logger.info("Feature engine service stopped gracefully.")


# Required for _process_kline_update if not already imported
import json
# Required for _to_nanoseconds_epoch if using timezone.utc
from datetime import timezone


async def main():
    service = FeatureEngineService()

    loop = asyncio.get_event_loop()

    def signal_handler(sig, frame):
        print(f"Signal {sig} received, initiating shutdown...")
        if not service.is_running:  # Already stopping
            return
        # Schedule stop() to be called by the loop
        loop.create_task(service.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await service.start()
        # Keep the main coroutine alive until stop() is called
        while service.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:  # Should be caught by signal_handler now
        print("KeyboardInterrupt caught in main, stopping...")
    except Exception as e:
        service.logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        # Ensure stop is called if not already handled by signal
        if service.is_running:  # If not stopped by signal handler or normal exit
            service.logger.info("Main function exiting, ensuring service stop...")
            await service.stop()
        print("Service shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
