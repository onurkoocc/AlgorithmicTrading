import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
from collections import defaultdict
import pandas as pd
import numpy as np
import pytz

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
        self.buffer_size = self.config.get('feature_engine.buffer_size', 5000)
        self.flush_interval = self.config.get('feature_engine.flush_interval', 10)

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
        self.logger.info("Initializing feature engine service")

        if not self._init_connections():
            raise Exception("Failed to initialize database connections")

        self._create_feature_tables()

        await self._calculate_historical_features()

    def _create_feature_tables(self):
        for interval in self.intervals:
            table_name = f"features_{interval}"

            drop_query = f"DROP TABLE IF EXISTS {table_name}"
            try:
                self.questdb.execute_query(drop_query)
                self.logger.info(f"Dropped existing table {table_name}")
            except:
                pass

            create_query = f"""
                CREATE TABLE {table_name} (
                    symbol SYMBOL capacity 256 CACHE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    sma_10 DOUBLE,
                    sma_20 DOUBLE,
                    sma_50 DOUBLE,
                    sma_100 DOUBLE,
                    sma_200 DOUBLE,
                    ema_10 DOUBLE,
                    ema_20 DOUBLE,
                    ema_50 DOUBLE,
                    ema_100 DOUBLE,
                    ema_200 DOUBLE,
                    rsi_14 DOUBLE,
                    rsi_21 DOUBLE,
                    macd DOUBLE,
                    macd_signal DOUBLE,
                    macd_hist DOUBLE,
                    bb_upper DOUBLE,
                    bb_middle DOUBLE,
                    bb_lower DOUBLE,
                    bb_width DOUBLE,
                    bb_percent DOUBLE,
                    atr_14 DOUBLE,
                    atr_21 DOUBLE,
                    adx_14 DOUBLE,
                    plus_di DOUBLE,
                    minus_di DOUBLE,
                    cci_20 DOUBLE,
                    stoch_k DOUBLE,
                    stoch_d DOUBLE,
                    williams_r DOUBLE,
                    mfi_14 DOUBLE,
                    obv DOUBLE,
                    vwap DOUBLE,
                    pivot DOUBLE,
                    resistance_1 DOUBLE,
                    resistance_2 DOUBLE,
                    resistance_3 DOUBLE,
                    support_1 DOUBLE,
                    support_2 DOUBLE,
                    support_3 DOUBLE,
                    volume_sma_20 DOUBLE,
                    volume_ratio DOUBLE,
                    price_change_1 DOUBLE,
                    price_change_5 DOUBLE,
                    price_change_10 DOUBLE,
                    price_change_20 DOUBLE,
                    log_return_1 DOUBLE,
                    log_return_5 DOUBLE,
                    log_return_10 DOUBLE,
                    log_return_20 DOUBLE,
                    volatility_20 DOUBLE,
                    volatility_50 DOUBLE,
                    sharpe_ratio_20 DOUBLE,
                    high_low_ratio DOUBLE,
                    close_open_ratio DOUBLE,
                    upper_shadow DOUBLE,
                    lower_shadow DOUBLE,
                    body_size DOUBLE,
                    is_bullish_candle BOOLEAN,
                    trend_strength DOUBLE,
                    momentum_score DOUBLE,
                    volume_momentum DOUBLE,
                    price_position DOUBLE,
                    market_regime LONG,
                    avg_spread DOUBLE,
                    spread_volatility DOUBLE,
                    relative_spread DOUBLE,
                    spread_momentum DOUBLE,
                    buy_pressure DOUBLE,
                    sell_pressure DOUBLE,
                    order_flow_imbalance DOUBLE,
                    volume_weighted_buy_pressure DOUBLE,
                    vwap_deviation_5 DOUBLE,
                    vwap_deviation_10 DOUBLE,
                    vwap_deviation_20 DOUBLE,
                    vwap_deviation_50 DOUBLE,
                    volume_concentration DOUBLE,
                    avg_trade_size DOUBLE,
                    trade_size_momentum DOUBLE,
                    price_impact DOUBLE,
                    kyle_lambda DOUBLE,
                    amihud_illiquidity DOUBLE,
                    volume_turnover DOUBLE,
                    effective_spread DOUBLE,
                    realized_volatility DOUBLE,
                    depth_imbalance DOUBLE,
                    feature_version LONG,
                    timestamp TIMESTAMP
                ) timestamp(timestamp) PARTITION BY DAY WAL;
            """

            try:
                self.questdb.execute_query(create_query)
                self.logger.info(f"Created features table {table_name}")

                time.sleep(2)

            except Exception as e:
                self.logger.error(f"Failed to create features table {table_name}: {e}")
                raise

    async def _calculate_historical_features(self):
        self.logger.info("Calculating features for historical data")

        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    max_rows = 5000 if interval in ['1m', '3m', '5m'] else 10000
                    df = self.questdb.get_klines_df(symbol, interval, limit=max_rows)

                    if df.empty:
                        continue

                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                    batch_size = 1000
                    total_rows = len(df)

                    for start_idx in range(0, total_rows, batch_size):
                        end_idx = min(start_idx + batch_size, total_rows)

                        lookback_start = max(0, start_idx - 200)
                        batch_df = df.iloc[lookback_start:end_idx].copy()

                        features_df = self._calculate_all_features(batch_df, symbol, interval)

                        if not features_df.empty:
                            features_to_store = features_df.iloc[-(end_idx - start_idx):].copy()
                            self._store_features(symbol, interval, features_to_store)

                        await asyncio.sleep(0.5)

                    self.logger.info(f"Completed historical features for {symbol} {interval}")
                    await asyncio.sleep(2)

                except Exception as e:
                    self.logger.error(f"Error calculating historical features for {symbol} {interval}: {e}")

    def _calculate_all_features(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        features_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

        try:
            technical_features = self.technical_calculator.calculate(df)
            features_df = pd.concat([features_df, technical_features], axis=1)
        except Exception as e:
            self.logger.warning(f"Error calculating technical features: {e}")

        try:
            custom_features = self.custom_calculator.calculate(df)
            features_df = pd.concat([features_df, custom_features], axis=1)
        except Exception as e:
            self.logger.warning(f"Error calculating custom features: {e}")

        try:
            microstructure_features = self.microstructure_calculator.calculate(df)
            features_df = pd.concat([features_df, microstructure_features], axis=1)
        except Exception as e:
            self.logger.warning(f"Error calculating microstructure features: {e}")

        features_df['symbol'] = symbol
        features_df['feature_version'] = 1

        features_df = features_df.replace([np.inf, -np.inf], np.nan)

        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].ffill().bfill()

        return features_df

    def _store_features(self, symbol: str, interval: str, features_df: pd.DataFrame):
        table_name = f"features_{interval}"

        if features_df.index.name != 'timestamp':
            features_df = features_df.reset_index(names=['timestamp'])
        else:
            features_df = features_df.reset_index()

        records = features_df.to_dict('records')

        batch_size = 50
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            for record in batch:
                if 'timestamp' not in record:
                    self.logger.error(f"Missing timestamp in record for {symbol} {interval}")
                    continue

                timestamp_val = record['timestamp']
                if pd.isna(timestamp_val):
                    self.logger.warning(f"NaN timestamp for {symbol} {interval}, skipping record")
                    continue

                if isinstance(timestamp_val, pd.Timestamp):
                    timestamp_ns = int(timestamp_val.timestamp() * 1_000_000_000)
                elif isinstance(timestamp_val, (int, float)):
                    timestamp_ns = int(timestamp_val * 1_000_000_000)
                else:
                    timestamp_ns = int(pd.Timestamp(timestamp_val).timestamp() * 1_000_000_000)

                numeric_fields = []
                for key, value in record.items():
                    if key in ['symbol', 'timestamp']:
                        continue

                    if pd.isna(value):
                        value = 0.0

                    if isinstance(value, bool):
                        numeric_fields.append(f"{key}={'true' if value else 'false'}")
                    elif isinstance(value, (int, float)):
                        if key in ['market_regime', 'feature_version']:
                            numeric_fields.append(f"{key}={int(value)}i")
                        else:
                            numeric_fields.append(f"{key}={float(value)}")

                line = f"{table_name},symbol={symbol} {','.join(numeric_fields)} {timestamp_ns}"

                self.questdb._send_line(line)

            if i + batch_size < len(records):
                time.sleep(0.05)

    async def process_realtime_updates(self):
        channels = []
        for symbol in self.symbols:
            for interval in self.intervals:
                channels.append(f"kline:{symbol}:{interval}")

        self.redis.subscribe(channels)

        while self.is_running:
            try:
                message = self.redis.get_message(timeout=1.0)

                if message and message['type'] == 'message':
                    await self._process_kline_update(message)

            except Exception as e:
                self.logger.error(f"Error processing realtime update: {e}")

            await asyncio.sleep(0.001)

    async def _process_kline_update(self, message: Dict[str, Any]):
        try:
            channel = message['channel'].decode('utf-8')
            parts = channel.split(':')
            symbol = parts[1]
            interval = parts[2]

            kline_data = message['data']

            lookback_periods = max(200, self.config.get('feature_engine.lookback_periods', 500))

            df = self.questdb.get_klines_df(symbol, interval, limit=lookback_periods)

            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                new_timestamp = pd.Timestamp(kline_data['timestamp'], unit='s')
                if new_timestamp.tz is not None:
                    new_timestamp = new_timestamp.tz_localize(None)

                new_row = pd.DataFrame([{
                    'open': kline_data['open'],
                    'high': kline_data['high'],
                    'low': kline_data['low'],
                    'close': kline_data['close'],
                    'volume': kline_data['volume'],
                    'quote_volume': kline_data.get('quote_volume', 0),
                    'trades': kline_data.get('trades', 0),
                    'taker_buy_volume': kline_data.get('taker_buy_volume', 0),
                    'taker_buy_quote_volume': kline_data.get('taker_buy_quote_volume', 0)
                }], index=[new_timestamp])

                df = pd.concat([df, new_row])
                df = df.iloc[-lookback_periods:]

                features_df = self._calculate_all_features(df, symbol, interval)

                if not features_df.empty:
                    latest_features = features_df.iloc[-1:].copy()
                    self.feature_buffer[symbol][interval].append(latest_features)

                    buffer_size = sum(
                        len(features_list)
                        for symbol_intervals in self.feature_buffer.values()
                        for features_list in symbol_intervals.values()
                    )

                    if buffer_size >= self.buffer_size:
                        await self.flush_buffer()

                    latest_dict = latest_features.reset_index().to_dict('records')[0]
                    if 'timestamp' in latest_dict:
                        if isinstance(latest_dict['timestamp'], pd.Timestamp):
                            latest_dict['timestamp'] = latest_dict['timestamp'].isoformat()

                    self.redis.publish(f"features:{symbol}:{interval}", latest_dict)

            self.metrics.record_ws_message(symbol, 'feature_update')

        except Exception as e:
            self.logger.error(f"Error processing kline update: {e}")

    async def flush_buffer(self):
        if not self.feature_buffer:
            return

        for symbol, intervals in self.feature_buffer.items():
            for interval, features_list in intervals.items():
                if not features_list:
                    continue

                try:
                    combined_df = pd.concat(features_list)
                    self._store_features(symbol, interval, combined_df)

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
                    for interval in self.intervals:
                        try:
                            features_df = self.questdb.get_features_df(symbol, interval, limit=500)

                            if not features_df.empty:
                                quality_metrics = self.quality_monitor.check_quality(features_df)

                                self.redis.set_json(
                                    f"quality:{symbol}:{interval}",
                                    quality_metrics,
                                    expire=3600
                                )

                                if 'missing_analysis' in quality_metrics and 'missing_percentage' in quality_metrics[
                                    'missing_analysis']:
                                    if quality_metrics['missing_analysis']['missing_percentage'] > 10:
                                        self.logger.warning(
                                            f"High missing data for {symbol} {interval}: "
                                            f"{quality_metrics['missing_analysis']['missing_percentage']:.2f}%"
                                        )
                        except Exception as e:
                            self.logger.warning(f"Skipping quality check for {symbol} {interval}: {e}")

            except Exception as e:
                self.logger.error(f"Quality monitoring error: {e}")

            await asyncio.sleep(300)

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