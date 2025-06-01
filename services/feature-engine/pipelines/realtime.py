import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


class RealtimeFeaturePipeline:
    def __init__(self):
        self.feature_cache = {}
        self.computation_times = deque(maxlen=1000)
        self.optimization_enabled = True

    def process_batch(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        start_time = time.time()

        df = self._validate_and_clean(df)

        if self.optimization_enabled and len(df) > 50:
            features = self._compute_features_optimized(df)
        else:
            features = self._compute_features_standard(df)

        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)

        return features

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df.replace([np.inf, -np.inf], np.nan)

        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].ffill().bfill()

        df['volume'] = df['volume'].fillna(0)

        return df

    def _compute_features_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values

        for period in [10, 20, 50, 100, 200]:
            if len(df) >= period:
                features[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)

        if len(df) >= 14:
            features['rsi_14'] = talib.RSI(close_prices, timeperiod=14)

        if len(df) >= 14:
            features['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

        return features

    def _compute_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values

        features = pd.DataFrame(index=df.index)

        periods = [10, 20, 50, 100, 200]
        for period in periods:
            if len(df) >= period:
                features[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)

        if len(df) >= 14:
            features['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
            features['atr_14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

        return features

    def get_performance_stats(self) -> Dict[str, float]:
        if not self.computation_times:
            return {}

        times = list(self.computation_times)
        return {
            'mean_computation_time': np.mean(times),
            'max_computation_time': np.max(times),
            'min_computation_time': np.min(times),
            'p95_computation_time': np.percentile(times, 95),
            'total_processed': len(times)
        }