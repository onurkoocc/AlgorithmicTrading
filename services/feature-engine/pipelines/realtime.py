import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
from numba import jit


class RealtimeFeaturePipeline:
    def __init__(self):
        self.feature_cache = {}
        self.computation_times = deque(maxlen=1000)
        self.optimization_enabled = True

    def process_batch(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        start_time = time.time()

        df = self._validate_and_clean(df)

        features = self._compute_features_optimized(df) if self.optimization_enabled else self._compute_features_standard(df)

        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)

        return features

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")

        df = df.replace([np.inf, -np.inf], np.nan)

        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].ffill().bfill()

        df['volume'] = df['volume'].fillna(0)

        return df

    def _compute_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        arrays = {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values
        }

        sma_values = self._compute_sma_vectorized(arrays['close'], [10, 20, 50, 100, 200])
        ema_values = self._compute_ema_vectorized(arrays['close'], [10, 20, 50, 100, 200])

        features = pd.DataFrame(index=df.index)

        for i, period in enumerate([10, 20, 50, 100, 200]):
            features[f'sma_{period}'] = sma_values[i]
            features[f'ema_{period}'] = ema_values[i]

        features['rsi_14'] = self._compute_rsi_optimized(arrays['close'], 14)
        features['atr_14'] = self._compute_atr_optimized(arrays['high'], arrays['low'], arrays['close'], 14)

        return features

    def _compute_features_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        for period in [10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        return features

    @staticmethod
    @jit(nopython=True)
    def _compute_sma_vectorized(prices: np.ndarray, periods: List[int]) -> List[np.ndarray]:
        results = []
        for period in periods:
            sma = np.full_like(prices, np.nan)
            for i in range(period - 1, len(prices)):
                sma[i] = np.mean(prices[i - period + 1:i + 1])
            results.append(sma)
        return results

    @staticmethod
    @jit(nopython=True)
    def _compute_ema_vectorized(prices: np.ndarray, periods: List[int]) -> List[np.ndarray]:
        results = []
        for period in periods:
            ema = np.full_like(prices, np.nan)
            if len(prices) > 0:
                ema[0] = prices[0]
                alpha = 2.0 / (period + 1)
                for i in range(1, len(prices)):
                    ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
            results.append(ema)
        return results

    @staticmethod
    @jit(nopython=True)
    def _compute_rsi_optimized(prices: np.ndarray, period: int) -> np.ndarray:
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.full_like(prices, np.nan)
        rsi[period] = 100 - 100 / (1 + rs)

        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100 - 100 / (1 + rs)

        return rsi

    @staticmethod
    @jit(nopython=True)
    def _compute_atr_optimized(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        tr = np.full_like(high, np.nan)
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.full_like(high, np.nan)
        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

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