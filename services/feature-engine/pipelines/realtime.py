import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
from numba import jit, float64, int64


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

        for period in [10, 20, 50, 100, 200]:
            if len(df) >= period:
                features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        if len(df) >= 14:
            features['rsi_14'] = self._calculate_rsi(df['close'].values, 14)

        if len(df) >= 14:
            features['atr_14'] = self._calculate_atr(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                14
            )

        return features

    def _compute_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values

        features = pd.DataFrame(index=df.index)

        periods = np.array([10, 20, 50, 100, 200], dtype=np.int64)
        sma_results = _compute_sma_vectorized(close_prices, periods)
        ema_results = _compute_ema_vectorized(close_prices, periods)

        for i, period in enumerate(periods):
            if len(df) >= period:
                features[f'sma_{period}'] = sma_results[i]
                features[f'ema_{period}'] = ema_results[i]

        if len(df) >= 14:
            features['rsi_14'] = _compute_rsi_optimized(close_prices, 14)
            features['atr_14'] = _compute_atr_optimized(high_prices, low_prices, close_prices, 14)

        return features

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        rsi = np.full(len(prices), np.nan)

        if down != 0:
            rs = up / down
            rsi[period] = 100 - 100 / (1 + rs)
        else:
            rsi[period] = 100

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

            if down != 0:
                rs = up / down
                rsi[i] = 100 - 100 / (1 + rs)
            else:
                rsi[i] = 100

        return rsi

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        if len(high) < period:
            return np.full(len(high), np.nan)

        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.full(len(high), np.nan)
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


@jit(nopython=True)
def _compute_sma_vectorized(prices: np.ndarray, periods: np.ndarray) -> List[np.ndarray]:
    results = []
    for period in periods:
        sma = np.full(len(prices), np.nan, dtype=float64)
        if len(prices) >= period:
            for i in range(period - 1, len(prices)):
                sma[i] = np.mean(prices[i - period + 1:i + 1])
        results.append(sma)
    return results


@jit(nopython=True)
def _compute_ema_vectorized(prices: np.ndarray, periods: np.ndarray) -> List[np.ndarray]:
    results = []
    for period in periods:
        ema = np.full(len(prices), np.nan, dtype=float64)
        if len(prices) > 0:
            ema[0] = prices[0]
            alpha = 2.0 / (period + 1)
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        results.append(ema)
    return results


@jit(nopython=True)
def _compute_rsi_optimized(prices: np.ndarray, period: int64) -> np.ndarray:
    if len(prices) < period + 1:
        return np.full(len(prices), np.nan, dtype=float64)

    rsi = np.full(len(prices), np.nan, dtype=float64)
    deltas = np.diff(prices)

    seed = deltas[:period]
    up = 0.0
    down = 0.0

    for delta in seed:
        if delta >= 0:
            up += delta
        else:
            down -= delta

    up /= period
    down /= period

    if down != 0:
        rs = up / down
        rsi[period] = 100 - 100 / (1 + rs)
    else:
        rsi[period] = 100

    for i in range(period, len(deltas)):
        delta = deltas[i]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        if down != 0:
            rs = up / down
            rsi[i + 1] = 100 - 100 / (1 + rs)
        else:
            rsi[i + 1] = 100

    return rsi


@jit(nopython=True)
def _compute_atr_optimized(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int64) -> np.ndarray:
    if len(high) < period:
        return np.full(len(high), np.nan, dtype=float64)

    tr = np.zeros(len(high), dtype=float64)
    tr[0] = high[0] - low[0]

    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.full(len(high), np.nan, dtype=float64)
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr