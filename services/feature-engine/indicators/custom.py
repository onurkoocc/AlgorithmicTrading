import pandas as pd
import numpy as np
from typing import Optional


class CustomIndicatorCalculator:
    def __init__(self):
        self.lookback_periods = [1, 5, 10, 20]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        features = self._calculate_price_changes(df, features)
        features = self._calculate_returns(df, features)
        features = self._calculate_volatility(df, features)
        features = self._calculate_candle_patterns(df, features)
        features = self._calculate_trend_features(df, features)
        features = self._calculate_momentum_features(df, features)
        features = self._calculate_market_regime(df, features)

        return features

    def _calculate_price_changes(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            shifted_close = df['close'].shift(period)
            features[f'price_change_{period}'] = np.where(
                shifted_close > 0,
                (df['close'] - shifted_close) / shifted_close,
                0
            )

        return features

    def _calculate_returns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            shifted_close = df['close'].shift(period)
            features[f'log_return_{period}'] = np.where(
                shifted_close > 0,
                np.log(df['close'] / shifted_close),
                0
            )

        return features

    def _calculate_volatility(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        shifted_close = df['close'].shift(1)
        returns = np.where(
            shifted_close > 0,
            np.log(df['close'] / shifted_close),
            0
        )
        returns_series = pd.Series(returns, index=df.index)

        features['volatility_20'] = returns_series.rolling(window=20).std() * np.sqrt(252)
        features['volatility_50'] = returns_series.rolling(window=50).std() * np.sqrt(252)

        returns_mean = returns_series.rolling(window=20).mean()
        returns_std = returns_series.rolling(window=20).std()
        features['sharpe_ratio_20'] = np.where(
            returns_std > 0,
            (returns_mean * 252) / (returns_std * np.sqrt(252)),
            0
        )

        return features

    def _calculate_candle_patterns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['high_low_ratio'] = np.where(
            df['low'] > 0,
            df['high'] / df['low'],
            1
        )

        features['close_open_ratio'] = np.where(
            df['open'] > 0,
            df['close'] / df['open'],
            1
        )

        hl_range = df['high'] - df['low']
        features['upper_shadow'] = np.where(
            hl_range > 0,
            (df['high'] - np.maximum(df['open'], df['close'])) / hl_range,
            0
        )

        features['lower_shadow'] = np.where(
            hl_range > 0,
            (np.minimum(df['open'], df['close']) - df['low']) / hl_range,
            0
        )

        features['body_size'] = np.where(
            hl_range > 0,
            np.abs(df['close'] - df['open']) / hl_range,
            0
        )

        features['is_bullish_candle'] = df['close'] > df['open']

        return features

    def _calculate_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()

        features['trend_strength'] = np.where(
            sma_200 > 0,
            (df['close'] - sma_200) / sma_200,
            0
        )

        ema_short = df['close'].ewm(span=12, adjust=False).mean()
        ema_long = df['close'].ewm(span=26, adjust=False).mean()
        features['momentum_score'] = np.where(
            ema_long > 0,
            (ema_short - ema_long) / ema_long,
            0
        )

        return features

    def _calculate_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        volume_sma_5 = df['volume'].rolling(window=5).mean()
        volume_sma_20 = df['volume'].rolling(window=20).mean()
        features['volume_momentum'] = np.where(
            volume_sma_20 > 0,
            volume_sma_5 / volume_sma_20,
            1
        )

        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        price_range = high_20 - low_20
        features['price_position'] = np.where(
            price_range > 0,
            (df['close'] - low_20) / price_range,
            0.5
        )

        return features

    def _calculate_market_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        shifted_close = df['close'].shift(1)
        returns = np.where(
            shifted_close > 0,
            np.log(df['close'] / shifted_close),
            0
        )
        returns_series = pd.Series(returns, index=df.index)

        volatility = returns_series.rolling(window=20).std()
        volatility_median = volatility.rolling(window=100).median()
        trend = df['close'].rolling(window=50).mean().diff(10)

        conditions = [
            (trend > 0) & (volatility < volatility_median),
            (trend > 0) & (volatility >= volatility_median),
            (trend <= 0) & (volatility < volatility_median),
            (trend <= 0) & (volatility >= volatility_median)
        ]

        values = [1, 2, 3, 4]

        features['market_regime'] = np.select(conditions, values, default=0)

        return features